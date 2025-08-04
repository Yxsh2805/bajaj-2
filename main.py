import os
import time
import logging
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document as DocxDocument
import email
import io
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# RAG imports
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_TOKEN = "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8"

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

class TimeConstrainedVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_time_constrained(self, documents: List[Document]):
        """Time-constrained embedding - max 15-20 seconds"""
        logger.info(f"TIME CONSTRAINED: Processing {len(documents)} chunks (target: 15-20s)")
        
        start_time = time.time()
        
        def embed_fast(doc):
            """Fast embedding with minimal retry"""
            try:
                return self.embeddings.embed_query(doc.page_content)
            except Exception as e:
                # Only ONE retry for speed
                try:
                    time.sleep(0.05)  # Very short delay
                    return self.embeddings.embed_query(doc.page_content)
                except:
                    logger.warning(f"Fast fail: {str(e)[:30]}")
                    return None
        
        # 12 workers - aggressive for speed
        with ThreadPoolExecutor(max_workers=12) as executor:
            vectors = list(executor.map(embed_fast, documents))
        
        # Store results
        successful_count = 0
        for doc, vector in zip(documents, vectors):
            if vector is not None:
                self.documents.append(doc)
                self.vectors.append(vector)
                successful_count += 1
        
        embedding_time = time.time() - start_time
        success_rate = (successful_count / len(documents)) * 100
        logger.info(f"TIME CONSTRAINED: {embedding_time:.1f}s, {successful_count}/{len(documents)} chunks ({success_rate:.1f}% success)")
        
        # Time budget check
        if embedding_time > 20:
            logger.warning(f"Embedding exceeded 20s target: {embedding_time:.1f}s")
    
    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """Fast similarity search"""
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Fast cosine similarity
            similarities = []
            query_norm = np.linalg.norm(query_vector)
            
            for i, vector in enumerate(self.vectors):
                vector_norm = np.linalg.norm(vector)
                if query_norm > 0 and vector_norm > 0:
                    cos_sim = np.dot(query_vector, vector) / (query_norm * vector_norm)
                    similarities.append((cos_sim, i))
                else:
                    similarities.append((0.0, i))
            
            # Sort and return top k
            similarities.sort(reverse=True)
            top_indices = [idx for _, idx in similarities[:k]]
            
            return [self.documents[i] for i in top_indices]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self.documents[:k] if len(self.documents) >= k else self.documents

def smart_document_loader(url: str) -> List[Document]:
    """Smart document loading with all pages but strategic chunking"""
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            
            if total_pages > 30:
                logger.warning(f"Document has {total_pages} pages - processing first 30")
                total_pages = 30
            
            # COMPLETE PROCESSING - all pages
            text = ""
            for i in range(total_pages):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text.strip():
                        text += f"{page_text}\n\n"  # Remove page markers to save space
                except Exception as e:
                    logger.warning(f"Error reading page {i+1}: {e}")
                    continue
            
            logger.info(f"SMART: Processed ALL {total_pages} pages")
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "pdf", "pages": total_pages})]
        
        elif url_lower.endswith('.docx') or 'wordprocessingml' in content_type:
            docx_file = io.BytesIO(response.content)
            doc = DocxDocument(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "docx"})]
        
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            return [Document(page_content=text[:120000], metadata={"source": url, "type": "html"})]
        
    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise

class TimeConstrainedRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing TIME CONSTRAINED RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=3200
            )

            # STRATEGIC chunking - larger chunks to reduce total count
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1750,  # LARGER chunks to reduce count
                chunk_overlap=175, # 10% overlap
                separators=["\n\n", "\n", ". ", " "]  # Fewer separators for speed
            )

            self.initialized = True
            logger.info("TIME CONSTRAINED RAG engine ready!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _time_constrained_query(self, vectorstore: TimeConstrainedVectorStore, query: str) -> str:
        """Time-efficient query"""
        docs = vectorstore.similarity_search(query, k=8)
        context = " ".join([doc.page_content for doc in docs])[:3200]
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Efficient system prompt
        system_content = """You are an expert insurance policy analyst.

INSTRUCTIONS:
- Input questions are separated by " | "
- Output answers MUST be separated by " | " in the same order
- Provide accurate answers based on the document content
- Include specific numbers, percentages, time periods, and conditions
- If information is not in the document, state "Information not available in provided document"
- Keep answers detailed but concise

CRITICAL: Separate each answer with " | " and maintain exact question order."""

        human_content = f"""Questions: {query}

Document: {context}

Provide detailed answers separated by " | " in the same order."""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        
        response = self.chat_model.invoke(messages)
        return response.content

    async def process_time_constrained(self, url: str, questions: List[str]) -> List[str]:
        """Time-constrained processing"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="30 second timeout exceeded")

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        total_start = time.time()
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("CACHED - INSTANT!")
        else:
            # Document loading
            docs = smart_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            # AGGRESSIVE CHUNK LIMITING for time constraint
            original_count = len(chunks)
            
            if len(chunks) > 100:  # HARD LIMIT: 75 chunks max
                # Smart selection strategy
                keep_first = int(len(chunks) * 0.45)  # 45% from start
                keep_middle = int(len(chunks) * 0.20) # 20% from middle  
                keep_last = int(len(chunks) * 0.35)   # 35% from end
                
                # Calculate middle section
                middle_start = len(chunks) // 2 - keep_middle // 2
                
                # Select chunks strategically
                selected_chunks = (
                    chunks[:keep_first] +
                    chunks[middle_start:middle_start + keep_middle] +
                    chunks[-keep_last:]
                )
                
                chunks = selected_chunks
                
                logger.info(f"CHUNK LIMIT: Reduced {original_count} → {len(chunks)} chunks for time constraint")
            
            # Ensure we don't exceed 75 chunks
            if len(chunks) > 100:
                chunks = chunks[:100]
                logger.info(f"HARD LIMIT: Truncated to 100 chunks")
            
            # Time-aware embedding
            embed_start = time.time()
            vectorstore = TimeConstrainedVectorStore(self.embeddings)
            vectorstore.add_documents_time_constrained(chunks)
            
            embed_time = time.time() - embed_start
            total_time_so_far = time.time() - total_start
            
            logger.info(f"Time budget: Embed {embed_time:.1f}s, Total {total_time_so_far:.1f}s, Remaining {30 - total_time_so_far:.1f}s")
            
            # Cache the vectorstore
            self.vectorstore_cache[url_hash] = vectorstore
        
        # Query phase
        batch_query = " | ".join(questions)
        
        query_start = time.time()
        response = self._time_constrained_query(vectorstore, batch_query)
        query_time = time.time() - query_start
        
        total_time = time.time() - total_start
        logger.info(f"TIME CONSTRAINED: Query {query_time:.1f}s, Total {total_time:.1f}s")
        
        # Fast answer parsing
        answers = []
        raw_splits = response.split(" | ")
        
        for split in raw_splits:
            cleaned = split.strip()
            if cleaned and len(cleaned) > 5:
                answers.append(cleaned)
        
        # Ensure correct count
        while len(answers) < len(questions):
            answers.append("Information not available in provided document.")
        
        return answers[:len(questions)]

# Global engine
rag_engine = TimeConstrainedRAGEngine()

def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    
    token = authorization.split("Bearer ")[-1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        rag_engine.initialize()
        logger.info("TIME CONSTRAINED RAG application ready")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="TIME CONSTRAINED RAG API", version="3.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Time-constrained processing - max 75 chunks, 15-20s embedding"""
    try:
        logger.info(f"TIME CONSTRAINED: {len(request.questions)} questions (75 chunk limit)")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        start_time = time.time()
        answers = await rag_engine.process_time_constrained(request.documents, request.questions)
        total_time = time.time() - start_time

        logger.info(f"TIME CONSTRAINED: Completed in {total_time:.1f}s")
        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cache_entries": len(rag_engine.vectorstore_cache),
        "mode": "time_constrained",
        "max_chunks": 100,
        "target_embed_time": "15-20_seconds",
        "embedding_provider": "Together.AI (Time Constrained)"
    }

@app.get("/")
async def root():
    return {"message": "TIME CONSTRAINED RAG API - Max 100 Chunks, 15-20s Embedding"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
