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

class Optimized65VectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_fast_65(self, documents: List[Document]):
        """Fast processing of up to 65 chunks for optimal coverage"""
        logger.info(f"FAST 65: Processing {len(documents)} chunks (target <30s)")
        
        start_time = time.time()
        
        def embed_reliable(doc):
            """Single-attempt reliable embedding"""
            try:
                return self.embeddings.embed_query(doc.page_content)
            except Exception as e:
                logger.warning(f"Embed failed: {str(e)[:30]}")
                return None
        
        # 6 workers for stability and speed balance
        with ThreadPoolExecutor(max_workers=6) as executor:
            vectors = list(executor.map(embed_reliable, documents))
        
        # Store successful results
        successful_count = 0
        for doc, vector in zip(documents, vectors):
            if vector is not None:
                self.documents.append(doc)
                self.vectors.append(vector)
                successful_count += 1
        
        embedding_time = time.time() - start_time
        success_rate = (successful_count / len(documents)) * 100
        logger.info(f"FAST 65: {embedding_time:.1f}s, {successful_count}/{len(documents)} chunks ({success_rate:.1f}% success)")
    
    def similarity_search(self, query: str, k: int = 7) -> List[Document]:
        """Fast similarity search optimized for 65 chunks"""
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Efficient cosine similarity
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

def fast_document_loader(url: str) -> List[Document]:
    """Optimized document loader for 25-page limit"""
    try:
        response = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = min(len(pdf_reader.pages), 25)  # Hard limit 25 pages
            
            # Fast batch processing
            text_parts = []
            for i in range(total_pages):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error reading page {i+1}: {e}")
                    continue
            
            text = "\n\n".join(text_parts)
            logger.info(f"FAST: Processed {total_pages} pages ({len(text)} chars)")
            return [Document(page_content=text[:120000], metadata={"source": url, "type": "pdf", "pages": total_pages})]
        
        elif url_lower.endswith('.docx') or 'wordprocessingml' in content_type:
            docx_file = io.BytesIO(response.content)
            doc = DocxDocument(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return [Document(page_content=text[:120000], metadata={"source": url, "type": "docx"})]
        
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

class Fast65RAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing FAST 65-CHUNK RAG engine with 8B model...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            
            # CRITICAL: Switch to 8B model for speed
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.1-8B-Instruct-Turbo",  # 8B instead of 70B
                temperature=0,
                max_tokens=2000  # Reduced for speed
            )

            # Optimized chunking for 65 chunks
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1800,    # Balanced size
                chunk_overlap=150,  # Reasonable overlap
                separators=["\n\n", "\n", ". ", " "]  # Efficient separators
            )

            self.initialized = True
            logger.info("FAST 65-CHUNK RAG engine ready with 8B model!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def smart_chunk_selection(self, chunks: List[Document], target_chunks: int = 65) -> List[Document]:
        """Smart adaptive selection for optimal 65-chunk coverage"""
        total = len(chunks)
        
        if total <= target_chunks:
            logger.info(f"COVERAGE: Using all {total} chunks")
            return chunks
        
        if total <= 100:
            # Dense sampling for smaller documents
            step = total / target_chunks
            indices = [int(i * step) for i in range(target_chunks)]
            selected = [chunks[i] for i in indices]
            logger.info(f"COVERAGE: Dense sampling {target_chunks}/{total} chunks")
            return selected
        else:
            # Strategic sampling for larger documents
            start_count = int(target_chunks * 0.38)  # 25 chunks from start
            middle_count = int(target_chunks * 0.24)  # 15 chunks from middle
            end_count = target_chunks - start_count - middle_count  # 25 chunks from end
            
            middle_start = total // 2 - middle_count // 2
            
            selected = (chunks[:start_count] + 
                       chunks[middle_start:middle_start + middle_count] + 
                       chunks[-end_count:])
            
            logger.info(f"COVERAGE: Strategic sampling {len(selected)}/{total} chunks")
            logger.info(f"Distribution: Start={start_count}, Middle={middle_count}, End={end_count}")
            return selected[:target_chunks]

    def _fast_65_query(self, vectorstore: Optimized65VectorStore, query: str) -> str:
        """Streamlined query optimized for 8B model"""
        docs = vectorstore.similarity_search(query, k=7)  # Good coverage from 65 chunks
        context = " ".join([doc.page_content for doc in docs])[:2400]  # Optimized context length
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Streamlined system prompt for 8B model
        system_content = """Insurance expert. Extract specific facts from document.

RULES:
- Questions separated by " | ", answers separated by " | "
- Include exact numbers, dates, percentages, conditions
- If not in document: "Information not available"
- Maintain exact question order
- Be precise and factual"""

        human_content = f"""Questions: {query}

Document Context: {context}

Provide answers separated by " | " in exact order:"""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        
        response = self.chat_model.invoke(messages)
        return response.content

    async def process_fast_65(self, url: str, questions: List[str]) -> List[str]:
        """Fast 65-chunk processing under 30 seconds"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=29.0  # Slightly under 30s for safety
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="29 second timeout exceeded")

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        total_start = time.time()
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("CACHED - INSTANT!")
        else:
            # Fast document loading
            load_start = time.time()
            docs = fast_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            load_time = time.time() - load_start
            
            # Smart chunk selection for optimal coverage
            select_start = time.time()
            selected_chunks = self.smart_chunk_selection(chunks, target_chunks=65)
            select_time = time.time() - select_start
            
            logger.info(f"TIMING: Load={load_time:.1f}s, Select={select_time:.1f}s")
            
            # Fast embedding
            embed_start = time.time()
            vectorstore = Optimized65VectorStore(self.embeddings)
            vectorstore.add_documents_fast_65(selected_chunks)
            embed_time = time.time() - embed_start
            
            # Cache for future use
            self.vectorstore_cache[url_hash] = vectorstore
            logger.info(f"TIMING: Embed={embed_time:.1f}s")
        
        # Fast query processing
        batch_query = " | ".join(questions)
        
        query_start = time.time()
        response = self._fast_65_query(vectorstore, batch_query)
        query_time = time.time() - query_start
        
        total_time = time.time() - total_start
        logger.info(f"FAST 65: Query={query_time:.1f}s, Total={total_time:.1f}s")
        
        # Quick answer parsing
        answers = []
        raw_splits = response.split(" | ")
        
        for split in raw_splits:
            cleaned = split.strip()
            # Filter out obvious non-answers
            if (cleaned and len(cleaned) > 5 and 
                not cleaned.lower().startswith(('question:', 'answer:', 'q:', 'a:'))):
                answers.append(cleaned)
        
        # Ensure correct count
        while len(answers) < len(questions):
            answers.append("Information not available in provided document.")
        
        logger.info(f"PARSED: {len(answers)} answers for {len(questions)} questions")
        return answers[:len(questions)]

# Global engine
rag_engine = Fast65RAGEngine()

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
        logger.info("FAST 65-CHUNK RAG application ready")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="FAST 65-CHUNK RAG API", version="4.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Fast 65-chunk processing with 8B model - guaranteed under 30 seconds"""
    try:
        logger.info(f"FAST 65: {len(request.questions)} questions (65 chunk limit, 8B model)")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        start_time = time.time()
        answers = await rag_engine.process_fast_65(request.documents, request.questions)
        total_time = time.time() - start_time

        logger.info(f"FAST 65: Completed in {total_time:.1f}s")
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
        "mode": "fast_65_chunks",
        "max_chunks": 65,
        "model": "Llama-3.1-8B-Instruct-Turbo",
        "chunk_distribution": "38% start, 24% middle, 38% end",
        "target_time": "<30_seconds",
        "embedding_provider": "Together.AI (Fast 65)"
    }

@app.get("/")
async def root():
    return {"message": "FAST 65-CHUNK RAG API - Optimal Coverage Under 30 Seconds"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
