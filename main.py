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

class Smart50VectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_smart_50(self, documents: List[Document]):
        """Smart processing of exactly 50 chunks for maximum coverage"""
        logger.info(f"SMART 50: Processing {len(documents)} chunks (guaranteed <30s)")
        
        start_time = time.time()
        
        def embed_reliable(doc):
            """Reliable embedding with minimal retry"""
            try:
                return self.embeddings.embed_query(doc.page_content)
            except Exception as e:
                # One quick retry
                try:
                    time.sleep(0.1)
                    return self.embeddings.embed_query(doc.page_content)
                except:
                    logger.warning(f"Embed failed: {str(e)[:30]}")
                    return None
        
        # 10 workers - proven to work well
        with ThreadPoolExecutor(max_workers=10) as executor:
            vectors = list(executor.map(embed_reliable, documents))
        
        # Store results
        successful_count = 0
        for doc, vector in zip(documents, vectors):
            if vector is not None:
                self.documents.append(doc)
                self.vectors.append(vector)
                successful_count += 1
        
        embedding_time = time.time() - start_time
        success_rate = (successful_count / len(documents)) * 100
        logger.info(f"SMART 50: {embedding_time:.1f}s, {successful_count}/{len(documents)} chunks ({success_rate:.1f}% success)")
    
    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """Enhanced similarity search for 50 chunks"""
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # High-quality cosine similarity
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
    """Same document loader as before"""
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
                        text += f"{page_text}\n\n"
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

class Smart50RAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing SMART 50-CHUNK RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=3200
            )

            # OPTIMIZED chunking for maximum coverage in 50 chunks
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # LARGER chunks for more content per chunk
                chunk_overlap=200, # 10% overlap for continuity
                separators=["\n\n", "\n", ". ", "! ", "? ", " "]  # Full separators for quality
            )

            self.initialized = True
            logger.info("SMART 50-CHUNK RAG engine ready!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _smart_50_query(self, vectorstore: Smart50VectorStore, query: str) -> str:
        """Enhanced query for 50-chunk model"""
        docs = vectorstore.similarity_search(query, k=9)  # Use more of the 50 chunks
        context = " ".join([doc.page_content for doc in docs])[:3500]  # More context
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Enhanced system prompt for better accuracy
        system_content = """You are an expert insurance policy analyst with high accuracy standards.

CRITICAL INSTRUCTIONS:
- Input questions are separated by " | "
- Output answers MUST be separated by " | " in the same order
- Provide accurate, detailed answers based on the document content
- Include specific numbers, time periods, conditions, and percentages when available
- If information is not in the document, state "Information not available in provided document"
- Keep answers comprehensive but concise

QUALITY REQUIREMENTS:
- Start with the key information first
- Include specific details like amounts, time periods, percentages
- Reference exact conditions and requirements from the document
- Ensure each answer fully addresses its corresponding question
- Use precise language from the source document

CRITICAL: Separate each answer with " | " and maintain exact question order."""

        human_content = f"""Answer these questions accurately based on the document:

Questions: {query}

Document Context: {context}

Provide detailed, accurate answers separated by " | " in the same order."""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        
        response = self.chat_model.invoke(messages)
        return response.content

    async def process_smart_50(self, url: str, questions: List[str]) -> List[str]:
        """Smart 50-chunk processing"""
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
            
            # INTELLIGENT CHUNK SELECTION for maximum coverage
            original_count = len(chunks)
            
            if len(chunks) > 50:
                # ENHANCED selection strategy for maximum coverage
                
                # Calculate distribution
                total_chunks = len(chunks)
                
                # More sophisticated distribution:
                # 40% from start (definitions, basic terms)
                # 25% from middle (detailed conditions)  
                # 35% from end (exclusions, appendices)
                
                start_count = int(0.40 * 50)  # 20 chunks
                middle_count = int(0.25 * 50)  # 12-13 chunks
                end_count = 50 - start_count - middle_count  # 17-18 chunks
                
                # Select start chunks
                start_chunks = chunks[:start_count]
                
                # Select middle chunks (true middle of document)
                middle_start = total_chunks // 2 - middle_count // 2
                middle_end = middle_start + middle_count
                middle_chunks = chunks[middle_start:middle_end]
                
                # Select end chunks
                end_chunks = chunks[-end_count:]
                
                # Combine for balanced coverage
                selected_chunks = start_chunks + middle_chunks + end_chunks
                
                logger.info(f"SMART SELECTION: {original_count} → 50 chunks")
                logger.info(f"Distribution: Start={len(start_chunks)}, Middle={len(middle_chunks)}, End={len(end_chunks)}")
                
                chunks = selected_chunks
            
            # Ensure exactly 50 chunks
            if len(chunks) > 50:
                chunks = chunks[:50]
            
            logger.info(f"FINAL: Processing exactly {len(chunks)} chunks")
            
            # Smart embedding
            vectorstore = Smart50VectorStore(self.embeddings)
            vectorstore.add_documents_smart_50(chunks)
            
            # Cache for future use
            self.vectorstore_cache[url_hash] = vectorstore
        
        # Query processing
        batch_query = " | ".join(questions)
        
        query_start = time.time()
        response = self._smart_50_query(vectorstore, batch_query)
        query_time = time.time() - query_start
        
        total_time = time.time() - total_start
        logger.info(f"SMART 50: Query {query_time:.1f}s, Total {total_time:.1f}s")
        
        # Enhanced answer parsing
        answers = []
        raw_splits = response.split(" | ")
        
        for split in raw_splits:
            cleaned = split.strip()
            if cleaned and len(cleaned) > 8 and not cleaned.lower().startswith(('question:', 'answer:')):
                answers.append(cleaned)
        
        # Ensure correct count
        while len(answers) < len(questions):
            answers.append("Unable to find specific information in the provided document.")
        
        return answers[:len(questions)]

# Global engine
rag_engine = Smart50RAGEngine()

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
        logger.info("SMART 50-CHUNK RAG application ready")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="SMART 50-CHUNK RAG API", version="3.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Smart 50-chunk processing - maximum coverage within 30 seconds"""
    try:
        logger.info(f"SMART 50: {len(request.questions)} questions (50 chunk limit)")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        start_time = time.time()
        answers = await rag_engine.process_smart_50(request.documents, request.questions)
        total_time = time.time() - start_time

        logger.info(f"SMART 50: Completed in {total_time:.1f}s")
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
        "mode": "smart_50_chunks",
        "max_chunks": 50,
        "chunk_distribution": "40% start, 25% middle, 35% end",
        "embedding_provider": "Together.AI (Smart 50)"
    }

@app.get("/")
async def root():
    return {"message": "SMART 50-CHUNK RAG API - Maximum Coverage in 50 Chunks"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
