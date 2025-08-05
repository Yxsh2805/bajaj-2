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

class UltraFastVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_ultra_fast(self, documents: List[Document]):
        """ULTRA FAST embedding with aggressive optimizations"""
        logger.info(f"ULTRA FAST: Processing {len(documents)} chunks")
        
        start_time = time.time()
        
        # Clean and validate texts first
        valid_texts = []
        valid_docs = []
        
        for doc in documents:
            # Clean text aggressively
            text = doc.page_content.strip()
            # Remove problematic characters
            text = ''.join(char for char in text if ord(char) < 65536)
            # Limit length
            text = text[:1500]
            
            if len(text) > 50:  # Only embed meaningful content
                valid_texts.append(text)
                valid_docs.append(Document(page_content=text, metadata=doc.metadata))
        
        logger.info(f"ULTRA FAST: {len(valid_texts)} valid chunks after cleaning")
        
        # Try batch first, with immediate fallback
        try:
            # Quick batch attempt with timeout
            vectors = asyncio.wait_for(
                asyncio.to_thread(self.embeddings.embed_documents, valid_texts[:30]),  # Limit to 30
                timeout=8.0
            )
            vectors = asyncio.run(vectors)
            
            # Store results
            for doc, vector in zip(valid_docs[:30], vectors):
                self.documents.append(doc)
                self.vectors.append(vector)
                
            embedding_time = time.time() - start_time
            logger.info(f"ULTRA FAST: BATCH SUCCESS! {embedding_time:.1f}s for {len(vectors)} chunks")
            return
            
        except Exception as e:
            logger.warning(f"Batch failed: {str(e)[:50]}, using ultra-fast individual...")
        
        # Ultra-fast individual with strict limits
        self._ultra_fast_individual(valid_docs[:25])  # Even fewer chunks
        
        embedding_time = time.time() - start_time
        logger.info(f"ULTRA FAST: {embedding_time:.1f}s total")
    
    def _ultra_fast_individual(self, documents: List[Document]):
        """Ultra-fast individual embedding with no retries"""
        successful = 0
        
        for i, doc in enumerate(documents):
            if i >= 25:  # Hard limit
                break
                
            try:
                vector = self.embeddings.embed_query(doc.page_content)
                self.documents.append(doc)
                self.vectors.append(vector)
                successful += 1
                
                # No delays, no retries - just fast
                if successful >= 20:  # Stop at 20 successful
                    break
                    
            except:
                continue  # Skip failures immediately
        
        logger.info(f"ULTRA FAST INDIVIDUAL: {successful} chunks embedded")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Ultra-fast search"""
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Simple dot product (faster than cosine)
            similarities = []
            for i, vector in enumerate(self.vectors):
                sim = np.dot(query_vector, vector)
                similarities.append((sim, i))
            
            # Quick sort and return
            similarities.sort(reverse=True)
            return [self.documents[i] for _, i in similarities[:k]]
            
        except:
            return self.documents[:k] if len(self.documents) >= k else self.documents

def ultra_fast_document_loader(url: str) -> List[Document]:
    """Ultra-fast document loading"""
    try:
        logger.info(f"LOADING: {url}")
        response = requests.get(url, timeout=6)  # Shorter timeout
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = min(len(pdf_reader.pages), 20)  # Limit to 20 pages
            
            # Fast page processing
            text_parts = []
            for i in range(total_pages):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text and len(page_text.strip()) > 20:
                        text_parts.append(page_text[:3000])  # Limit per page
                except:
                    continue
            
            full_text = "\n\n".join(text_parts)[:80000]  # Hard limit
            logger.info(f"PDF: {total_pages} pages, {len(full_text)} chars")
            
            return [Document(page_content=full_text, metadata={"source": url, "type": "pdf"})]
        
        else:
            # Handle other formats simply
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()[:80000]
            logger.info(f"TEXT: {len(text)} chars")
            return [Document(page_content=text, metadata={"source": url, "type": "html"})]
        
    except Exception as e:
        logger.error(f"Load error: {e}")
        raise

class UltraFastRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing ULTRA-FAST RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            
            # WORKING 8B model
            self.chat_model = ChatTogether(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # Correct model name
                temperature=0,
                max_tokens=1500
            )

            # Ultra-fast chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Smaller for speed
                chunk_overlap=100,
                separators=["\n\n", "\n", ". "]
            )

            self.initialized = True
            logger.info("ULTRA-FAST RAG engine ready!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _ultra_fast_query(self, vectorstore: UltraFastVectorStore, query: str) -> str:
        """Ultra-fast query"""
        docs = vectorstore.similarity_search(query, k=5)
        context = " ".join([doc.page_content for doc in docs])[:1800]  # Smaller context
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Minimal prompt for speed
        system_content = """Insurance expert. Extract facts from document.
Questions/answers separated by " | ". If not found: "Not available"."""

        human_content = f"Questions: {query}\nDocument: {context}\nAnswers:"

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        
        response = self.chat_model.invoke(messages)
        return response.content

    async def process_ultra_fast(self, url: str, questions: List[str]) -> List[str]:
        """Ultra-fast processing under 30 seconds guaranteed"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=25.0  # Even shorter timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="25 second timeout exceeded")

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        total_start = time.time()
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("CACHED!")
        else:
            # Ultra-fast processing
            docs = ultra_fast_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            # Limit chunks aggressively
            if len(chunks) > 30:
                # Take first 15, last 15
                chunks = chunks[:15] + chunks[-15:]
            
            logger.info(f"PROCESSING: {len(chunks)} chunks")
            
            # Ultra-fast embedding
            vectorstore = UltraFastVectorStore(self.embeddings)
            vectorstore.add_documents_ultra_fast(chunks)
            
            self.vectorstore_cache[url_hash] = vectorstore
        
        # Ultra-fast query
        batch_query = " | ".join(questions)
        response = self._ultra_fast_query(vectorstore, batch_query)
        
        total_time = time.time() - total_start
        logger.info(f"ULTRA FAST: Total {total_time:.1f}s")
        
        # Quick parsing
        answers = [a.strip() for a in response.split(" | ") if a.strip()]
        while len(answers) < len(questions):
            answers.append("Information not available")
        
        return answers[:len(questions)]

# Global engine
rag_engine = UltraFastRAGEngine()

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
        logger.info("ULTRA-FAST RAG ready")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="ULTRA-FAST RAG API", version="6.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Ultra-fast processing - guaranteed under 25 seconds"""
    try:
        logger.info(f"ULTRA FAST: {len(request.questions)} questions")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        start_time = time.time()
        answers = await rag_engine.process_ultra_fast(request.documents, request.questions)
        total_time = time.time() - start_time

        logger.info(f"ULTRA FAST: Completed in {total_time:.1f}s")
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
        "mode": "ultra_fast",
        "max_chunks": 30,
        "model": "Meta-Llama-3.1-8B-Instruct-Turbo",
        "target_time": "<25_seconds"
    }

@app.get("/")
async def root():
    return {"message": "ULTRA-FAST RAG API - Guaranteed Under 25 Seconds"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
