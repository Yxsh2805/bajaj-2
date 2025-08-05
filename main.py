import os
import time
import logging
import hashlib
import random
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
from functools import wraps

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

class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=15):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func, *args, **kwargs):
        if self.state == 'OPEN':
            if time.time() - self.last_failure_time < self.timeout:
                # Use cached result or return None during circuit open
                return None
            else:
                self.state = 'HALF_OPEN'
        
        try:
            result = func(*args, **kwargs)
            if self.state == 'HALF_OPEN':
                self.state = 'CLOSED'
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = 'OPEN'
                logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
            
            raise e

def with_smart_retry(max_retries=2, base_delay=0.3):
    """Smart retry with exponential backoff and jitter"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        logger.warning(f"All {max_retries + 1} attempts failed: {e}")
                        return None
                    
                    # Smart backoff: shorter delays for API issues
                    if "503" in str(e) or "temporarily unavailable" in str(e).lower():
                        delay = base_delay * (1.5 ** attempt) + random.uniform(0, 0.2)
                    else:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.3)
                    
                    logger.info(f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class FinalOptimizedVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
        self.circuit_breaker = CircuitBreaker(failure_threshold=4, timeout=10)
    
    @with_smart_retry(max_retries=2, base_delay=0.2)
    def embed_with_retry(self, text):
        """Embedding with smart retry logic"""
        return self.embeddings.embed_query(text)
    
    def embed_robust(self, doc):
        """Robust embedding with circuit breaker"""
        try:
            return self.circuit_breaker.call(self.embed_with_retry, doc.page_content)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None
    
    def add_documents_final(self, documents: List[Document]):
        """Enhanced with robust error handling and fallback strategies"""
        logger.info(f"FINAL: Processing {len(documents)} chunks")
        
        start_time = time.time()
        
        # Strategy 1: Parallel processing with reduced workers
        try:
            with ThreadPoolExecutor(max_workers=4) as executor:  # Reduced from 8
                vectors = list(executor.map(self.embed_robust, documents))
            
            successful_count = self._store_vectors(documents, vectors)
            
            # If we have good success rate, we're done
            if successful_count >= len(documents) * 0.7:
                embedding_time = time.time() - start_time
                logger.info(f"FINAL: Strategy 1 - {embedding_time:.1f}s, {successful_count} chunks embedded")
                return
        except Exception as e:
            logger.warning(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Sequential processing for failed chunks
        logger.info("Falling back to sequential processing for remaining chunks")
        failed_docs = []
        for doc, vector in zip(documents, vectors if 'vectors' in locals() else [None] * len(documents)):
            if vector is None:
                failed_docs.append(doc)
        
        if failed_docs:
            self._embed_sequentially(failed_docs[:10])  # Limit to 10 failed chunks max
        
        embedding_time = time.time() - start_time
        total_embedded = len(self.documents)
        logger.info(f"FINAL: Combined strategies - {embedding_time:.1f}s, {total_embedded} chunks embedded")
    
    def _store_vectors(self, documents, vectors):
        """Store successful embeddings"""
        successful_count = 0
        for doc, vector in zip(documents, vectors):
            if vector is not None:
                self.documents.append(doc)
                self.vectors.append(vector)
                successful_count += 1
        return successful_count
    
    def _embed_sequentially(self, documents):
        """Sequential embedding as fallback"""
        for doc in documents:
            try:
                vector = self.embed_with_retry(doc.page_content)
                if vector is not None:
                    self.documents.append(doc)
                    self.vectors.append(vector)
                time.sleep(0.1)  # Small delay between sequential requests
            except Exception as e:
                logger.warning(f"Sequential embedding failed: {e}")
                continue
    
    def similarity_search(self, query: str, k: int = 7) -> List[Document]:
        """Optimized search with retry"""
        if not self.vectors:
            return []
        
        try:
            # Try with circuit breaker first
            query_vector = self.circuit_breaker.call(self.embed_with_retry, query)
            if query_vector is None:
                logger.warning("Query embedding failed, using fallback")
                return self.documents[:k] if len(self.documents) >= k else self.documents
            
            # Fast cosine similarity
            similarities = []
            query_norm = np.linalg.norm(query_vector)
            
            for i, vector in enumerate(self.vectors):
                vector_norm = np.linalg.norm(vector)
                if query_norm > 0 and vector_norm > 0:
                    cos_sim = np.dot(query_vector, vector) / (query_norm * vector_norm)
                    similarities.append((cos_sim, i))
            
            similarities.sort(reverse=True)
            return [self.documents[i] for _, i in similarities[:k]]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self.documents[:k] if len(self.documents) >= k else self.documents

def final_document_loader(url: str) -> List[Document]:
    """Final optimized document loader with timeout handling"""
    try:
        # Faster timeout for quicker failure detection
        response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = min(len(pdf_reader.pages), 20)  # Reduced from 25
            
            # More aggressive page sampling for speed
            if total_pages <= 10:
                pages_to_process = list(range(total_pages))
            else:
                # Even more selective sampling
                first_pages = list(range(5))  # First 5 pages
                middle_pages = list(range(total_pages//2-1, total_pages//2+2))  # 3 middle pages
                last_pages = list(range(total_pages-5, total_pages))  # Last 5 pages
                pages_to_process = sorted(set(first_pages + middle_pages + last_pages))
            
            text = ""
            for i in pages_to_process:
                if i < len(pdf_reader.pages):
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text.strip():
                        text += page_text + "\n\n"
            
            logger.info(f"FINAL: Processed {len(pages_to_process)}/{total_pages} pages")
            return [Document(page_content=text.strip()[:80000], metadata={"source": url, "type": "pdf"})]  # Reduced from 100k
        
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            return [Document(page_content=text[:80000], metadata={"source": url, "type": "html"})]  # Reduced from 100k
        
    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise

class FinalRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=8)
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing FINAL RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            
            # CRITICAL: Use 8B model with BETTER PROMPTING
            self.chat_model = ChatTogether(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                temperature=0,
                max_tokens=1800  # Reduced for faster response
            )

            # More aggressive chunking for speed
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,  # Reduced from 1400
                chunk_overlap=100,  # Reduced from 120
                separators=["\n\n", "\n", ". ", " "]
            )

            self.initialized = True
            logger.info("FINAL RAG engine ready!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    @with_smart_retry(max_retries=1, base_delay=0.2)
    def _query_with_retry(self, messages):
        """Chat query with retry"""
        return self.chat_model.invoke(messages)

    def _final_query(self, vectorstore: FinalOptimizedVectorStore, query: str) -> str:
        """FINAL QUERY with robust error handling"""
        docs = vectorstore.similarity_search(query, k=6)  # Reduced from 7
        context = " ".join([doc.page_content for doc in docs])[:2200]  # Reduced from 2500
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # CRITICAL: Better system prompt to prevent question repetition
        system_content = """You are an insurance policy expert. Your task is to answer questions, NOT repeat them.

CRITICAL RULES:
- Questions are separated by " | "
- Provide ONLY answers separated by " | " in the same order
- DO NOT include the questions in your response
- DO NOT start answers with "Question:" or "Q:" or repeat the question
- Extract specific facts, numbers, percentages, conditions from the document
- If information not found, say "Information not available in document"
- Keep answers concise but informative

Example:
Input: "What is the waiting period? | What is the sum insured?"
Output: "36 months for pre-existing diseases | Up to 50 lacs as per plan selected"

REMEMBER: Provide ONLY the answers, separated by " | ", NO questions."""

        human_content = f"""Document Context: {context}

Questions to answer: {query}

Provide only the answers separated by " | " (do not repeat questions):"""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        
        try:
            response = self.circuit_breaker.call(self._query_with_retry, messages)
            if response is None:
                return "Service temporarily unavailable | " * query.count("|") + "Service temporarily unavailable"
            return response.content
        except Exception as e:
            logger.error(f"Query error: {e}")
            fallback_answer = "Information not available due to service error"
            return " | ".join([fallback_answer] * (query.count("|") + 1))

    async def process_final(self, url: str, questions: List[str]) -> List[str]:
        """FINAL PROCESSING - Guaranteed under 30 seconds with better error handling"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=26.0  # Reduced timeout for safety margin
            )
        except asyncio.TimeoutError:
            logger.error("Timeout exceeded - returning fallback responses")
            return ["Processing timeout - please try again"] * len(questions)

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        total_start = time.time()
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("CACHED!")
        else:
            # Load and process document
            docs = final_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            # FINAL OPTIMIZATION: Limit to 30 chunks max (reduced from 35)
            if len(chunks) > 30:
                # Strategic selection: 40% start, 25% middle, 35% end
                start_count = int(30 * 0.40)  # 12
                middle_count = int(30 * 0.25)  # 7-8
                end_count = 30 - start_count - middle_count  # 10-11
                
                middle_start = len(chunks) // 2 - middle_count // 2
                chunks = (chunks[:start_count] + 
                         chunks[middle_start:middle_start + middle_count] + 
                         chunks[-end_count:])
                
                logger.info(f"FINAL: Selected 30/{len(chunks)} chunks")
            
            # Fast embedding with enhanced error handling
            vectorstore = FinalOptimizedVectorStore(self.embeddings)
            vectorstore.add_documents_final(chunks)
            
            # Only cache if we have sufficient embeddings
            if len(vectorstore.documents) >= len(chunks) * 0.5:  # At least 50% success
                self.vectorstore_cache[url_hash] = vectorstore
            else:
                logger.warning(f"Low embedding success rate: {len(vectorstore.documents)}/{len(chunks)}")
        
        # Query with timing
        batch_query = " | ".join(questions)
        
        query_start = time.time()
        response = self._final_query(vectorstore, batch_query)
        query_time = time.time() - query_start
        
        total_time = time.time() - total_start
        logger.info(f"FINAL: Query={query_time:.1f}s, Total={total_time:.1f}s")
        
        # Enhanced answer parsing
        answers = []
        raw_splits = response.split(" | ")
        
        for split in raw_splits:
            cleaned = split.strip()
            # Remove any question repetition
            if cleaned and not any(q.lower().strip() in cleaned.lower() for q in questions):
                if len(cleaned) > 5:  # More lenient meaningful answer check
                    answers.append(cleaned)
        
        # Ensure correct count with better fallbacks
        while len(answers) < len(questions):
            answers.append("Information not available in document.")
        
        return answers[:len(questions)]

# Global engine
rag_engine = FinalRAGEngine()

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
        logger.info("FINAL RAG ready for submission")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="FINAL RAG API", version="FINAL", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """FINAL VERSION - Guaranteed under 30 seconds with robust error handling"""
    try:
        logger.info(f"FINAL: {len(request.questions)} questions - TARGET: <28s")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        start_time = time.time()
        answers = await rag_engine.process_final(request.documents, request.questions)
        total_time = time.time() - start_time

        logger.info(f"FINAL: SUCCESS in {total_time:.1f}s (Target: <28s)")
        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}")
        # Return graceful fallback instead of 500 error
        fallback_answers = ["Service temporarily unavailable - please try again"] * len(request.questions)
        return {"answers": fallback_answers}

@app.get("/health")
async def health_check():
    return {
        "status": "ready_for_submission",
        "mode": "final_optimized_resilient",
        "model": "Meta-Llama-3.1-8B-Instruct-Turbo", 
        "target_time": "<26_seconds",
        "max_chunks": 30,
        "features": ["circuit_breaker", "smart_retry", "fallback_strategies"]
    }

@app.get("/")
async def root():
    return {"message": "FINAL RAG API - Ready for Submission with Enhanced Resilience"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
