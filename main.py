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
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# RAG imports
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected Bearer token for authentication
EXPECTED_TOKEN = "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8"

# ----- Request/Response Models -----
class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# ----- Optimized Vector Store -----
class OptimizedVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store with batch processing"""
        for doc in documents:
            vector = self.embeddings.embed_query(doc.page_content)
            self.documents.append(doc)
            self.vectors.append(vector)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """High-quality similarity search with threshold filtering"""
        if not self.vectors:
            return []
        
        query_vector = self.embeddings.embed_query(query)
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        
        # Use similarity threshold to ensure quality results
        threshold = 0.1
        top_indices = np.argsort(similarities)[-8:][::-1]  # Get top 8, filter by threshold
        
        filtered_indices = [i for i in top_indices if similarities[i] > threshold][:k]
        return [self.documents[i] for i in filtered_indices if i < len(self.documents)]

# ----- Optimized Document Loader -----
def load_document_content_optimized(url: str) -> List[Document]:
    """Load content with smart size management and speed optimization"""
    try:
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        # Smart size management - allow reasonable documents
        if len(response.content) > 8 * 1024 * 1024:  # 8MB limit for speed
            logger.warning("Large document detected - processing first 8MB for speed")
            content = response.content[:8 * 1024 * 1024]
        else:
            content = response.content
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        # Handle PDF files
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            # Limit pages for speed while maintaining quality
            max_pages = min(50, len(pdf_reader.pages))  # Process up to 50 pages
            for i, page in enumerate(pdf_reader.pages[:max_pages]):
                text += page.extract_text() + "\n"
                if i > 0 and i % 10 == 0:  # Progress check every 10 pages
                    logger.info(f"Processed {i+1}/{max_pages} pages")
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "pdf"})]
        
        # Handle DOCX files
        elif url_lower.endswith('.docx') or 'wordprocessingml' in content_type:
            docx_file = io.BytesIO(content)
            doc = DocxDocument(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "docx"})]
        
        # Handle EML files
        elif url_lower.endswith('.eml') or 'message/rfc822' in content_type:
            eml_content = content.decode('utf-8', errors='ignore')
            msg = email.message_from_string(eml_content)
            
            text = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                text = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "eml"})]
        
        # Handle HTML/Text files (default)
        else:
            soup = BeautifulSoup(content, 'html.parser')
            
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            text = ' '.join(text.split())
            
            return [Document(page_content=text, metadata={"source": url, "type": "html"})]
        
    except Exception as e:
        logger.error(f"Failed to load document {url}: {e}")
        raise

# ----- High-Performance RAG Engine -----
class HighPerformanceRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.document_cache: Dict[str, Any] = {}
        self.max_cache_size = 3  # Slightly larger for better caching
    
    def _get_url_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def initialize(self):
        """Initialize optimized RAG components"""
        if self.initialized:
            return
            
        logger.info("Initializing high-performance RAG engine...")
        
        try:
            # Set environment variables
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            # Initialize embeddings
            self.embeddings = TogetherEmbeddings(
                model="BAAI/bge-base-en-v1.5"
            )
            
            # Test embeddings quickly
            logger.info("Testing embeddings...")
            test_embedding = self.embeddings.embed_query("test")
            logger.info(f"Embeddings working - dimension: {len(test_embedding)}")

            # Initialize chat model
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=3000
            )

            # Optimized text splitter for speed + quality
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Sweet spot for accuracy vs speed
                chunk_overlap=75,  # Maintains context overlap
                separators=["\n\n", "\n", ". ", "!", "?", " ", ""]  # Better semantic breaks
            )

            self.initialized = True
            logger.info("High-performance RAG engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {str(e)}")
            raise

    def _high_accuracy_rag_query(self, vectorstore: OptimizedVectorStore, query: str) -> str:
        """High-accuracy batch processing with optimized context"""
        try:
            # Get relevant documents for the combined query
            docs = vectorstore.similarity_search(query, k=4)
            
            # Maintain quality context - balanced size for speed
            context = " ".join([doc.page_content for doc in docs])[:2500]
            
            # High-quality system prompt for accuracy
            system_prompt = """You are an expert document assistant with high accuracy standards.

CRITICAL INSTRUCTIONS:
- Input questions are separated by " | "
- Output answers MUST be separated by " | " in the same order
- Provide detailed, accurate answers based strictly on the document content
- If information is not in the document, clearly state "Information not found in document"
- Maintain the exact question order in your responses
- Use specific details and quotes from the document when available

Quality Guidelines:
- Prioritize accuracy over brevity
- Include relevant context and details
- Reference specific sections when possible
- Ensure each answer directly addresses its corresponding question"""

            human_prompt = f"""Questions: {query}

Document Context: {context}

Please provide detailed, accurate answers to each question based on the document content."""

            # Use optimized message structure
            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.chat_model.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"RAG query error: {e}")
            raise

    def _load_and_process_document_fast(self, url: str) -> tuple:
        """Fast document loading with caching"""
        if url in self.document_cache:
            logger.info(f"Using cached document for {url}")
            return self.document_cache[url]
        
        logger.info(f"Loading document: {url}")
        start_time = time.time()
        
        try:
            docs = load_document_content_optimized(url)
            chunks = self.text_splitter.split_documents(docs)
            
            load_time = time.time() - start_time
            logger.info(f"Document loaded and chunked in {load_time:.2f}s ({len(chunks)} chunks)")
            
            # Cache management
            if len(self.document_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.document_cache))
                del self.document_cache[oldest_key]
            
            self.document_cache[url] = (docs, chunks)
            return docs, chunks
            
        except Exception as e:
            logger.error(f"Failed to load document {url}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load document: {str(e)}")

    def _create_vectorstore_optimized(self, url: str, chunks: List) -> OptimizedVectorStore:
        """High-speed vectorstore creation with parallel processing"""
        logger.info(f"Creating optimized vectorstore for {url}")
        start_time = time.time()
        
        try:
            vectorstore = OptimizedVectorStore(self.embeddings)
            
            # Optimal batch processing for speed
            batch_size = 12  # Optimal for Together.AI rate limits and speed
            total_batches = (len(chunks) - 1) // batch_size + 1
            
            logger.info(f"Processing {len(chunks)} chunks in {total_batches} batches")
            
            # Use parallel processing for speed
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = []
                
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i + batch_size]
                    future = executor.submit(vectorstore.add_documents, batch)
                    futures.append(future)
                
                # Wait for completion with progress tracking
                for i, future in enumerate(futures):
                    future.result(timeout=8)  # 8 second timeout per batch
                    if i % 2 == 0 or i == len(futures) - 1:  # Log progress
                        logger.info(f"Completed batch {i+1}/{len(futures)}")
            
            creation_time = time.time() - start_time
            logger.info(f"Optimized vectorstore created in {creation_time:.2f}s")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Vectorstore creation error: {e}")
            raise HTTPException(status_code=500, detail="Failed to create vectorstore")

    def process_document_questions_optimized(self, url: str, questions: List[str]) -> List[str]:
        """High-performance processing with timeout protection"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        # Timeout protection - 27 seconds to stay under 30
        def timeout_handler(signum, frame):
            raise TimeoutError("Processing timeout - document too large or complex")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(27)  # 27-second timeout
        
        total_start_time = time.time()
        
        try:
            # Fast document loading and processing
            docs, chunks = self._load_and_process_document_fast(url)
            
            # Check if we're running out of time
            if time.time() - total_start_time > 18:
                raise TimeoutError("Document processing taking too long")
            
            # Create optimized vectorstore
            vectorstore = self._create_vectorstore_optimized(url, chunks)
            
            # Process ALL questions together for consistency and speed
            logger.info(f"Processing {len(questions)} questions with high accuracy...")
            batch_query = " | ".join(questions)
            
            query_start_time = time.time()
            batch_result = self._high_accuracy_rag_query(vectorstore, batch_query)
            query_time = time.time() - query_start_time
            
            logger.info(f"High-quality LLM query completed in {query_time:.2f}s")
            
            # Careful answer parsing to maintain accuracy
            answers = [answer.strip() for answer in batch_result.split(" | ")]
            
            # Quality validation - ensure we have good answers
            if len(answers) != len(questions):
                logger.warning(f"Answer count mismatch: {len(questions)} questions, {len(answers)} answers")
                if len(answers) < len(questions):
                    for i in range(len(answers), len(questions)):
                        answers.append("Unable to generate answer - please try rephrasing the question.")
                answers = answers[:len(questions)]
            
            total_time = time.time() - total_start_time
            logger.info(f"High-performance processing completed in {total_time:.2f}s")
            
            return answers

        except TimeoutError as e:
            logger.error(f"Timeout error: {str(e)}")
            raise HTTPException(status_code=408, detail="Request timeout - document too large or complex")
        except Exception as e:
            logger.error(f"Error in high-performance processing: {str(e)}")
            raise
        finally:
            signal.alarm(0)  # Cancel the alarm

# Global RAG engine instance
rag_engine = HighPerformanceRAGEngine()

# Token Verifier
def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid format")
    
    token = authorization.split("Bearer ")[-1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Bearer token")

# Lifespan Management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        rag_engine.initialize()
        logger.info("High-performance application startup completed successfully")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        logger.info("Continuing with limited functionality...")
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")

# FastAPI App
app = FastAPI(title="High-Performance Railway RAG API", version="2.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Optimized endpoint for high-performance document processing"""
    try:
        logger.info(f"Received request with {len(request.questions)} questions")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="Questions list is empty")
        if len(request.questions) > 15:  # Limit for performance
            raise HTTPException(status_code=400, detail="Maximum 15 questions allowed per request")

        answers = rag_engine.process_document_questions_optimized(request.documents, request.questions)

        logger.info(f"Successfully processed {len(answers)} answers")
        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_initialized": rag_engine.initialized,
        "cached_documents": len(rag_engine.document_cache),
        "timestamp": datetime.now().isoformat(),
        "performance_mode": "optimized"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "High-Performance RAG API is running", "endpoints": ["/hackrx/run", "/health"]}

# Railway startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
