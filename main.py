import os
import time
import logging
import hashlib
import requests
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# Try importing PDF libraries
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expected Bearer token for authentication
EXPECTED_TOKEN = os.getenv("API_TOKEN", "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8")

# ----- Request/Response Models -----
class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# ----- Simple RAG Engine using Together API directly -----
class SimpleRAGEngine:
    def __init__(self):
        self.together_api_key = None
        self.initialized = False
        self.document_cache: Dict[str, str] = {}
        self.max_cache_size = 3
    
    def _get_url_hash(self, url: str) -> str:
        """Generate hash for URL for caching purposes"""
        return hashlib.md5(url.encode()).hexdigest()[:8]
    
    def initialize(self):
        """Initialize the engine"""
        if self.initialized:
            return
            
        logger.info("Initializing Simple RAG engine...")
        
        # Get API key
        self.together_api_key = os.getenv("TOGETHER_API_KEY")
        if not self.together_api_key:
            raise ValueError("TOGETHER_API_KEY environment variable is required")

        self.initialized = True
        logger.info("Simple RAG engine initialized successfully")

    def _extract_pdf_text(self, url: str) -> str:
        """Extract text from PDF URL"""
        try:
            logger.info(f"Downloading PDF from: {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            
            if PDF_SUPPORT:
                try:
                    pdf_file = BytesIO(response.content)
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = ""
                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    
                    if text.strip():
                        logger.info(f"Successfully extracted PDF text ({len(text)} characters)")
                        return text
                except Exception as e:
                    logger.error(f"PyPDF2 extraction failed: {e}")
            
            # Fallback: treat as text
            content = response.content.decode('utf-8', errors='ignore')
            if len(content) > 100:
                return content
                
            raise Exception("Could not extract readable content from PDF")
            
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}")
            raise

    def _get_document_text(self, url: str) -> str:
        """Get document text with caching"""
        # Check cache first
        if url in self.document_cache:
            logger.info("Using cached document")
            return self.document_cache[url]
        
        # Extract text based on URL type
        if url.lower().endswith('.pdf') or 'pdf' in url.lower():
            text = self._extract_pdf_text(url)
        else:
            # Handle regular web pages
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text = response.text
        
        # Cache management
        if len(self.document_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.document_cache))
            del self.document_cache[oldest_key]
        
        self.document_cache[url] = text
        return text

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Simple text chunking"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def _find_relevant_chunks(self, chunks: List[str], questions: List[str], max_chunks: int = 5) -> List[str]:
        """Simple keyword-based chunk selection"""
        # Combine all questions to find keywords
        all_questions = " ".join(questions).lower()
        question_words = set(all_questions.split())
        
        # Score chunks based on keyword overlap
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            score = sum(1 for word in question_words if word in chunk_lower and len(word) > 3)
            chunk_scores.append((score, i, chunk))
        
        # Return top chunks
        chunk_scores.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, _, chunk in chunk_scores[:max_chunks]]

    def _call_together_api(self, prompt: str) -> str:
        """Call Together API directly"""
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.together_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 3000
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Together API call failed: {e}")
            raise

    def process_questions(self, url: str, questions: List[str]) -> List[str]:
        """Process document and answer questions"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            # Get document text
            document_text = self._get_document_text(url)
            
            # Chunk the document
            chunks = self._chunk_text(document_text)
            logger.info(f"Document split into {len(chunks)} chunks")
            
            # Find relevant chunks
            relevant_chunks = self._find_relevant_chunks(chunks, questions)
            context = "\n\n".join(relevant_chunks)
            
            # Prepare prompt
            questions_text = " | ".join(questions)
            
            prompt = f"""You are an expert insurance policy assistant. Answer the following questions based ONLY on the provided policy document context.

CRITICAL FORMAT: The questions are separated by " | ". Your answers MUST also be separated by " | " in the exact same order.

Questions: {questions_text}

Policy Document Context:
{context[:4000]}

Instructions:
- Answer each question based solely on the policy document provided
- Give direct, clear answers
- If information is not found in the document, say "Information not available in the policy document"
- Maintain the exact order of questions
- Separate each answer with " | "
- Be specific about waiting periods, coverage amounts, and conditions when mentioned

Answers:"""

            # Get response from Together API
            response = self._call_together_api(prompt)
            
            # Parse answers
            if " | " in response:
                answers = [answer.strip() for answer in response.split(" | ")]
            else:
                # Fallback: split by newlines or return single answer
                answers = [line.strip() for line in response.split('\n') if line.strip()]
                if not answers:
                    answers = [response.strip()]
            
            # Ensure we have the right number of answers
            while len(answers) < len(questions):
                answers.append("Unable to determine from the policy document.")
            answers = answers[:len(questions)]
            
            logger.info(f"Successfully processed {len(answers)} answers")
            return answers
            
        except Exception as e:
            logger.error(f"Error processing questions: {str(e)}")
            raise

# Global RAG engine instance
rag_engine = SimpleRAGEngine()

# ----- Token Verifier -----
def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    token = authorization.split("Bearer ")[-1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# ----- Lifespan Event Handler -----
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        rag_engine.initialize()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")

# ----- FastAPI App -----
app = FastAPI(
    title="Simple RAG Question Answering API",
    version="3.0.0",
    lifespan=lifespan
)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    try:
        logger.info(f"Received request with {len(request.questions)} questions")

        # Validation
        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="Questions list is empty")
        if len(request.questions) > 10:
            raise HTTPException(status_code=400, detail="Too many questions (max 10)")

        # Process questions
        answers = rag_engine.process_questions(request.documents, request.questions)

        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "initialized": rag_engine.initialized,
        "cached_docs": len(rag_engine.document_cache),
        "pdf_support": PDF_SUPPORT,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Simple RAG API is running", "version": "3.0.0", "pdf_support": PDF_SUPPORT}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
