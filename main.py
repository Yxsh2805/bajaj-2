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
import asyncio

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
    
    def add_documents_single_batch(self, documents: List[Document]):
        """Add all documents in maximum batch size for ultimate speed"""
        logger.info(f"Processing {len(documents)} chunks in single ultra-batch")
        
        start_time = time.time()
        for doc in documents:
            vector = self.embeddings.embed_query(doc.page_content)
            self.documents.append(doc)
            self.vectors.append(vector)
        
        batch_time = time.time() - start_time
        logger.info(f"Ultra-batch embedding completed in {batch_time:.2f}s")
    
    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """Ultra-fast similarity search with maximum coverage"""
        if not self.vectors:
            return []
        
        query_vector = self.embeddings.embed_query(query)
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        
        # Maximum coverage strategy
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_indices if i < len(self.documents)]

def ultra_fast_document_loader(url: str) -> List[Document]:
    """Ultra-fast document loading with aggressive optimization"""
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        # Handle PDF with ultra-fast processing
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            
            # Ultra-aggressive sampling for maximum speed
            if total_pages <= 30:
                # Process all pages for reasonable documents
                pages_to_process = list(range(total_pages))
            else:
                # Ultra-fast sampling: strategic page selection
                first_third = list(range(0, min(10, total_pages)))
                middle_third = list(range(total_pages//3, total_pages//3 + 10))
                last_third = list(range(max(0, total_pages-10), total_pages))
                pages_to_process = sorted(set(first_third + middle_third + last_third))
            
            text = ""
            for i in pages_to_process:
                if i < len(pdf_reader.pages):
                    text += pdf_reader.pages[i].extract_text() + "\n\n"
            
            logger.info(f"Ultra-fast processing: {len(pages_to_process)}/{total_pages} pages")
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "pdf"})]
        
        # Handle other file types (same as Solution 1)
        elif url_lower.endswith('.docx') or 'wordprocessingml' in content_type:
            docx_file = io.BytesIO(response.content)
            doc = DocxDocument(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "docx"})]
        
        elif url_lower.endswith('.eml') or 'message/rfc822' in content_type:
            eml_content = response.content.decode('utf-8', errors='ignore')
            msg = email.message_from_string(eml_content)
            text = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
            else:
                text = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "eml"})]
        
        else:  # HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return [Document(page_content=' '.join(text.split()), metadata={"source": url, "type": "html"})]
        
    except Exception as e:
        logger.error(f"Failed to load document {url}: {e}")
        raise

class UltraFastRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.document_cache: Dict[str, Any] = {}
        self.max_cache_size = 3
    
    def _get_url_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def initialize(self):
        """Initialize for maximum speed processing"""
        if self.initialized:
            return
            
        logger.info("Initializing ultra-fast RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=4000  # Increased for handling many questions
            )

            # Ultra-optimized chunking for maximum speed
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Larger chunks = fewer total chunks
                chunk_overlap=150, # Balanced overlap
                separators=["\n\n", "\n", ". ", " "]  # Simplified separators
            )

            self.initialized = True
            logger.info("Ultra-fast RAG engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {str(e)}")
            raise

    def _ultra_fast_query(self, vectorstore: UltraFastVectorStore, query: str) -> str:
        """Ultra-fast query processing - handles unlimited questions"""
        try:
            docs = vectorstore.similarity_search(query, k=8)  # Maximum context
            context = " ".join([doc.page_content for doc in docs])[:4000]  # Large context window
            
            system_prompt = """You are an ultra-fast, high-accuracy document analysis expert.

CRITICAL INSTRUCTIONS:
- Input questions are separated by " | "
- Output answers MUST be separated by " | " in EXACT same order
- Handle ANY number of questions (no limits)
- Provide precise, detailed answers based strictly on document content
- Include specific numbers, percentages, dates, and technical details
- If information missing, state "Information not available in document"
- Maintain perfect question-answer order matching

Ultra-Fast Processing Guidelines:
- Prioritize speed while maintaining accuracy
- Use document-specific terminology and exact phrases
- Include relevant context for each answer
- Handle complex multi-part questions comprehensively"""

            human_prompt = f"""Document Context: {context}

Questions (separated by " | "): {query}

Provide comprehensive, accurate answers separated by " | " in the exact same order."""

            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.chat_model.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Ultra-fast query error: {e}")
            raise

    def _load_and_process_document(self, url: str) -> tuple:
        """Ultra-fast document loading with caching"""
        if url in self.document_cache:
            logger.info(f"Using cached document for {url}")
            return self.document_cache[url]
        
        logger.info(f"Loading document with ultra-fast processing: {url}")
        start_time = time.time()
        
        try:
            docs = ultra_fast_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            load_time = time.time() - start_time
            logger.info(f"Ultra-fast document processing: {load_time:.2f}s ({len(chunks)} chunks)")
            
            # Cache management
            if len(self.document_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.document_cache))
                del self.document_cache[oldest_key]
            
            self.document_cache[url] = (docs, chunks)
            return docs, chunks
            
        except Exception as e:
            logger.error(f"Failed to load document {url}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load document: {str(e)}")

    def _create_vectorstore_ultrafast(self, url: str, chunks: List) -> UltraFastVectorStore:
        """Ultra-fast vectorstore creation - single batch processing"""
        logger.info(f"Creating ultra-fast vectorstore for {url}")
        start_time = time.time()
        
        try:
            vectorstore = UltraFastVectorStore(self.embeddings)
            vectorstore.add_documents_single_batch(chunks)
            
            creation_time = time.time() - start_time
            logger.info(f"Ultra-fast vectorstore created in {creation_time:.2f}s")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Ultra-fast vectorstore creation error: {e}")
            raise HTTPException(status_code=500, detail="Failed to create vectorstore")

    async def process_unlimited_questions(self, url: str, questions: List[str]) -> List[str]:
        """Process unlimited questions in single batch - maximum speed"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        total_start_time = time.time()
        
        try:
            # Ultra-fast processing with 25-second timeout
            return await asyncio.wait_for(
                self._process_questions_ultrafast(url, questions),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            logger.error("Ultra-fast processing timeout")
            raise HTTPException(status_code=408, detail="Request timeout - even ultra-fast processing couldn't complete in time")

    async def _process_questions_ultrafast(self, url: str, questions: List[str]) -> List[str]:
        """Internal ultra-fast processing logic"""
        docs, chunks = self._load_and_process_document(url)
        vectorstore = self._create_vectorstore_ultrafast(url, chunks)
        
        logger.info(f"Processing {len(questions)} questions in single ultra-fast batch...")
        batch_query = " | ".join(questions)
        
        query_start_time = time.time()
        batch_result = self._ultra_fast_query(vectorstore, batch_query)
        query_time = time.time() - query_start_time
        
        logger.info(f"Ultra-fast query completed in {query_time:.2f}s")
        
        answers = [answer.strip() for answer in batch_result.split(" | ")]
        
        # Ultra-fast quality validation
        if len(answers) != len(questions):
            logger.warning(f"Answer count mismatch: {len(questions)} questions, {len(answers)} answers")
            if len(answers) < len(questions):
                for i in range(len(answers), len(questions)):
                    answers.append("Processing incomplete - question may be too complex.")
            answers = answers[:len(questions)]
        
        total_time = time.time() - total_start_time
        logger.info(f"Ultra-fast processing completed in {total_time:.2f}s")
        
        return answers

# Global engine instance
rag_engine = UltraFastRAGEngine()

def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid format")
    
    token = authorization.split("Bearer ")[-1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Bearer token")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        rag_engine.initialize()
        logger.info("Ultra-fast RAG application startup completed")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
    yield
    logger.info("Application shutting down...")

app = FastAPI(title="Ultra-Fast Single Batch RAG API", version="2.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Ultra-fast single batch processing - handles unlimited questions"""
    try:
        logger.info(f"Received request with {len(request.questions)} questions for ultra-fast processing")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="Questions list is empty")
        # No question limit - handles unlimited questions!

        answers = await rag_engine.process_unlimited_questions(request.documents, request.questions)

        logger.info(f"Successfully processed {len(answers)} answers with ultra-fast processing")
        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_initialized": rag_engine.initialized,
        "cached_documents": len(rag_engine.document_cache),
        "timestamp": datetime.now().isoformat(),
        "mode": "ultra_fast_single_batch"
    }

@app.get("/")
async def root():
    return {"message": "Ultra-Fast Single Batch RAG API", "endpoints": ["/hackrx/run", "/health"]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
