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
from concurrent.futures import ThreadPoolExecutor
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

class IntelligentVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_batch(self, documents: List[Document]):
        """Add documents in optimized batches"""
        batch_size = 20  # Aggressive batching for speed
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_vectors = []
            
            # Process batch of documents
            for doc in batch:
                vector = self.embeddings.embed_query(doc.page_content)
                batch_vectors.append(vector)
                self.documents.append(doc)
            
            self.vectors.extend(batch_vectors)
            
            if (i // batch_size + 1) % 3 == 0:  # Log every 3rd batch
                logger.info(f"Processed embedding batch {i // batch_size + 1}")
    
    def similarity_search(self, query: str, k: int = 6) -> List[Document]:
        """High-quality similarity search"""
        if not self.vectors:
            return []
        
        query_vector = self.embeddings.embed_query(query)
        similarities = cosine_similarity([query_vector], self.vectors)[0]
        
        # Enhanced filtering for better results
        threshold = 0.05  # Lower threshold for more coverage
        top_indices = np.argsort(similarities)[-12:][::-1]  # Get top 12, filter to 6
        
        filtered_indices = [i for i in top_indices if similarities[i] > threshold][:k]
        return [self.documents[i] for i in filtered_indices if i < len(self.documents)]

def smart_document_loader(url: str) -> List[Document]:
    """Intelligent document loading with smart sampling"""
    try:
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        # Handle PDF with intelligent processing
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            
            # Smart page selection strategy
            if total_pages <= 25:
                pages_to_process = list(range(total_pages))
            else:
                # Intelligent sampling: first 15, last 10, middle samples
                first_pages = list(range(15))
                last_pages = list(range(total_pages-10, total_pages))
                middle_step = max(1, (total_pages-25)//10)
                middle_pages = list(range(15, total_pages-10, middle_step))[:10]
                pages_to_process = sorted(set(first_pages + middle_pages + last_pages))
            
            text = ""
            for i in pages_to_process:
                if i < len(pdf_reader.pages):
                    page_text = pdf_reader.pages[i].extract_text()
                    text += page_text + "\n\n"  # Double newline for better chunking
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i+1} pages")
            
            logger.info(f"Smart sampling: processed {len(pages_to_process)}/{total_pages} pages")
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "pdf", "pages": len(pages_to_process)})]
        
        # Handle other file types (DOCX, EML, HTML) - same as before
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

class IntelligentRAGEngine:
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
        """Initialize with optimal settings for 30-second processing"""
        if self.initialized:
            return
            
        logger.info("Initializing intelligent RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=3000
            )

            # Optimized chunking for speed and quality balance
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,  # Larger chunks = fewer embeddings
                chunk_overlap=120, # Sufficient overlap for context
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )

            self.initialized = True
            logger.info("Intelligent RAG engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {str(e)}")
            raise

    def _intelligent_query(self, vectorstore: IntelligentVectorStore, query: str) -> str:
        """Intelligent query processing with enhanced context"""
        try:
            docs = vectorstore.similarity_search(query, k=6)  # More context for accuracy
            context = " ".join([doc.page_content for doc in docs])[:3000]  # Larger context window
            
            system_prompt = """You are an expert document analysis assistant with exceptional accuracy.

CRITICAL INSTRUCTIONS:
- Input questions are separated by " | "
- Output answers MUST be separated by " | " in the same order
- Provide comprehensive, accurate answers based strictly on document content
- Include specific details, numbers, percentages, and quotes when available
- If information is not in document, state "Information not available in provided document"
- Maintain exact question order in responses
- Use technical precision and specific terminology from the document

Quality Standards:
- Prioritize accuracy and completeness
- Include relevant context and supporting details
- Reference specific sections, clauses, or provisions when possible
- Ensure each answer comprehensively addresses its question"""

            human_prompt = f"""Document Context: {context}

Questions: {query}

Provide detailed, accurate answers based on the document content."""

            from langchain_core.messages import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.chat_model.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            raise

    def _load_and_process_document(self, url: str) -> tuple:
        """Load and process with intelligent caching"""
        if url in self.document_cache:
            logger.info(f"Using cached document for {url}")
            return self.document_cache[url]
        
        logger.info(f"Loading document: {url}")
        start_time = time.time()
        
        try:
            docs = smart_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            load_time = time.time() - start_time
            logger.info(f"Document processed in {load_time:.2f}s ({len(chunks)} chunks)")
            
            # Cache management
            if len(self.document_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.document_cache))
                del self.document_cache[oldest_key]
            
            self.document_cache[url] = (docs, chunks)
            return docs, chunks
            
        except Exception as e:
            logger.error(f"Failed to load document {url}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load document: {str(e)}")

    def _create_vectorstore_fast(self, url: str, chunks: List) -> IntelligentVectorStore:
        """Fast vectorstore creation with intelligent batching"""
        logger.info(f"Creating vectorstore for {url}")
        start_time = time.time()
        
        try:
            vectorstore = IntelligentVectorStore(self.embeddings)
            vectorstore.add_documents_batch(chunks)
            
            creation_time = time.time() - start_time
            logger.info(f"Vectorstore created in {creation_time:.2f}s")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Vectorstore creation error: {e}")
            raise HTTPException(status_code=500, detail="Failed to create vectorstore")

    async def process_questions_intelligent(self, url: str, questions: List[str]) -> List[str]:
        """Intelligent processing with timeout protection"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        total_start_time = time.time()
        
        try:
            # Process with 25-second timeout for safety
            return await asyncio.wait_for(
                self._process_questions_internal(url, questions),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            logger.error("Processing timeout - document too complex")
            raise HTTPException(status_code=408, detail="Request timeout - please try a smaller document or fewer questions")

    async def _process_questions_internal(self, url: str, questions: List[str]) -> List[str]:
        """Internal processing logic"""
        docs, chunks = self._load_and_process_document(url)
        vectorstore = self._create_vectorstore_fast(url, chunks)
        
        logger.info(f"Processing {len(questions)} questions intelligently...")
        batch_query = " | ".join(questions)
        
        query_start_time = time.time()
        batch_result = self._intelligent_query(vectorstore, batch_query)
        query_time = time.time() - query_start_time
        
        logger.info(f"Intelligent query completed in {query_time:.2f}s")
        
        answers = [answer.strip() for answer in batch_result.split(" | ")]
        
        # Quality validation
        if len(answers) != len(questions):
            logger.warning(f"Answer count mismatch: {len(questions)} questions, {len(answers)} answers")
            if len(answers) < len(questions):
                for i in range(len(answers), len(questions)):
                    answers.append("Unable to process this question - please rephrase or try individually.")
            answers = answers[:len(questions)]
        
        return answers

# Global engine instance
rag_engine = IntelligentRAGEngine()

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
        logger.info("Intelligent RAG application startup completed")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
    yield
    logger.info("Application shutting down...")

app = FastAPI(title="Intelligent Batch RAG API", version="2.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Intelligent batch processing endpoint"""
    try:
        logger.info(f"Received request with {len(request.questions)} questions")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="Questions list is empty")
        if len(request.questions) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 questions allowed per request")

        answers = await rag_engine.process_questions_intelligent(request.documents, request.questions)

        logger.info(f"Successfully processed {len(answers)} answers")
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
        "mode": "intelligent_batch"
    }

@app.get("/")
async def root():
    return {"message": "Intelligent Batch RAG API", "endpoints": ["/hackrx/run", "/health"]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
