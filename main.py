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

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# RAG imports
from langchain_together import ChatTogether
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LOCAL EMBEDDING imports
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_TOKEN = "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8"

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

class LocalEmbeddingVectorStore:
    def __init__(self):
        # Load the local embedding model ONCE when the class is created
        logger.info("Loading local embedding model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # 80MB model, fast and accurate
        logger.info("Local embedding model loaded successfully")
        
        self.documents = []
        self.vectors = []
    
    def add_documents_local_batch(self, documents: List[Document]):
        """ULTRA-FAST: Process ALL documents in ONE batch call"""
        logger.info(f"LOCAL BATCH: Processing {len(documents)} chunks in ONE call")
        
        start_time = time.time()
        
        # Extract all text content
        texts = [doc.page_content for doc in documents]
        
        # SINGLE batch embedding call - NO individual API calls!
        logger.info("Generating embeddings with local model...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,  # Process 32 chunks simultaneously
            show_progress_bar=False,  # Disable for cleaner logs
            convert_to_numpy=True
        )
        
        # Store results
        self.documents = documents
        self.vectors = embeddings
        
        embedding_time = time.time() - start_time
        logger.info(f"LOCAL BATCH: Completed in {embedding_time:.2f}s ({len(documents)} chunks)")
    
    def similarity_search(self, query: str, k: int = 6) -> List[Document]:
        """Fast local similarity search"""
        if len(self.vectors) == 0:
            return []
        
        # Single query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Fast similarity calculation
        similarities = cosine_similarity(query_embedding, self.vectors)[0]
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        return [self.documents[i] for i in top_indices]

def optimized_document_loader(url: str) -> List[Document]:
    """Optimized document loading"""
    try:
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            
            # Process all pages for documents up to 30 pages
            if total_pages <= 30:
                pages_to_process = list(range(total_pages))
            else:
                # Smart sampling for larger documents
                first_pages = list(range(15))
                middle_pages = list(range(total_pages//3, total_pages//3 + 10))
                last_pages = list(range(total_pages-10, total_pages))
                pages_to_process = sorted(set(first_pages + middle_pages + last_pages))
            
            text = ""
            for i in pages_to_process:
                if i < len(pdf_reader.pages):
                    page_text = pdf_reader.pages[i].extract_text()
                    text += page_text + "\n\n"
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i+1} pages")
            
            logger.info(f"Document processing: {len(pages_to_process)}/{total_pages} pages")
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "pdf"})]
        
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

class LocalEmbeddingRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.text_splitter = None
        self.initialized = False
        self.document_cache: Dict[str, Any] = {}
        self.max_cache_size = 3
    
    def _get_url_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def initialize(self):
        """Initialize LOCAL embedding RAG engine"""
        if self.initialized:
            return
            
        logger.info("Initializing LOCAL embedding RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            # Only need Together.AI for the chat model (NOT for embeddings)
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=4000
            )

            # Text splitter - same as before
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", "! ", "? ", " "]
            )

            self.initialized = True
            logger.info("LOCAL embedding RAG engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {str(e)}")
            raise

    def _single_call_query(self, vectorstore: LocalEmbeddingVectorStore, query: str) -> str:
        """Single LLM call for all questions"""
        try:
            docs = vectorstore.similarity_search(query, k=6)
            context = " ".join([doc.page_content for doc in docs])[:3500]
            
            # Use direct message approach
            from langchain_core.messages import HumanMessage, SystemMessage
            
            system_content = """You are a helpful assistant who is an expert in explaining insurance policies and their application to general queries.
Help the human with their queries related to the context of the policy provided. If you don't feel you have a concrete answer, say so.
Do not provide false information.

IMPORTANT OUTPUT FORMAT: You will receive multiple questions separated by " | ". Answer each question in the same order and separate your answers with " | ". Do not include any other separators, explanations, or formatting.

Example:
Input Questions: "What is the grace period for premium payment? | What is the waiting period for PED? | Does this policy cover maternity?"
Output Answers: "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits. | There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered. | Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months."

Guidelines:
- Keep answers short and direct, explain if needed—meat of the answer at the front.
- No generalizations; caveats must be clear and explicit.
- Example bad: "if adherence to certain rules and meets certain conditions" (too vague/generic).
- Example good: "if the insured is under the age of 60 and income is over 7lpa" (if supported by context!).
- Replicate definitions in the language of the source document.
- CRITICAL: Separate each answer with " | " and maintain the exact order of questions."""

            human_content = f"""Answer the following questions (separated by " | "):
{query}

Here are some relevant excerpts that might be useful for you in answering the questions:
{context}"""

            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=human_content)
            ]
            
            logger.info("Making SINGLE LLM API call for all questions...")
            response = self.chat_model.invoke(messages)
            logger.info("SINGLE LLM API call completed successfully")
            
            return response.content
            
        except Exception as e:
            logger.error(f"Single call query error: {e}")
            raise

    def _load_and_process_document(self, url: str) -> tuple:
        """Load and process document with caching"""
        if url in self.document_cache:
            logger.info(f"Using cached document for {url}")
            return self.document_cache[url]
        
        logger.info(f"Loading document for LOCAL processing: {url}")
        start_time = time.time()
        
        try:
            docs = optimized_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            load_time = time.time() - start_time
            logger.info(f"Document processed in {load_time:.2f}s ({len(chunks)} chunks)")
            
            if len(self.document_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.document_cache))
                del self.document_cache[oldest_key]
            
            self.document_cache[url] = (docs, chunks)
            return docs, chunks
            
        except Exception as e:
            logger.error(f"Failed to load document {url}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load document: {str(e)}")

    def _create_vectorstore_local(self, url: str, chunks: List) -> LocalEmbeddingVectorStore:
        """Create vectorstore with LOCAL embeddings"""
        logger.info(f"Creating LOCAL vectorstore")
        start_time = time.time()
        
        try:
            vectorstore = LocalEmbeddingVectorStore()
            vectorstore.add_documents_local_batch(chunks)
            
            creation_time = time.time() - start_time
            logger.info(f"LOCAL vectorstore created in {creation_time:.2f}s")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"LOCAL vectorstore creation error: {e}")
            raise HTTPException(status_code=500, detail=f"Vectorstore creation failed: {str(e)}")

    async def process_questions_local(self, url: str, questions: List[str]) -> List[str]:
        """Process ALL questions with LOCAL embeddings"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_questions_internal(url, questions),
                timeout=28.0
            )
        except asyncio.TimeoutError:
            logger.error("LOCAL processing timeout")
            raise HTTPException(status_code=408, detail="Request timeout")

    async def _process_questions_internal(self, url: str, questions: List[str]) -> List[str]:
        """Internal processing logic"""
        docs, chunks = self._load_and_process_document(url)
        vectorstore = self._create_vectorstore_local(url, chunks)
        
        logger.info(f"Processing {len(questions)} questions with LOCAL embeddings...")
        batch_query = " | ".join(questions)
        
        query_start_time = time.time()
        single_response = self._single_call_query(vectorstore, batch_query)
        query_time = time.time() - query_start_time
        
        logger.info(f"SINGLE LLM call completed in {query_time:.2f}s")
        
        answers = [answer.strip() for answer in single_response.split(" | ")]
        
        if len(answers) != len(questions):
            logger.warning(f"Answer count mismatch: {len(questions)} questions, {len(answers)} answers")
            if len(answers) < len(questions):
                for i in range(len(answers), len(questions)):
                    answers.append("Unable to provide answer based on available context.")
            answers = answers[:len(questions)]
        
        return answers

# Global RAG engine instance
rag_engine = LocalEmbeddingRAGEngine()

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
        logger.info("LOCAL embedding RAG application startup completed")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
    yield
    logger.info("Application shutting down...")

app = FastAPI(title="LOCAL Embedding RAG API", version="3.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """LOCAL embedding processing - ULTRA FAST"""
    try:
        logger.info(f"Received request with {len(request.questions)} questions for LOCAL processing")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="Questions list is empty")

        answers = await rag_engine.process_questions_local(request.documents, request.questions)

        logger.info(f"Successfully processed {len(answers)} answers with LOCAL embeddings")
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
        "mode": "local_embeddings"
    }

@app.get("/")
async def root():
    return {"message": "LOCAL Embedding RAG API - Ultra Fast Processing", "endpoints": ["/hackrx/run", "/health"]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
