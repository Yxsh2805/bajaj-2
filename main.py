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
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableAssign, RunnableLambda

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_TOKEN = "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8"

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

class OptimizedSingleCallVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_optimized(self, documents: List[Document]):
        """Optimized embedding generation with controlled batching"""
        logger.info(f"Optimized processing {len(documents)} chunks for single LLM call")
        
        # Process embeddings in controlled batches
        batch_size = 12  # Optimal batch size for Together.AI stability
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_start = time.time()
            
            # Process batch with micro-delays for stability
            for j, doc in enumerate(batch):
                try:
                    vector = self.embeddings.embed_query(doc.page_content)
                    self.documents.append(doc)
                    self.vectors.append(vector)
                    
                    # Micro-delay to prevent rate limiting
                    if j < len(batch) - 1:
                        time.sleep(0.05)  # 50ms delay between calls
                        
                except Exception as e:
                    logger.error(f"Embedding error for chunk {len(self.documents)}: {e}")
                    continue
            
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) - 1) // batch_size + 1
            batch_time = time.time() - batch_start
            
            logger.info(f"Embedding batch {batch_num}/{total_batches} completed in {batch_time:.2f}s")
            
            # Inter-batch delay for stability
            if batch_num < total_batches:
                time.sleep(0.8)  # 800ms between batches
    
    def similarity_search(self, query: str, k: int = 6) -> List[Document]:
        """Fast similarity search for single LLM call"""
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Efficient similarity calculation using numpy
            similarities = []
            for i, vector in enumerate(self.vectors):
                # Compute cosine similarity efficiently
                dot_product = np.dot(query_vector, vector)
                norm_product = np.linalg.norm(query_vector) * np.linalg.norm(vector)
                similarity = dot_product / norm_product if norm_product > 0 else 0
                similarities.append((similarity, i))
            
            # Get top k documents efficiently
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_indices = [idx for _, idx in similarities[:k]]
            
            return [self.documents[i] for i in top_indices]
            
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return self.documents[:k] if len(self.documents) >= k else self.documents

def optimized_document_loader(url: str) -> List[Document]:
    """Optimized document loading for single call processing"""
    try:
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        # Handle PDF with optimized processing
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            
            # Smart page processing for speed
            if total_pages <= 30:
                pages_to_process = list(range(total_pages))
            else:
                # Strategic sampling for large documents
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
            
            logger.info(f"Optimized processing: {len(pages_to_process)}/{total_pages} pages")
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "pdf"})]
        
        # Handle DOCX files
        elif url_lower.endswith('.docx') or 'wordprocessingml' in content_type:
            docx_file = io.BytesIO(response.content)
            doc = DocxDocument(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "docx"})]
        
        # Handle EML files
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
        
        # Handle HTML files
        else:
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

class SingleCallRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.policy_prompt = None
        self.initialized = False
        self.document_cache: Dict[str, Any] = {}
        self.max_cache_size = 3
    
    def _get_url_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def initialize(self):
        """Initialize for single LLM call processing"""
        if self.initialized:
            return
            
        logger.info("Initializing single-call RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=4000
            )

            # Optimized chunking for single call
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Larger chunks for better context
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", "! ", "? ", " "]
            )

            # Your PERFECT prompt template
            self.policy_prompt = ChatPromptTemplate([
                ("system", """You are a helpful assistant who is an expert in explaining insurance policies and their application to general queries.
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
- CRITICAL: Separate each answer with " | " and maintain the exact order of questions."""),
                ("human", """Answer the following questions (separated by " | "):
{query}

Here are some relevant excerpts that might be useful for you in answering the questions:
{context}"""),
            ])

            self.initialized = True
            logger.info("Single-call RAG engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {str(e)}")
            raise

    def _single_call_query(self, vectorstore: OptimizedSingleCallVectorStore, query: str) -> str:
        """PERFECT: Single LLM call for all questions"""
        try:
            # Get relevant context documents
            docs = vectorstore.similarity_search(query, k=6)
            context = " ".join([doc.page_content for doc in docs])[:3500]
            
            # Build the chain for single LLM call
            def get_context(x):
                return context
            
            chain = (
                RunnableAssign({"context": RunnableLambda(get_context)}) |
                self.policy_prompt |
                self.chat_model |
                StrOutputParser()
            )
            
            # SINGLE API CALL for ALL questions
            logger.info("Making SINGLE LLM API call for all questions...")
            response = chain.invoke({"query": query})
            logger.info("SINGLE LLM API call completed successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"Single call query error: {e}")
            raise

    def _load_and_process_document(self, url: str) -> tuple:
        """Load and process document with caching"""
        if url in self.document_cache:
            logger.info(f"Using cached document for {url}")
            return self.document_cache[url]
        
        logger.info(f"Loading document for single-call processing: {url}")
        start_time = time.time()
        
        try:
            docs = optimized_document_loader(url)
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

    def _create_vectorstore_optimized(self, url: str, chunks: List) -> OptimizedSingleCallVectorStore:
        """Create optimized vectorstore for single LLM call"""
        logger.info(f"Creating optimized vectorstore for single LLM call")
        start_time = time.time()
        
        try:
            vectorstore = OptimizedSingleCallVectorStore(self.embeddings)
            vectorstore.add_documents_optimized(chunks)
            
            creation_time = time.time() - start_time
            logger.info(f"Optimized vectorstore created in {creation_time:.2f}s")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Optimized vectorstore creation error: {e}")
            raise HTTPException(status_code=500, detail=f"Vectorstore creation failed: {str(e)}")

    async def process_questions_single_call(self, url: str, questions: List[str]) -> List[str]:
        """Process ALL questions in SINGLE LLM call"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        total_start_time = time.time()
        
        try:
            # Process with timeout for Railway
            return await asyncio.wait_for(
                self._process_questions_internal(url, questions),
                timeout=28.0  # 28-second timeout for Railway
            )
        except asyncio.TimeoutError:
            logger.error("Single-call processing timeout")
            raise HTTPException(status_code=408, detail="Request timeout - document processing took too long")

    async def _process_questions_internal(self, url: str, questions: List[str]) -> List[str]:
        """Internal processing logic"""
        # Load and process document (optimized embedding generation)
        docs, chunks = self._load_and_process_document(url)
        vectorstore = self._create_vectorstore_optimized(url, chunks)
        
        # SINGLE LLM CALL for ALL questions
        logger.info(f"Processing {len(questions)} questions in SINGLE LLM call...")
        batch_query = " | ".join(questions)  # Your perfect approach
        
        query_start_time = time.time()
        single_response = self._single_call_query(vectorstore, batch_query)
        query_time = time.time() - query_start_time
        
        logger.info(f"SINGLE LLM call completed in {query_time:.2f}s")
        
        # Parse single response into individual answers
        answers = [answer.strip() for answer in single_response.split(" | ")]
        
        # Validate answer count
        if len(answers) != len(questions):
            logger.warning(f"Answer count mismatch: {len(questions)} questions, {len(answers)} answers")
            if len(answers) < len(questions):
                for i in range(len(answers), len(questions)):
                    answers.append("Unable to provide answer based on available context.")
            answers = answers[:len(questions)]
        
        return answers

# Global RAG engine instance
rag_engine = SingleCallRAGEngine()

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
        logger.info("Single-call RAG application startup completed")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
    yield
    logger.info("Application shutting down...")

app = FastAPI(title="Single-Call RAG API", version="3.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Single LLM call processing - handles unlimited questions"""
    try:
        logger.info(f"Received request with {len(request.questions)} questions for single-call processing")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="Questions list is empty")

        answers = await rag_engine.process_questions_single_call(request.documents, request.questions)

        logger.info(f"Successfully processed {len(answers)} answers with SINGLE LLM call")
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
        "mode": "single_llm_call"
    }

@app.get("/")
async def root():
    return {"message": "Single-Call RAG API - One LLM call for all questions", "endpoints": ["/hackrx/run", "/health"]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
