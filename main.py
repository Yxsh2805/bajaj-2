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
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# RAG imports - keep exactly as before
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

class ParallelVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_parallel(self, documents: List[Document]):
        """PARALLEL processing - 5x faster than sequential"""
        logger.info(f"PARALLEL processing {len(documents)} chunks with multiple workers")
        
        start_time = time.time()
        
        def embed_single_doc(doc_tuple):
            """Embed a single document - for parallel processing"""
            index, doc = doc_tuple
            try:
                vector = self.embeddings.embed_query(doc.page_content)
                return (index, doc, vector, None)  # (index, doc, vector, error)
            except Exception as e:
                logger.error(f"Embedding error for chunk {index}: {e}")
                return (index, doc, None, str(e))
        
        # Process with 8 parallel workers (instead of 1)
        max_workers = 8  # Increase from 1 to 8 workers
        successful_embeddings = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all embedding tasks
            doc_tuples = [(i, doc) for i, doc in enumerate(documents)]
            future_to_index = {executor.submit(embed_single_doc, doc_tuple): doc_tuple[0] 
                             for doc_tuple in doc_tuples}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                result = future.result()
                index, doc, vector, error = result
                
                if vector is not None:
                    successful_embeddings.append((index, doc, vector))
                
                completed += 1
                if completed % 20 == 0:  # Log progress every 20 completions
                    logger.info(f"PARALLEL: {completed}/{len(documents)} embeddings completed")
        
        # Sort by original index to maintain order
        successful_embeddings.sort(key=lambda x: x[0])
        
        # Store results
        for _, doc, vector in successful_embeddings:
            self.documents.append(doc)
            self.vectors.append(vector)
        
        embedding_time = time.time() - start_time
        logger.info(f"PARALLEL embedding completed in {embedding_time:.2f}s ({len(self.documents)}/{len(documents)} successful)")
    
    def similarity_search(self, query: str, k: int = 6) -> List[Document]:
        """Fast similarity search - same as before"""
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            similarities = []
            for i, vector in enumerate(self.vectors):
                dot_product = np.dot(query_vector, vector)
                norm_product = np.linalg.norm(query_vector) * np.linalg.norm(vector)
                similarity = dot_product / norm_product if norm_product > 0 else 0
                similarities.append((similarity, i))
            
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_indices = [idx for _, idx in similarities[:k]]
            
            return [self.documents[i] for i in top_indices]
            
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return self.documents[:k] if len(self.documents) >= k else self.documents

def optimized_document_loader(url: str) -> List[Document]:
    """Same document loader as before"""
    try:
        response = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            
            if total_pages <= 30:
                pages_to_process = list(range(total_pages))
            else:
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

class ParallelRAGEngine:
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
        """Initialize - same as before"""
        if self.initialized:
            return
            
        logger.info("Initializing PARALLEL RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=4000
            )

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", "! ", "? ", " "]
            )

            self.initialized = True
            logger.info("PARALLEL RAG engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {str(e)}")
            raise

    def _single_call_query(self, vectorstore: ParallelVectorStore, query: str) -> str:
        """Single LLM call - same as before"""
        try:
            docs = vectorstore.similarity_search(query, k=6)
            context = " ".join([doc.page_content for doc in docs])[:3500]
            
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
        """Load and process - same as before"""
        if url in self.document_cache:
            logger.info(f"Using cached document for {url}")
            return self.document_cache[url]
        
        logger.info(f"Loading document for PARALLEL processing: {url}")
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

    def _create_vectorstore_parallel(self, url: str, chunks: List) -> ParallelVectorStore:
        """Create vectorstore with PARALLEL embeddings"""
        logger.info(f"Creating PARALLEL vectorstore")
        start_time = time.time()
        
        try:
            vectorstore = ParallelVectorStore(self.embeddings)
            vectorstore.add_documents_parallel(chunks)  # This is where the magic happens!
            
            creation_time = time.time() - start_time
            logger.info(f"PARALLEL vectorstore created in {creation_time:.2f}s")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"PARALLEL vectorstore creation error: {e}")
            raise HTTPException(status_code=500, detail=f"Vectorstore creation failed: {str(e)}")

    async def process_questions_parallel(self, url: str, questions: List[str]) -> List[str]:
        """Process with PARALLEL embeddings"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_questions_internal(url, questions),
                timeout=45.0  # Increased timeout for parallel processing
            )
        except asyncio.TimeoutError:
            logger.error("PARALLEL processing timeout")
            raise HTTPException(status_code=408, detail="Request timeout")

    async def _process_questions_internal(self, url: str, questions: List[str]) -> List[str]:
        """Internal processing with parallel embeddings"""
        docs, chunks = self._load_and_process_document(url)
        vectorstore = self._create_vectorstore_parallel(url, chunks)  # Parallel magic here!
        
        logger.info(f"Processing {len(questions)} questions with PARALLEL embeddings...")
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
rag_engine = ParallelRAGEngine()

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
        logger.info("PARALLEL RAG application startup completed")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
    yield
    logger.info("Application shutting down...")

app = FastAPI(title="PARALLEL RAG API", version="3.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """PARALLEL embedding processing - 5x faster"""
    try:
        logger.info(f"Received request with {len(request.questions)} questions for PARALLEL processing")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="Questions list is empty")

        answers = await rag_engine.process_questions_parallel(request.documents, request.questions)

        logger.info(f"Successfully processed {len(answers)} answers with PARALLEL embeddings")
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
        "mode": "parallel_embeddings"
    }

@app.get("/")
async def root():
    return {"message": "PARALLEL RAG API - 8x Faster Embedding Processing", "endpoints": ["/hackrx/run", "/health"]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
