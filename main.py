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
from concurrent.futures import ThreadPoolExecutor, as_completed

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

class QualityParallelVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_quality_parallel(self, documents: List[Document]):
        """Quality-focused parallel processing with robust error handling"""
        logger.info(f"QUALITY PARALLEL processing {len(documents)} chunks")
        
        start_time = time.time()
        
        def embed_with_quality_retries(doc_tuple):
            """Embed with quality-focused retries and error handling"""
            index, doc = doc_tuple
            max_retries = 3
            base_delay = 0.15  # Slightly more conservative delay
            
            for attempt in range(max_retries):
                try:
                    # Add jitter to prevent rate limiting
                    jitter = random.uniform(0, 0.08)
                    time.sleep(base_delay + jitter)
                    
                    vector = self.embeddings.embed_query(doc.page_content)
                    return (index, doc, vector, None, attempt + 1)  # Include attempt count
                    
                except Exception as e:
                    if "rate limit" in str(e).lower() or "429" in str(e):
                        # Exponential backoff for rate limits
                        wait_time = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
                        logger.warning(f"Rate limit hit - retry {attempt + 1} for chunk {index} after {wait_time:.1f}s")
                        time.sleep(wait_time)
                    elif attempt < max_retries - 1:
                        # Regular retry for other errors
                        wait_time = base_delay * (1.5 ** attempt)
                        logger.warning(f"Retry {attempt + 1} for chunk {index}: {str(e)[:100]}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Final failure for chunk {index} after {max_retries} attempts: {e}")
                        return (index, doc, None, str(e), max_retries)
        
        # Balanced workers for quality + speed
        max_workers = 6  # Sweet spot - fast but not overwhelming APIs
        successful_embeddings = []
        failed_chunks = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            doc_tuples = [(i, doc) for i, doc in enumerate(documents)]
            future_to_index = {executor.submit(embed_with_quality_retries, doc_tuple): doc_tuple[0] 
                             for doc_tuple in doc_tuples}
            
            completed = 0
            total_attempts = 0
            
            for future in as_completed(future_to_index):
                result = future.result()
                index, doc, vector, error, attempts = result
                total_attempts += attempts
                
                if vector is not None:
                    successful_embeddings.append((index, doc, vector))
                else:
                    failed_chunks.append((index, error))
                
                completed += 1
                if completed % 20 == 0:
                    success_rate = (completed - len(failed_chunks)) / completed * 100
                    logger.info(f"QUALITY: {completed}/{len(documents)} processed ({success_rate:.1f}% success)")
        
        # Quality check - ensure we have enough successful embeddings
        success_rate = len(successful_embeddings) / len(documents) * 100
        avg_attempts = total_attempts / len(documents)
        
        if success_rate < 85:  # Quality threshold
            logger.error(f"Low embedding success rate: {success_rate:.1f}% - this will impact answer quality")
            raise HTTPException(status_code=502, detail=f"Embedding quality too low: {success_rate:.1f}% success rate")
        
        # Sort by original index to maintain order
        successful_embeddings.sort(key=lambda x: x[0])
        
        # Store results
        for _, doc, vector in successful_embeddings:
            self.documents.append(doc)
            self.vectors.append(vector)
        
        embedding_time = time.time() - start_time
        logger.info(f"QUALITY embedding completed in {embedding_time:.2f}s ({len(self.documents)}/{len(documents)} successful - {success_rate:.1f}%, avg {avg_attempts:.1f} attempts)")
        
        if failed_chunks:
            logger.warning(f"Failed chunks: {[idx for idx, _ in failed_chunks[:5]]}{'...' if len(failed_chunks) > 5 else ''}")
    
    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """Enhanced similarity search for better quality"""
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # More sophisticated similarity calculation
            similarities = []
            query_norm = np.linalg.norm(query_vector)
            
            for i, vector in enumerate(self.vectors):
                vector_norm = np.linalg.norm(vector)
                if query_norm > 0 and vector_norm > 0:
                    # Cosine similarity
                    dot_product = np.dot(query_vector, vector)
                    similarity = dot_product / (query_norm * vector_norm)
                else:
                    similarity = 0.0
                similarities.append((similarity, i))
            
            # Sort and apply quality threshold
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Quality filter - only return chunks with decent similarity
            quality_threshold = 0.1  # Minimum similarity threshold
            quality_results = [(sim, idx) for sim, idx in similarities if sim > quality_threshold]
            
            # Return top k, but at least 4 for context richness
            num_results = min(max(k, 4), len(quality_results))
            top_indices = [idx for _, idx in quality_results[:num_results]]
            
            selected_docs = [self.documents[i] for i in top_indices]
            
            # Log quality metrics
            if quality_results:
                avg_similarity = np.mean([sim for sim, _ in quality_results[:num_results]])
                logger.info(f"Retrieved {num_results} chunks with avg similarity {avg_similarity:.3f}")
            
            return selected_docs
            
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            # Return first k documents as fallback
            return self.documents[:k] if len(self.documents) >= k else self.documents

# Same document loader - keep as is
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
            
            if total_pages <= 35:  # Slightly increased for quality
                pages_to_process = list(range(total_pages))
            else:
                first_pages = list(range(20))  # More pages from start
                middle_pages = list(range(total_pages//3, total_pages//3 + 10))
                last_pages = list(range(total_pages-15, total_pages))  # More from end
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

class QualityRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.document_cache: Dict[str, Any] = {}
        self.vectorstore_cache: Dict[str, Any] = {}  # Cache vectorstores too
        self.max_cache_size = 3
    
    def _get_url_hash(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def initialize(self):
        """Initialize with quality focus"""
        if self.initialized:
            return
            
        logger.info("Initializing QUALITY RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=4000
            )

            # Optimized chunking for quality
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,  # Slightly larger for better context
                chunk_overlap=150,  # More overlap for continuity
                separators=["\n\n", "\n", ". ", "! ", "? ", " "]
            )

            self.initialized = True
            logger.info("QUALITY RAG engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {str(e)}")
            raise

    def _enhanced_single_call_query(self, vectorstore: QualityParallelVectorStore, query: str) -> str:
        """Enhanced LLM call with better quality control"""
        try:
            docs = vectorstore.similarity_search(query, k=8)  # More context for quality
            context = " ".join([doc.page_content for doc in docs])[:4000]  # Larger context
            
            from langchain_core.messages import HumanMessage, SystemMessage
            
            # Enhanced system prompt for better quality
            system_content = """You are an expert insurance policy analyst with exceptional attention to detail and accuracy.

CRITICAL INSTRUCTIONS FOR HIGH-QUALITY RESPONSES:
- Input questions are separated by " | "
- Output answers MUST be separated by " | " in the exact same order
- Provide comprehensive, accurate answers based strictly on the document content
- Include specific details, numbers, percentages, time periods, and exact conditions
- If information is not in the document, clearly state "Information not available in the provided document"
- Maintain the exact question order in your responses
- Use precise terminology and specific references from the document

QUALITY GUIDELINES:
- Start each answer with the key information first
- Include specific details like time periods, amounts, percentages, and conditions
- Quote exact phrases from the document when relevant
- Avoid vague terms like "certain conditions" - be specific about what those conditions are
- If multiple conditions apply, list them clearly
- Ensure each answer directly and completely addresses its corresponding question

Example of HIGH-QUALITY formatting:
Input: "What is the grace period? | What is the waiting period for PED?"
Output: "A grace period of thirty (30) days is provided for premium payment after the due date, during which the policy remains in force without loss of continuity benefits. | There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception date for pre-existing diseases and their direct complications to be covered under this policy."

CRITICAL: Separate each answer with " | " and maintain exact question order."""

            human_content = f"""Based on the document context provided, answer the following questions with high accuracy and detail:

Questions (separated by " | "): {query}

Document Context:
{context}

Provide comprehensive, accurate answers separated by " | " in the exact same order as the questions."""

            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=human_content)
            ]
            
            logger.info("Making ENHANCED LLM API call for all questions...")
            response = self.chat_model.invoke(messages)
            logger.info("ENHANCED LLM API call completed successfully")
            
            return response.content
            
        except Exception as e:
            logger.error(f"Enhanced query error: {e}")
            raise

    def _load_and_process_document(self, url: str) -> tuple:
        """Load and process with caching"""
        if url in self.document_cache:
            logger.info(f"Using cached document for {url}")
            return self.document_cache[url]
        
        logger.info(f"Loading document for QUALITY processing: {url}")
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
                # Also clean vectorstore cache
                if oldest_key in self.vectorstore_cache:
                    del self.vectorstore_cache[oldest_key]
            
            self.document_cache[url] = (docs, chunks)
            return docs, chunks
            
        except Exception as e:
            logger.error(f"Failed to load document {url}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load document: {str(e)}")

    def _create_vectorstore_quality(self, url: str, chunks: List) -> QualityParallelVectorStore:
        """Create vectorstore with quality focus and caching"""
        url_hash = self._get_url_hash(url)
        
        # Check vectorstore cache
        if url_hash in self.vectorstore_cache:
            logger.info(f"Using cached vectorstore for {url}")
            return self.vectorstore_cache[url_hash]
        
        logger.info(f"Creating QUALITY vectorstore")
        start_time = time.time()
        
        try:
            vectorstore = QualityParallelVectorStore(self.embeddings)
            vectorstore.add_documents_quality_parallel(chunks)
            
            creation_time = time.time() - start_time
            logger.info(f"QUALITY vectorstore created in {creation_time:.2f}s")
            
            # Cache the vectorstore
            self.vectorstore_cache[url_hash] = vectorstore
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"QUALITY vectorstore creation error: {e}")
            raise HTTPException(status_code=500, detail=f"Vectorstore creation failed: {str(e)}")

    async def process_questions_quality(self, url: str, questions: List[str]) -> List[str]:
        """Process with quality focus"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_questions_internal(url, questions),
                timeout=75.0  # Generous timeout for quality
            )
        except asyncio.TimeoutError:
            logger.error("QUALITY processing timeout")
            raise HTTPException(status_code=408, detail="Request timeout - document processing took too long")

    async def _process_questions_internal(self, url: str, questions: List[str]) -> List[str]:
        """Internal processing with quality focus"""
        docs, chunks = self._load_and_process_document(url)
        vectorstore = self._create_vectorstore_quality(url, chunks)
        
        logger.info(f"Processing {len(questions)} questions with QUALITY focus...")
        batch_query = " | ".join(questions)
        
        query_start_time = time.time()
        single_response = self._enhanced_single_call_query(vectorstore, batch_query)
        query_time = time.time() - query_start_time
        
        logger.info(f"ENHANCED LLM call completed in {query_time:.2f}s")
        
        # Enhanced answer parsing with better splitting
        import re
        # Try multiple splitting strategies for robustness
        answers = []
        
        # Primary split on " | "
        raw_answers = single_response.split(" | ")
        
        # If count doesn't match, try regex split
        if len(raw_answers) != len(questions):
            raw_answers = re.split(r'\s*\|\s*', single_response.strip())
        
        # Clean and validate answers
        for ans in raw_answers:
            cleaned = ans.strip()
            if cleaned:
                answers.append(cleaned)
        
        # Ensure we have the right number of answers
        if len(answers) != len(questions):
            logger.warning(f"Answer count mismatch: {len(questions)} questions, {len(answers)} answers")
            if len(answers) < len(questions):
                # Add placeholder answers for missing ones
                for i in range(len(answers), len(questions)):
                    answers.append("Unable to provide a complete answer based on the available document content. Please verify the question or try rephrasing.")
            answers = answers[:len(questions)]
        
        # Quality check - ensure no answers are too short (likely parsing errors)
        for i, answer in enumerate(answers):
            if len(answer.strip()) < 20:  # Very short answers might be parsing errors
                logger.warning(f"Suspiciously short answer {i+1}: '{answer}'")
                answers[i] = "The answer appears incomplete. Please try rephrasing this question."
        
        return answers

# Global RAG engine instance
rag_engine = QualityRAGEngine()

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
        logger.info("QUALITY RAG application startup completed")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
    yield
    logger.info("Application shutting down...")

app = FastAPI(title="QUALITY RAG API", version="3.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Quality-focused processing with improved robustness"""
    try:
        logger.info(f"Received request with {len(request.questions)} questions for QUALITY processing")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="Questions list is empty")
        if len(request.questions) > 15:
            raise HTTPException(status_code=400, detail="Maximum 15 questions allowed for quality processing")

        answers = await rag_engine.process_questions_quality(request.documents, request.questions)

        logger.info(f"Successfully processed {len(answers)} QUALITY answers")
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
        "cached_vectorstores": len(rag_engine.vectorstore_cache),
        "timestamp": datetime.now().isoformat(),
        "mode": "quality_embeddings"
    }

@app.get("/")
async def root():
    return {"message": "QUALITY RAG API - High-Quality Answers with Optimized Speed", "endpoints": ["/hackrx/run", "/health"]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=120
    )
