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

class BatchOptimizedVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_batch_optimized(self, documents: List[Document]):
        """BATCH embedding optimization - much faster"""
        logger.info(f"BATCH: Processing {len(documents)} chunks in batches")
        
        start_time = time.time()
        
        # Extract all texts for batch processing
        texts = [doc.page_content for doc in documents]
        
        try:
            # BATCH EMBEDDING - single API call for all chunks
            logger.info("BATCH: Making single API call for all embeddings...")
            vectors = self.embeddings.embed_documents(texts)  # Batch call
            
            # Store all results
            for doc, vector in zip(documents, vectors):
                self.documents.append(doc)
                self.vectors.append(vector)
            
            embedding_time = time.time() - start_time
            logger.info(f"BATCH: {embedding_time:.1f}s for {len(documents)} chunks (MUCH FASTER!)")
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Fallback to individual processing if batch fails
            logger.info("Falling back to individual embedding...")
            self._fallback_individual_embedding(documents)
    
    def _fallback_individual_embedding(self, documents: List[Document]):
        """Fallback individual embedding with rate limiting"""
        successful_count = 0
        
        for i, doc in enumerate(documents):
            try:
                if i > 0 and i % 10 == 0:  # Rate limiting
                    time.sleep(0.5)
                
                vector = self.embeddings.embed_query(doc.page_content)
                self.documents.append(doc)
                self.vectors.append(vector)
                successful_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to embed chunk {i}: {str(e)[:30]}")
                continue
        
        logger.info(f"FALLBACK: {successful_count}/{len(documents)} chunks embedded")
    
    def similarity_search(self, query: str, k: int = 7) -> List[Document]:
        """Fast similarity search"""
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Efficient cosine similarity
            similarities = []
            query_norm = np.linalg.norm(query_vector)
            
            for i, vector in enumerate(self.vectors):
                vector_norm = np.linalg.norm(vector)
                if query_norm > 0 and vector_norm > 0:
                    cos_sim = np.dot(query_vector, vector) / (query_norm * vector_norm)
                    similarities.append((cos_sim, i))
                else:
                    similarities.append((0.0, i))
            
            # Sort and return top k
            similarities.sort(reverse=True)
            top_indices = [idx for _, idx in similarities[:k]]
            
            return [self.documents[i] for i in top_indices]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self.documents[:k] if len(self.documents) >= k else self.documents

def fast_document_loader_with_logging(url: str) -> List[Document]:
    """Document loader with detailed logging"""
    try:
        logger.info(f"DOCUMENT: Loading from {url}")
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = min(len(pdf_reader.pages), 25)
            
            # Process all pages
            text_parts = []
            for i in range(total_pages):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Error reading page {i+1}: {e}")
                    continue
            
            full_text = "\n\n".join(text_parts)
            char_count = len(full_text)
            logger.info(f"DOCUMENT: PDF - {total_pages} pages, {char_count:,} characters")
            
            # Log document preview
            preview = full_text[:500].replace('\n', ' ')
            logger.info(f"DOCUMENT PREVIEW: {preview}...")
            
            return [Document(page_content=full_text[:120000], metadata={
                "source": url, 
                "type": "pdf", 
                "pages": total_pages,
                "char_count": char_count
            })]
        
        elif url_lower.endswith('.docx') or 'wordprocessingml' in content_type:
            docx_file = io.BytesIO(response.content)
            doc = DocxDocument(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            logger.info(f"DOCUMENT: DOCX - {len(text):,} characters")
            preview = text[:500].replace('\n', ' ')
            logger.info(f"DOCUMENT PREVIEW: {preview}...")
            
            return [Document(page_content=text[:120000], metadata={"source": url, "type": "docx"})]
        
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            clean_text = '\n'.join(line for line in lines if line)
            
            logger.info(f"DOCUMENT: HTML - {len(clean_text):,} characters")
            preview = clean_text[:500].replace('\n', ' ')
            logger.info(f"DOCUMENT PREVIEW: {preview}...")
            
            return [Document(page_content=clean_text[:120000], metadata={"source": url, "type": "html"})]
        
    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise

class BatchOptimizedRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing BATCH-OPTIMIZED RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            
            # 8B model for speed
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.1-8B-Instruct-Turbo",
                temperature=0,
                max_tokens=2000
            )

            # Optimized chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1800,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " "]
            )

            self.initialized = True
            logger.info("BATCH-OPTIMIZED RAG engine ready!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def smart_chunk_selection(self, chunks: List[Document], target_chunks: int = 50) -> List[Document]:
        """Reduced to 50 chunks for faster batch embedding"""
        total = len(chunks)
        
        if total <= target_chunks:
            logger.info(f"COVERAGE: Using all {total} chunks")
            return chunks
        
        if total <= 80:
            # Dense sampling for smaller documents
            step = total / target_chunks
            indices = [int(i * step) for i in range(target_chunks)]
            selected = [chunks[i] for i in indices]
            logger.info(f"COVERAGE: Dense sampling {target_chunks}/{total} chunks")
            return selected
        else:
            # Strategic sampling for larger documents
            start_count = int(target_chunks * 0.40)  # 20 chunks
            middle_count = int(target_chunks * 0.25)  # 12-13 chunks  
            end_count = target_chunks - start_count - middle_count  # 17-18 chunks
            
            middle_start = total // 2 - middle_count // 2
            
            selected = (chunks[:start_count] + 
                       chunks[middle_start:middle_start + middle_count] + 
                       chunks[-end_count:])
            
            logger.info(f"COVERAGE: Strategic sampling {len(selected)}/{total} chunks")
            logger.info(f"Distribution: Start={start_count}, Middle={middle_count}, End={end_count}")
            return selected[:target_chunks]

    def _batch_optimized_query(self, vectorstore: BatchOptimizedVectorStore, query: str) -> str:
        """Optimized query for batch processing"""
        docs = vectorstore.similarity_search(query, k=7)
        context = " ".join([doc.page_content for doc in docs])[:2400]
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        system_content = """Insurance expert. Extract specific facts from document.

RULES:
- Questions separated by " | ", answers separated by " | "
- Include exact numbers, dates, percentages, conditions
- If not in document: "Information not available"
- Maintain exact question order
- Be precise and factual"""

        human_content = f"""Questions: {query}

Document Context: {context}

Provide answers separated by " | " in exact order:"""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        
        response = self.chat_model.invoke(messages)
        return response.content

    async def process_batch_optimized(self, url: str, questions: List[str]) -> List[str]:
        """Batch-optimized processing"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=29.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="29 second timeout exceeded")

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        total_start = time.time()
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("CACHED - INSTANT!")
        else:
            # Document loading with logging
            load_start = time.time()
            docs = fast_document_loader_with_logging(url)
            chunks = self.text_splitter.split_documents(docs)
            load_time = time.time() - load_start
            
            # Smart chunk selection (reduced to 50 for batch speed)
            select_start = time.time()
            selected_chunks = self.smart_chunk_selection(chunks, target_chunks=50)
            select_time = time.time() - select_start
            
            logger.info(f"TIMING: Load={load_time:.1f}s, Select={select_time:.1f}s")
            
            # BATCH embedding (single API call)
            embed_start = time.time()
            vectorstore = BatchOptimizedVectorStore(self.embeddings)
            vectorstore.add_documents_batch_optimized(selected_chunks)
            embed_time = time.time() - embed_start
            
            # Cache for future use
            self.vectorstore_cache[url_hash] = vectorstore
            logger.info(f"TIMING: Embed={embed_time:.1f}s")
        
        # Query processing
        batch_query = " | ".join(questions)
        
        query_start = time.time()
        response = self._batch_optimized_query(vectorstore, batch_query)
        query_time = time.time() - query_start
        
        total_time = time.time() - total_start
        logger.info(f"BATCH: Query={query_time:.1f}s, Total={total_time:.1f}s")
        
        # Answer parsing
        answers = []
        raw_splits = response.split(" | ")
        
        for split in raw_splits:
            cleaned = split.strip()
            if (cleaned and len(cleaned) > 5 and 
                not cleaned.lower().startswith(('question:', 'answer:', 'q:', 'a:'))):
                answers.append(cleaned)
        
        # Ensure correct count
        while len(answers) < len(questions):
            answers.append("Information not available in provided document.")
        
        logger.info(f"PARSED: {len(answers)} answers for {len(questions)} questions")
        return answers[:len(questions)]

# Global engine
rag_engine = BatchOptimizedRAGEngine()

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
        logger.info("BATCH-OPTIMIZED RAG application ready")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="BATCH-OPTIMIZED RAG API", version="5.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Batch-optimized processing - single API call for embeddings"""
    try:
        logger.info(f"BATCH: {len(request.questions)} questions (50 chunks, batch embedding)")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        start_time = time.time()
        answers = await rag_engine.process_batch_optimized(request.documents, request.questions)
        total_time = time.time() - start_time

        logger.info(f"BATCH: Completed in {total_time:.1f}s")
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
        "cache_entries": len(rag_engine.vectorstore_cache),
        "mode": "batch_optimized",
        "max_chunks": 50,
        "model": "Llama-3.1-8B-Instruct-Turbo",
        "embedding_strategy": "batch_single_api_call",
        "target_time": "<20_seconds"
    }

@app.get("/")
async def root():
    return {"message": "BATCH-OPTIMIZED RAG API - Single API Call for All Embeddings"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
