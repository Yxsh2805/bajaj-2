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

class OptimalVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_optimal(self, documents: List[Document]):
        """Optimal processing for ≤30 pages within 30s"""
        logger.info(f"OPTIMAL: Processing ALL {len(documents)} chunks (≤30 pages, 30s limit)")
        
        start_time = time.time()
        
        def embed_time_aware(doc):
            """Time-aware embedding with smart retry"""
            for attempt in range(2):  # 2 attempts to balance speed vs accuracy
                try:
                    if attempt > 0:
                        time.sleep(0.1)  # Short delay for retry
                    
                    return self.embeddings.embed_query(doc.page_content)
                    
                except Exception as e:
                    if attempt == 1:  # Final attempt
                        logger.warning(f"Embed failed: {str(e)[:40]}")
                        return None
        
        # 10 workers - optimal for Together.AI speed vs stability
        with ThreadPoolExecutor(max_workers=10) as executor:
            vectors = list(executor.map(embed_time_aware, documents))
        
        # Store results
        successful_count = 0
        for doc, vector in zip(documents, vectors):
            if vector is not None:
                self.documents.append(doc)
                self.vectors.append(vector)
                successful_count += 1
        
        embedding_time = time.time() - start_time
        success_rate = (successful_count / len(documents)) * 100
        logger.info(f"OPTIMAL: {embedding_time:.1f}s, {successful_count}/{len(documents)} chunks ({success_rate:.1f}% success)")
        
        # Time warning if embedding takes too long
        if embedding_time > 20:
            logger.warning(f"Embedding took {embedding_time:.1f}s - may exceed 30s total")
    
    def similarity_search(self, query: str, k: int = 9) -> List[Document]:
        """Optimal similarity search for 30-page docs"""
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

def complete_30page_loader(url: str) -> List[Document]:
    """Complete processing for ≤30 pages with time awareness"""
    try:
        load_start = time.time()
        response = requests.get(url, timeout=12, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            
            if total_pages > 30:
                logger.warning(f"Document has {total_pages} pages - expected ≤30. Processing first 30 pages.")
                total_pages = 30  # Hard limit for time constraint
            
            # COMPLETE PROCESSING - all pages for accuracy
            text = ""
            for i in range(total_pages):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text.strip():
                        # Add page markers for better context
                        text += f"\n=== PAGE {i+1} ===\n{page_text}\n"
                except Exception as e:
                    logger.warning(f"Error reading page {i+1}: {e}")
                    continue
            
            load_time = time.time() - load_start
            logger.info(f"COMPLETE: Processed ALL {total_pages} pages in {load_time:.1f}s")
            
            return [Document(
                page_content=text.strip(), 
                metadata={"source": url, "type": "pdf", "pages": total_pages, "load_time": load_time}
            )]
        
        elif url_lower.endswith('.docx') or 'wordprocessingml' in content_type:
            docx_file = io.BytesIO(response.content)
            doc = DocxDocument(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            load_time = time.time() - load_start
            return [Document(
                page_content=text.strip(), 
                metadata={"source": url, "type": "docx", "load_time": load_time}
            )]
        
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            load_time = time.time() - load_start
            return [Document(
                page_content=text[:150000],  # Reasonable limit for HTML
                metadata={"source": url, "type": "html", "load_time": load_time}
            )]
        
    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise

class Optimal30PageRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing OPTIMAL RAG engine (≤30 pages, 30s limit)...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=3500  # Balanced for speed vs detail
            )

            # OPTIMAL chunking for 30-page docs
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=900,   # Sweet spot: detail + speed
                chunk_overlap=135, # 15% overlap for continuity
                separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " "]  # Good separation
            )

            self.initialized = True
            logger.info("OPTIMAL RAG engine ready!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _optimal_query(self, vectorstore: OptimalVectorStore, query: str) -> str:
        """Optimal query balancing accuracy and speed"""
        docs = vectorstore.similarity_search(query, k=9)  # Good context coverage
        context = " ".join([doc.page_content for doc in docs])[:3500]  # Optimal context size
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Optimized system prompt for 30-page docs
        system_content = """You are an expert insurance policy analyst specializing in comprehensive document analysis.

CRITICAL INSTRUCTIONS:
- Input questions are separated by " | "
- Output answers MUST be separated by " | " in the exact same order
- Provide accurate, detailed answers based on the complete document content
- Include specific numbers, time periods, conditions, percentages, and exact requirements
- Quote exact phrases when providing specific terms or conditions
- If information is not in the document, state "Information not available in provided document"
- Ensure each answer is comprehensive and addresses all aspects of the question

QUALITY REQUIREMENTS:
- Start with the most important information first
- Include specific details: amounts, time periods, percentages, conditions, exceptions
- Reference exact conditions, prerequisites, and limitations from the document
- Use precise terminology from the source document
- For complex conditions, explain the requirements step-by-step
- Include relevant cross-references when applicable

CRITICAL: Separate each answer with " | " and maintain exact question order."""

        human_content = f"""Based on the complete insurance policy document, provide comprehensive answers to these questions:

Questions: {query}

Complete Document Context:
{context}

Provide detailed, accurate answers separated by " | " in the same order."""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        
        response = self.chat_model.invoke(messages)
        return response.content

    async def process_optimal(self, url: str, questions: List[str]) -> List[str]:
        """Optimal processing for ≤30 pages within 30s"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=30.0  # Hard 30s limit
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="30 second timeout exceeded")

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        total_start = time.time()
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("CACHED - INSTANT OPTIMAL!")
        else:
            # Document loading phase
            docs = complete_30page_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            # Check time budget
            load_time = time.time() - total_start
            if load_time > 8:
                logger.warning(f"Document loading took {load_time:.1f}s - time budget tight")
            
            # NO CHUNK LIMITING for ≤30 pages - process ALL for max accuracy
            logger.info(f"OPTIMAL: Processing ALL {len(chunks)} chunks for maximum accuracy")
            
            # Time-aware embedding
            embed_start = time.time()
            vectorstore = OptimalVectorStore(self.embeddings)
            vectorstore.add_documents_optimal(chunks)
            
            embed_time = time.time() - embed_start
            total_time_so_far = time.time() - total_start
            
            logger.info(f"Time budget: {total_time_so_far:.1f}s used, {30 - total_time_so_far:.1f}s remaining")
            
            # Cache the complete vectorstore
            self.vectorstore_cache[url_hash] = vectorstore
        
        # Query phase
        batch_query = " | ".join(questions)
        
        query_start = time.time()
        response = self._optimal_query(vectorstore, batch_query)
        query_time = time.time() - query_start
        
        total_time = time.time() - total_start
        logger.info(f"OPTIMAL: Query {query_time:.1f}s, Total {total_time:.1f}s")
        
        # Enhanced answer parsing
        answers = []
        raw_splits = response.split(" | ")
        
        for split in raw_splits:
            cleaned = split.strip()
            # Filter out non-answers
            if (cleaned and 
                not cleaned.lower().startswith(('question:', 'answer:', 'based on', 'according to')) and
                len(cleaned) > 10):  # Minimum substantive length
                answers.append(cleaned)
        
        # Ensure correct count
        while len(answers) < len(questions):
            answers.append("Unable to find specific information in the provided document.")
        
        return answers[:len(questions)]

# Global engine
rag_engine = Optimal30PageRAGEngine()

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
        logger.info("OPTIMAL 30-PAGE RAG application ready")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="OPTIMAL 30-PAGE RAG API", version="3.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Optimal accuracy processing for documents ≤30 pages within 30s"""
    try:
        logger.info(f"OPTIMAL: {len(request.questions)} questions for ≤30 page document")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        start_time = time.time()
        answers = await rag_engine.process_optimal(request.documents, request.questions)
        total_time = time.time() - start_time

        logger.info(f"OPTIMAL: Completed in {total_time:.1f}s with full document processing")
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
        "mode": "optimal_30_pages",
        "target_pages": "≤30_pages",
        "time_limit": "30_seconds",
        "processing": "complete_document",
        "embedding_provider": "Together.AI (Optimal)"
    }

@app.get("/")
async def root():
    return {"message": "OPTIMAL RAG API - Max Accuracy for ≤30 Pages in ≤30 Seconds"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
