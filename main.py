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

class FinalOptimizedVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_final(self, documents: List[Document]):
        """FINAL OPTIMIZATION - Maximum speed with good accuracy"""
        logger.info(f"FINAL: Processing {len(documents)} chunks")
        
        start_time = time.time()
        
        def embed_fast_fail(doc):
            """One attempt only - no retries for maximum speed"""
            try:
                return self.embeddings.embed_query(doc.page_content)
            except Exception as e:
                return None  # Immediate failure, no retry
        
        # 8 workers - optimal for Together.AI
        with ThreadPoolExecutor(max_workers=8) as executor:
            vectors = list(executor.map(embed_fast_fail, documents))
        
        # Store only successful embeddings
        successful_count = 0
        for doc, vector in zip(documents, vectors):
            if vector is not None:
                self.documents.append(doc)
                self.vectors.append(vector)
                successful_count += 1
        
        embedding_time = time.time() - start_time
        logger.info(f"FINAL: {embedding_time:.1f}s, {successful_count} chunks embedded")
    
    def similarity_search(self, query: str, k: int = 7) -> List[Document]:
        """Optimized search"""
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Fast cosine similarity
            similarities = []
            query_norm = np.linalg.norm(query_vector)
            
            for i, vector in enumerate(self.vectors):
                vector_norm = np.linalg.norm(vector)
                if query_norm > 0 and vector_norm > 0:
                    cos_sim = np.dot(query_vector, vector) / (query_norm * vector_norm)
                    similarities.append((cos_sim, i))
            
            similarities.sort(reverse=True)
            return [self.documents[i] for _, i in similarities[:k]]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self.documents[:k] if len(self.documents) >= k else self.documents

def final_document_loader(url: str) -> List[Document]:
    """Final optimized document loader"""
    try:
        response = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = min(len(pdf_reader.pages), 25)
            
            # Aggressive page sampling for speed
            if total_pages <= 15:
                pages_to_process = list(range(total_pages))
            else:
                # Minimal high-value pages
                first_pages = list(range(8))  # First 8 pages
                middle_pages = list(range(total_pages//2-2, total_pages//2+3))  # 5 middle pages
                last_pages = list(range(total_pages-7, total_pages))  # Last 7 pages
                pages_to_process = sorted(set(first_pages + middle_pages + last_pages))
            
            text = ""
            for i in pages_to_process:
                if i < len(pdf_reader.pages):
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text.strip():
                        text += page_text + "\n\n"
            
            logger.info(f"FINAL: Processed {len(pages_to_process)}/{total_pages} pages")
            return [Document(page_content=text.strip()[:100000], metadata={"source": url, "type": "pdf"})]
        
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            return [Document(page_content=text[:100000], metadata={"source": url, "type": "html"})]
        
    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise

class FinalRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing FINAL RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            
            # CRITICAL: Use 8B model with BETTER PROMPTING
            self.chat_model = ChatTogether(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                temperature=0,
                max_tokens=2000
            )

            # Optimized chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1400,
                chunk_overlap=120,
                separators=["\n\n", "\n", ". ", " "]
            )

            self.initialized = True
            logger.info("FINAL RAG engine ready!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _final_query(self, vectorstore: FinalOptimizedVectorStore, query: str) -> str:
        """FINAL QUERY - Better prompting to prevent question repetition"""
        docs = vectorstore.similarity_search(query, k=7)
        context = " ".join([doc.page_content for doc in docs])[:2500]
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # CRITICAL: Better system prompt to prevent question repetition
        system_content = """You are an insurance policy expert. Your task is to answer questions, NOT repeat them.

CRITICAL RULES:
- Questions are separated by " | "
- Provide ONLY answers separated by " | " in the same order
- DO NOT include the questions in your response
- DO NOT start answers with "Question:" or "Q:" or repeat the question
- Extract specific facts, numbers, percentages, conditions from the document
- If information not found, say "Information not available in document"

Example:
Input: "What is the waiting period? | What is the sum insured?"
Output: "36 months for pre-existing diseases | Up to 50 lacs as per plan selected"

REMEMBER: Provide ONLY the answers, separated by " | ", NO questions."""

        human_content = f"""Document Context: {context}

Questions to answer: {query}

Provide only the answers separated by " | " (do not repeat questions):"""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        
        response = self.chat_model.invoke(messages)
        return response.content

    async def process_final(self, url: str, questions: List[str]) -> List[str]:
        """FINAL PROCESSING - Guaranteed under 30 seconds"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=28.0  # 28s timeout for safety
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="28 second timeout exceeded")

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        total_start = time.time()
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("CACHED!")
        else:
            # Load and process document
            docs = final_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            # FINAL OPTIMIZATION: Limit to 35 chunks max
            if len(chunks) > 35:
                # Strategic selection: 40% start, 25% middle, 35% end
                start_count = int(35 * 0.40)  # 14
                middle_count = int(35 * 0.25)  # 8-9
                end_count = 35 - start_count - middle_count  # 12-13
                
                middle_start = len(chunks) // 2 - middle_count // 2
                chunks = (chunks[:start_count] + 
                         chunks[middle_start:middle_start + middle_count] + 
                         chunks[-end_count:])
                
                logger.info(f"FINAL: Selected 35/{len(chunks)} chunks")
            
            # Fast embedding
            vectorstore = FinalOptimizedVectorStore(self.embeddings)
            vectorstore.add_documents_final(chunks)
            
            self.vectorstore_cache[url_hash] = vectorstore
        
        # Query with timing
        batch_query = " | ".join(questions)
        
        query_start = time.time()
        response = self._final_query(vectorstore, batch_query)
        query_time = time.time() - query_start
        
        total_time = time.time() - total_start
        logger.info(f"FINAL: Query={query_time:.1f}s, Total={total_time:.1f}s")
        
        # Clean answer parsing
        answers = []
        raw_splits = response.split(" | ")
        
        for split in raw_splits:
            cleaned = split.strip()
            # Remove any question repetition
            if cleaned and not any(q.lower().strip() in cleaned.lower() for q in questions):
                if len(cleaned) > 8:  # Meaningful answers only
                    answers.append(cleaned)
        
        # Ensure correct count
        while len(answers) < len(questions):
            answers.append("Information not available in document.")
        
        return answers[:len(questions)]

# Global engine
rag_engine = FinalRAGEngine()

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
        logger.info("FINAL RAG ready for submission")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="FINAL RAG API", version="FINAL", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """FINAL VERSION - Guaranteed under 30 seconds"""
    try:
        logger.info(f"FINAL: {len(request.questions)} questions - TARGET: <28s")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        start_time = time.time()
        answers = await rag_engine.process_final(request.documents, request.questions)
        total_time = time.time() - start_time

        logger.info(f"FINAL: SUCCESS in {total_time:.1f}s (Target: <28s)")
        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

@app.get("/health")
async def health_check():
    return {
        "status": "ready_for_submission",
        "mode": "final_optimized",
        "model": "Meta-Llama-3.1-8B-Instruct-Turbo", 
        "target_time": "<28_seconds",
        "max_chunks": 35
    }

@app.get("/")
async def root():
    return {"message": "FINAL RAG API - Ready for Submission"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
