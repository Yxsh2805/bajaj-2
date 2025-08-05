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

# RAG imports - back to Together.AI for speed
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

class SpeedOptimizedVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_fast(self, documents: List[Document]):
        """SPEED OPTIMIZED - 30 second target"""
        logger.info(f"SPEED MODE: Processing {len(documents)} chunks with 10 workers")
        
        start_time = time.time()
        
        def embed_minimal_retry(doc):
            """Minimal retry for maximum speed"""
            try:
                return self.embeddings.embed_query(doc.page_content)
            except Exception as e:
                # Only ONE retry to keep speed up
                try:
                    time.sleep(0.1)  # Very short pause
                    return self.embeddings.embed_query(doc.page_content)
                except:
                    logger.warning(f"Fast fail: {str(e)[:30]}")
                    return None
        
        # 10 workers for speed (Together.AI can handle this)
        with ThreadPoolExecutor(max_workers=10) as executor:
            vectors = list(executor.map(embed_minimal_retry, documents))
        
        # Store results quickly
        successful_count = 0
        for doc, vector in zip(documents, vectors):
            if vector is not None:
                self.documents.append(doc)
                self.vectors.append(vector)
                successful_count += 1
        
        embedding_time = time.time() - start_time
        success_rate = (successful_count / len(documents)) * 100
        logger.info(f"SPEED MODE: {embedding_time:.1f}s, {successful_count}/{len(documents)} chunks ({success_rate:.1f}% success)")
    
    def similarity_search(self, query: str, k: int = 6) -> List[Document]:
        """Fast similarity search"""
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Fast dot product similarity (skip normalization for speed)
            similarities = [np.dot(query_vector, vec) for vec in self.vectors]
            top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
            
            return [self.documents[i] for i in top_indices]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self.documents[:k] if len(self.documents) >= k else self.documents

def speed_document_loader(url: str) -> List[Document]:
    """Speed optimized document loader"""
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})  # Reduced timeout
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            
            # SPEED: Process fewer pages
            if total_pages <= 20:
                pages_to_process = list(range(total_pages))
            else:
                # Minimal sampling for speed
                first_pages = list(range(12))  # Fewer pages
                middle_pages = list(range(total_pages//3, total_pages//3 + 5))
                last_pages = list(range(total_pages-8, total_pages))
                pages_to_process = sorted(set(first_pages + middle_pages + last_pages))
            
            text = ""
            for i in pages_to_process:
                if i < len(pdf_reader.pages):
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text.strip():
                        text += page_text + "\n\n"
            
            logger.info(f"SPEED: Processed {len(pages_to_process)}/{total_pages} pages")
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "pdf"})]
        
        elif url_lower.endswith('.docx') or 'wordprocessingml' in content_type:
            docx_file = io.BytesIO(response.content)
            doc = DocxDocument(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "docx"})]
        
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            return [Document(page_content=text[:80000], metadata={"source": url, "type": "html"})]  # Reduced text limit
        
    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise

class SpeedRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing SPEED RAG engine...")
        
        try:
            # Together.AI for both chat and embeddings (SPEED!)
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            # Back to Together.AI embeddings - faster and more reliable for your use case
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=3000  # Reduced for speed
            )

            # SPEED OPTIMIZED chunking
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Smaller chunks for speed
                chunk_overlap=80,   # Less overlap
                separators=["\n\n", "\n", ". ", " "]  # Fewer separators
            )

            self.initialized = True
            logger.info("SPEED RAG engine ready!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _speed_query(self, vectorstore: SpeedOptimizedVectorStore, query: str) -> str:
        """Speed optimized query"""
        docs = vectorstore.similarity_search(query, k=6)  # Fewer docs for speed
        context = " ".join([doc.page_content for doc in docs])[:2800]  # Shorter context
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Shorter system prompt for speed
        system_content = """You are an insurance policy expert. Answer questions separated by " | " with answers separated by " | " in the same order. Be accurate and include specific details from the document."""

        human_content = f"""Questions: {query}\n\nContext: {context}\n\nAnswers separated by " | "."""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        
        response = self.chat_model.invoke(messages)
        return response.content

    async def process_speed(self, url: str, questions: List[str]) -> List[str]:
        """SPEED processing - 30 second target"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=30.0  # HARD 30 second limit
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="30 second timeout exceeded")

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("CACHED - INSTANT!")
        else:
            docs = speed_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            # AGGRESSIVE chunk limiting for 30-second target
            if len(chunks) > 50:
                # Take first 30 and last 20 chunks only
                chunks = chunks[:30] + chunks[-20:]
                logger.info(f"SPEED: Limited to {len(chunks)} chunks for 30s target")
            
            vectorstore = SpeedOptimizedVectorStore(self.embeddings)
            vectorstore.add_documents_fast(chunks)
            
            # Cache aggressively
            self.vectorstore_cache[url_hash] = vectorstore
        
        batch_query = " | ".join(questions)
        
        query_start = time.time()
        response = self._speed_query(vectorstore, batch_query)
        query_time = time.time() - query_start
        
        logger.info(f"LLM: {query_time:.1f}s")
        
        # Fast answer parsing
        answers = [ans.strip() for ans in response.split(" | ")]
        
        while len(answers) < len(questions):
            answers.append("Information not found.")
        
        return answers[:len(questions)]

# Global engine
rag_engine = SpeedRAGEngine()

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
        logger.info("SPEED RAG application ready")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="SPEED RAG API", version="3.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """SPEED processing - 30 second target"""
    try:
        logger.info(f"SPEED MODE: {len(request.questions)} questions - 30s target")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        start_time = time.time()
        answers = await rag_engine.process_speed(request.documents, request.questions)
        total_time = time.time() - start_time

        logger.info(f"SPEED MODE: Completed in {total_time:.1f}s - Target: 30s")
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
        "mode": "speed_optimized",
        "target_time": "30_seconds",
        "embedding_provider": "Together.AI (Speed Mode)"
    }

@app.get("/")
async def root():
    return {"message": "SPEED RAG API - 30 Second Target"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)



