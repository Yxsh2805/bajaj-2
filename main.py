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

class AccuracyImprovedVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_improved(self, documents: List[Document]):
        """Improved accuracy while maintaining speed"""
        logger.info(f"ACCURACY+SPEED: Processing {len(documents)} chunks with 8 workers")
        
        start_time = time.time()
        
        def embed_with_smart_retry(doc):
            """Smart retry - 2 attempts with different strategies"""
            for attempt in range(2):
                try:
                    if attempt == 0:
                        # First attempt - normal
                        return self.embeddings.embed_query(doc.page_content)
                    else:
                        # Second attempt - shorter delay
                        time.sleep(0.15)
                        return self.embeddings.embed_query(doc.page_content)
                except Exception as e:
                    if attempt == 1:
                        logger.warning(f"Embed failed after 2 attempts: {str(e)[:40]}")
                        return None
        
        # 8 workers - balance between speed and stability
        with ThreadPoolExecutor(max_workers=8) as executor:
            vectors = list(executor.map(embed_with_smart_retry, documents))
        
        # Store results
        successful_count = 0
        for doc, vector in zip(documents, vectors):
            if vector is not None:
                self.documents.append(doc)
                self.vectors.append(vector)
                successful_count += 1
        
        embedding_time = time.time() - start_time
        success_rate = (successful_count / len(documents)) * 100
        logger.info(f"ACCURACY+SPEED: {embedding_time:.1f}s, {successful_count}/{len(documents)} chunks ({success_rate:.1f}% success)")
    
    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """Improved similarity search - back to cosine similarity"""
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Back to proper cosine similarity for accuracy
            similarities = []
            query_norm = np.linalg.norm(query_vector)
            
            for i, vector in enumerate(self.vectors):
                vector_norm = np.linalg.norm(vector)
                if query_norm > 0 and vector_norm > 0:
                    cos_sim = np.dot(query_vector, vector) / (query_norm * vector_norm)
                    similarities.append((cos_sim, i))
                else:
                    similarities.append((0.0, i))
            
            # Sort and return top k (increased from 6 to 8)
            similarities.sort(reverse=True)
            top_indices = [idx for _, idx in similarities[:k]]
            
            return [self.documents[i] for i in top_indices]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return self.documents[:k] if len(self.documents) >= k else self.documents

def improved_document_loader(url: str) -> List[Document]:
    """Improved document loader - better page coverage"""
    try:
        response = requests.get(url, timeout=12, headers={'User-Agent': 'Mozilla/5.0'})  # Restored timeout
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            
            # IMPROVED: Better page sampling for accuracy
            if total_pages <= 25:
                pages_to_process = list(range(total_pages))
            else:
                # More balanced sampling
                first_pages = list(range(15))  # More from start (was 12)
                middle_pages = list(range(total_pages//3, total_pages//3 + 8))  # More from middle (was 5)
                last_pages = list(range(total_pages-10, total_pages))  # More from end (was 8)
                pages_to_process = sorted(set(first_pages + middle_pages + last_pages))
            
            text = ""
            for i in pages_to_process:
                if i < len(pdf_reader.pages):
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text.strip():
                        text += page_text + "\n\n"
            
            logger.info(f"IMPROVED: Processed {len(pages_to_process)}/{total_pages} pages")
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
            return [Document(page_content=text[:90000], metadata={"source": url, "type": "html"})]  # Increased limit
        
    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise

class AccuracyImprovedRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing ACCURACY IMPROVED RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=3200  # Slightly increased for better answers
            )

            # IMPROVED chunking - better balance
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1050,    # Increased from 1000
                chunk_overlap=100,  # Increased from 80
                separators=["\n\n", "\n", ". ", "! ", "? ", " "]  # Full separators back
            )

            self.initialized = True
            logger.info("ACCURACY IMPROVED RAG engine ready!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _improved_query(self, vectorstore: AccuracyImprovedVectorStore, query: str) -> str:
        """Improved query with better context"""
        docs = vectorstore.similarity_search(query, k=8)  # More context docs
        context = " ".join([doc.page_content for doc in docs])[:3000]  # More context chars
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Better system prompt for accuracy
        system_content = """You are an expert insurance policy analyst with high accuracy standards.

CRITICAL INSTRUCTIONS:
- Input questions are separated by " | "
- Output answers MUST be separated by " | " in the same order
- Provide accurate, detailed answers based on the document content
- Include specific numbers, time periods, conditions, and percentages when available
- If information is not in the document, state "Information not available in provided document"
- Keep answers comprehensive but concise

QUALITY REQUIREMENTS:
- Start with the key information first
- Include specific details like amounts, time periods, percentages
- Reference exact conditions and requirements from the document
- Ensure each answer fully addresses its corresponding question
- Use precise language from the source document

CRITICAL: Separate each answer with " | " and maintain exact question order."""

        human_content = f"""Answer these questions accurately based on the document:

Questions: {query}

Document Context: {context}

Provide detailed, accurate answers separated by " | " in the same order."""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        
        response = self.chat_model.invoke(messages)
        return response.content

    async def process_improved(self, url: str, questions: List[str]) -> List[str]:
        """Improved processing - better accuracy within time budget"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=30.0  # Keep 30s hard limit
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="30 second timeout exceeded")

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("CACHED - INSTANT!")
        else:
            docs = improved_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            # IMPROVED: More chunks for better accuracy
            if len(chunks) > 65:  # Increased from 50 to 65
                # Better chunk selection strategy
                keep_first = int(len(chunks) * 0.5)   # 50% from start (was 60%)
                keep_middle = int(len(chunks) * 0.15) # 15% from true middle
                keep_last = int(len(chunks) * 0.25)   # 25% from end (was 40%)
                
                middle_start = len(chunks) // 2 - keep_middle // 2
                chunks = (chunks[:keep_first] + 
                         chunks[middle_start:middle_start + keep_middle] + 
                         chunks[-keep_last:])
                         
                logger.info(f"IMPROVED: {len(chunks)} chunks selected (better coverage)")
            
            vectorstore = AccuracyImprovedVectorStore(self.embeddings)
            vectorstore.add_documents_improved(chunks)
            
            # Cache for future use
            self.vectorstore_cache[url_hash] = vectorstore
        
        batch_query = " | ".join(questions)
        
        query_start = time.time()
        response = self._improved_query(vectorstore, batch_query)
        query_time = time.time() - query_start
        
        logger.info(f"LLM: {query_time:.1f}s")
        
        # Improved answer parsing
        answers = []
        raw_splits = response.split(" | ")
        
        for split in raw_splits:
            cleaned = split.strip()
            if cleaned and not cleaned.lower().startswith(('question:', 'answer:')):
                answers.append(cleaned)
        
        # Ensure correct count
        while len(answers) < len(questions):
            answers.append("Unable to find specific information in the provided document.")
        
        return answers[:len(questions)]

# Global engine
rag_engine = AccuracyImprovedRAGEngine()

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
        logger.info("ACCURACY IMPROVED RAG application ready")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="ACCURACY IMPROVED RAG API", version="3.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Improved accuracy while maintaining speed"""
    try:
        logger.info(f"ACCURACY+SPEED: {len(request.questions)} questions - targeting 25-28s")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        start_time = time.time()
        answers = await rag_engine.process_improved(request.documents, request.questions)
        total_time = time.time() - start_time

        logger.info(f"ACCURACY+SPEED: Completed in {total_time:.1f}s")
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
        "mode": "accuracy_improved",
        "target_time": "25-28_seconds",
        "embedding_provider": "Together.AI (Accuracy+Speed Mode)"
    }

@app.get("/")
async def root():
    return {"message": "ACCURACY IMPROVED RAG API - Better Balance"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
