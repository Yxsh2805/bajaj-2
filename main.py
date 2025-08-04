import os
import time
import logging
import hashlib
import re
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

def clean_text_for_embedding(text: str) -> str:
    """Clean text to prevent 400 errors"""
    if not text or not text.strip():
        return "Empty content"
    
    # Remove problematic characters
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/]', ' ', text)
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Ensure minimum length
    if len(text.strip()) < 10:
        text = f"Short content: {text}"
    
    # Truncate if too long (Together.AI limit is ~8192 tokens, ~32k chars)
    if len(text) > 30000:
        text = text[:30000] + "..."
        logger.warning(f"Truncated long text to 30k chars")
    
    return text.strip()

class RobustVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_robust(self, documents: List[Document]):
        """Robust embedding with input validation"""
        logger.info(f"ROBUST: Processing {len(documents)} chunks with validation")
        
        start_time = time.time()
        
        def embed_with_validation(doc_tuple):
            """Embed with thorough validation and retry"""
            index, doc = doc_tuple
            
            # Clean and validate text
            clean_text = clean_text_for_embedding(doc.page_content)
            
            # Skip if text is too short or empty
            if len(clean_text.strip()) < 10:
                logger.warning(f"Skipping chunk {index}: too short")
                return None
            
            for attempt in range(3):  # 3 attempts with different strategies
                try:
                    if attempt == 0:
                        # First attempt - full text
                        return self.embeddings.embed_query(clean_text)
                    elif attempt == 1:
                        # Second attempt - truncate more aggressively
                        shorter_text = clean_text[:15000]
                        time.sleep(0.1)
                        return self.embeddings.embed_query(shorter_text)
                    else:
                        # Third attempt - much shorter
                        shortest_text = clean_text[:5000]
                        time.sleep(0.2)
                        return self.embeddings.embed_query(shortest_text)
                        
                except Exception as e:
                    error_str = str(e)
                    if "400" in error_str:
                        logger.warning(f"Chunk {index} attempt {attempt+1}: 400 error - {error_str[:60]}")
                        if attempt == 2:
                            return None
                        continue
                    elif "503" in error_str or "502" in error_str:
                        logger.warning(f"Chunk {index} attempt {attempt+1}: Server error - retrying")
                        time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        logger.warning(f"Chunk {index} attempt {attempt+1}: {error_str[:40]}")
                        if attempt == 2:
                            return None
                        continue
            
            return None
        
        # Prepare documents with indices
        doc_tuples = [(i, doc) for i, doc in enumerate(documents)]
        
        # 8 workers to reduce API pressure
        successful_embeddings = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            vectors = list(executor.map(embed_with_validation, doc_tuples))
        
        # Store successful results
        for i, (doc, vector) in enumerate(zip(documents, vectors)):
            if vector is not None:
                self.documents.append(doc)
                self.vectors.append(vector)
                successful_embeddings.append(i)
        
        embedding_time = time.time() - start_time
        success_rate = (len(successful_embeddings) / len(documents)) * 100
        logger.info(f"ROBUST: {embedding_time:.1f}s, {len(successful_embeddings)}/{len(documents)} chunks ({success_rate:.1f}% success)")
        
        # Quality check
        if len(successful_embeddings) < len(documents) * 0.7:  # Less than 70% success
            logger.error(f"Low success rate: {success_rate:.1f}% - may affect accuracy")
        
        if embedding_time > 25:
            logger.warning(f"Embedding time exceeded target: {embedding_time:.1f}s")
    
    def similarity_search(self, query: str, k: int = 8) -> List[Document]:
        """Robust similarity search"""
        if not self.vectors:
            logger.warning("No vectors available for search")
            return []
        
        try:
            # Clean query text
            clean_query = clean_text_for_embedding(query)
            query_vector = self.embeddings.embed_query(clean_query)
            
            # Calculate similarities
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

def robust_document_loader(url: str) -> List[Document]:
    """Robust document loading with text cleaning"""
    try:
        response = requests.get(url, timeout=12, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = min(len(pdf_reader.pages), 30)  # Hard limit 30 pages
            
            # Clean text extraction
            text_parts = []
            for i in range(total_pages):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text and page_text.strip():
                        # Clean page text
                        clean_page = clean_text_for_embedding(page_text)
                        if len(clean_page.strip()) > 20:  # Only add substantial content
                            text_parts.append(clean_page)
                except Exception as e:
                    logger.warning(f"Error reading page {i+1}: {e}")
                    continue
            
            full_text = "\n\n".join(text_parts)
            logger.info(f"ROBUST: Processed {total_pages} pages, {len(full_text)} chars")
            
            return [Document(
                page_content=full_text, 
                metadata={"source": url, "type": "pdf", "pages": total_pages}
            )]
        
        elif url_lower.endswith('.docx') or 'wordprocessingml' in content_type:
            docx_file = io.BytesIO(response.content)
            doc = DocxDocument(docx_file)
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            clean_text = clean_text_for_embedding(text)
            
            return [Document(
                page_content=clean_text, 
                metadata={"source": url, "type": "docx"}
            )]
        
        else:
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            text = soup.get_text()
            clean_text = clean_text_for_embedding(text)
            
            return [Document(
                page_content=clean_text[:100000],  # Limit HTML content
                metadata={"source": url, "type": "html"}
            )]
        
    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise

class RobustRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing ROBUST RAG engine...")
        
        try:
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=3200
            )

            # Conservative chunking to avoid 400 errors
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1400,   # Conservative size
                chunk_overlap=140, # 10% overlap
                separators=["\n\n", "\n", ". ", " "]
            )

            self.initialized = True
            logger.info("ROBUST RAG engine ready!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _robust_query(self, vectorstore: RobustVectorStore, query: str) -> str:
        """Robust query processing"""
        docs = vectorstore.similarity_search(query, k=8)
        context = " ".join([doc.page_content for doc in docs])[:3000]
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        system_content = """You are an expert insurance policy analyst.

INSTRUCTIONS:
- Input questions are separated by " | "
- Output answers MUST be separated by " | " in the same order
- Provide accurate answers based on the document content
- Include specific numbers, percentages, time periods, and conditions
- If information is not in the document, state "Information not available in provided document"

CRITICAL: Separate each answer with " | " and maintain exact question order."""

        human_content = f"""Questions: {query}

Document: {context}

Provide answers separated by " | " in the same order."""

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content)
        ]
        
        response = self.chat_model.invoke(messages)
        return response.content

    async def process_robust(self, url: str, questions: List[str]) -> List[str]:
        """Robust processing with error handling"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="30 second timeout exceeded")

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        total_start = time.time()
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("CACHED - INSTANT!")
        else:
            # Document loading with validation
            docs = robust_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            # Aggressive chunk limiting to avoid timeouts
            original_count = len(chunks)
            if len(chunks) > 90:  # REDUCED from 75 to 60
                # Strategic selection
                keep_first = int(len(chunks) * 0.5)   # 50% from start
                keep_middle = int(len(chunks) * 0.15) # 15% from middle
                keep_last = int(len(chunks) * 0.35)   # 35% from end
                
                middle_start = len(chunks) // 2 - keep_middle // 2
                chunks = (
                    chunks[:keep_first] +
                    chunks[middle_start:middle_start + keep_middle] +
                    chunks[-keep_last:]
                )
                
                logger.info(f"CHUNK LIMIT: {original_count} → {len(chunks)} chunks")
            
            # Robust embedding with validation
            vectorstore = RobustVectorStore(self.embeddings)
            vectorstore.add_documents_robust(chunks)
            
            # Cache result
            self.vectorstore_cache[url_hash] = vectorstore
        
        # Query processing
        batch_query = " | ".join(questions)
        
        query_start = time.time()
        response = self._robust_query(vectorstore, batch_query)
        query_time = time.time() - query_start
        
        total_time = time.time() - total_start
        logger.info(f"ROBUST: Query {query_time:.1f}s, Total {total_time:.1f}s")
        
        # Answer parsing
        answers = []
        raw_splits = response.split(" | ")
        
        for split in raw_splits:
            cleaned = split.strip()
            if cleaned and len(cleaned) > 5:
                answers.append(cleaned)
        
        while len(answers) < len(questions):
            answers.append("Information not available in provided document.")
        
        return answers[:len(questions)]

# Global engine
rag_engine = RobustRAGEngine()

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
        logger.info("ROBUST RAG application ready")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="ROBUST RAG API", version="3.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Robust processing with input validation"""
    try:
        logger.info(f"ROBUST: {len(request.questions)} questions (max 60 chunks)")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        start_time = time.time()
        answers = await rag_engine.process_robust(request.documents, request.questions)
        total_time = time.time() - start_time

        logger.info(f"ROBUST: Completed in {total_time:.1f}s")
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
        "mode": "robust_validated",
        "max_chunks": 90,
        "validation": "enabled",
        "embedding_provider": "Together.AI (Robust)"
    }

@app.get("/")
async def root():
    return {"message": "ROBUST RAG API - Input Validation & Error Handling"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
