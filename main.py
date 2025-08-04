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

# RAG imports - using reliable packages
from langchain_together import ChatTogether
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

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

class OptimizedVectorStore:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []
    
    def add_documents_fast(self, documents: List[Document]):
        """Fast parallel embedding with your reliable HF token"""
        logger.info(f"HUGGING FACE: Processing {len(documents)} chunks with {len(documents)} workers")
        
        start_time = time.time()
        
        def embed_with_retry(doc):
            """Embed with simple retry for reliability"""
            for attempt in range(2):  # Only 2 attempts to keep it fast
                try:
                    return self.embeddings.embed_query(doc.page_content)
                except Exception as e:
                    if attempt == 0:  # Only retry once
                        time.sleep(0.3)
                        continue
                    logger.warning(f"Embedding failed after retry: {str(e)[:60]}")
                    return None
        
        # Use optimal worker count - tested to be fast
        max_workers = min(10, len(documents))  # Max 10 workers or document count
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            vectors = list(executor.map(embed_with_retry, documents))
        
        # Store successful results only
        successful_count = 0
        for doc, vector in zip(documents, vectors):
            if vector is not None:
                self.documents.append(doc)
                self.vectors.append(vector)
                successful_count += 1
        
        embedding_time = time.time() - start_time
        success_rate = (successful_count / len(documents)) * 100
        logger.info(f"HF EMBEDDINGS: {embedding_time:.1f}s, {successful_count}/{len(documents)} chunks ({success_rate:.1f}% success)")
    
    def similarity_search(self, query: str, k: int = 7) -> List[Document]:
        """Fast similarity search - same techniques as before"""
        if not self.vectors:
            return []
        
        try:
            query_vector = self.embeddings.embed_query(query)
            
            # Fast cosine similarity calculation
            similarities = []
            query_norm = np.linalg.norm(query_vector)
            
            for i, vector in enumerate(self.vectors):
                vector_norm = np.linalg.norm(vector)
                if query_norm > 0 and vector_norm > 0:
                    cos_sim = np.dot(query_vector, vector) / (query_norm * vector_norm)
                    similarities.append((cos_sim, i))
                else:
                    similarities.append((0.0, i))
            
            # Get top k results
            similarities.sort(reverse=True)
            top_indices = [idx for _, idx in similarities[:k]]
            
            return [self.documents[i] for i in top_indices]
            
        except Exception as e:
            logger.error(f"Similarity search error: {e}")
            return self.documents[:k] if len(self.documents) >= k else self.documents

def fast_document_loader(url: str) -> List[Document]:
    """Optimized document loader - same proven approach"""
    try:
        response = requests.get(url, timeout=12, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        url_lower = url.lower()
        content_type = response.headers.get('content-type', '').lower()
        
        if url_lower.endswith('.pdf') or 'pdf' in content_type:
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            
            # Smart page selection for speed + accuracy
            if total_pages <= 30:
                pages_to_process = list(range(total_pages))
            else:
                # Proven sampling strategy
                first_pages = list(range(18))
                middle_pages = list(range(total_pages//3, total_pages//3 + 8))
                last_pages = list(range(total_pages-12, total_pages))
                pages_to_process = sorted(set(first_pages + middle_pages + last_pages))
            
            text = ""
            for i in pages_to_process:
                if i < len(pdf_reader.pages):
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text.strip():
                        text += page_text + "\n\n"
            
            logger.info(f"PDF: Processed {len(pages_to_process)}/{total_pages} pages")
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "pdf"})]
        
        elif url_lower.endswith('.docx') or 'wordprocessingml' in content_type:
            docx_file = io.BytesIO(response.content)
            doc = DocxDocument(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return [Document(page_content=text.strip(), metadata={"source": url, "type": "docx"})]
        
        else:  # HTML/other
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

class OptimizedRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.text_splitter = None
        self.initialized = False
        self.vectorstore_cache = {}
    
    def initialize(self):
        if self.initialized:
            return
            
        logger.info("Initializing OPTIMIZED RAG engine...")
        
        try:
            # Together AI for chat (proven to work)
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
            
            # Your Hugging Face token - reliable embeddings
            self.embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key="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",  # Your actual token
                model_name="sentence-transformers/all-MiniLM-L6-v2"  # Fast and accurate
            )
            
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=3500
            )

            # Optimized chunking - proven settings
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1100,  # Sweet spot for speed + accuracy
                chunk_overlap=110,  # 10% overlap
                separators=["\n\n", "\n", ". ", "! ", "? ", " "]
            )

            self.initialized = True
            logger.info("OPTIMIZED RAG engine ready with your HF token!")
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _optimized_query(self, vectorstore: OptimizedVectorStore, query: str) -> str:
        """Optimized LLM query - proven approach"""
        docs = vectorstore.similarity_search(query, k=7)
        context = " ".join([doc.page_content for doc in docs])[:3200]
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Proven system prompt that works well
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

    async def process_optimized(self, url: str, questions: List[str]) -> List[str]:
        """Optimized processing with proven timeout"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            return await asyncio.wait_for(
                self._process_internal(url, questions),
                timeout=50.0  # Proven timeout that works
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Processing timeout")

    async def _process_internal(self, url: str, questions: List[str]) -> List[str]:
        """Internal processing - optimized approach"""
        url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
        
        if url_hash in self.vectorstore_cache:
            vectorstore = self.vectorstore_cache[url_hash]
            logger.info("Using cached vectorstore - FAST!")
        else:
            # Load and process document
            docs = fast_document_loader(url)
            chunks = self.text_splitter.split_documents(docs)
            
            # Smart chunk limiting for speed
            if len(chunks) > 85:  # Slightly reduced for speed
                keep_first = int(len(chunks) * 0.6)
                keep_last = int(len(chunks) * 0.3)
                chunks = chunks[:keep_first] + chunks[-keep_last:]
                logger.info(f"Optimized: {len(chunks)} chunks selected for speed")
            
            vectorstore = OptimizedVectorStore(self.embeddings)
            vectorstore.add_documents_fast(chunks)
            
            # Cache for future use
            self.vectorstore_cache[url_hash] = vectorstore
        
        # Process all questions in one call
        batch_query = " | ".join(questions)
        
        query_start = time.time()
        response = self._optimized_query(vectorstore, batch_query)
        query_time = time.time() - query_start
        
        logger.info(f"LLM query completed: {query_time:.1f}s")
        
        # Parse answers with robust logic
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

# Global engine instance
rag_engine = OptimizedRAGEngine()

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
        logger.info("OPTIMIZED RAG APPLICATION READY")
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield

app = FastAPI(title="OPTIMIZED RAG API", version="3.0.0", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Optimized processing with your HF token - Fast & Reliable"""
    try:
        logger.info(f"OPTIMIZED processing: {len(request.questions)} questions")

        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="No questions provided")

        answers = await rag_engine.process_optimized(request.documents, request.questions)

        logger.info(f"OPTIMIZED: Successfully completed {len(answers)} answers")
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
        "mode": "optimized_huggingface",
        "embedding_provider": "Hugging Face (Your Token)",
        "version": "optimized_v3"
    }

@app.get("/")
async def root():
    return {"message": "OPTIMIZED RAG API - Fast, Reliable & Accurate"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
