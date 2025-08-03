import os
import time
import logging
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# RAG imports
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableAssign, RunnableLambda
import chromadb
from langchain_chroma import Chroma

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expected Bearer token for authentication
EXPECTED_TOKEN = "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8"

# ----- Request/Response Models -----
class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# ----- Railway-Optimized RAG Engine -----
class RailwayRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.persistent_client = None
        self.text_splitter = None
        self.policy_prompt = None
        self.initialized = False
        
        # Reduced caching for Railway's memory constraints
        self.document_cache: Dict[str, Any] = {}
        self.vectorstore_cache: Dict[str, Any] = {}
        self.max_cache_size = 2  # Reduced for Railway
    
    def _get_url_hash(self, url: str) -> str:
        """Generate hash for URL for caching purposes"""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def initialize(self):
        """Initialize RAG components for Railway environment"""
        if self.initialized:
            return
            
        logger.info("Initializing Railway RAG engine...")
        
        # Set environment variables with Railway-friendly defaults
        os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "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")
        os.environ["LANGCHAIN_PROJECT"] = "railway-rag-deployment"

        # Initialize LLM and embeddings
        self.chat_model = ChatTogether(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            temperature=0,
            max_tokens=3000  # Reduced for Railway
        )
        self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")

        # Railway-friendly persistent client with fallback
        try:
            vector_path = os.environ.get("VECTOR_STORE_PATH", "./vectorstore")
            os.makedirs(vector_path, exist_ok=True)
            self.persistent_client = chromadb.PersistentClient(
                path=vector_path,
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Using persistent storage at: {vector_path}")
        except Exception as e:
            logger.warning(f"Failed to create persistent client, using in-memory: {e}")
            self.persistent_client = chromadb.Client()  # In-memory fallback

        # Optimized text splitter for Railway
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # Smaller chunks for Railway
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Prompt template
        self.policy_prompt = ChatPromptTemplate([
            ("system", """You are an expert document assistant. Answer questions concisely and accurately.

CRITICAL FORMAT: Input questions are separated by " | ". Output answers MUST be separated by " | " in the same order.

Guidelines:
- Direct, concise answers
- Lead with key information
- Cite specific document content when available
- If unsure, state limitations clearly
- Maintain exact order and use " | " separator between answers"""),
            ("human", """Questions: {query}
Context: {context}"""),
        ])

        self.initialized = True
        logger.info("Railway RAG engine initialized successfully")

    def build_chain(self, retriever):
        """Build Railway-optimized RAG chain"""
        def retrieve(state):
            query = state["query"]
            results = retriever.invoke(query, k=6)  # Reduced for Railway
            context = " ".join([doc.page_content for doc in results])
            return context[:3000]  # Limit context length for Railway
        
        return (
            RunnableAssign({"context": RunnableLambda(retrieve)}) |
            self.policy_prompt |
            self.chat_model |
            StrOutputParser()
        )

    def _load_and_process_document(self, url: str) -> tuple:
        """Load and process document with Railway-friendly caching"""
        url_hash = self._get_url_hash(url)
        
        if url in self.document_cache:
            logger.info(f"Using cached document for {url}")
            return self.document_cache[url]
        
        logger.info(f"Loading document: {url}")
        start_time = time.time()
        
        try:
            loader = UnstructuredURLLoader(urls=[url])
            docs = loader.load()
            chunks = self.text_splitter.split_documents(docs)
            
            load_time = time.time() - start_time
            logger.info(f"Document loaded and chunked in {load_time:.2f}s ({len(chunks)} chunks)")
            
            # Railway-friendly cache management
            if len(self.document_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.document_cache))
                del self.document_cache[oldest_key]
                logger.info(f"Removed oldest cached document: {oldest_key}")
            
            self.document_cache[url] = (docs, chunks)
            return docs, chunks
            
        except Exception as e:
            logger.error(f"Failed to load document {url}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to load document: {str(e)}")

    def _create_or_get_vectorstore(self, url: str, chunks: List) -> tuple:
        """Create vectorstore with Railway optimization"""
        url_hash = self._get_url_hash(url)
        collection_name = f"doc_{url_hash}"
        
        try:
            existing_collection = self.persistent_client.get_collection(collection_name)
            if existing_collection.count() > 0:
                logger.info(f"Reusing existing vectorstore: {collection_name}")
                vectorstore = Chroma(
                    client=self.persistent_client,
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                )
                return vectorstore, False
        except Exception:
            pass
        
        logger.info(f"Creating new vectorstore: {collection_name}")
        start_time = time.time()
        
        vectorstore = Chroma(
            client=self.persistent_client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
        
        # Railway-optimized batch processing
        batch_size = 25  # Smaller batches for Railway
        total_chunks = len(chunks)
        
        try:
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i:i + batch_size]
                vectorstore.add_documents(batch)
                logger.info(f"Processed batch {i//batch_size + 1}/{(total_chunks-1)//batch_size + 1}")
        except Exception as e:
            logger.error(f"Vectorstore creation error: {e}")
            raise HTTPException(status_code=500, detail="Failed to create vectorstore")
        
        creation_time = time.time() - start_time
        logger.info(f"Vectorstore created in {creation_time:.2f}s")
        
        return vectorstore, True

    def process_document_questions(self, url: str, questions: List[str]) -> List[str]:
        """Railway-optimized document processing and question answering"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        total_start_time = time.time()
        
        try:
            # Load and process document
            docs, chunks = self._load_and_process_document(url)
            
            # Create or reuse vectorstore
            vectorstore, is_new = self._create_or_get_vectorstore(url, chunks)
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Optimized for Railway
            )
            
            # Build chain
            rag_chain = self.build_chain(retriever)
            
            # Process questions in batch
            logger.info(f"Processing {len(questions)} questions in batch...")
            batch_query = " | ".join(questions)
            
            query_start_time = time.time()
            batch_result = rag_chain.invoke({"query": batch_query})
            query_time = time.time() - query_start_time
            
            logger.info(f"LLM query completed in {query_time:.2f}s")
            
            # Parse results
            answers = [answer.strip() for answer in batch_result.split(" | ")]
            
            # Validate answer count
            if len(answers) != len(questions):
                logger.warning(f"Answer count mismatch: {len(questions)} questions, {len(answers)} answers")
                while len(answers) < len(questions):
                    answers.append("Unable to generate answer for this question.")
                answers = answers[:len(questions)]
            
            total_time = time.time() - total_start_time
            logger.info(f"Total processing time: {total_time:.2f}s")
            
            return answers

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise

    def cleanup_old_collections(self, keep_recent: int = 1):
        """Clean up old collections for Railway memory management"""
        try:
            collections = self.persistent_client.list_collections()
            if len(collections) > keep_recent:
                sorted_collections = sorted(collections, key=lambda x: x.name, reverse=True)
                to_delete = sorted_collections[keep_recent:]
                
                for collection in to_delete:
                    try:
                        self.persistent_client.delete_collection(collection.name)
                        logger.info(f"Cleaned up old collection: {collection.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete collection {collection.name}: {e}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

# Global RAG engine instance
rag_engine = RailwayRAGEngine()

# ----- Token Verifier -----
def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid format")
    
    token = authorization.split("Bearer ")[-1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Bearer token")

# ----- FastAPI App -----
app = FastAPI(title="Railway RAG API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    try:
        rag_engine.initialize()
        logger.info("Railway application startup completed successfully")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        logger.info("Continuing with limited functionality...")

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    """Main endpoint for processing documents and answering questions"""
    try:
        logger.info(f"Received request with {len(request.questions)} questions")

        # Validation
        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="Questions list is empty")

        # Process questions
        answers = rag_engine.process_document_questions(request.documents, request.questions)

        # Periodic cleanup for Railway memory management
        if len(rag_engine.document_cache) >= rag_engine.max_cache_size:
            rag_engine.cleanup_old_collections(keep_recent=1)

        logger.info(f"Successfully processed {len(answers)} answers")
        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_initialized": rag_engine.initialized,
        "cached_documents": len(rag_engine.document_cache),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/clear-cache")
async def clear_cache(authorization: str = Depends(verify_token)):
    """Clear document cache and old collections"""
    try:
        rag_engine.document_cache.clear()
        rag_engine.cleanup_old_collections(keep_recent=0)
        return {"status": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Cache clear error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

# Railway-specific startup
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Railway sets this automatically
    uvicorn.run(app, host="0.0.0.0", port=port)
