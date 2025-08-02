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

# ----- Optimized RAG Engine -----
class OptimizedRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.persistent_client = None
        self.text_splitter = None
        self.policy_prompt = None
        self.initialized = False
        
        # Caching for optimization
        self.document_cache: Dict[str, Any] = {}  # URL -> processed document data
        self.vectorstore_cache: Dict[str, Any] = {}  # URL hash -> vectorstore info
        self.max_cache_size = 5  # Keep last 5 documents in memory
    
    def _get_url_hash(self, url: str) -> str:
        """Generate hash for URL for caching purposes"""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def initialize(self):
        """Initialize RAG components once at startup"""
        if self.initialized:
            return
            
        logger.info("Initializing optimized RAG engine...")
        
        # Set environment variables
        os.environ["TOGETHER_API_KEY"] = "deb14836869b48e01e1853f49381b9eb7885e231ead3bc4f6bbb4a5fc4570b78"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = "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"
        os.environ["LANGCHAIN_PROJECT"] = "chunking and rag bajaj"

        # Initialize LLM and embeddings with optimized settings
        self.chat_model = ChatTogether(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            temperature=0,  # Consistent outputs
            max_tokens=4000  # Reasonable limit
        )
        self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")

        # Initialize persistent client with optimized settings
        self.persistent_client = chromadb.PersistentClient(
            path="/content/vectorstore_optimized",
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Optimized text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Larger chunks = fewer embeddings
            chunk_overlap=100,  # Reduced overlap
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Optimized prompt template for batch processing
        self.policy_prompt = ChatPromptTemplate([
            ("system", """You are an expert insurance policy assistant. Answer questions concisely and accurately.

CRITICAL FORMAT: Input questions are separated by " | ". Output answers MUST be separated by " | " in the same order.

Guidelines:
- Direct, concise answers
- Lead with key information
- Cite specific policy terms when available
- If unsure, state limitations clearly
- Maintain exact order and use " | " separator between answers"""),
            ("human", """Questions: {query}
Context: {context}"""),
        ])

        self.initialized = True
        logger.info("Optimized RAG engine initialized successfully")

    def build_chain(self, retriever):
        """Build optimized RAG chain"""
        def retrieve(state):
            query = state["query"]
            # Retrieve more relevant chunks but limit processing
            results = retriever.invoke(query, k=8)  # Limit to top 8 most relevant
            # Concatenate with length limit to avoid token overflow
            context = " ".join([doc.page_content for doc in results])
            return context[:4000]  # Limit context length
        
        return (
            RunnableAssign({"context": RunnableLambda(retrieve)}) |
            self.policy_prompt |
            self.chat_model |
            StrOutputParser()
        )

    def _load_and_process_document(self, url: str) -> tuple:
        """Load and process document with caching"""
        url_hash = self._get_url_hash(url)
        
        # Check if document is already cached
        if url in self.document_cache:
            logger.info(f"Using cached document for {url}")
            return self.document_cache[url]
        
        logger.info(f"Loading document: {url}")
        start_time = time.time()
        
        # Load document
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(docs)
        
        load_time = time.time() - start_time
        logger.info(f"Document loaded and chunked in {load_time:.2f}s ({len(chunks)} chunks)")
        
        # Cache the result (limit cache size)
        if len(self.document_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.document_cache))
            del self.document_cache[oldest_key]
        
        self.document_cache[url] = (docs, chunks)
        return docs, chunks

    def _create_or_get_vectorstore(self, url: str, chunks: List) -> tuple:
        """Create vectorstore with optimized batch processing"""
        url_hash = self._get_url_hash(url)
        collection_name = f"doc_{url_hash}"
        
        # Check if vectorstore already exists and is valid
        try:
            existing_collection = self.persistent_client.get_collection(collection_name)
            if existing_collection.count() > 0:
                logger.info(f"Reusing existing vectorstore: {collection_name}")
                vectorstore = Chroma(
                    client=self.persistent_client,
                    collection_name=collection_name,
                    embedding_function=self.embeddings,
                )
                return vectorstore, False  # False = not newly created
        except Exception:
            pass  # Collection doesn't exist, create new one
        
        logger.info(f"Creating new vectorstore: {collection_name}")
        start_time = time.time()
        
        # Create fresh vectorstore
        vectorstore = Chroma(
            client=self.persistent_client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
        
        # Optimized batch insertion
        batch_size = 50  # Process in batches
        total_chunks = len(chunks)
        
        def add_batch(batch_chunks):
            vectorstore.add_documents(batch_chunks)
        
        # Process chunks in parallel batches
        with ThreadPoolExecutor(max_workers=4) as executor:  # Reduced workers
            futures = []
            for i in range(0, total_chunks, batch_size):
                batch = chunks[i:i + batch_size]
                future = executor.submit(add_batch, batch)
                futures.append(future)
            
            # Wait for all batches to complete
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
        
        creation_time = time.time() - start_time
        logger.info(f"Vectorstore created in {creation_time:.2f}s")
        
        return vectorstore, True  # True = newly created

    def process_document_questions(self, url: str, questions: List[str]) -> List[str]:
        """Optimized document processing and question answering"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        total_start_time = time.time()
        
        try:
            # Step 1: Load and process document (with caching)
            docs, chunks = self._load_and_process_document(url)
            
            # Step 2: Create or reuse vectorstore
            vectorstore, is_new = self._create_or_get_vectorstore(url, chunks)
            
            # Step 3: Create retriever with optimized settings
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}  # Retrieve top 6 most relevant chunks
            )
            
            # Step 4: Build chain
            rag_chain = self.build_chain(retriever)
            
            # Step 5: Process all questions in batch
            logger.info(f"Processing {len(questions)} questions in batch...")
            batch_query = " | ".join(questions)
            
            query_start_time = time.time()
            batch_result = rag_chain.invoke({"query": batch_query})
            query_time = time.time() - query_start_time
            
            logger.info(f"LLM query completed in {query_time:.2f}s")
            
            # Step 6: Parse results
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

    def cleanup_old_collections(self, keep_recent: int = 3):
        """Clean up old collections to free memory"""
        try:
            collections = self.persistent_client.list_collections()
            if len(collections) > keep_recent:
                # Sort by name (which includes timestamp) and keep most recent
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
rag_engine = OptimizedRAGEngine()

# ----- Token Verifier -----
def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid format")
    
    token = authorization.split("Bearer ")[-1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Bearer token")

# ----- FastAPI App -----
app = FastAPI(title="Optimized RAG Question Answering API", version="2.0.0")

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    try:
        rag_engine.initialize()
        logger.info("Optimized application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {str(e)}")
        raise

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    try:
        logger.info(f"Received request with {len(request.questions)} questions")

        # Basic validation
        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="Questions list is empty")

        # Process questions using optimized RAG engine
        answers = rag_engine.process_document_questions(request.documents, request.questions)

        # Periodic cleanup
        if len(rag_engine.document_cache) >= rag_engine.max_cache_size:
            rag_engine.cleanup_old_collections(keep_recent=2)

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

# ----- For Local Testing -----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)