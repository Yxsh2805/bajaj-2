import os
import time
import logging
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Expected Bearer token for authentication
EXPECTED_TOKEN = os.getenv("API_TOKEN", "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8")

# ----- Request/Response Models -----
class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# ----- Simplified RAG Engine -----
class SimpleRAGEngine:
    def __init__(self):
        self.chat_model = None
        self.embeddings = None
        self.persistent_client = None
        self.text_splitter = None
        self.policy_prompt = None
        self.initialized = False
        
        # Simplified caching
        self.document_cache: Dict[str, Any] = {}
        self.max_cache_size = 3  # Reduced for Railway limits
    
    def _get_url_hash(self, url: str) -> str:
        """Generate hash for URL for caching purposes"""
        return hashlib.md5(url.encode()).hexdigest()[:8]
    
    def initialize(self):
        """Initialize RAG components"""
        if self.initialized:
            return
            
        logger.info("Initializing RAG engine...")
        
        # Get environment variables with defaults
        together_api_key = os.getenv("TOGETHER_API_KEY")
        langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
        
        if not together_api_key:
            raise ValueError("TOGETHER_API_KEY environment variable is required")
        if not langchain_api_key:
            raise ValueError("LANGCHAIN_API_KEY environment variable is required")

        # Set environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "rag-railway")

        try:
            # Initialize LLM and embeddings
            self.chat_model = ChatTogether(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                temperature=0,
                max_tokens=3000
            )
            self.embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
            logger.info("LLM and embeddings initialized")

        except Exception as e:
            logger.error(f"Failed to initialize LLM/embeddings: {str(e)}")
            raise

        try:
            # Use in-memory ChromaDB for Railway (ephemeral storage)
            self.persistent_client = chromadb.Client()
            logger.info("ChromaDB client initialized (in-memory)")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )

        # Prompt template
        self.policy_prompt = ChatPromptTemplate([
            ("system", """You are an expert assistant. Answer questions accurately and concisely.

FORMAT: Input questions are separated by " | ". Output answers MUST be separated by " | " in the same order.

Guidelines:
- Give direct, clear answers
- Use information from the context
- If unsure, state limitations
- Maintain exact order and use " | " separator"""),
            ("human", """Questions: {query}
Context: {context}"""),
        ])

        self.initialized = True
        logger.info("RAG engine initialized successfully")

    def build_chain(self, retriever):
        """Build RAG chain"""
        def retrieve(state):
            query = state["query"]
            results = retriever.invoke(query)
            context = " ".join([doc.page_content for doc in results])
            return context[:3000]  # Limit context length
        
        return (
            RunnableAssign({"context": RunnableLambda(retrieve)}) |
            self.policy_prompt |
            self.chat_model |
            StrOutputParser()
        )

    def _load_document(self, url: str):
        """Load and process document"""
        url_hash = self._get_url_hash(url)
        
        # Check cache
        if url in self.document_cache:
            logger.info(f"Using cached document for {url}")
            return self.document_cache[url]
        
        logger.info(f"Loading document: {url}")
        
        try:
            # Load document
            loader = UnstructuredURLLoader(urls=[url])
            docs = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(docs)
            
            logger.info(f"Document loaded ({len(chunks)} chunks)")
            
            # Simple cache management
            if len(self.document_cache) >= self.max_cache_size:
                # Remove first entry (FIFO)
                oldest_key = next(iter(self.document_cache))
                del self.document_cache[oldest_key]
            
            self.document_cache[url] = chunks
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load document from {url}: {str(e)}")
            raise

    def _create_vectorstore(self, chunks: List, collection_name: str):
        """Create vectorstore"""
        logger.info(f"Creating vectorstore: {collection_name}")
        
        try:
            # Create collection
            collection = self.persistent_client.create_collection(
                name=collection_name,
                get_or_create=True
            )
            
            # Create vectorstore
            vectorstore = Chroma(
                client=self.persistent_client,
                collection_name=collection_name,
                embedding_function=self.embeddings,
            )
            
            # Add documents in batches
            batch_size = 20
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                vectorstore.add_documents(batch)
            
            logger.info("Vectorstore created successfully")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create vectorstore: {str(e)}")
            raise

    def process_questions(self, url: str, questions: List[str]) -> List[str]:
        """Process document and answer questions"""
        if not self.initialized:
            raise RuntimeError("RAG engine not initialized")
        
        try:
            # Load document
            chunks = self._load_document(url)
            
            # Create unique collection name
            url_hash = self._get_url_hash(url)
            collection_name = f"doc_{url_hash}_{int(time.time())}"
            
            # Create vectorstore
            vectorstore = self._create_vectorstore(chunks, collection_name)
            
            # Create retriever
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            # Build chain
            rag_chain = self.build_chain(retriever)
            
            # Process questions
            logger.info(f"Processing {len(questions)} questions")
            batch_query = " | ".join(questions)
            
            batch_result = rag_chain.invoke({"query": batch_query})
            
            # Parse results
            answers = [answer.strip() for answer in batch_result.split(" | ")]
            
            # Ensure answer count matches question count
            if len(answers) != len(questions):
                logger.warning("Answer count mismatch, adjusting...")
                while len(answers) < len(questions):
                    answers.append("Unable to generate answer.")
                answers = answers[:len(questions)]
            
            logger.info(f"Successfully processed {len(answers)} answers")
            
            # Cleanup
            try:
                self.persistent_client.delete_collection(collection_name)
            except:
                pass  # Ignore cleanup errors
            
            return answers

        except Exception as e:
            logger.error(f"Error processing questions: {str(e)}")
            raise

# Global RAG engine instance
rag_engine = SimpleRAGEngine()

# ----- Token Verifier -----
def verify_token(authorization: Optional[str] = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    token = authorization.split("Bearer ")[-1]
    if token != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# ----- Lifespan Event Handler -----
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        rag_engine.initialize()
        logger.info("Application startup completed")
    except Exception as e:
        logger.error(f"Failed to initialize: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Application shutting down")

# ----- FastAPI App -----
app = FastAPI(
    title="RAG Question Answering API",
    version="2.1.0",
    lifespan=lifespan
)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask_questions(
    request: QuestionRequest,
    authorization: str = Depends(verify_token)
):
    try:
        logger.info(f"Received request with {len(request.questions)} questions")

        # Validation
        if not request.documents.startswith(('http://', 'https://')):
            raise HTTPException(status_code=400, detail="Invalid document URL")
        if not request.questions:
            raise HTTPException(status_code=400, detail="Questions list is empty")
        if len(request.questions) > 10:  # Limit for Railway
            raise HTTPException(status_code=400, detail="Too many questions (max 10)")

        # Process questions
        answers = rag_engine.process_questions(request.documents, request.questions)

        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "initialized": rag_engine.initialized,
        "cached_docs": len(rag_engine.document_cache),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG API is running", "version": "2.1.0"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
