import os, time, logging, hashlib, random, io, asyncio
from typing import List, Optional
from contextlib import asynccontextmanager
from functools import wraps
import requests, PyPDF2, numpy as np
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# LangChain / Together
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_TOKEN = "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8"

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# ─────────────────── AGGRESSIVE TIMEOUT WRAPPER ───────────────
class AggressiveTimeoutWrapper:
    """Wraps any function with a hard process-level timeout"""
    
    @staticmethod
    def call_with_hard_timeout(func, args, kwargs, timeout_seconds=3):
        """Call function with absolutely hard timeout using multiprocessing"""
        import multiprocessing
        
        def target(queue, func, args, kwargs):
            try:
                result = func(*args, **kwargs)
                queue.put(('success', result))
            except Exception as e:
                queue.put(('error', str(e)))
        
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=target, args=(queue, func, args, kwargs))
        process.start()
        process.join(timeout=timeout_seconds)
        
        if process.is_alive():
            process.terminate()
            process.join()
            raise TimeoutError(f"Hard timeout after {timeout_seconds}s")
        
        try:
            result_type, result = queue.get_nowait()
            if result_type == 'success':
                return result
            else:
                raise Exception(result)
        except:
            raise TimeoutError("Process terminated unexpectedly")

# ─────────────────── MINIMAL CIRCUIT BREAKER ───────────────
class MinimalCircuitBreaker:
    def __init__(self, max_failures=3):
        self.failures = 0
        self.max_failures = max_failures
        self.last_failure_time = 0
        
    def should_skip(self):
        # Skip for 10 seconds after max failures
        if self.failures >= self.max_failures:
            if time.time() - self.last_failure_time < 10:
                return True
            else:
                self.failures = 0  # Reset after cooldown
        return False
    
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        
    def record_success(self):
        self.failures = 0

# ───────────────────── LIGHTNING FAST VECTOR STORE ─────────────────
class LightningFastVectorStore:
    def __init__(self, embeddings):
        self.emb = embeddings
        self.docs, self.vecs = [], []
        self.cb = MinimalCircuitBreaker(2)
        
        # Try to configure underlying client for faster timeouts
        try:
            if hasattr(embeddings, 'client'):
                # Set aggressive timeouts on the HTTP client
                embeddings.client.timeout = 3.0
            # Also try to set on the Together client if it exists
            if hasattr(embeddings, 'together_client'):
                embeddings.together_client.timeout = 3.0
        except Exception as e:
            logger.warning(f"Could not set client timeout: {e}")

    def _embed_one_aggressive(self, doc: Document):
        """Single embedding with multiple timeout strategies"""
        if self.cb.should_skip():
            logger.warning("Circuit breaker active - skipping embedding")
            return None
            
        try:
            # Strategy 1: Use process-level timeout (most aggressive)
            try:
                result = AggressiveTimeoutWrapper.call_with_hard_timeout(
                    self.emb.embed_query, 
                    (doc.page_content,), 
                    {}, 
                    timeout_seconds=4
                )
                self.cb.record_success()
                return result
            except Exception as e:
                logger.warning(f"Hard timeout embedding: {e}")
                self.cb.record_failure()
                return None
                
        except Exception as e:
            self.cb.record_failure()
            logger.warning(f"Embedding completely failed: {e}")
            return None

    def add_documents_final(self, docs: List[Document]):
        logger.info(f"LIGHTNING: embedding {len(docs)} chunks")
        global_start = time.time()
        
        # Process sequentially with strict timing
        for i, doc in enumerate(docs):
            elapsed = time.time() - global_start
            
            # Hard stop after 8 seconds total
            if elapsed > 8:
                logger.warning(f"Embedding budget exhausted after {elapsed:.1f}s - stopping at {i}/{len(docs)}")
                break
            
            # Try to embed this document
            vector = self._embed_one_aggressive(doc)
            if vector is not None:
                self.docs.append(doc)
                self.vecs.append(vector)
                logger.info(f"Embedded {i+1}/{len(docs)} in {elapsed:.1f}s")
            else:
                logger.warning(f"Failed to embed chunk {i+1}")
                
            # If we're getting slow, be more aggressive about stopping
            if elapsed > 5 and len(self.vecs) >= 3:
                logger.info(f"Have {len(self.vecs)} vectors after {elapsed:.1f}s - stopping early")
                break
        
        total_time = time.time() - global_start
        logger.info(f"LIGHTNING: {total_time:.1f}s, {len(self.vecs)}/{len(docs)} vectors")
        
        # Fallback: store docs for text search if we have very few vectors
        if len(self.vecs) < 2:
            logger.warning("Very few vectors - storing docs for text search")
            self.docs = docs[:15]

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vecs:
            return self.docs[:k]
            
        try:
            # Try to embed query with timeout
            if self.cb.should_skip():
                return self.docs[:k]
                
            qv = AggressiveTimeoutWrapper.call_with_hard_timeout(
                self.emb.embed_query, (query,), {}, timeout_seconds=3
            )
            
            # Fast similarity
            qn = np.linalg.norm(qv) + 1e-9
            sims = [(np.dot(qv, v)/(qn*np.linalg.norm(v)+1e-9), i) 
                    for i, v in enumerate(self.vecs)]
            sims.sort(reverse=True)
            return [self.docs[i] for _, i in sims[:k]]
            
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return self.docs[:k]

# ───────────────────── TEXT SEARCH FALLBACK ───────────────────────
def emergency_text_search(docs: List[Document], query: str, k=5):
    if not docs:
        return []
    qw = set(query.lower().split())
    scored = []
    for d in docs:
        words = set(d.page_content.lower().split())
        overlap = len(qw & words)
        if overlap:
            scored.append((overlap, d))
    scored.sort(reverse=True)
    return [d for _, d in scored[:k]]

# ───────────────── DOCUMENT LOADER ──────────────────────
def final_document_loader(url: str) -> List[Document]:
    r = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
    r.raise_for_status()
    ctype, url_l = r.headers.get("content-type","").lower(), url.lower()

    if url_l.endswith(".pdf") or "pdf" in ctype:
        pdf = PyPDF2.PdfReader(io.BytesIO(r.content))
        pages = min(len(pdf.pages), 15)  # Reduced further
        pick = list(range(pages)) if pages<=8 else \
               sorted(set(list(range(4)) + list(range(pages//2-1, pages//2+2)) + 
                         list(range(pages-4, pages))))
        text = "".join((pdf.pages[i].extract_text() or "")+"\n\n" for i in pick)
        return [Document(page_content=text[:60000], metadata={"source": url, "type": "pdf"})]

    soup = BeautifulSoup(r.content, "html.parser")
    for tag in soup(["script","style","nav","footer","header"]): 
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())
    return [Document(page_content=text[:60000], metadata={"source": url, "type": "html"})]

# ────────────────────── RAG ENGINE ─────────────────────────
class FinalRAGEngine:
    def __init__(self):
        self.initialized = False
        self.vec_cache = {}
        self.llm_cb = MinimalCircuitBreaker(2)

    def initialize(self):
        if self.initialized: 
            return
        logger.info("Initializing LIGHTNING RAG…")
        
        os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")
        
        # Initialize with minimal retry settings
        self.emb = TogetherEmbeddings(
            model="BAAI/bge-base-en-v1.5",
            # Add explicit timeout settings if supported
        )
        
        self.llm = ChatTogether(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            temperature=0,
            max_tokens=1200  # Reduced for faster response
        )
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks
            chunk_overlap=80,
            separators=["\n\n","\n",". "," "]
        )
        
        self.initialized = True
        logger.info("LIGHTNING Engine ready!")

    def _final_query(self, vs: LightningFastVectorStore, q: str) -> str:
        # Get documents
        docs = vs.similarity_search(q, k=4) 
        if not docs:
            docs = emergency_text_search(vs.docs, q, k=4)
        if not docs:
            return "No content available"
            
        ctx = " ".join(d.page_content for d in docs)[:1500]  # Smaller context
        
        prompt = f"Context: {ctx}\n\nQuestions: {q}\n\nAnswer each question separated by ' | ':"
        
        try:
            if self.llm_cb.should_skip():
                return "Service temporarily unavailable"
                
            # LLM call with process timeout
            result = AggressiveTimeoutWrapper.call_with_hard_timeout(
                self.llm.invoke, (prompt,), {}, timeout_seconds=6
            )
            self.llm_cb.record_success()
            return result.content
            
        except Exception as e:
            self.llm_cb.record_failure()
            logger.warning(f"LLM call failed: {e}")
            return "Information not available due to service error"

    async def process_final(self, url: str, qs: List[str]) -> List[str]:
        if not self.initialized:
            raise RuntimeError("Engine not initialized")

        async def inner():
            hid = hashlib.md5(url.encode()).hexdigest()[:8]
            
            if hid in self.vec_cache:
                vs = self.vec_cache[hid]
                logger.info("Using cached vectorstore")
            else:
                try:
                    docs = final_document_loader(url)
                    chunks = self.splitter.split_documents(docs)
                    
                    # Limit to 15 chunks maximum
                    if len(chunks) > 15:
                        s, e = 8, 7
                        chunks = chunks[:s] + chunks[-e:]
                    
                    vs = LightningFastVectorStore(self.emb)
                    vs.add_documents_final(chunks)
                    
                    # Only cache if reasonably successful
                    if len(vs.vecs) > 0:
                        self.vec_cache[hid] = vs
                    
                except Exception as e:
                    logger.error(f"Document processing failed: {e}")
                    return ["Document processing failed"] * len(qs)

            # Query processing
            batch_q = " | ".join(qs)
            raw = self._final_query(vs, batch_q)
            
            # Parse answers
            parts = [p.strip() for p in raw.split(" | ")]
            ans = [p if p and len(p) > 3 else "Information not available in document." 
                   for p in parts]
            
            while len(ans) < len(qs):
                ans.append("Information not available in document.")
            
            return ans[:len(qs)]

        try:
            return await asyncio.wait_for(inner(), timeout=22.0)  # 22s total timeout
        except asyncio.TimeoutError:
            logger.error("Total timeout exceeded")
            return ["Processing timeout - please try again"] * len(qs)

# ───────────────────── FASTAPI ────────────────────────
rag = FinalRAGEngine()

def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization required")
    if authorization.split("Bearer ")[-1] != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.initialize()
    yield

app = FastAPI(title="LIGHTNING RAG API", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def run(request: QuestionRequest, _: str = Depends(verify_token)):
    if not request.documents.startswith(("http://","https://")):
        raise HTTPException(status_code=400, detail="Invalid document URL")
    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions provided")
    
    t0 = time.time()
    answers = await rag.process_final(request.documents, request.questions)
    total_time = time.time() - t0
    
    logger.info(f"LIGHTNING: Total {total_time:.1f}s")
    return {"answers": answers}

@app.get("/health")
async def health():
    return {"status": "lightning", "max_latency": "<25s", "timeout": "process_level"}

@app.get("/")
async def root():
    return {"message": "LIGHTNING RAG API - Process-level timeouts"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
