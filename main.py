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

# ───────────────────────── logging ─────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_TOKEN = "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8"

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# ─────────────────── TIMEOUT-AWARE CIRCUIT BREAKER ───────────────
class TimeoutAwareCircuitBreaker:
    def __init__(self, failure_threshold=2, timeout=5):
        self.threshold, self.timeout = failure_threshold, timeout
        self.failures, self.last_fail, self.state = 0, 0.0, "CLOSED"

    def call(self, fn, *a, timeout_seconds=3, **kw):
        """Call function with hard timeout limit"""
        if self.state == "OPEN":
            if time.time() - self.last_fail < self.timeout:
                logger.warning("Circuit OPEN - skipping call")
                return None
            self.state = "HALF_OPEN"

        try:
            # Use ThreadPoolExecutor with timeout for ANY function call
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fn, *a, **kw)
                try:
                    result = future.result(timeout=timeout_seconds)
                    if self.state == "HALF_OPEN":
                        self.state, self.failures = "CLOSED", 0
                    return result
                except FutureTimeoutError:
                    logger.warning(f"Function timeout after {timeout_seconds}s")
                    raise TimeoutError(f"Call timeout after {timeout_seconds}s")
                    
        except Exception as e:
            self.failures, self.last_fail = self.failures + 1, time.time()
            if self.failures >= self.threshold:
                self.state = "OPEN"
                logger.warning(f"Circuit breaker OPEN after {self.failures} failures")
            raise e

# ─────────────────── AGGRESSIVE RETRY WITH TIMEOUTS ───────────────
def with_timeout_retry(max_retries=1, base_delay=0.1, call_timeout=3):
    def deco(fn):
        @wraps(fn)
        def wrapped(*a, **kw):
            for attempt in range(max_retries + 1):
                try:
                    # Each attempt gets its own timeout
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(fn, *a, **kw)
                        return future.result(timeout=call_timeout)
                        
                except (FutureTimeoutError, TimeoutError) as e:
                    logger.warning(f"Attempt {attempt+1} timeout: {e}")
                    if attempt == max_retries:
                        return None
                    time.sleep(base_delay * (2 ** attempt))
                    
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1} failed: {e}")
                    if attempt == max_retries:
                        return None
                    time.sleep(base_delay * (2 ** attempt))
            return None
        return wrapped
    return deco

# ───────────────────── ULTRA-FAST VECTOR STORE ─────────────────
class BulletproofVectorStore:
    def __init__(self, embeddings):
        self.emb = embeddings
        self.docs, self.vecs = [], []
        self.cb = TimeoutAwareCircuitBreaker(2, 5)
        
        # Pre-configure embeddings with timeout if possible
        if hasattr(embeddings, 'client'):
            # Set timeout on the underlying client if available
            try:
                embeddings.client.timeout = 5  # 5 second timeout
            except:
                pass

    @with_timeout_retry(max_retries=1, base_delay=0.1, call_timeout=3)
    def _embed_raw(self, text: str):
        """Raw embedding call with no additional timeout"""
        return self.emb.embed_query(text)

    def _embed_one_safe(self, doc: Document):
        """Embed one document with multiple timeout layers"""
        try:
            # Layer 1: Circuit breaker with timeout
            result = self.cb.call(
                self._embed_raw, 
                doc.page_content, 
                timeout_seconds=4  # 4s max per call
            )
            return result
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def add_documents_final(self, docs: List[Document]):
        logger.info(f"BULLETPROOF: embedding {len(docs)} chunks")
        global_start = time.time()
        
        # Process in small batches with strict timing
        batch_size = 5
        max_total_time = 10  # 10 seconds max for all embeddings
        
        for i in range(0, len(docs), batch_size):
            if time.time() - global_start > max_total_time:
                logger.warning("Embedding time budget exhausted - stopping early")
                break
                
            batch = docs[i:i + batch_size]
            batch_start = time.time()
            
            # Process batch with timeout
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(self._embed_one_safe, doc) for doc in batch]
                
                for doc, future in zip(batch, futures):
                    try:
                        # Each future gets max 5 seconds
                        vector = future.result(timeout=5)
                        if vector is not None:
                            self.docs.append(doc)
                            self.vecs.append(vector)
                    except FutureTimeoutError:
                        logger.warning("Batch future timeout")
                        continue
                    except Exception as e:
                        logger.warning(f"Batch processing error: {e}")
                        continue
            
            batch_time = time.time() - batch_start
            logger.info(f"Batch {i//batch_size + 1}: {batch_time:.1f}s, {len(self.vecs)} total vectors")
            
            # If batch took too long, reduce workers or stop
            if batch_time > 3:
                logger.warning("Batch too slow - may stop early")
        
        total_time = time.time() - global_start
        logger.info(f"BULLETPROOF: {total_time:.1f}s, {len(self.vecs)}/{len(docs)} vectors")
        
        # Ensure we have at least some vectors or fall back to text storage
        if len(self.vecs) < 3:
            logger.warning("Very few vectors - storing docs for text search")
            self.docs = docs[:20]  # Store first 20 for text search

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vecs:
            return self.docs[:k] if self.docs else []
            
        try:
            # Query embedding with timeout
            qv = self.cb.call(self._embed_raw, query, timeout_seconds=3)
            if qv is None:
                return self.docs[:k] if self.docs else []
                
            # Fast similarity
            qn = np.linalg.norm(qv) + 1e-9
            sims = [(np.dot(qv, v)/(qn*np.linalg.norm(v)+1e-9), i) 
                    for i, v in enumerate(self.vecs)]
            sims.sort(reverse=True)
            return [self.docs[i] for _, i in sims[:k]]
            
        except Exception as e:
            logger.warning(f"Similarity search failed: {e}")
            return self.docs[:k] if self.docs else []

# ───────────────────── EMERGENCY TEXT SEARCH ───────────────────────
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

# ───────────────── DOCUMENT LOADER (unchanged) ──────────────────────
def final_document_loader(url: str) -> List[Document]:
    r = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
    r.raise_for_status()
    ctype, url_l = r.headers.get("content-type","").lower(), url.lower()

    if url_l.endswith(".pdf") or "pdf" in ctype:
        pdf = PyPDF2.PdfReader(io.BytesIO(r.content))
        pages = min(len(pdf.pages), 20)
        pick = list(range(pages)) if pages<=10 else \
               sorted(set(list(range(5)) + list(range(pages//2-1, pages//2+2)) + 
                         list(range(pages-5, pages))))
        text = "".join((pdf.pages[i].extract_text() or "")+"\n\n" for i in pick)
        return [Document(page_content=text[:80000], metadata={"source": url, "type": "pdf"})]

    soup = BeautifulSoup(r.content, "html.parser")
    for tag in soup(["script","style","nav","footer","header"]): 
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())
    return [Document(page_content=text[:80000], metadata={"source": url, "type": "html"})]

# ────────────────────── RAG ENGINE ─────────────────────────
class FinalRAGEngine:
    def __init__(self):
        self.initialized = False
        self.vec_cache = {}
        self.cb = TimeoutAwareCircuitBreaker(2, 5)

    def initialize(self):
        if self.initialized: 
            return
        logger.info("Initializing BULLETPROOF RAG…")
        
        # Initialize with explicit timeouts
        os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")
        
        self.emb = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")
        self.llm = ChatTogether(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            temperature=0,
            max_tokens=1_500
        )
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1_200, chunk_overlap=100,
            separators=["\n\n","\n",". "," "]
        )
        
        self.initialized = True
        logger.info("BULLETPROOF Engine ready!")

    @with_timeout_retry(max_retries=1, base_delay=0.2, call_timeout=5)
    def _invoke_llm_raw(self, prompt):
        return self.llm.invoke(prompt)

    def _final_query(self, vs: BulletproofVectorStore, q: str) -> str:
        # Get documents (with fallback)
        docs = vs.similarity_search(q, k=4) 
        if not docs:
            docs = emergency_text_search(vs.docs, q, k=4)
        if not docs:
            return "No content available"
            
        ctx = " ".join(d.page_content for d in docs)[:1_800]
        prompt = f"""Context: {ctx}

Questions: {q}

Answer each question separated by ' | '. If data not in context, reply 'Information not available in document'."""

        try:
            # LLM call with circuit breaker and timeout
            res = self.cb.call(self._invoke_llm_raw, prompt, timeout_seconds=6)
            return res.content if res else "Service unavailable"
        except Exception as e:
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
                # Document processing with timeout
                try:
                    docs = final_document_loader(url)
                    chunks = self.splitter.split_documents(docs)
                    
                    # Limit chunks more aggressively
                    if len(chunks) > 20:
                        s, m = int(20*.4), int(20*.25)
                        e = 20 - s - m
                        mid = len(chunks)//2 - m//2
                        chunks = chunks[:s] + chunks[mid:mid+m] + chunks[-e:]
                    
                    vs = BulletproofVectorStore(self.emb)
                    vs.add_documents_final(chunks)
                    self.vec_cache[hid] = vs
                    
                except Exception as e:
                    logger.error(f"Document processing failed: {e}")
                    # Return fallback answers
                    return ["Document processing failed"] * len(qs)

            # Query processing
            batch_q = " | ".join(qs)
            raw = self._final_query(vs, batch_q)
            
            # Parse answers
            parts = [p.strip() for p in raw.split(" | ")]
            ans = []
            for p in parts:
                if p and len(p) > 3:
                    ans.append(p)
                else:
                    ans.append("Information not available in document.")
            
            # Ensure we have right number of answers
            while len(ans) < len(qs):
                ans.append("Information not available in document.")
            
            return ans[:len(qs)]

        try:
            return await asyncio.wait_for(inner(), timeout=24.0)  # 24s timeout
        except asyncio.TimeoutError:
            logger.error("Total timeout exceeded")
            return ["Processing timeout - please try again"] * len(qs)

# ───────────────────── FASTAPI (unchanged) ────────────────────────
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

app = FastAPI(title="BULLETPROOF RAG API", lifespan=lifespan)

@app.post("/hackrx/run", response_model=AnswerResponse)
async def run(request: QuestionRequest, _: str = Depends(verify_token)):
    if not request.documents.startswith(("http://","https://")):
        raise HTTPException(status_code=400, detail="Invalid document URL")
    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions provided")
    
    t0 = time.time()
    answers = await rag.process_final(request.documents, request.questions)
    total_time = time.time() - t0
    
    logger.info(f"BULLETPROOF: Total {total_time:.1f}s")
    return {"answers": answers}

@app.get("/health")
async def health():
    return {"status": "bulletproof", "max_latency": "<25s", "timeout_layers": "multiple"}

@app.get("/")
async def root():
    return {"message": "BULLETPROOF RAG API - Multiple timeout layers"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
