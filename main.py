# ==========================================================
#  FINAL RAG – resilient + accurate (≤30 s end-to-end)
# ==========================================================
#  • Smart-retry & circuit-breaker
#  • Multi-Query Expansion (MQE)
#  • MMR de-dup/diversify re-rank
#  • Optional Cross-Encoder re-rank
#
#  Accuracy knobs (set at top):
#  ─────────────────────────────
#  USE_MQE = True   # +≈120 ms
#  USE_MMR = True   # +≈2 ms
#  USE_CE  = False  # +≈250 ms (turn on if budget allows)
# ----------------------------------------------------------

import os, time, logging, hashlib, random, io, asyncio
from typing import List, Optional
from contextlib import asynccontextmanager
from functools import wraps
import requests, PyPDF2, numpy as np
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# LangChain / Together
from langchain_together import ChatTogether, TogetherEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------- accuracy knobs --------------------
USE_MQE = True   # Multi-Query expansion
USE_MMR = True   # Max-Marginal-Relevance
USE_CE  = False  # Cross-encoder re-rank
# -----------------------------------------------------------

if USE_CE:
    from sentence_transformers import CrossEncoder   # pip install sentence-transformers

# ----------------------- logging ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# -----------------------------------------------------------

EXPECTED_TOKEN = "5aa05ad358e859e92978582cde20423149f28beb49da7a2bbb487afa8fce1be8"

class QuestionRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# ======================= infra helpers =====================
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=15):
        self.threshold = failure_threshold
        self.timeout = timeout
        self.failures, self.last_failure, self.state = 0, None, "CLOSED"

    def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure < self.timeout:
                return None
            self.state = "HALF_OPEN"

        try:
            res = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state, self.failures = "CLOSED", 0
            return res
        except Exception as e:
            self.failures, self.last_failure = self.failures + 1, time.time()
            if self.failures >= self.threshold:
                self.state = "OPEN"
                logger.warning("Circuit breaker OPEN")
            raise e

def with_smart_retry(max_retries=2, base_delay=0.3):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kw):
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kw)
                except Exception as e:
                    if attempt == max_retries:
                        logger.warning(f"Retries exhausted: {e}")
                        return None
                    backoff = (base_delay * (1.5 ** attempt) +
                               random.uniform(0, 0.2))
                    time.sleep(backoff)
        return wrapped
    return deco
# ===========================================================


# ================= vector store (fast + robust) ============
class FinalOptimizedVectorStore:
    def __init__(self, embeddings):
        self.embeddings, self.documents, self.vectors = embeddings, [], []
        self.circuit = CircuitBreaker(4, 10)

    @with_smart_retry(2, 0.2)
    def _embed(self, text):
        return self.embeddings.embed_query(text)

    def _embed_doc(self, doc: Document):
        try:
            return self.circuit.call(self._embed, doc.page_content)
        except Exception:
            return None

    def add_documents_final(self, docs: List[Document]):
        logger.info(f"FINAL: Processing {len(docs)} chunks")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=4) as pool:
            vecs = list(pool.map(self._embed_doc, docs))
        ok = self._store(docs, vecs)
        logger.info(f"FINAL: {time.time()-t0:.1f}s, {ok} chunks embedded")

    def similarity_search(self, query: str, k: int = 7):
        if not self.vectors:
            return []
        qvec = self.circuit.call(self._embed, query)
        if qvec is None:
            return self.documents[:k]
        sims, qn = [], np.linalg.norm(qvec)
        for i, v in enumerate(self.vectors):
            sims.append((np.dot(qvec, v) / (qn * np.linalg.norm(v) + 1e-9), i))
        sims.sort(reverse=True)
        return [self.documents[i] for _, i in sims[:k]]

    def _store(self, docs, vecs):
        cnt = 0
        for d, v in zip(docs, vecs):
            if v is not None:
                self.documents.append(d)
                self.vectors.append(v)
                cnt += 1
        return cnt
# ===========================================================


# ================ fast PDF / HTML loader ===================
def final_document_loader(url: str) -> List[Document]:
    try:
        r = requests.get(url, timeout=5,
                         headers={'User-Agent': 'Mozilla/5.0'})
        r.raise_for_status()
        ctype, url_l = r.headers.get("content-type", "").lower(), url.lower()

        # -------- PDF --------
        if url_l.endswith(".pdf") or "pdf" in ctype:
            pdf = PyPDF2.PdfReader(io.BytesIO(r.content))
            pages = min(len(pdf.pages), 20)
            pick = list(range(pages)) if pages <= 10 else \
                   sorted(set(range(5) +
                              list(range(pages//2-1, pages//2+2)) +
                              list(range(pages-5, pages))))
            txt = ""
            for i in pick:
                txt += (pdf.pages[i].extract_text() or "") + "\n\n"
            return [Document(page_content=txt[:80000],
                             metadata={"source": url, "type": "pdf"})]

        # -------- HTML/others --------
        soup = BeautifulSoup(r.content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = "\n".join(l.strip() for l in soup.get_text().splitlines() if l.strip())
        return [Document(page_content=text[:80000],
                         metadata={"source": url, "type": "html"})]

    except Exception as e:
        logger.error(f"Document load error: {e}")
        raise
# ===========================================================


# ======================== MAIN ENGINE ======================
class FinalRAGEngine:
    def __init__(self):
        self.initialized = False
        self.vector_cache = {}
        self.circuit = CircuitBreaker(3, 8)

    # ---------- INIT ----------
    def initialize(self):
        if self.initialized:
            return
        logger.info("Initializing FINAL RAG…")
        os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")
        self.emb = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")

        self.llm = ChatTogether(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            temperature=0,
            max_tokens=1_800
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1_200, chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " "]
        )

        if USE_CE:
            self.ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.initialized = True
        logger.info("Engine ready!")

    # ---------- accuracy helpers ----------
    @with_smart_retry(2, 0.2)
    def _embed(self, text):
        return self.emb.embed_query(text)

    def _expand_queries(self, q: str, n: int = 2) -> List[str]:
        if not USE_MQE:
            return [q]
        prompt = (f"Provide {n} very short paraphrases (≤12 words each) "
                  f"for the following query:\n\n{q}\nParaphrases:")
        try:
            resp = self.llm.invoke(prompt).content
            alts = [s.strip("-• ").strip() for s in resp.split("\n") if s.strip()]
            return [q] + alts[:n]
        except Exception as e:
            logger.warning(f"MQE failed: {e}")
            return [q]

    def _mmr(self, qvec, doc_vecs, docs, lam: float = 0.6, top_k: int = 6):
        if not USE_MMR or len(docs) <= top_k:
            return docs[:top_k]
        sim_qd = doc_vecs @ qvec / \
                 (np.linalg.norm(doc_vecs, axis=1) * np.linalg.norm(qvec) + 1e-9)
        sel, cand = [], list(range(len(docs)))
        while len(sel) < top_k and cand:
            if not sel:
                idx = cand[int(np.argmax(sim_qd[cand]))]
            else:
                sim_dd = doc_vecs[cand] @ doc_vecs[sel].T
                mmr = lam * sim_qd[cand] - (1-lam) * sim_dd.max(axis=1)
                idx = cand[int(np.argmax(mmr))]
            sel.append(idx)
            cand.remove(idx)
        return [docs[i] for i in sel]

    def _ce_rerank(self, q: str, docs: List[Document], top_k: int = 5):
        if not USE_CE:
            return docs[:top_k]
        pairs = [[q, d.page_content[:350]] for d in docs]
        scores = self.ce.predict(pairs)
        return [d for _, d in sorted(zip(scores, docs), reverse=True)][:top_k]

    # ---------- core query ----------
    @with_smart_retry(1, 0.2)
    def _invoke_llm(self, messages):
        return self.llm.invoke(messages)

    def _final_query(self, vs: FinalOptimizedVectorStore, q: str) -> str:
        # -------- retrieval pipeline --------
        cand_docs = []
        for aq in self._expand_queries(q):
            for d in vs.similarity_search(aq, k=4):
                if d.page_content not in {c.page_content for c in cand_docs}:
                    cand_docs.append(d)

        if USE_MMR and len(cand_docs) > 6:
            qvec = self._embed(q)
            dvecs = np.vstack([self._embed(d.page_content[:512]) for d in cand_docs])
            cand_docs = self._mmr(qvec, dvecs, cand_docs, top_k=6)

        cand_docs = self._ce_rerank(q, cand_docs, top_k=5)

        ctx = " ".join(d.page_content for d in cand_docs)[:2_200]

        from langchain_core.messages import HumanMessage, SystemMessage
        sys_msg = (
            "You are an insurance policy expert.\n"
            "- Questions are separated by ' | '.\n"
            "- Provide ONLY answers separated by ' | ' in the same order.\n"
            "- If info missing, output 'Information not available in document'."
        )
        human = (
            f"Document Context: {ctx}\n\n"
            f"Questions to answer: {q}\n\n"
            f"Provide ONLY the answers separated by ' | ':"
        )
        msgs = [SystemMessage(content=sys_msg), HumanMessage(content=human)]

        try:
            resp = self.circuit.call(self._invoke_llm, msgs)
            if resp is None:
                return "Service temporarily unavailable | " * q.count("|") + \
                       "Service temporarily unavailable"
            return resp.content
        except Exception:
            return "Information not available due to service error"

    # ---------- public wrapper ----------
    async def process(self, url: str, questions: List[str]) -> List[str]:
        if not self.initialized:
            raise RuntimeError("Engine not initialized")

        async def inner():
            h = hashlib.md5(url.encode()).hexdigest()[:8]
            if h in self.vector_cache:
                vs = self.vector_cache[h]
                logger.info("CACHED vectorstore")
            else:
                docs = final_document_loader(url)
                chunks = self.splitter.split_documents(docs)
                if len(chunks) > 30:
                    s, m = int(30*0.4), int(30*0.25)
                    e = 30 - s - m
                    mid = len(chunks)//2 - m//2
                    chunks = (chunks[:s] + chunks[mid:mid+m] + chunks[-e:])
                vs = FinalOptimizedVectorStore(self.emb)
                vs.add_documents_final(chunks)
                if len(vs.documents) >= max(10, len(chunks)//2):
                    self.vector_cache[h] = vs
            batch_q = " | ".join(questions)
            answer_blob = self._final_query(vs, batch_q)
            parts = [p.strip() for p in answer_blob.split(" | ")]
            out = []
            for p in parts:
                if p and not any(q.lower() in p.lower() for q in questions):
                    out.append(p)
            while len(out) < len(questions):
                out.append("Information not available in document.")
            return out[:len(questions)]

        try:
            return await asyncio.wait_for(inner(), timeout=26.0)
        except asyncio.TimeoutError:
            logger.error("Timeout – returning fallbacks")
            return ["Processing timeout - please try again"] * len(questions)

# --------------- FastAPI glue ------------------------------
rag = FinalRAGEngine()

def verify_token(auth: Optional[str] = Header(None)):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Auth required")
    if auth.split("Bearer ")[-1] != EXPECTED_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag.initialize()
    yield

app = FastAPI(title="FINAL RAG API", lifespan=lifespan)  # ← FastAPI instance is now 'app'

@app.post("/hackrx/run", response_model=AnswerResponse)
async def ask(request: QuestionRequest, _: str = Depends(verify_token)):
    if not request.documents.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL")
    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions")
    t0 = time.time()
    answers = await rag.process(request.documents, request.questions)
    logger.info(f"Total {time.time()-t0:.1f}s")
    return {"answers": answers}

@app.get("/health")
async def health():
    return {
        "status": "ready",
        "features": ["MQE" if USE_MQE else "",
                     "MMR" if USE_MMR else "",
                     "CrossEncoder" if USE_CE else "",
                     "circuit_breaker", "smart_retry"]
    }

@app.get("/")
async def root():
    return {"message": "FINAL RAG API – resilient & accurate"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
