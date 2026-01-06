# ============================================================
# config.py  (SINGLE-FILE PROJECT)
# WE RAG Chatbot (Streamlit) + Playwright Crawl + FAISS RAG
# - Start from ONE seed URL and auto-discover links
# - STRICT SCOPE to avoid noise (Business/Sustainability/Investors/etc.)
# - Focus on WE Pay + Services only (Arabic/English)
# - Chat: Extractive (no OpenAI key) OR OpenAI grounded answer (with key)
# ============================================================

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

import numpy as np
import faiss
import streamlit as st
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1) SEED URL (put ONE link here)
# -----------------------------
START_URLS = [
    # ضع هنا لينك واحد فقط (صفحة الخدمات / WE Pay hub)
    # مثال: صفحة الخدمات اللي انت جبت منها القائمة الكبيرة
    "https://te.eg/wps/portal/te/Personal/Mobile/Other-Services/!ut/p/z1/rVFNT8MwDP0r47BjFDdd2_RYVFTG6KZRPtZcptC4LNCvtWGwf082IXHaEBK-PFnv2X62qaArKhq50y_S6LaRlc1z4a89iNkUMrjllwmHZfZwn3jptQspo49UUFE0pjMbmhscQ4-DVtgYLasx1O2zrnBUt-q9styA_U4XOIyhNRvsyU_-gaNO7g-4tnjo2RVa0bxABaXDfOJxFpCJQpfIiaeIUrwE6aIDAdKn30wKS8OJiMDWi6NktoyZs0hgtgj9EKJ0GkN4FzDg7FtwpkduPQQnPVy5NPvjUjfHiWeWsq_Rr9utiOz928bgp6Grf3tAV9fc3ZO3cj4nIo8uvgCX85wm/dz/d5/L0lDUmlTUSEhL3dHa0FKRnNBLzROV3FpQSEhL2Fy/"
]

# -----------------------------
# 2) STRICT SCOPE SETTINGS
# -----------------------------
# مسارات مسموح نمشي فيها (شخصي فقط)
ALLOW_PATHS = [
    "/wps/portal/te/personal/",
    "/te/personal/",
    "/personal/",
]

# كلمات مفتاحية "لازم" تكون موجودة في URL أو نص اللينك عشان نعتبره relevant
FOCUS_HINTS = [
    # WE Pay
    "we pay", "wepay", "wallet", "محفظ", "المحفظة", "محفظة",
    # Services
    "services", "service", "خدمات", "other-services", "mobile/services", "mobile/other-services",
    # Common WE Pay topics
    "faq", "أسئلة", "limits", "الحدود", "fees", "المصاريف",
    "qr", "رمز", "تحويل", "إيداع", "سحب", "فوري", "fawry",
    "كود", "أكواد", "electric", "gas", "water", "bill", "فاتورة",
]

# أقسام ممنوعة بالكامل (تشتيت)
BLOCK_HINTS = [
    "investor", "relations", "sustainability", "csr", "responsibility",
    "jobs", "careers", "media", "news", "press",
    "business", "enterprise",
    "afcon",
    # لو مش عايز عروض/تسوق:
    "promotions", "promotion", "shop.te.eg", "/shop", "stores", "sahel",
    # صفحات عامة مش مطلوبة غالبًا
    "board", "executive", "vision", "mission",
]

# -----------------------------
# 3) CRAWL LIMITS
# -----------------------------
MAX_PAGES_DEFAULT = 120
MAX_DEPTH_DEFAULT = 3

# -----------------------------
# 4) Storage
# -----------------------------
DATA_DIR = Path("data")
RAW_JSONL = DATA_DIR / "we_docs.jsonl"
INDEX_DIR = DATA_DIR / "index"
FAISS_INDEX = INDEX_DIR / "faiss.index"
META_JSON = INDEX_DIR / "meta.json"

# -----------------------------
# 5) RAG settings
# -----------------------------
CHUNK_MAX_CHARS = 2400
CHUNK_OVERLAP_CHARS = 300
TOP_K = 6

# -----------------------------
# 6) Playwright settings
# -----------------------------
PAGE_TIMEOUT_MS = 60000
WAIT_UNTIL = "networkidle"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ============================================================
# Helpers
# ============================================================
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

def normalize_url(url: str) -> str:
    u, _ = urldefrag(url)
    return (u or "").strip()

def host_allowed(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        return host.endswith("te.eg") or host.endswith("www.te.eg")
    except Exception:
        return False

def is_in_scope(url: str, anchor_text: str = "") -> bool:
    """
    Strict scope:
    - host must be te.eg
    - must contain allowed personal paths
    - must NOT contain blocked hints
    - must match focus hints (either in url or anchor text)
    """
    try:
        u = (url or "").lower()
        t = (anchor_text or "").lower()

        if not host_allowed(url):
            return False

        if not any(p in u for p in ALLOW_PATHS):
            return False

        if any(b in u for b in BLOCK_HINTS):
            return False

        # must match focus (either url or anchor text)
        if not any(h in u or h in t for h in FOCUS_HINTS):
            return False

        return True
    except Exception:
        return False

def clean_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    for el in soup.select("nav, footer, header"):
        el.decompose()

    text = soup.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if len(ln) > 2]
    return "\n".join(lines)

def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

def extract_scoped_links(base_url: str, html: str) -> List[str]:
    """
    Extract all links from page, then apply strict is_in_scope filter.
    """
    soup = BeautifulSoup(html, "lxml")
    out = []
    seen = set()

    for a in soup.find_all("a"):
        href = a.get("href")
        if not href:
            continue

        text = " ".join([
            a.get_text(" ", strip=True) or "",
            a.get("title") or "",
            a.get("aria-label") or ""
        ]).strip()

        abs_url = normalize_url(urljoin(base_url, href))
        if not abs_url:
            continue

        if abs_url.startswith("mailto:") or abs_url.startswith("tel:"):
            continue

        if is_in_scope(abs_url, text) and abs_url not in seen:
            out.append(abs_url)
            seen.add(abs_url)

    return out


# ============================================================
# Crawl (BFS with Depth + Limits)
# ============================================================
def crawl_sites(
    start_urls: List[str],
    max_pages: int = MAX_PAGES_DEFAULT,
    max_depth: int = MAX_DEPTH_DEFAULT,
    headless: bool = True
) -> List[Dict]:
    ensure_dirs()

    queue: List[Tuple[str, int]] = []
    seen = set()

    for u in start_urls:
        u = normalize_url(u)
        if u and host_allowed(u) and u not in seen:
            queue.append((u, 0))
            seen.add(u)

    docs: List[Dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 720}
        )
        page = context.new_page()

        while queue and len(docs) < max_pages:
            url, depth = queue.pop(0)

            try:
                page.goto(url, wait_until=WAIT_UNTIL, timeout=PAGE_TIMEOUT_MS)
                for _ in range(2):
                    page.mouse.wheel(0, 2500)
                    page.wait_for_timeout(700)

                html = page.content()
                title = page.title()
                text = clean_text_from_html(html)

                docs.append({
                    "url": url,
                    "title": title,
                    "text": text,
                    "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "depth": depth
                })

                if depth >= max_depth:
                    continue

                links = extract_scoped_links(url, html)
                for link in links:
                    if link not in seen:
                        seen.add(link)
                        queue.append((link, depth + 1))

            except Exception as e:
                docs.append({
                    "url": url,
                    "title": "",
                    "text": "",
                    "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "depth": depth,
                    "error": str(e)
                })

        browser.close()

    return docs

def save_jsonl(docs: List[Dict], path: Path):
    ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def read_jsonl(path: Path) -> List[Dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


# ============================================================
# Indexing (Embeddings + FAISS)
# ============================================================
def build_faiss_index(jsonl_path: Path) -> Tuple[int, int]:
    if not jsonl_path.exists():
        raise FileNotFoundError("No crawled data found. Run crawl first.")

    ensure_dirs()
    docs = read_jsonl(jsonl_path)

    chunks = []
    meta = []
    for d in docs:
        if not d.get("text"):
            continue
        parts = chunk_text(d["text"], CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS)
        for i, part in enumerate(parts):
            chunks.append(part)
            meta.append({
                "url": d.get("url"),
                "title": d.get("title"),
                "crawled_at": d.get("crawled_at"),
                "chunk_id": i,
            })

    if not chunks:
        raise RuntimeError("No chunks created. Pages may be blocked or empty.")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, str(FAISS_INDEX))
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump([{"text": t, "meta": m} for t, m in zip(chunks, meta)], f, ensure_ascii=False, indent=2)

    return (len(docs), len(chunks))


# ============================================================
# Retrieval + Answering
# ============================================================
class RAGStore:
    def __init__(self):
        if not FAISS_INDEX.exists() or not META_JSON.exists():
            raise FileNotFoundError("Index not found. Build index first.")
        self.index = faiss.read_index(str(FAISS_INDEX))
        with open(META_JSON, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        q = self.model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        scores, ids = self.index.search(q, top_k)
        out = []
        for sc, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            it = self.items[idx]
            out.append({
                "score": float(sc),
                "text": it["text"],
                "meta": it["meta"],
            })
        return out

def unique_sources(hits: List[Dict]) -> List[str]:
    sources = []
    for h in hits:
        url = h.get("meta", {}).get("url")
        if url and url not in sources:
            sources.append(url)
    return sources

def grounded_answer_extractive(query: str, hits: List[Dict]) -> Tuple[str, List[str]]:
    if not hits:
        return "مش لاقي معلومات كفاية في بيانات WE الحالية.", []

    if hits[0]["score"] < 0.25:
        return "مش لاقي معلومات كفاية في بيانات WE الحالية للإجابة بدقة. حدّث الداتا أو غيّر صيغة السؤال.", []

    top = hits[:3]
    bullets = []
    for h in top:
        snippet = h["text"]
        if len(snippet) > 800:
            snippet = snippet[:800].rsplit("\n", 1)[0] + "..."
        title = h["meta"].get("title") or "WE page"
        bullets.append(f"- **{title}**\n{snippet}")

    sources = unique_sources(top)
    answer = (
        "ده أقرب محتوى من بيانات WE مرتبط بسؤالك:\n\n"
        + "\n\n".join(bullets)
        + "\n\n**ملاحظة:** الرد مبني فقط على صفحات WE اللي اتسحبت واتعمل لها فهرسة."
    )
    return answer, sources

def grounded_answer_openai(query: str, hits: List[Dict], model_name: str) -> Tuple[str, List[str]]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return grounded_answer_extractive(query, hits)

    if not hits or hits[0]["score"] < 0.25:
        return "مش لاقي معلومات كفاية في بيانات WE الحالية للإجابة بدقة. حدّث الداتا أو اسأل بشكل مختلف.", []

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    context_blocks = []
    for i, h in enumerate(hits[:5], start=1):
        meta = h.get("meta", {})
        url = meta.get("url", "")
        title = meta.get("title", "WE page")
        text = (h.get("text") or "")
        if len(text) > 1400:
            text = text[:1400] + "..."
        context_blocks.append(f"[{i}] TITLE: {title}\nURL: {url}\nCONTENT:\n{text}")

    context = "\n\n".join(context_blocks)
    sources = unique_sources(hits[:5])

    system = (
        "You are a customer-support assistant for WE (Egypt) using RAG.\n"
        "CRITICAL RULES:\n"
        "- Use ONLY the provided CONTEXT from WE pages.\n"
        "- If the answer is not explicitly in the context, say you don't have enough information from WE data.\n"
        "- Do NOT use outside knowledge.\n"
        "- Respond in Arabic if the user asked in Arabic; English if asked in English. If both are requested, provide both.\n"
        "- Always include a Sources section listing relevant WE URLs.\n"
    )

    user = f"USER QUESTION:\n{query}\n\nCONTEXT (WE only):\n{context}"

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    answer = resp.choices[0].message.content.strip()
    return answer, sources


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="WE RAG Bot", layout="wide")
st.title("📡 WE RAG Chatbot (WE Pay/Services only) — Single File")

with st.sidebar:
    st.header("Controls")

    headless = st.toggle("Headless Crawl", value=True)
    max_pages = st.slider("Max pages to crawl", 20, 250, MAX_PAGES_DEFAULT, 10)
    max_depth = st.slider("Max depth", 1, 6, MAX_DEPTH_DEFAULT, 1)

    use_openai = st.toggle("Use OpenAI for clean chat answer", value=False)
    model_name = st.text_input("OpenAI Model", value=DEFAULT_OPENAI_MODEL)

    st.divider()
    st.caption("STRICT SCOPE: Personal + WE Pay/Services only. (Business/Investors/etc. blocked)")

    if st.button("🔄 Crawl Now"):
        with st.spinner("Crawling (auto-discovering relevant links)..."):
            docs = crawl_sites(START_URLS, max_pages=max_pages, max_depth=max_depth, headless=headless)
            save_jsonl(docs, RAW_JSONL)
        st.success(f"Crawl done. Saved {len(docs)} pages to {RAW_JSONL}")

    if st.button("📦 Build / Refresh Index"):
        with st.spinner("Building FAISS index..."):
            num_docs, num_chunks = build_faiss_index(RAW_JSONL)
        st.success(f"Index ready. Docs: {num_docs}, Chunks: {num_chunks}")

    st.divider()
    st.caption("Run order: Crawl → Build Index → Ask in chat")

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.write(content)

q = st.chat_input("اسأل عن خدمات WE Pay / الأكواد / الحدود / المصاريف… عربي أو English")
if q:
    st.session_state.messages.append(("user", q))
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        try:
            store = RAGStore()
            hits = store.retrieve(q, top_k=TOP_K)

            if use_openai:
                ans, sources = grounded_answer_openai(q, hits, model_name=model_name)
            else:
                ans, sources = grounded_answer_extractive(q, hits)

            st.write(ans)

            if sources:
                st.subheader("Sources (WE)")
                for s in sources:
                    st.write(s)

            saved = ans + ("\n\nSources:\n" + "\n".join(sources) if sources else "")
            st.session_state.messages.append(("assistant", saved))

        except Exception as e:
            err = f"حصل خطأ: {e}\n\nنفّذ Crawl ثم Build Index من السايدبار الأول."
            st.error(err)
            st.session_state.messages.append(("assistant", err))
