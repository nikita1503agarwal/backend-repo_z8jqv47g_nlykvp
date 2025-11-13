from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import datetime as dt
import requests
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from schemas import Submission, AnalysisResult
from database import db, create_document, get_documents

app = FastAPI(title="AI Guardian API", version="1.0.0")

# CORS for frontend preview
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NEWS_FEEDS = [
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.reuters.com/rssFeed/world",
]

class NewsItem(BaseModel):
    title: str
    summary: Optional[str] = None
    link: Optional[str] = None

class AnalyzeResponse(BaseModel):
    id: str
    plagiarism_score: float
    fake_news_score: float
    verdict: str
    compared_articles: List[NewsItem]

@app.get("/test")
async def test() -> Dict[str, Any]:
    # quick db roundtrip
    try:
        _id = await create_document("ping", {"ok": True, "at": dt.datetime.utcnow().isoformat()})
        docs = await get_documents("ping", limit=1)
        return {"status": "ok", "db": True, "doc_count": len(docs), "id": _id}
    except Exception as e:
        return {"status": "ok", "db": False, "error": str(e)}

async def fetch_feed(url: str) -> List[Dict[str, Any]]:
    try:
        parsed = await asyncio.to_thread(feedparser.parse, url)
        items = []
        for e in parsed.entries[:20]:
            items.append({
                "title": getattr(e, "title", ""),
                "summary": getattr(e, "summary", ""),
                "link": getattr(e, "link", ""),
            })
        return items
    except Exception:
        return []

async def gather_latest_news() -> List[Dict[str, Any]]:
    results = await asyncio.gather(*[fetch_feed(u) for u in NEWS_FEEDS])
    flat: List[Dict[str, Any]] = []
    for r in results:
        flat.extend(r)
    # dedupe by title
    seen = set()
    unique = []
    for it in flat:
        t = it.get("title", "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        unique.append(it)
    return unique[:60]

@app.get("/news-sources", response_model=List[str])
async def news_sources():
    return NEWS_FEEDS

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: Submission):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Fetch latest news
    articles = await gather_latest_news()

    corpus = [text]
    titles = []
    for a in articles:
        combined = f"{a.get('title','')} {a.get('summary','')}"
        titles.append(a.get("title", ""))
        corpus.append(combined)

    # Compute similarities
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        X = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(X[0:1], X[1:]).flatten() if X.shape[0] > 1 else []
    except Exception:
        sims = []

    # Heuristics:
    # - plagiarism_score: max similarity to any recent article
    # - fake_news_score: 1 - average of top-5 similarities (if it's closer to reputable articles, it's likely not fake)
    plagiarism_score = float(max(sims) if len(sims) else 0.0)
    top_k = sorted(sims, reverse=True)[:5] if len(sims) else []
    agreement = sum(top_k) / len(top_k) if top_k else 0.0
    fake_news_score = float(max(0.0, 1.0 - agreement))

    verdict = "Likely Authentic"
    if plagiarism_score > 0.7:
        verdict = "Possible Plagiarism"
    if fake_news_score > 0.6:
        verdict = "Potentially Misleading"
    if plagiarism_score > 0.7 and fake_news_score > 0.6:
        verdict = "High Risk: Copied & Misleading"

    # Store
    doc = {
        "text": text,
        "source_url": payload.url,
        "plagiarism_score": plagiarism_score,
        "fake_news_score": fake_news_score,
        "verdict": verdict,
        "articles_used": articles[:10],
        "_created_at": dt.datetime.utcnow().isoformat(),
        "_updated_at": dt.datetime.utcnow().isoformat(),
    }
    doc_id = await create_document("analysis", doc)

    compared = []
    for idx, a in enumerate(articles[:5]):
        compared.append(NewsItem(title=a.get("title", ""), summary=a.get("summary"), link=a.get("link")))

    return AnalyzeResponse(
        id=doc_id,
        plagiarism_score=round(plagiarism_score, 3),
        fake_news_score=round(fake_news_score, 3),
        verdict=verdict,
        compared_articles=compared,
    )

