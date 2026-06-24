import os
import requests
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from openai import OpenAI

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN","").strip()

# ==========================================
# 1. DATABASE CONFIGURATION
# ==========================================
DATABASE_URL = "sqlite:///./database.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class SearchHistory(Base):
    __tablename__ = "search_history"
    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String, index=True)
    created_at = Column(DateTime, server_default=func.now())


class ProfileRecord(Base):
    __tablename__ = "profile_records"
    id = Column(Integer, primary_key=True, index=True)
    search_id = Column(Integer, ForeignKey("search_history.id"))
    name = Column(String)
    headline = Column(String)
    company = Column(String)
    location = Column(String)
    profile_url = Column(String)
    source = Column(String)
    rank = Column(Integer)


Base.metadata.create_all(bind=engine)

# ==========================================
# 2. AI CLIENT CONFIGURATION (GitHub Models)
# ==========================================

AI_MODEL = "openai/gpt-4o-mini"  # any model id available under GitHub Models
_ai_client = None


def get_ai_client() -> OpenAI:
    global _ai_client
    if _ai_client is None:
        if not GITHUB_TOKEN:
            raise RuntimeError("GITHUB_TOKEN is not set; cannot call GitHub Models.")
        _ai_client = OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=GITHUB_TOKEN,
        )
    return _ai_client

# ==========================================
# 3. GITHUB PUBLIC PROFILE SEARCH (replaces MOCK_PROFILES)
# ==========================================
GITHUB_API_BASE = "https://api.github.com"

def github_headers():
    headers = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers

def search_github_users(query: str, per_page: int = 10) -> List[dict]:
   
    url = f"{GITHUB_API_BASE}/search/users"
    # Wrap multi-word terms in quotes so GitHub treats them as one phrase
    # (otherwise "machine learning" becomes two separate required terms).
    phrase = f'"{query}"' if " " in query else query
    search_query = f"{phrase} in:bio,login,name,type:user"
    params = {"q": search_query, "per_page": per_page}
    resp = requests.get(url, headers=github_headers(), params=params, timeout=15)

    if resp.status_code == 401:
        raise HTTPException(status_code=401, detail="GitHub token invalid or missing required scope.")
    if resp.status_code == 403:
        # Usually rate limiting (60/hr unauthenticated, 30/min for search, 5000/hr authenticated)
        raise HTTPException(status_code=429, detail="GitHub API rate limit hit. Wait and retry, or check token.")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"GitHub search failed: {resp.text}")

    return resp.json().get("items", [])


def get_github_user_details(username: str) -> Optional[dict]:
   
    url = f"{GITHUB_API_BASE}/users/{username}"
    resp = requests.get(url, headers=github_headers(), timeout=15)
    if resp.status_code != 200:
        remaining = resp.headers.get("x-ratelimit-remaining")
        print(f"GitHub user details fetch failed for '{username}': "
              f"status={resp.status_code} rate_limit_remaining={remaining}")
        return None
    return resp.json()


def fetch_github_profiles(search_terms: List[str], limit_per_term: int = 5) -> List[dict]:
  
    seen_usernames = set()
    profiles = []

    for term in search_terms:
        items = search_github_users(term, per_page=limit_per_term)
        for item in items:
            username = item.get("login")
            if not username or username in seen_usernames:
                continue
            seen_usernames.add(username)

            details = get_github_user_details(username)
            if not details:
                continue

            headline = details.get("bio") or "GitHub Developer"

            profiles.append({
                "name": details.get("name") or username,
                "headline": headline,
                "company": details.get("company") or "N/A",
                "location": details.get("location") or "N/A",
                "url": details.get("html_url"),
                "source": "GitHub",
                # extra signal used only for ranking, not returned to client directly
                "_followers": details.get("followers", 0),
                "_matched_term": term,
            })

    return profiles

# ==========================================
# 4. FASTAPI SETUP
# ==========================================
app = FastAPI(title="AI-Powered Public Profile Discovery Module")

class SearchRequest(BaseModel):
    keyword: str

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 5. AI QUERY EXPANSION
# ==========================================
def expand_keyword_with_ai(keyword: str) -> List[str]:
    prompt = (
        f'Provide exactly 3 professional job titles or search terms closely '
        f'related to "{keyword}" that would be useful as GitHub search queries '
        f'(e.g. for matching bios or usernames). Output only a clean, '
        f'comma-separated list, no numbering, no extra text.'
    )
    try:
        client = get_ai_client()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=AI_MODEL,
            temperature=0.3,
        )
        ai_output = response.choices[0].message.content.strip()
        expanded = [kw.strip().lower() for kw in ai_output.split(",") if kw.strip()]
        return expanded if expanded else [keyword.lower()]
    except Exception as e:
        print(f"AI expansion failed, falling back to raw keyword. Error: {e}")
        return [keyword.lower()]

# ==========================================
# 6. API ENDPOINTS
# ==========================================
# API 1: POST /api/profile-search
@app.post("/api/profile-search")
def profile_search(request: SearchRequest, db: Session = Depends(get_db)):
    user_keyword = request.keyword.lower().strip()
    if not user_keyword:
        raise HTTPException(status_code=400, detail="keyword must not be empty")

    # Save history first
    new_history = SearchHistory(keyword=request.keyword)
    db.add(new_history)
    db.commit()
    db.refresh(new_history)

    # AI Expansion
    expanded_keywords = expand_keyword_with_ai(user_keyword)
    all_search_terms = [user_keyword] + expanded_keywords

    # Discover real public profiles from GitHub
    raw_profiles = fetch_github_profiles(all_search_terms)

    # Ranking: keyword match in bio/headline + matched on primary term + followers as tiebreaker
    scored_profiles = []
    for profile in raw_profiles:
        headline_lower = profile["headline"].lower()
        score = 0

        if user_keyword in headline_lower:
            score += 10
        for term in expanded_keywords:
            if term in headline_lower:
                score += 2
        if profile["_matched_term"] == user_keyword:
            score += 3

        score += min(profile.get("_followers", 0) / 100, 5)

        if score > 0:
            scored_profiles.append({"profile_data": profile, "score": score})

    if not scored_profiles and raw_profiles:
        scored_profiles = [
            {"profile_data": p, "score": p.get("_followers", 0)} for p in raw_profiles
        ]

    scored_profiles.sort(key=lambda x: x["score"], reverse=True)

    # Save + build response
    final_results = []
    for rank_idx, item in enumerate(scored_profiles, start=1):
        prof = item["profile_data"]
        db_profile = ProfileRecord(
            search_id=new_history.id,
            name=prof["name"],
            headline=prof["headline"],
            company=prof["company"],
            location=prof["location"],
            profile_url=prof["url"],
            source=prof["source"],
            rank=rank_idx,
        )
        db.add(db_profile)

        final_results.append({
            "rank": rank_idx,
            "name": prof["name"],
            "headline": prof["headline"],
            "company": prof["company"],
            "location": prof["location"],
            "profile_url": prof["url"],
            "source": prof["source"],
            "relevance_score": round(item["score"], 2),
        })

    db.commit()

    return {
        "status": "success",
        "search_id": new_history.id,
        "original_keyword": request.keyword,
        "ai_expanded_terms": expanded_keywords,
        "ranked_profiles": final_results,
    }

# API 2: GET /api/profile-search/history
@app.get("/api/profile-search/history")
def get_search_history(db: Session = Depends(get_db)):
    history = db.query(SearchHistory).order_by(SearchHistory.id.desc()).all()
    return {"search_history": history}

# API 3: GET /api/profile-search/{id}
@app.get("/api/profile-search/{id}")
def get_search_details(id: int, db: Session = Depends(get_db)):
    search_meta = db.query(SearchHistory).filter(SearchHistory.id == id).first()
    if not search_meta:
        raise HTTPException(status_code=404, detail="Search record not found")

    profiles = (
        db.query(ProfileRecord)
        .filter(ProfileRecord.search_id == id)
        .order_by(ProfileRecord.rank)
        .all()
    )
    return {
        "search_id": search_meta.id,
        "searched_keyword": search_meta.keyword,
        "discovered_profiles": profiles,
    }

@app.get("/")
def root():
    return {"status": "ok", "message": "Profile Discovery API is running"}
