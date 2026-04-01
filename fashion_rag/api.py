"""
FastAPI server — production entry point.
Start: uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1

Keep --workers 1: multiple workers each load Mistral into RAM.
"""

import hashlib
import json
import time

import redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_chain import get_chain
import os

load_dotenv()

app = FastAPI(title="Fashion RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    decode_responses=True,
)


class QueryRequest(BaseModel):
    question: str
    gender  : str | None = None
    category: str | None = None


class QueryResponse(BaseModel):
    answer    : str
    sources   : list[dict]
    cached    : bool
    latency_ms: float


@app.post("/query", response_model=QueryResponse)
async def query_fashion(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    cache_key = hashlib.md5(
        f"{req.question.lower().strip()}{req.gender}{req.category}".encode()
    ).hexdigest()

    # ── Cache hit ──────────────────────────────────────────────────────────
    cached_val = cache.get(cache_key)
    if cached_val:
        data = json.loads(cached_val)
        data["cached"]     = True
        data["latency_ms"] = 0.0
        return data

    # ── RAG retrieval ──────────────────────────────────────────────────────
    t0    = time.perf_counter()
    chain = get_chain(gender=req.gender, category=req.category)
    result= chain.invoke({"query": req.question})
    ms    = round((time.perf_counter() - t0) * 1000, 1)

    response = {
        "answer"    : result["result"],
        "sources"   : [doc.metadata for doc in result["source_documents"]],
        "cached"    : False,
        "latency_ms": ms,
    }

    cache.setex(cache_key, 3600, json.dumps(response))
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.delete("/cache")
def clear_cache():
    cache.flushdb()
    return {"message": "Cache cleared."}
