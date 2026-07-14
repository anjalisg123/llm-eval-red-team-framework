"""FastAPI wrapper that exposes the RAG system as a black box over HTTP.

The eval framework only ever talks to this endpoint. It never imports target.rag
directly — that boundary is what lets the framework evaluate *any* RAG service that
speaks this contract, and keeps the harness honest (it sees only what a real client
sees: the answer, the retrieved contexts, and latency).
"""

from __future__ import annotations

import time

from fastapi import FastAPI
from pydantic import BaseModel

from target import rag

app = FastAPI(title="Meridian Assistant (target RAG system)")


class Turn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    query: str
    history: list[Turn] = []
    k: int | None = None


class ContextChunk(BaseModel):
    text: str
    source: str
    distance: float


class QueryResponse(BaseModel):
    answer: str
    contexts: list[ContextChunk]
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    start = time.perf_counter()
    history = [t.model_dump() for t in req.history]
    result = rag.answer(req.query, history=history, k=req.k)
    latency_ms = (time.perf_counter() - start) * 1000
    return QueryResponse(
        answer=result["answer"],
        contexts=[ContextChunk(**c) for c in result["contexts"]],
        latency_ms=round(latency_ms, 1),
    )
