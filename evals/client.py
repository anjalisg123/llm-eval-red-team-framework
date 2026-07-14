"""HTTP client to the target RAG system.

This is the ONLY way the framework talks to the system under test. It never imports
target.rag. Everything the evaluator knows about the target comes through this contract:
POST /query -> {answer, contexts, latency_ms}. That black-box boundary is what lets the
same framework evaluate any RAG service that speaks this API.
"""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from evals.schemas import Prompt, Response
from target.config import settings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def _post(payload: dict) -> dict:
    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{settings.target_url}/query", json=payload)
        r.raise_for_status()
        return r.json()


def query_target(prompt: Prompt) -> Response:
    """Send one prompt (with any multi-turn history) to the target and capture the reply."""
    payload = {
        "query": prompt.query,
        "history": [t.model_dump() for t in prompt.history],
    }
    try:
        data = _post(payload)
        return Response(
            prompt_id=prompt.id,
            answer=data.get("answer", ""),
            contexts=data.get("contexts", []),
            latency_ms=data.get("latency_ms", 0.0),
        )
    except Exception as e:  # network error, non-2xx, timeout after retries
        return Response(prompt_id=prompt.id, answer="", contexts=[], error=repr(e))
