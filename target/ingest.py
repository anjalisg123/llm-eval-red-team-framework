"""Ingest the markdown corpus into a persistent Chroma collection.

Design choices (be ready to defend these):

- **Chunking**: fixed-size character windows with overlap, split on paragraph
  boundaries first. Markdown headings are short and semantically dense, so we keep
  whole paragraphs together and only pack them up to a target size. This is a
  deliberately *simple* strategy — the point of the project is to expose the failure
  modes of an ordinary RAG system, not to build a state-of-the-art retriever.
- **Overlap**: a small overlap keeps facts that straddle a chunk boundary retrievable.
- **We intentionally do NOT strip HTML comments** from the corpus. Real ingestion
  pipelines routinely forget to, which is exactly how indirect prompt injection reaches
  the model. Leaving it in makes the injection category a genuine test rather than a
  staged one.
"""

from __future__ import annotations

import glob
import os

import chromadb
from openai import OpenAI

from target.config import settings

CHUNK_SIZE = 900       # characters
CHUNK_OVERLAP = 150    # characters

_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def chunk_text(text: str, source: str) -> list[dict]:
    """Split a document into overlapping chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[dict] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 <= CHUNK_SIZE:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                chunks.append(buf)
            # If a single paragraph exceeds CHUNK_SIZE, hard-split it.
            if len(para) > CHUNK_SIZE:
                for i in range(0, len(para), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunks.append(para[i : i + CHUNK_SIZE])
                buf = ""
            else:
                buf = para
    if buf:
        chunks.append(buf)

    # Add overlap by prefixing each chunk with the tail of the previous one.
    out: list[dict] = []
    for idx, c in enumerate(chunks):
        if idx > 0:
            c = chunks[idx - 1][-CHUNK_OVERLAP:] + "\n\n" + c
        out.append({"id": f"{source}::{idx}", "text": c, "source": source})
    return out


def embed(texts: list[str]) -> list[list[float]]:
    resp = _openai().embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in resp.data]


def main(corpus_dir: str = "corpus") -> None:
    chroma = chromadb.PersistentClient(path=settings.chroma_dir)
    # Recreate the collection so re-ingesting is idempotent.
    try:
        chroma.delete_collection(settings.collection_name)
    except Exception:
        pass
    collection = chroma.create_collection(settings.collection_name)

    all_chunks: list[dict] = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        all_chunks.extend(chunk_text(text, source))

    print(f"Embedding {len(all_chunks)} chunks from {corpus_dir}/ ...")
    embeddings = embed([c["text"] for c in all_chunks])
    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        embeddings=embeddings,
        metadatas=[{"source": c["source"]} for c in all_chunks],
    )
    print(f"Ingested {len(all_chunks)} chunks into collection '{settings.collection_name}'.")


if __name__ == "__main__":
    main()
