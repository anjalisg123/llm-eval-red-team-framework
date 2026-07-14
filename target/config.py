"""Central configuration, loaded from environment / .env.

Both the target RAG system and the eval framework import from here so that
model names, retrieval-k, and paths are defined in exactly one place.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""

    target_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    chroma_dir: str = ".chroma"
    collection_name: str = "corpus"
    retrieval_k: int = 4
    target_url: str = "http://127.0.0.1:8000"

    consistency_paraphrases: int = 5


settings = Settings()
