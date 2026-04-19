"""
Session message embeddings for semantic search.

Supports pluggable providers:
  - fastembed (recommended, local, ~130MB model download)
  - sentence-transformers
  - openai (API)
  - mock (deterministic hash fallback, zero dependency)

Configure in ~/.hermes/config.yaml under `session_search:`.
"""

import json
import hashlib
import logging
import math
import os
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_fastembed_model = None
_st_model = None


def _get_config() -> Dict:
    """Read session_search config from ~/.hermes/config.yaml"""
    cfg = {}
    try:
        import yaml
        from hermes_cli.config import get_hermes_home

        config_path = get_hermes_home() / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("session_search", {})
    except Exception:
        pass
    return {}


def embed_text(text: str) -> List[float]:
    """
    Generate a normalized embedding vector for text.
    Falls back to mock embedding if the configured provider fails.
    """
    cfg = _get_config()
    provider = cfg.get("embedding_provider", "fastembed")
    model_name = cfg.get("embedding_model")

    text = (text or "").strip()
    if not text:
        return []

    emb = None
    if provider == "fastembed":
        emb = _embed_fastembed(text, model_name)
    elif provider == "sentence-transformers":
        emb = _embed_sentence_transformers(text, model_name)
    elif provider == "openai":
        emb = _embed_openai(text, model_name)

    if emb:
        return emb
    logger.warning("All embedding providers failed for '%s...' — falling back to mock (no semantic meaning)", text[:40])
    return _embed_mock(text)


def _embed_fastembed(text: str, model_name: Optional[str] = None) -> Optional[List[float]]:
    global _fastembed_model
    try:
        from fastembed import TextEmbedding

        if _fastembed_model is None:
            mn = model_name or "BAAI/bge-small-en-v1.5"
            _fastembed_model = TextEmbedding(model_name=mn)
        vec = next(_fastembed_model.embed([text]))
        return _normalize(vec.tolist())
    except Exception:
        return None


def _embed_sentence_transformers(text: str, model_name: Optional[str] = None) -> Optional[List[float]]:
    global _st_model
    try:
        from sentence_transformers import SentenceTransformer

        if _st_model is None:
            mn = model_name or "all-MiniLM-L6-v2"
            _st_model = SentenceTransformer(mn)
        vec = _st_model.encode(text, convert_to_numpy=True)
        return _normalize(vec.tolist())
    except Exception:
        return None


def _embed_openai(text: str, model_name: Optional[str] = None) -> Optional[List[float]]:
    try:
        import openai

        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        resp = client.embeddings.create(
            input=text,
            model=model_name or "text-embedding-3-small",
        )
        return _normalize(resp.data[0].embedding)
    except Exception:
        return None


def _embed_mock(text: str, dim: int = 384) -> List[float]:
    """Deterministic hash-based embedding. Zero-dependency fallback."""
    h = hashlib.sha256(text.encode()).digest()
    vec = []
    for i in range(dim):
        byte = h[i % len(h)]
        val = (byte / 255.0) * 2 - 1
        vec.append(val)
    return _normalize(vec)


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
