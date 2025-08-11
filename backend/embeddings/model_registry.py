# backend/embeddings/model_registry.py
"""
Return a LangChain-compatible Embeddings object from a short key.

Supported keys
--------------
local-mini         -> sentence-transformers/all-MiniLM-L6-v2
openai-ada-002     -> OpenAI text-embedding-ada-002
together-e5        -> Together-AI  embedding/e5-large-v2    (if TogetherEmbeddings present)
together-short     -> Together-AI  ts-embedding-1           (if TogetherEmbeddings present)
huggingface/...    -> any Sentence-Transformers model on HF Hub
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, Dict

from langchain.embeddings.base import Embeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------#
# Try every known location for TogetherEmbeddings                            #
# ---------------------------------------------------------------------------#
try:
    # new path (≥ 0.0.40)
    from langchain_community.embeddings.together import TogetherEmbeddings  # type: ignore
except ImportError:
    try:
        # very new path (some nightly builds)
        from langchain_community.embeddings.together_ai import TogetherEmbeddings  # type: ignore
    except ImportError:
        try:
            # old pre-community path (≤ 0.1.0 of langchain)
            from langchain.embeddings.together import TogetherEmbeddings  # type: ignore
        except ImportError:
            TogetherEmbeddings = None  # Together AI support not installed

# ---------------------------------------------------------------------------#
# Public factory                                                             #
# ---------------------------------------------------------------------------#
def get_embedder(provider_key: str = "local-mini") -> Embeddings:
    """
    Return a ready-to-use Embeddings instance.

    Raises
    ------
    ValueError
        If the key is unknown or a required API key is missing.
    """
    key = provider_key.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown embedder '{provider_key}'. "
            f"Valid keys: {', '.join(sorted(_REGISTRY))} "
            "or 'huggingface/<model-id>'."
        )
    return _REGISTRY[key]()

# ---------------------------------------------------------------------------#
# Lazy factory helpers                                                       #
# ---------------------------------------------------------------------------#
@lru_cache(maxsize=None)
def _local_mini() -> Embeddings:
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@lru_cache(maxsize=None)
def _openai_ada() -> Embeddings:
    _require_env("OPENAI_API_KEY")
    return OpenAIEmbeddings(model="text-embedding-ada-002")


@lru_cache(maxsize=None)
def _together_e5() -> Embeddings:
    _require_env("TOGETHER_API_KEY")
    return TogetherEmbeddings(model="embedding/e5-large-v2")  # type: ignore[name-defined]


@lru_cache(maxsize=None)
def _together_short() -> Embeddings:
    _require_env("TOGETHER_API_KEY")
    return TogetherEmbeddings(model="ts-embedding-1")  # type: ignore[name-defined]


def _hf_generic(model_id: str) -> Embeddings:
    """Return embeddings for any Sentence-Transformers model on HF Hub."""
    return HuggingFaceEmbeddings(model_name=model_id)

# ---------------------------------------------------------------------------#
# Build the registry                                                         #
# ---------------------------------------------------------------------------#
_REGISTRY: Dict[str, Callable[[], Embeddings]] = {
    "local-mini":     _local_mini,
    "openai-ada-002": _openai_ada,
}

# Add Together keys only if the class is available
if TogetherEmbeddings is not None:  # type: ignore[name-defined]
    _REGISTRY.update(
        {
            "together-e5":   _together_e5,
            "together-short": _together_short,
        }
    )

# ---------------------------------------------------------------------------#
# Dynamic “huggingface/<model-id>” support                                   #
# ---------------------------------------------------------------------------#
def __getattr__(name: str):
    if name.startswith("huggingface/"):
        model_id = name.split("huggingface/", 1)[1]
        return lambda: _hf_generic(model_id)
    raise AttributeError(name)

# ---------------------------------------------------------------------------#
# Utility                                                                    #
# ---------------------------------------------------------------------------#
def _require_env(var: str) -> None:
    if not os.getenv(var):
        raise ValueError(
            f"Environment variable '{var}' must be set to use this embedding provider."
        )
