"""
Application configuration.

This module centralizes all configurable parameters used across the
RAG pipeline. Environment-specific secrets (API keys, database URLs)
should remain in `.env`, while application behaviour belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

BENCHMARK_DIR = DATA_DIR / "benchmarks"

LOG_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalConfig:
    chunk_size: int = 500
    chunk_overlap: int = 100

    top_k: int = 5


# ---------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "models/text-embedding-004"

    batch_size: int = 32


# ---------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class VectorStoreConfig:
    collection_name: str = "financial_rag"

    persist_directory: Path = (
        PROCESSED_DATA_DIR / "vector_store"
    )


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    retrieval: RetrievalConfig = RetrievalConfig()

    embedding: EmbeddingConfig = EmbeddingConfig()

    vector_store: VectorStoreConfig = VectorStoreConfig()

    logging: LoggingConfig = LoggingConfig()

@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "models/text-embedding-004"

    batch_size: int = 32

    max_retries: int = 3

    timeout_seconds: int = 30
    
settings = Settings()