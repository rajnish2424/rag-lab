"""
Shared data models used throughout the RAG pipeline.

These models provide a consistent interface between ingestion,
retrieval, evaluation, and generation modules.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    """
    Represents a raw document loaded from disk.

    Attributes:
        document_id:
            Unique identifier for the document.

        text:
            Raw document content.

        metadata:
            Arbitrary metadata associated with the document
            (category, filename, source, etc.).
    """

    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """
    Represents a retrieval unit produced from a document.
    """

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

class EmbeddedChunk:
    """
    A chunk together with its embedding vector.
    """

    chunk: Chunk

    embedding: list[float]