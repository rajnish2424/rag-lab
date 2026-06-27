"""
Chunk documents into retrieval-friendly text segments.

Responsibilities
----------------
- Split documents using recursive chunking.
- Preserve document metadata.
- Generate deterministic chunk IDs.
- Validate chunking configuration.
- Return structured Chunk objects.

This module intentionally does NOT:
- Generate embeddings.
- Store vectors.
- Perform retrieval.
"""

from __future__ import annotations

import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter

from common.config import settings
from common.constants import CHUNK_ID_TEMPLATE
from common.models import Chunk, Document

logger = logging.getLogger(__name__)


class Chunker:
    """
    Splits documents into overlapping chunks suitable for retrieval.
    """

    DEFAULT_SEPARATORS = [
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ]

    def __init__(
        self,
        chunk_size: int = settings.retrieval.chunk_size,
        chunk_overlap: int = settings.retrieval.chunk_overlap,
        separators: list[str] | None = None,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or self.DEFAULT_SEPARATORS,
            length_function=len,
        )

        logger.info(
            "Chunker initialized "
            "(chunk_size=%s, chunk_overlap=%s)",
            chunk_size,
            chunk_overlap,
        )

    def chunk_documents(
        self,
        documents: list[Document],
    ) -> list[Chunk]:
        """
        Split multiple documents into chunks.
        """

        logger.info(
            "Starting chunking for %d document(s).",
            len(documents),
        )

        chunks: list[Chunk] = []

        for document in documents:
            chunks.extend(self._chunk_document(document))

        logger.info(
            "Chunking complete. Generated %d chunks.",
            len(chunks),
        )

        return chunks

    def _chunk_document(
        self,
        document: Document,
    ) -> list[Chunk]:
        """
        Split a single document into chunks.
        """

        split_text = self.text_splitter.split_text(document.text)

        logger.debug(
            "Document '%s' produced %d chunk(s).",
            document.document_id,
            len(split_text),
        )

        chunks: list[Chunk] = []

        for index, text in enumerate(split_text):
            chunks.append(
                self._build_chunk(
                    document=document,
                    text=text,
                    index=index,
                )
            )

        return chunks

    def _build_chunk(
        self,
        document: Document,
        text: str,
        index: int,
    ) -> Chunk:
        """
        Create a Chunk object from split text.
        """

        return Chunk(
            chunk_id=CHUNK_ID_TEMPLATE.format(
                document_id=document.document_id,
                index=index,
            ),
            document_id=document.document_id,
            chunk_index=index,
            text=text,
            metadata=document.metadata.copy(),
        )