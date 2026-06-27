"""
loader.py

Loads Markdown documents from the raw data directory.

Responsibilities:
- Read markdown files
- Generate document IDs
- Return structured document objects

Does NOT:
- Chunk documents
- Clean text
- Generate embeddings
"""

from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads markdown documents from a directory.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {self.data_dir}"
            )

    def load_documents(self) -> List[Dict]:
        """
        Loads all markdown files from the directory.

        Returns:
            List of document dictionaries.
        """

        documents = []

        markdown_files = sorted(self.data_dir.glob("*.md"))

        logger.info(f"Found {len(markdown_files)} markdown files.")

        for file_path in markdown_files:

            try:
                content = file_path.read_text(
                    encoding="utf-8"
                )

                document = {
                    "document_id": file_path.stem,
                    "file_name": file_path.name,
                    "content": content,
                    "source_path": str(file_path.resolve())
                }

                documents.append(document)

                logger.info(f"Loaded {file_path.name}")

            except Exception as e:
                logger.exception(
                    f"Failed loading {file_path.name}: {e}"
                )

        logger.info(
            f"Successfully loaded {len(documents)} documents."
        )

        return documents