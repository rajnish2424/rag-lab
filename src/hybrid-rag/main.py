import logging

from shared.ingestion.loader import DocumentLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

loader = DocumentLoader(
    "shared/data/raw"
)

documents = loader.load_documents()

print(f"\nLoaded {len(documents)} documents.\n")

print(documents[0]["document_id"])
print(documents[0]["file_name"])
print(documents[0]["content"][:300])