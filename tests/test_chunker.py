from common.models import Document
from hybrid_rag.ingestion.chunker import Chunker


def test_chunker_generates_chunks():

    document = Document(
        document_id="test_doc",
        text="Lorem ipsum " * 200,
        metadata={"category": "test"},
    )

    chunker = Chunker(
        chunk_size=200,
        chunk_overlap=50,
    )

    chunks = chunker.chunk_documents([document])

    assert len(chunks) > 1

    assert chunks[0].document_id == "test_doc"

    assert chunks[0].metadata["category"] == "test"

    assert chunks[0].chunk_id == "test_doc_chunk_000"