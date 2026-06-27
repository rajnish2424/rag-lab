class EmbeddingModel:

    def embed_documents(
        self,
        chunks: list[Chunk],
    ) -> list[EmbeddedChunk]:
        """
        Generate embeddings for multiple chunks.
        """

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate an embedding for a user query.
        """