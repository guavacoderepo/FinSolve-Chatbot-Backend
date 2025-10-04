from qdrant_client.models import VectorParams, Distance
from qdrant_client.http.models import PointStruct
from haystack.nodes import PreProcessor
import uuid
from typing import List

class VectorService:
    def __init__(self, model, client):
        """
        Initialize the vector service:
        - Load the sentence transformer model for embeddings.
        - Set chunk size and overlap for text splitting.
        - Connect to Qdrant vector database.
        """
        self.model = model
        self.chunk_size = 300  # Tokens per chunk
        self.overlap = 40      # Overlap tokens between chunks
        self.search_limit = 8  # Number of search results to return
        self.client = client

    def load_text(self, path: str) -> str:
        """
        Read raw text from a file.
        Return empty string if file reading fails.
        """
        try:
            with open(path, encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """
        Split long text into overlapping chunks using Haystack PreProcessor.
        Respects sentence boundaries for better coherence.
        """
        preprocessor = PreProcessor(
            split_length=self.chunk_size,
            split_overlap=self.overlap,
            split_respect_sentence_boundary=True
        )
        processed = preprocessor.process([{"content": text}])
        # Extract content from processed chunks
        return [str(d.content) for d in processed]
    
    def embed_text(self, text: str) -> List:
        """
        Generate embedding vector for a single text string.
        """
        return self.model.encode(text).tolist()

    def embed_chunks(self, chunks: List[str], source: str) -> List[PointStruct]:
        """
        Embed multiple text chunks and prepare them as Qdrant points.
        Each chunk gets a unique UUID and metadata including original text and source.
        """
        sources = [source] * len(chunks)  # Same source for all chunks
        # Ensure all chunks are strings before encoding
        str_chunks = [str(c) for c in chunks]
        vectors = self.model.encode(str_chunks, show_progress_bar=False)  # Batch encode

        # Create Qdrant point payloads with id, vector, and metadata
        return [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=v.tolist(),
                payload={"text": c, "source": s}
            )
            for c, v, s in zip(str_chunks, vectors, sources)
        ]

    def save_vectors(self, collection: str, points: List[PointStruct]) -> None:
        """
        Save vectors to a Qdrant collection.
        Creates the collection if it doesn't exist.
        """
        if not self.client.collection_exists(collection_name=collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=384,           # Embedding dimension of MiniLM
                    distance=Distance.COSINE   # Similarity metric for search
                ),
                timeout=120
            )
        self.client.upsert(collection, points)  # Insert points

        print(f"Save in collect ->  {collection}")

    def search(self, collection: str, query_vector: List[float]) -> List[dict]:
        """
        Search the vector collection for top matching vectors to the query.
        Returns a list of results with id, similarity score, text, and source.
        """
        try:
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=self.search_limit
            )
            return [{
                "id": str(result.id),
                "score": result.score,
                "text": result.payload.get("text", "") if result.payload else "",
                "source": result.payload.get("source", "") if result.payload else ""
            } for result in results]
        except Exception as e:
            print(f"[search] Error: {e}")
            return []

    def delete_all_collections(self) -> None:
        """
        Delete all collections in the Qdrant database.
        Use with caution as this removes all data.
        """
        for c in self.client.get_collections().collections:
            self.client.delete_collection(c.name)

    def get_collections(self):
        """
        Get a list of all collections in the Qdrant database.
        """
        return self.client.get_collections().collections
