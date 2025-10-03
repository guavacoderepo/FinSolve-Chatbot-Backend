import glob
from .vectorServices import VectorService
from typing import List, Dict
from ..schemas.schemes import Roles

class RAGService:
    def __init__(self, model, client, documents: List[Dict] = []):
        """
        Initialize with optional list of documents.
        VectorService handles text loading, chunking, embedding, and vector storage.
        """
        self.vector = VectorService(model, client)
        self.documents: List[Dict] = documents

    def retrie_text(self) ->  List[Dict]:
        """
        Load text files from 'resources/data/*/*' folder,
        read content, add metadata, and store in self.documents.
        """
        paths: List[str] = glob.glob('backend/resources/data/*/*')  # Get all nested files
        for path in paths:
            text = self.vector.load_text(path)  # Load file content
            if not text:
                continue  # Skip if empty
            self.documents.append({
                "text": text,
                "source": path,
                "collection": path.split('/')[-2]  # Extract collection name from path
            })
        return self.documents

    def save_document(self) -> None:
        """
        For each document:
        - Split text into chunks
        - Generate vector embeddings for chunks
        - Save embeddings to the respective collection
        """
        for document in self.documents:
            chunks: List[str] = self.vector.chunk_text(document['text'])  # Chunk the text
            embeddings = self.vector.embed_chunks(chunks, document['source'])  # Create embeddings
            self.vector.save_vectors(document['collection'], embeddings)  # Save to vector DB

    def retrive_vectors(self, prompt: str, collection: Roles) -> List:
        """
        Retrieve relevant vectors for a prompt:
        - Embed the prompt
        - If collection is 'executives', search all collections
        - Otherwise, search only in specified collection
        - Return vectors sorted by similarity score descending
        """
        prompt_embedding = self.vector.embed_text(text=prompt)

        # For executives, search across all collections
        if collection.value == "executives":
            vectors = []
            for role in self.vector.get_collections():
                vectors.extend(self.vector.search(role.name, prompt_embedding))
            # Sort by score descending
            return sorted(vectors, key=lambda a: a['score'], reverse=True)
        else:
            # Search in specified collection only
            return self.vector.search(collection.value, prompt_embedding)
