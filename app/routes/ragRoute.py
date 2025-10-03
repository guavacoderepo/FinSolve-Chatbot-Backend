from fastapi import APIRouter, Request
from ..services.ragServices import RAGService

# Create a new API router for RAG-related endpoints
rag_router = APIRouter()

# Endpoint to trigger RAG vector indexing process
@rag_router.get("/")
def rag_contents(req: Request):
    try:
        model = req.app.state.chunk_model
        client = req.app.state.qdrant_client

        rag = RAGService(model, client)   # Initialize the RAG service
        rag.retrie_text()    # Retrieve text to be indexed
        rag.save_document()  # Save the processed document/vector


        
        print("RAGing done ✅") 

        return "RAGging done ✅ --- proceed" # Return a simple confirmation message
    except Exception as e:
        print(f"Rag error -> {e}")
