from fastapi import APIRouter, status, Request, HTTPException, Depends
from ..schemas.schemes import ChatModel, User
from ..utils.markdownPDF import download_pdf
from ..db.base import get_db
from ..services.llmServices import LLMServices
from ..services.ragServices import RAGService
from ..utils.jwtAuth import AccessTokenBearer
import re


chat_router = APIRouter()

@chat_router.post('/start')
async def handle_chat(request: ChatModel, req: Request, auth=Depends(AccessTokenBearer()), db=Depends(get_db)):
    try:
        prompt = request.prompt
        user = User(**auth["user"])    # Convert dict to User model

        # Regex to detect download/export commands in prompt
        pattern = re.compile(
            r"\b(download|get|save|export|fetch|send|give me|copy|retrieve|dl|grab)\b",
            re.IGNORECASE
        )

        llm = LLMServices(prompt=prompt, user=user, db=db, api_key=auth['api_key'])

        # If prompt requests a download action and chat history exists, generate PDF
        if re.search(pattern, prompt):
            if llm.history:
                history = llm.history[-1]['response']
                return await download_pdf(history)
            # No history to generate PDF from
            raise HTTPException(detail="Empty chat history", status_code=status.HTTP_400_BAD_REQUEST)
        
        # Otherwise, perform Retrieval-Augmented Generation (RAG) workflow
        model = req.app.state.chunk_model
        client = req.app.state.qdrant_client

        rag = RAGService(model, client)   # Initialize the RAG service
        vectors = rag.retrive_vectors(prompt, user.role)

        # Combine retrieved text as context
        rag_context = "\n\n".join([v['text'] for v in vectors]) if vectors else ""
        source_path = vectors[0]['source'] if vectors else "no_source_found"

        # Link to source repo or docs
        source_url = f"[🔗 View source](https://raw.githubusercontent.com/guavacoderepo/Role-based-chatbot/refs/heads/main/backend/{source_path})"

        # Generate answer from LLM using context and source URL
        response_text = llm.gpt_conversation_prompt(rag_context=rag_context)

        # Append source URL if response has valid info
        if "That information is not available" not in response_text:
            response_text = f"{response_text}\n\n{source_url}"

        # Save chat conversation to history
        llm.save_conversation(response=response_text)

        return {"response": response_text}

    except Exception as e:
        # Return error as HTTP 400 with error message
        raise HTTPException(detail=str(e), status_code=status.HTTP_400_BAD_REQUEST)


@chat_router.get('/history')
def handle_retrieve_history(auth=Depends(AccessTokenBearer()), db=Depends(get_db)):
    # Initialize LLM service with authenticated user and db
    llm = LLMServices(user=User(**auth['user']), db=db)
    # Return stored conversation history for user
    return llm.history
