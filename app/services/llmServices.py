from fastapi import HTTPException, status
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from ..schemas.schemes import User, ConversationModel
from datetime import datetime
from typing import List, Dict, Optional
from ..config.settings import Settings
from ..services.chatServices import ChatServices

settings = Settings() # type: ignore

class LLMServices:
    def __init__(self, user: User, api_key:Optional[str] = None, prompt: str = "", db=None):
        """
        Initialize with user, optional prompt, and DB connection.
        Also load previous conversation history from the DB.
        """
        self.user = user
        self.prompt = prompt
        self.api_key = api_key
        self.chat_service = ChatServices(db) 
        self.history = self.retrieve_conversations()
    def prompt_formate(self, rag_context: Optional[str] = None) -> List[ChatCompletionMessageParam]:
        """
        Build the messages list for the LLM:
        - system instructions with user's role
        - optionally add RAG context
        - include last 3 conversation exchanges
        - finally add current user prompt
        """
        system_prompt = (
            f"You are a helpful and knowledgeable AI assistant specializing in {self.user.role.value} at FinSolve Technologies. "
            "Your role is to provide accurate and concise answers to user questions."
        )

        if rag_context:
            system_prompt += (
                "\n\nUse the following context and the full chat history to answer the user's current question. "
                "If the question is a follow-up (e.g., 'explain more', 'why?', or builds on the last answer), you must look at the most recent exchanges in the history to formulate your response. "
                "Only say `That information is not available in the current context or knowledge base.` if neither the provided context nor any part of the chat history help answer the question. "
                "Do not use any external or assumed knowledge beyond the context or chat history."
                f"\n\nContext:\n{rag_context.strip()}"
            )
        
        prompt:List[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]

        # Include up to last 3 conversation exchanges for context
        history = self.history[-3:] if len(self.history) > 3 else self.history
        for message in history:
            prompt.extend([
                {"role": "user", "content": message["prompt"]}, 
                {"role": "assistant", "content": message["response"]}
            ])

        # Add current user prompt last
        prompt.append({"role": "user", "content": self.prompt})
        return prompt 

    def gpt_conversation_prompt(self, rag_context: Optional[str] = None) -> str:
        """
        Send the prompt to OpenAI's GPT API and return the assistant's reply.
        Raises any errors encountered during the API call.
        """
        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model='gpt-4o',
                messages=self.prompt_formate(rag_context)
            )
            return response.choices[0].message.content or ""
        
        except Exception as e:
            print(f"[gpt_conversation_prompt] Error: {e}")
            raise  HTTPException(detail=f"[gpt_conversation_prompt] Error: {e}", status_code=status.HTTP_400_BAD_REQUEST)

    def retrieve_conversations(self) -> List[Dict]:
        """
        Fetch past conversation history for the current user from DB.
        Returns list of dicts with prompt and response.
        """
        chats = self.chat_service.fetch_chats_by_user(self.user.id)
        return [{"prompt": chat["prompt"], "response": chat["response"]} for chat in chats]

    def save_conversation(self, response: str) -> None:
        """
        Save a new conversation (prompt + response) to the database.
        """
        new_chat = ConversationModel(
            userId=self.user.id,
            prompt=self.prompt or "",
            response=response,
            date=datetime.now().isoformat()
        )
        self.chat_service.insert_chat(new_chat)
