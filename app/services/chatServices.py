from typing import Dict, Any, List
import sqlite3
from ..db.queries import Queries
from ..schemas.schemes import ConversationModel
from fastapi import HTTPException, status


class ChatServices:
    def __init__(self, db: Any):
        # Initialize with a database connection object
        self.db = db

    def insert_chat(self, chat: ConversationModel) -> Dict:
        """
        Insert a new chat record into the database.
        Returns the inserted chat record.
        """
        try:
            # Execute insert query with chat data
            cursor = self.db.execute(
                Queries.insert_chat, 
                (chat.userId, chat.prompt, chat.response, chat.date)
            )
            self.db.commit()  # Commit transaction

            # Get last inserted row ID and fetch the full record
            last_id = cursor.lastrowid 
            return self.get_chat_by_id(last_id)

        except sqlite3.IntegrityError:
            # Raise HTTP error if insertion fails
            raise HTTPException(detail="Failed to insert chat", status_code=status.HTTP_400_BAD_REQUEST)

    def get_chat_by_id(self, chat_id: int) -> Dict:
        """
        Retrieve a single chat record by its ID.
        Returns the chat as a dict or empty dict if not found.
        """
        cursor = self.db.execute(Queries.get_chat_by_id, (chat_id,))
        row = cursor.fetchone()

        # Convert row to dictionary using cursor column names
        return dict(zip([column[0] for column in cursor.description], row)) if row else {}

    def fetch_chats_by_user(self, user_id: int) -> List[Dict]:
        """
        Retrieve all chats for a specific user.
        Returns a list of chat dictionaries.
        """
        cursor = self.db.execute(Queries.fetch_all_chat, (user_id,))
        rows = cursor.fetchall()

        # Convert all rows to list of dicts or empty list if none found
        return [
            dict(zip([column[0] for column in cursor.description], row)) for row in rows
        ] if rows else []
