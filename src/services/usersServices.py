from typing import Dict, Any
import sqlite3
from src.db.queries import Queries
from src.schemas.schemes import RegisterModel
from fastapi import HTTPException, status

class UsersServices:
    def __init__(self, db: Any):
        self.db = db  # Database connection or cursor

    def insert_user(self, user: RegisterModel) -> Dict:
        """
        Insert a new user into the database.
        """
        try:
            # Execute insert query with username, role, and password
            cursor = self.db.execute(
                Queries.insert_user, 
                (user.username, user.role.value, user.password)
            )
            self.db.commit()  # Commit transaction

            last_id = cursor.lastrowid  # Get inserted user's ID

            # Retrieve inserted user details by ID
            inserted_user = self.get_user_by_id(last_id)

            return inserted_user

        except sqlite3.IntegrityError as e:
            # Raise HTTP 400 if user already exists or invalid input
            raise HTTPException(detail="User already exists or invalid input", status_code=status.HTTP_400_BAD_REQUEST)

    def get_user_by_id(self, user_id: int) -> Dict:
        """
        Retrieve a user by their ID.
        """
        cursor = self.db.execute(Queries.get_user_by_id, (user_id,))
        row = cursor.fetchone()
        # Return user data as dict if found, else empty dict
        return dict(zip([column[0] for column in cursor.description], row)) if row else {}

    def get_user_by_username(self, username: str) -> Dict:
        """
        Retrieve a user by their username.
        """
        cursor = self.db.execute(Queries.get_user_by_username, (username,))
        row = cursor.fetchone()
        # Return user data as dict if found, else empty dict
        return dict(zip([column[0] for column in cursor.description], row)) if row else {}
