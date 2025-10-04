import sqlite3
from pathlib import Path
from .queries import Queries
from config.settings import Settings

settings = Settings() # type: ignore

def create_tables():
    # Create the database file if it doesn't exist
    Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(settings.DB_PATH) as conn:
        cursor = conn.cursor()

        # Create users table
        cursor.execute(Queries.create_user_table)

        # Create conversations table
        cursor.execute(Queries.create_conversation_table)

        conn.commit()
