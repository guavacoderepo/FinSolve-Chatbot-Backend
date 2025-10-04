import sqlite3
from config.settings import Settings

settings = Settings() # type: ignore

def get_db():
    # Connect to SQLite database (thread-safe disabled for FastAPI compatibility)
    conn = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
    try:
        # Yield the connection object for use (e.g., in dependency injection)
        yield conn
    finally:
        # Ensure the connection is closed after use
        conn.close()
