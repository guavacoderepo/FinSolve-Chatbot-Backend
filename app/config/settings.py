from pydantic import BaseSettings

class Settings(BaseSettings):
    SECRET_KEY:str
    DB_PATH:str
    QDRANT_KEY:str
    QDRANT_URL:str
    env: str = "development"

    class Config:
        env_file =  "backend/.env"
