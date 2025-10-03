from fastapi import FastAPI
from contextlib import asynccontextmanager
from .app.routes.authRoute import auth_router
from .app.routes.chatRoute import chat_router
from .app.routes.ragRoute import rag_router
from .app.middlewares.errorHandler import register_global_exception_handlers
from sentence_transformers import SentenceTransformer
from .app.db.db_init import create_tables
from .app.config.settings import Settings
from qdrant_client import QdrantClient

settings = Settings() # type: ignore

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Startup logic
        create_tables()

        # Load sentence transformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

        # Qdrant client connection
        qdrant_client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_KEY)

        # Attach to app.state for reuse across routes
        app.state.chunk_model = model
        app.state.qdrant_client = qdrant_client

        yield  # Hand over control to the app

        # Shutdown logic
        print("App is shutting down")
    except Exception as e:
        print(f"Init error: -> {e}")

# Initialize FastAPI app with lifespan context manager
app = FastAPI(lifespan=lifespan)

# Register custom global error handler middleware
register_global_exception_handlers(app)

# Register routers with prefixes and tags
app.include_router(router=chat_router, prefix='/api/v1/chat', tags=['chat'])
app.include_router(router=auth_router, prefix='/api/v1/auth', tags=['auth'])
app.include_router(router=rag_router, prefix='/api/v1/rag', tags=['rag'])
