from fastapi import Request, FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


def format_validation_error(exc: RequestValidationError) -> str:
    """
    Format validation errors into a readable string.
    """
    error_messages = []
    for err in exc.errors():
        loc = " -> ".join(str(x) for x in err.get("loc", []))  # Location of error
        msg = err.get("msg", "")  # Error message
        error_messages.append(f"{loc}: {msg}")
    return " | ".join(error_messages)  # Join all error messages


def register_global_exception_handlers(app: FastAPI):
    """
    Register global error handlers for the FastAPI app.
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        # Handle HTTP errors like 404, 401, etc.
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": False, "msg": exc.detail},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # Handle validation errors and return a 422 status
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"status": False, "msg": format_validation_error(exc)},
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        # Catch-all handler for any other exceptions (500 error)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": False, "msg": str(exc)},
        )
