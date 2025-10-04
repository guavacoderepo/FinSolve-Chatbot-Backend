from typing import Any, Dict, Optional
from fastapi import Request, status, Depends
from fastapi.exceptions import HTTPException
from fastapi.security import HTTPBearer
from .authutil import decode_token
from src.services.usersServices import UsersServices
from src.db.base import get_db

class AccessTokenBearer(HTTPBearer):
    def __init__(self, auto_error=True):
        # Initialize the HTTPBearer with optional auto error handling
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request, db=Depends(get_db)) -> Any:
        try:
            # Extract credentials (token) from the Authorization header
            creds = await super().__call__(request)
            if creds is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing authentication credentials"
                )

            # Decode JWT token
            token_data = decode_token(creds.credentials)

            # Verify token and fetch user
            user_service = UsersServices(db)
            user = user_service.get_user_by_id(token_data.get("id"))

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found or invalid token"
                )
            print(creds)
            # Attach extra info
            token = creds.credentials
            api_key = token_data.get("api_key")
        
            return {"user":user, "api_key": api_key, "token":token}
        
        except Exception as e:
            # Catch any unexpected errors
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Authentication error: {str(e)}"
            )

    def verify_token_data(self, token_data: Dict, db: Any) -> Optional[Dict]:
        # Use the UsersServices to get user info by id from token data
        user_service = UsersServices(db)
        user_id = token_data.get('id')
        if not isinstance(user_id, int):
            return None
        user = user_service.get_user_by_id(user_id)
        return user
