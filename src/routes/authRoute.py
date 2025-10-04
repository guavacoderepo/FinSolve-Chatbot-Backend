from fastapi import APIRouter, Depends, HTTPException, status
from src.utils.jwtAuth import AccessTokenBearer
from src.schemas.schemes import User, LoginModel, RegisterModel, AuthResponseModel
from src.services.usersServices import UsersServices
from src.utils.authutil import hash_password, check_password, generate_token
from src.db.base import get_db

auth_router = APIRouter()

@auth_router.post('/login')
def handle_login(request: LoginModel, db=Depends(get_db)):
    """
    Authenticate user and return JWT token if successful.
    """
    user_service = UsersServices(db)

    # Find user by username
    user = user_service.get_user_by_username(request.username)

    # If no user found, raise Unauthorized error
    if not user:
        raise HTTPException(
            detail="Invalid username or password",
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    
    # Check if password matches stored hash
    check_pwd = check_password(request.password, user['password'])

    if not check_pwd:
        raise HTTPException(
            detail="Invalid username or password",
            status_code=status.HTTP_401_UNAUTHORIZED
        )
    

    # Generate JWT token with user ID
    token = generate_token({"id": user['id'], "api_key":request.api_key})
    
    # Create User schema instance for response
    user = User(id=user['id'], role=user['role'], username=user['username'])
    # Return success status, user info, and token
    return AuthResponseModel(status=True, user=user, token=token)


@auth_router.post('/register', response_model=AuthResponseModel)
async def handle_register(request: RegisterModel, db=Depends(get_db)):
    """
    Register a new user after hashing the password.
    Prevent duplicate usernames.
    """
    user_service = UsersServices(db)
    
    # Check if username already exists
    existing_user = user_service.get_user_by_username(request.username)

    if existing_user:
        raise HTTPException(
            detail="User already exist",
            status_code=status.HTTP_400_BAD_REQUEST
        )
    
    # Hash password before saving
    request.password = hash_password(request.password)

    # Insert user into DB and get inserted user data
    result = user_service.insert_user(request)

    # Create User schema instance for response
    user = User(id=result['id'], role=result['role'], username=result['username'])

    # Return success status and user info (no token yet)
    return AuthResponseModel(status=True, user=user)


@auth_router.get('/me')
def handle_current_user(user = Depends(AccessTokenBearer())):
    """
    Return details of authenticated user.
    Requires valid access token.
    """
    # Optionally refresh the token
    token = generate_token({"id": user['id']})

    # Return user info and new token
    return AuthResponseModel(status=True, user=user, token=token)
