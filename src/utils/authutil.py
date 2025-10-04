import bcrypt
import datetime
import jwt  # Ensure this is PyJWT, not python-jwt or jwt-simple
from config.settings import Settings
from fastapi import HTTPException, status

settings = Settings()  # type: ignore # Load app settings (e.g., SECRET_KEY)

def hash_password(plain_password: str) -> str:
    # Generate a salt and hash the plain password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password.encode('utf-8'), salt)
    return hashed.decode('utf-8')  # Return hashed password as string for DB storage

def check_password(plain_password: str, hashed_password: str) -> bool:
    # Check if plain password matches the hashed password
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def generate_token(payload, expires_in=36000):
    # Add expiration time to payload
    payload['exp'] = datetime.datetime.now() + datetime.timedelta(seconds=expires_in)
    # Encode JWT token with secret key and HS256 algorithm
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
    return token

def decode_token(token):
    try:
        # Decode JWT token to get the payload
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload
    except Exception as e:
        # Raise 401 Unauthorized if token is invalid or expired
        raise HTTPException(detail="Unauthorized Access", status_code=status.HTTP_401_UNAUTHORIZED)
