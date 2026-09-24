from typing import Optional
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
 
try:
    from .config import JWT_SECRET_KEY, JWT_ALGORITHM
except ImportError:
    from config import JWT_SECRET_KEY, JWT_ALGORITHM

# OAuth2 scheme for token extraction (tokenUrl is just for OpenAPI docs)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=True)
 
 
def decode_access_token(token: str) -> Optional[int]:
    """
    Decode and validate a JWT token from the main backend.
   
    Args:
        token: JWT token string
   
    Returns:
        user_id as an integer from token if valid, None otherwise
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            return None
        return int(user_id)
    except (JWTError, ValueError, TypeError):
        return None

def get_current_user(token: str = Depends(oauth2_scheme)) -> int:
    """Dependency to authenticate the user and return their User ID."""
    user_id = decode_access_token(token)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not find user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id
    
    
    
