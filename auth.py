import os
from fastapi import HTTPException, status
from jose import JWTError, jwt

SECRET_KEY = os.getenv("AUTH_SECRET_KEY", "CHANGE_ME")
ALGORITHM = "HS256"

def verify_token(token_header: str) -> None:
    """
    Ověří 'Authorization: Bearer <token>'.
    Pokud chybí nebo je neplatný, vyhodí HTTP 401.
    """
    if not token_header.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token missing or invalid.")
    token = token_header.split(" ", 1)[1]
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Token invalid or expired.")
