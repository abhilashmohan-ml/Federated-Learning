"""JWT authentication Pydantic schemas."""
from pydantic import BaseModel


class TokenRequest(BaseModel):
    site_id: str
    site_secret: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 900   # seconds


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenClaims(BaseModel):
    sub: str        # site_id
    role: str       # "client" or "server"
    round_id: int = 0
    exp: int = 0
