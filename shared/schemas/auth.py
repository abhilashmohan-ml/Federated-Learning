"""
JWT authentication Pydantic schemas.

WHY SCHEMAS?
------------
When a site sends data to the server over HTTP, the body arrives as raw JSON text.
We need to:
  1. Parse that text into Python objects
  2. Validate that all required fields are present
  3. Check that values are the right type (e.g. a string, not a number)

Pydantic "schemas" (also called "models") do all three automatically.
Define a class that inherits from `BaseModel`, list the fields with their types,
and Pydantic handles parsing + validation for you.

PYTHON CONCEPT: Type hints (str, int, etc.)
  Python normally doesn't enforce types at runtime, but Pydantic DOES.
  If a field is declared `site_id: str` and you try to set it to the
  number 42, Pydantic will raise a validation error.

PYTHON CONCEPT: = "bearer"  (default value)
  Fields with a default value are optional in the request body.
  Fields without a default are required — omitting them causes a 422 error.

HOW THEY CONNECT TO THE API
---------------------------
  POST /auth/token receives a request body → parsed into TokenRequest
  POST /auth/token returns a response body → serialised from TokenResponse
  POST /auth/refresh receives → RefreshRequest
  JWT payload decoded → TokenClaims
"""

from pydantic import BaseModel  # base class for all schemas


class TokenRequest(BaseModel):
    """
    Sent by a site when it wants to authenticate.

    Example request body (JSON):
        {"site_id": "site_1", "site_secret": "my-secret-value"}

    The server looks up site_id in the database and verifies site_secret
    against the stored bcrypt hash.
    """
    site_id:     str  # identifies which site is authenticating, e.g. "site_1"
    site_secret: str  # the plaintext secret — compared against the bcrypt hash in DB


class TokenResponse(BaseModel):
    """
    Returned by the server after successful authentication.

    The site stores both tokens:
      - access_token:  used in the Authorization header for every API call
                       (short-lived: 15 minutes)
      - refresh_token: used to get a new access_token when it expires
                       (long-lived: 7 days, but single-use — consumed on use)

    WHY TWO TOKENS?
    ---------------
    Access tokens expire quickly (15 min) to limit damage if one is stolen.
    Refresh tokens last 7 days so sites don't need to re-enter their secret
    every 15 minutes. Refresh tokens are single-use so that if one is stolen
    and used, the legitimate holder's next refresh attempt will fail, alerting
    the system to a possible compromise.
    """
    access_token:  str          # short-lived JWT for API calls
    refresh_token: str          # long-lived JWT for obtaining new token pairs
    token_type:    str = "bearer"  # always "bearer" — tells clients the token type
    expires_in:    int = 900    # access token lifetime in seconds (15 × 60 = 900)


class RefreshRequest(BaseModel):
    """
    Sent when a site's access token has expired and it needs a fresh one.

    The site sends its refresh_token → server validates it, marks the old
    token as "used" (revoked), and issues a completely new token pair.

    Example request body:
        {"refresh_token": "eyJhbGciOiJIUzI1NiJ9..."}
    """
    refresh_token: str  # the refresh JWT previously obtained from /auth/token


class TokenClaims(BaseModel):
    """
    The data encoded inside a JWT (JSON Web Token).

    A JWT has three sections separated by dots: header.payload.signature
    The payload (middle section) is base64-encoded JSON containing these fields.
    The server signs the payload with its SECRET_KEY so it cannot be forged.

    PYTHON CONCEPT: Default values
      `round_id: int = 0` means if the claim is absent the field defaults to 0.
      `exp: int = 0` is similar — exp holds the expiry as a Unix timestamp.

    Fields
    ------
    sub   : "subject" — standard JWT claim, here holds the site_id
    role  : "client" for site nodes, "server" reserved for server-issued tokens
    round_id : which round this token was issued for (informational)
    exp   : expiry time as a Unix timestamp (seconds since 1970-01-01 00:00 UTC)
    """
    sub:      str       # subject = site_id e.g. "site_1"
    role:     str       # "client" | "server"
    round_id: int = 0   # FL round context (informational, not enforced)
    exp:      int = 0   # expiry timestamp — checked by the JWT library automatically
