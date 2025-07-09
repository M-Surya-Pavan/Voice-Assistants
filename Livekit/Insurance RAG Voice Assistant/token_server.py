import os
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

# Load environment variables
load_dotenv(override=True)

# Get LiveKit credentials from environment
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")

if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
    raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET environment variables are required")

app = FastAPI(title="Insurance Agent Token Server")

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TokenRequest(BaseModel):
    room_name: Optional[str] = None
    participant_name: Optional[str] = None
    session_id: Optional[str] = None

class TokenResponse(BaseModel):
    token: str
    room_name: str
    livekit_url: str

@app.post("/api/insurance/token", response_model=TokenResponse)
async def generate_insurance_token(request: TokenRequest):
    """
    Generate a LiveKit access token for the insurance consultation session.
    """
    try:
        # Generate unique identifiers if not provided
        session_id = request.session_id or str(uuid.uuid4())
        room_name = request.room_name or f"insurance-session-{session_id}"
        participant_name = request.participant_name or f"client-{session_id[:8]}"
        
        # Create access token with grants
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        token = token.with_identity(participant_name).with_grants(VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True
        ))
        
        # Generate JWT token
        jwt_token = token.to_jwt()
        
        return TokenResponse(
            token=jwt_token,
            room_name=room_name,
            livekit_url=LIVEKIT_URL
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate token: {str(e)}")

@app.get("/api/insurance/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "insurance-token-server"}

@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Insurance Agent Token Server",
        "endpoints": {
            "generate_token": "/api/insurance/token",
            "health": "/api/insurance/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)