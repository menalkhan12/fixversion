import os
from livekit.api import AccessToken, VideoGrants


def create_livekit_token(identity: str, room: str) -> str:
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("LIVEKIT_API_KEY/LIVEKIT_API_SECRET not configured")

    grants = VideoGrants(room_join=True, room=room)
    token = AccessToken(api_key, api_secret, identity=identity)
    token.add_grant(grants)
    return token.to_jwt()
