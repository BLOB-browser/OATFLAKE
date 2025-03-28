from fastapi import APIRouter, Request
from scripts.integrations.slack import handler
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/slack", tags=["slack"])  # Changed from /api/slack to /slack

@router.post("/events")
async def endpoint(request: Request):
    """Handle incoming Slack events"""
    try:
        # Get the JSON body from the request
        body = await request.json()
        
        # Handle URL verification challenge
        if body.get("type") == "url_verification":
            logger.info("Received Slack URL verification challenge")
            return {"challenge": body.get("challenge")}
            
        # Pass other events to the Slack handler
        return await handler.handle(request)
        
    except Exception as e:
        logger.error(f"Error handling Slack event: {e}")
        return {"error": str(e)}

@router.get("/health")
async def health_check():
    """Check if Slack integration is running"""
    return {"status": "ok", "service": "slack"}
