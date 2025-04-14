from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse  # Add this import to fix the errors
from pathlib import Path
from typing import List, Optional, Dict
from langchain.schema import Document
from api.routes import auth_router, data_router, system_router, training_router, slack, questions, markdown_router, knowledge
from api.routes.openrouter import router as openrouter_router
from api.routes.openrouter_models import router as openrouter_models_router  # Import OpenRouter models router
# Add import for goals router
from api.routes.goals import router as goals_router
from scripts.models.schemas import Definition, JsonRequest, ConnectionRequest, ConnectionResponse, Project, Method, Resource, WebRequest, WebResponse, AnswerRequest  # Add AnswerRequest
from scripts.services import storage, connection
from utils.config import BACKEND_CONFIG
from scripts.llm.ollama_client import OllamaClient
from scripts.llm.open_router_client import OpenRouterClient
from scripts.storage.supabase import SupabaseClient
from scripts.models.settings import ModelSettings
from scripts.services.settings_manager import SettingsManager
from scripts.data.data_processor import DataProcessor
from scripts.services.training_scheduler import start as training_scheduler_start
from scripts.services.training_scheduler import stop as training_scheduler_stop

# Add the import for uvicorn at the top
import uvicorn
import os
import httpx
import logging
import json
import shutil
from datetime import datetime

# Import the proper FastAPI router from api.routes.ollama
from api.routes.ollama import router as ollama_router

# Import the search router
from api.routes.search import router as search_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create base directories if they don't exist
for dir_path in ['static', 'static/css', 'static/js', 'static/img', 'templates']:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Create base directories and copy template if it doesn't exist
def ensure_template_exists():
    template_dir = Path('templates')
    template_dir.mkdir(parents=True, exist_ok=True)
    
    index_file = template_dir / 'index.html'
    if not index_file.exists():
        index_file.write_text("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blob Backend</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white">
    <div class="container mx-auto px-4 py-8">
        <!-- Your existing content -->
    </div>
    <script>
        // Your existing JavaScript
    </script>
</body>
</html>""")

# Create app
app = FastAPI(
    title="Blob Backend",
    description="API for Blob backend services",
    version="0.1.0"
)

# Ensure template exists before mounting
ensure_template_exists()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def get_allowed_origins():
    origins = [
        "http://localhost:3000",         # Local frontend
        "http://localhost:8999",         # Local backend
        "https://blob.iaac.net",         # DigitalOcean frontend
        "https://api.blob.iaac.net",     # DigitalOcean backend
        "https://blob-7z6z9.ondigitalocean.app",
        "https://blob-browser.net"      
    ]
    # Add ngrok URL if available
    if hasattr(app.state, 'ngrok_url'):
        origins.append(app.state.ngrok_url)
    return origins

# Update CORS middleware with both local and production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,  # Set back to True for production
    allow_methods=["*"],     # Allow all methods
    allow_headers=[
        "Content-Type",
        "Accept",
        "Authorization",     # Add Authorization for production
        "ngrok-skip-browser-warning"
    ]
)

# Add global variable for authenticated client
app.state.supabase_client = None

# Create a simple training scheduler object to match the old API
class TrainingSchedulerCompatibility:
    @staticmethod
    def start():
        return training_scheduler_start()
    
    @staticmethod
    def stop():
        return training_scheduler_stop()
    
    @staticmethod
    def get_status():
        # Import here to avoid circular imports
        from scripts.services.training_scheduler import get_status as get_scheduler_status
        return get_scheduler_status()

# Create the compatibility instance
training_scheduler = TrainingSchedulerCompatibility()

# Update startup event to refresh CORS settings when ngrok starts
@app.on_event("startup")
async def startup_event():
    """Initialize server components"""
    try:
        # Initialize only training scheduler using the compatibility layer
        training_scheduler.start()
        
        # Initialize Ollama client for chat functionality
        app.state.ollama_client = OllamaClient()
        
        # Pre-initialize Mistral model for resource processing
        try:
            import requests
            logger.info("Pre-loading Mistral model for resource processing...")
            response = requests.post(
                "http://localhost:11434/api/pull",
                json={"name": "mistral:7b-instruct-v0.2-q4_0"},
                timeout=10.0  # Just to check if Ollama is responding, not for the full download
            )
            if response.status_code == 200:
                logger.info("Mistral model is being loaded/verified in the background")
            else:
                logger.warning(f"Failed to initiate Mistral model loading: {response.status_code}")
        except Exception as e:
            logger.warning(f"Error pre-loading Mistral model: {e}")
        
        # Initialize OpenRouter client (without API key, it will check for environment variable)
        app.state.openrouter_client = OpenRouterClient()
        
        app.state.server_running = True
        
        # Try to load stored credentials and authenticate
        # Use local config.json instead of ~/.blob/config.json
        config_path = Path("config.json")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get('supabase_token'):
                    try:
                        supabase = SupabaseClient(
                            auth_token=config['supabase_token']
                        )
                        await supabase.test_connection()
                        app.state.supabase_client = supabase
                        logger.info("Restored Supabase session")
                    except Exception as token_error:
                        logger.error(f"Token validation failed: {token_error}")
                        # Clear invalid tokens from config
                        if 'supabase_token' in config:
                            del config['supabase_token']
                        if 'supabase_refresh_token' in config:
                            del config['supabase_refresh_token']
                        with open(config_path, 'w') as f:
                            json.dump(config, f)
            except Exception as e:
                logger.error(f"Failed to restore Supabase session: {e}")
                
    except Exception as e:
        logger.error(f"Startup error: {e}")

    # Update CORS settings
    app.middleware_stack = None  # Clear middleware to rebuild
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=[
            "Content-Type",
            "Accept",
            "Authorization",
            "ngrok-skip-browser-warning"
        ]
    )

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup server components"""
    try:
        # Stop the training scheduler first
        logger.info("Stopping training scheduler...")
        try:
            # Stop with more verbose logging
            from scripts.services.training_scheduler import _running, _scheduler_thread
            logger.info(f"Scheduler status before stopping: Running={_running}, Thread alive={_scheduler_thread and _scheduler_thread.is_alive()}")
            
            # Stop the scheduler with a longer timeout for keyboard interrupt cases
            training_scheduler.stop(timeout=30.0)  # Give more time during intentional shutdown
            
            # Double-check it stopped
            import time
            time.sleep(1)  # Give it a moment
            
            # Check if thread is still running
            if _running or (_scheduler_thread and _scheduler_thread.is_alive()):
                logger.warning(f"Scheduler still running after stop attempt: Running={_running}, Thread alive={_scheduler_thread and _scheduler_thread.is_alive()}")
                # Force stop as a last resort
                if _scheduler_thread:
                    logger.warning("Attempting to force thread termination...")
                    import ctypes
                    if hasattr(ctypes, 'pythonapi'):
                        try:
                            thread_id = _scheduler_thread.ident
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                ctypes.c_long(thread_id), 
                                ctypes.py_object(SystemExit)
                            )
                            logger.info("Force termination command sent")
                        except Exception as force_error:
                            logger.error(f"Error in force termination: {force_error}")
            else:
                logger.info("Scheduler successfully stopped")
        except Exception as stop_error:
            logger.error(f"Error stopping scheduler: {stop_error}")
        
        # Stop other services
        logger.info("Stopped all services")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Register all routers
app.include_router(auth_router)
app.include_router(data_router)
app.include_router(system_router, prefix="")  # Remove prefix for system_router
app.include_router(training_router)
app.include_router(questions.router)  # Add questions router
app.include_router(slack.router, prefix="")  # Remove any prefix when including the router
app.include_router(openrouter_router)  # Add OpenRouter router
app.include_router(openrouter_models_router, prefix="/api/openrouter", tags=["openrouter"])  # Include OpenRouter models router
app.include_router(markdown_router)  # Add Markdown router
app.include_router(knowledge.router)  # This router now uses prefix="/api/knowledge"
app.include_router(goals_router)  # Add goals router

# Register the Ollama router properly for FastAPI
app.include_router(ollama_router)

# Add the search router
app.include_router(search_router)

# Routes
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Blob Backend"
    })

@app.get("/api/config-test")
async def config_test():
    return {"message": "Config API test route"}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Skip logging for status endpoint
    if request.url.path != "/api/status":
        logger.info(f"Local server received: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        # Skip logging for status endpoint
        if request.url.path != "/api/status":
            logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request error: {e}")
        raise

# Add this to your main FastAPI app file

import uvicorn

@app.get("/debug/routes")
async def debug_routes():
    """Return all registered routes for debugging"""
    routes = []
    for route in app.routes:
        routes.append({
            "path": route.path,
            "name": route.name,
            "methods": route.methods if hasattr(route, "methods") else None
        })
    return {"routes": routes}

# Simple direct endpoint for setting storage path
from pydantic import BaseModel
from fastapi import Body

class PathUpdate(BaseModel):
    path: str

@app.post("/api/storage/set")
async def set_storage_path(data: PathUpdate):
    try:
        path = data.path
        
        # Platform-aware path handling
        import platform
        system = platform.system()
        
        if not os.path.isabs(path):
            # If it's not an absolute path, create one in a safe location
            if system == "Darwin":  # macOS
                path = os.path.join(os.path.expanduser("~/Documents"), path)
            elif system == "Linux":  # Linux/Raspberry Pi
                path = os.path.join(os.path.expanduser("~"), path)
            elif system == "Windows":  # Windows
                path = os.path.join(os.path.expanduser("~/Documents"), path)
        
        # Check if we can write to the directory
        parent_dir = os.path.dirname(path)
        if not os.access(parent_dir, os.W_OK):
            # If not writable, use a fallback in the home directory
            logger.warning(f"Directory {parent_dir} is not writable, using home directory instead")
            path = os.path.join(os.path.expanduser("~"), os.path.basename(path))
        
        logger.info(f"Storage path update request received: {path}")
        
        # Update local config.json instead of ~/.blob/config.json
        project_config_path = 'config.json'
        config = {}
        
        if os.path.exists(project_config_path):
            with open(project_config_path, 'r') as f:
                config = json.load(f)
        
        previous_path = config.get('data_path', 'none')
        config['data_path'] = path
        
        # Create directory if it doesn't exist
        try:
            os.makedirs(path, exist_ok=True)
        except PermissionError:
            # If we can't create the directory, use a fallback in home dir
            path = os.path.join(os.path.expanduser("~"), "blob_data")
            os.makedirs(path, exist_ok=True)
            config['data_path'] = path
            logger.warning(f"Permission denied, using fallback path: {path}")
        
        # Save config
        with open(project_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update BACKEND_CONFIG so other components use the new path
        from utils.config import BACKEND_CONFIG
        BACKEND_CONFIG['data_path'] = path
        
        logger.info(f"Updated data_path from '{previous_path}' to '{path}'")
            
        return {"success": True, "path": path}
    except Exception as e:
        logger.error(f"Error setting storage path: {e}")
        return {"success": False, "error": str(e)}

# Add alias for system-settings endpoint
@app.get("/api/system-settings")
async def system_settings_alias(request: Request, group_id: str = None):
    """Alias for /api/system/settings to maintain backwards compatibility"""
    try:
        settings_manager = SettingsManager()
        settings = settings_manager.load_settings()
        
        # Add current training schedule using our compatible object
        try:
            scheduler_status = training_scheduler.get_status()
            if scheduler_status["active"]:
                from scripts.models.settings import TrainingSchedule
                # Add default values if this is missing from the response
                settings.training = TrainingSchedule(
                    start="09:00",
                    stop="17:00"
                )
        except Exception as e:
            logger.error(f"Error getting scheduler status: {e}")
        
        return settings
    except Exception as e:
        logger.error(f"Settings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add alias for /api/answers to point to the existing questions/answers endpoint
@app.post("/api/answers")
async def answers_alias(request: Request):
    """Alias for /api/questions/answers to maintain backwards compatibility"""
    try:
        # Parse the incoming request
        body = await request.body()
        logger.info(f"Received answer request: {body.decode()}")
        
        data = await request.json()
        
        # Extract fields from the frontend format
        question_id = data.get("question_id")
        answer_text = data.get("text")  # Frontend sends "text"
        user_id = data.get("user_id", "blob")  # User ID or default to "blob"
        
        if not question_id:
            return JSONResponse(
                status_code=422,
                content={"detail": "Missing required field: question_id"}
            )
            
        if not answer_text:
            return JSONResponse(
                status_code=422,
                content={"detail": "Missing required field: text"}
            )
            
        # Create AnswerRequest model manually with the right field names
        from scripts.models.schemas import AnswerRequest
        answer_req = AnswerRequest(
            question_id=question_id,
            answer=answer_text  # Map "text" to "answer" expected by backend
        )
        
        # Set user info in request state to be used by create_answer
        request.state.user_id = user_id
        request.state.user_email = data.get("user_email", user_id)
        
        # Call the existing endpoint function directly
        from api.routes.questions import create_answer
        return await create_answer(answer_req, request)
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request: {e}")
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid JSON: {str(e)}"}
        )
    except Exception as e:
        logger.error(f"Error in answers_alias: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

# Define main function for Poetry entry point
def main():
    """Entry point for the application."""
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", 8999))
    
    # Set up signal handlers for keyboard interrupts
    import signal
    
    def handle_keyboard_interrupt(sig, frame):
        """Handle Ctrl+C by triggering a clean shutdown"""
        logger.info("Keyboard interrupt received, shutting down server...")
        print("\nShutting down server due to keyboard interrupt...")
        
        # We just return and let uvicorn handle the shutdown
        # This will trigger the shutdown event which cleans up resources
        pass
    
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    
    try:
        logger.info(f"Starting server on {host}:{port}...")
        uvicorn.run(app, host=host, port=port)
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt")
        # The shutdown event will be triggered automatically

# For local development
if __name__ == "__main__":
    main()