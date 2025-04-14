#!/usr/bin/env python3
import uvicorn
import webbrowser
import logging
import threading
import time
from dotenv import load_dotenv
from pyngrok import ngrok, conf
import os
import json
from pathlib import Path
import signal
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the global server variable at the module level
server = None

# Set up signal handling to gracefully handle Ctrl+C
def signal_handler(sig, frame):
    """Handle interrupt signals gracefully to ensure proper cleanup"""
    print("\nReceived keyboard interrupt (Ctrl+C)...")
    logger.info("Keyboard interrupt received, shutting down...")
    
    # First try to cancel any active knowledge processing
    try:
        print("Cancelling any active knowledge processing...")
        import requests
        try:
            # Send cancellation request to knowledge process
            cancel_response = requests.post(
                "http://localhost:8999/api/knowledge/cancel",
                json={"force": True},
                timeout=2
            )
            if cancel_response.status_code == 200:
                print("Cancellation request sent successfully")
        except:
            print("Could not send cancellation request - server may not be running")
    except ImportError:
        pass
        
    # Set cancellation flag in MainProcessor if available
    try:
        from scripts.analysis.main_processor import MainProcessor
        MainProcessor._cancel_processing = True
        print("Set internal cancellation flag for processor")
    except:
        pass
    
    # Wait a moment for processing to stop
    import time
    time.sleep(1)
    
    # First stop any running schedulers
    try:
        from scripts.services import training_scheduler
        if training_scheduler._running:
            print("Stopping training scheduler...")
            logger.info("Stopping training scheduler due to keyboard interrupt")
            # Use a short timeout since we're shutting down
            training_scheduler.stop(timeout=5.0)
    except Exception as e:
        print(f"Error stopping scheduler: {e}")
        logger.error(f"Error stopping scheduler during shutdown: {e}")
    
    # Then stop the application
    try:
        global server
        # Send termination signal to uvicorn
        if server is not None:
            print("Stopping web server...")
            server.should_exit = True
            
            # For harder shutdown if needed
            server.force_exit = True
    except Exception as e:
        print(f"Error stopping server: {e}")
        logger.error(f"Error stopping server during shutdown: {e}")
    
    print("Exiting gracefully...")
    # Force immediate exit as a last resort
    os._exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)  # Keyboard interrupt (Ctrl+C)
signal.signal(signal.SIGTERM, signal_handler) # Termination signal

class ServerManager:
    def __init__(self):
        self.ngrok_tunnel = None
        self.config_path = Path.home() / '.blob' / 'tunnel_config.json'
        
    def _cleanup_existing_sessions(self):
        """Kill any existing ngrok processes"""
        try:
            ngrok.kill()  # Kill any existing processes
            # Wait a moment for processes to clean up
            time.sleep(1)
        except Exception as e:
            logger.warning(f"Error cleaning up existing ngrok sessions: {e}")
        
    def _setup_ngrok(self):
        """Configure ngrok with auth token"""
        auth_token = os.getenv('NGROK_AUTH_TOKEN')
        if not auth_token:
            logger.warning("NGROK_AUTH_TOKEN not found in environment variables. Tunnel will not be available.")
            return False
            
        try:
            # Set new auth token
            conf.get_default().auth_token = auth_token
            logger.info("Ngrok auth token configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure ngrok: {e}")
            return False
        
    def _load_tunnel_config(self):
        """Load existing tunnel configuration"""
        try:
            if self.config_path.exists():
                config = json.loads(self.config_path.read_text())
                return config.get('tunnel_url')
        except Exception as e:
            logger.error(f"Error loading tunnel config: {e}")
        return None

    def _save_tunnel_config(self, url):
        """Save tunnel configuration"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {'tunnel_url': url}
            self.config_path.write_text(json.dumps(config))
        except Exception as e:
            logger.error(f"Error saving tunnel config: {e}")
        
    def start_ngrok(self):
        try:
            # Clean up existing sessions first
            self._cleanup_existing_sessions()
            
            # Setup ngrok, but make it optional
            if not self._setup_ngrok():
                logger.info("Ngrok setup skipped - continuing without tunnel")
                return None
            
            # Create new tunnel
            self.ngrok_tunnel = ngrok.connect(8999, "http")
            
            # Save new tunnel URL
            self._save_tunnel_config(self.ngrok_tunnel.public_url)
            
            logger.info(f"Created new tunnel: {self.ngrok_tunnel.public_url}")
            return self.ngrok_tunnel.public_url
            
        except Exception as e:
            logger.error(f"Ngrok error: {e}")
            return None
            
    def cleanup(self):
        try:
            if self.ngrok_tunnel:
                ngrok.disconnect(self.ngrok_tunnel.public_url)
            ngrok.kill()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def run_server(app):
    """Run the FastAPI server"""
    try:
        uvicorn.run(app, host="127.0.0.1", port=8999)
    except Exception as e:
        logger.error(f"Server error: {e}")

def main():
    """Entry point for the application."""
    global server
    
    server = None
    try:
        # Import app module
        from app import app
        
        # Initialize server manager
        server = ServerManager()
        app.state.server_running = True
        
        # Start ngrok, but continue even if it fails
        public_url = server.start_ngrok()
        if public_url:
            app.state.ngrok_url = public_url
            logger.info(f"Ngrok tunnel available at: {public_url}")
        else:
            logger.warning("Ngrok tunnel not available - continuing without it")
            app.state.ngrok_url = None
            
        # Open browser after a delay
        threading.Thread(
            target=lambda: (time.sleep(1.5), webbrowser.open('http://localhost:8999')),
            daemon=True
        ).start()
        
        # Get host and port from environment variables or use defaults
        host = os.environ.get("APP_HOST", "127.0.0.1")
        port = int(os.environ.get("APP_PORT", 8999))
        
        # Configure uvicorn to handle signals internally
        server = uvicorn.Server(uvicorn.Config(
            app='app:app',
            host=host,
            port=port,
            log_level="info",
            loop="auto"
        ))
        
        # Run server with modified signal handling that integrates with our custom handler
        try:
            # Run the server in a context that can be interrupted
            print(f"Starting server on {host}:{port}...")
            server.run()
        except KeyboardInterrupt:
            # Our signal handler should catch this, but just in case
            signal_handler(signal.SIGINT, None)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    finally:
        if server:
            server.cleanup()
    
    return 0

if __name__ == "__main__":
    # First, print the current working directory for debugging
    cwd = os.getcwd()
    logger.info(f"Current working directory: {cwd}")
    
    # Load from default .env file in the project root first with verbose output
    project_env_path = Path(__file__).parent / '.env'
    if project_env_path.exists():
        load_dotenv(dotenv_path=str(project_env_path))
        logger.info(f"Loaded environment from project root: {project_env_path}")

        # After loading, check if token is actually set
        if os.getenv('NGROK_AUTH_TOKEN'):
            logger.info("NGROK_AUTH_TOKEN successfully loaded from project .env file")
    else:
        logger.warning(f"Project .env file not found at {project_env_path}")
        
    # Then try to load from ~/.blob/.env if it exists
    blob_env_path = Path.home() / '.blob' / '.env'
    if blob_env_path.exists():
        load_dotenv(dotenv_path=str(blob_env_path), override=True)
        logger.info(f"Loaded environment from user directory: {blob_env_path}")

        # After loading, check if token is actually set
        if os.getenv('NGROK_AUTH_TOKEN'):
            logger.info("NGROK_AUTH_TOKEN successfully loaded from ~/.blob/.env file")
    
    exit(main())
