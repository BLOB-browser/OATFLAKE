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

# Setup logging with reduced noise
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

# Reduce noise from various loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("app").setLevel(logging.WARNING)
logging.getLogger("api").setLevel(logging.WARNING)
logging.getLogger("scripts").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Define the global server variable at the module level
server = None

# Set up signal handling to gracefully handle Ctrl+C
def signal_handler(sig, frame):
    """Handle interrupt signals gracefully to ensure proper cleanup"""
    print("\n⏹️  Shutting down...")
    
    # Cancel any active knowledge processing
    try:
        import requests
        requests.post("http://localhost:8999/api/knowledge/cancel", json={"force": True}, timeout=1)
    except:
        pass
        
    # Set cancellation flag
    try:
        from scripts.analysis.main_processor import MainProcessor
        MainProcessor._cancel_processing = True
    except:
        pass
    
    time.sleep(0.5)  # Brief pause for cleanup
    
    # Stop schedulers
    try:
        from scripts.services import training_scheduler
        if training_scheduler._running:
            training_scheduler.stop(timeout=3.0)
    except:
        pass
    
    # Stop the server
    try:
        global server
        if server is not None:
            server.should_exit = True
            server.force_exit = True
    except:
        pass
    
    print("✅ Shutdown complete")
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
            time.sleep(1)  # Wait a moment for processes to clean up
        except Exception:
            pass  # Silently handle cleanup errors
        
    def _setup_ngrok(self):
        """Configure ngrok with auth token"""
        auth_token = os.getenv('NGROK_AUTH_TOKEN')
        if not auth_token:
            return False  # Silently skip if no token
            
        try:
            conf.get_default().auth_token = auth_token
            return True
        except Exception:
            return False
        
    def _load_tunnel_config(self):
        """Load existing tunnel configuration"""
        try:
            if self.config_path.exists():
                config = json.loads(self.config_path.read_text())
                return config.get('tunnel_url')
        except Exception:
            pass  # Silently handle errors
        return None

    def _save_tunnel_config(self, url):
        """Save tunnel configuration"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {'tunnel_url': url}
            self.config_path.write_text(json.dumps(config))
        except Exception:
            pass  # Silently handle errors
        
    def start_ngrok(self):
        try:
            self._cleanup_existing_sessions()
            
            if not self._setup_ngrok():
                return None  # Skip ngrok silently
            
            self.ngrok_tunnel = ngrok.connect(8999, "http")
            self._save_tunnel_config(self.ngrok_tunnel.public_url)
            
            print(f"🌐 Tunnel: {self.ngrok_tunnel.public_url}")
            return self.ngrok_tunnel.public_url
            
        except Exception:
            return None  # Fail silently
            
    def cleanup(self):
        try:
            if self.ngrok_tunnel:
                ngrok.disconnect(self.ngrok_tunnel.public_url)
            ngrok.kill()
        except Exception:
            pass  # Silently handle cleanup errors

def run_server(app):
    """Run the FastAPI server"""
    try:
        uvicorn.run(app, host="127.0.0.1", port=8999, log_level="warning", access_log=False)
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
        else:
            app.state.ngrok_url = None
            
        # Open browser after a delay
        threading.Thread(
            target=lambda: (time.sleep(1.5), webbrowser.open('http://localhost:8999')),
            daemon=True
        ).start()
        
        # Get host and port from environment variables or use defaults
        host = os.environ.get("APP_HOST", "127.0.0.1")
        port = int(os.environ.get("APP_PORT", 8999))
        
        # Configure uvicorn with minimal logging
        server = uvicorn.Server(uvicorn.Config(
            app='app:app',
            host=host,
            port=port,
            log_level="warning",  # Reduce from "info" to "warning"
            access_log=False,     # Disable access logging
            loop="auto"
        ))
        
        # Run server with minimal logging
        try:
            print(f"🚀 Starting server on {host}:{port}")
            print(f"📱 Open: http://localhost:{port}")
            server.run()
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    finally:
        if server:
            server.cleanup()
    
    return 0

if __name__ == "__main__":
    # Load environment variables
    project_env_path = Path(__file__).parent / '.env'
    if project_env_path.exists():
        load_dotenv(dotenv_path=str(project_env_path))

    blob_env_path = Path.home() / '.blob' / '.env'
    if blob_env_path.exists():
        load_dotenv(dotenv_path=str(blob_env_path), override=True)
    
    exit(main())
