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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            raise ValueError("NGROK_AUTH_TOKEN not found in environment variables")
            
        try:
            # Set new auth token
            conf.get_default().auth_token = auth_token
            logger.info("Ngrok auth token configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to configure ngrok: {e}")
            raise
        
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
            
            self._setup_ngrok()
            
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
    server = None
    try:
        # Import app module
        from app import app
        
        # Initialize server manager
        server = ServerManager()
        app.state.server_running = True
        
        # Start ngrok
        public_url = server.start_ngrok()
        if not public_url:
            logger.error("Failed to start ngrok tunnel")
            return 1
            
        app.state.ngrok_url = public_url
        
        # Open browser after a delay
        threading.Thread(
            target=lambda: (time.sleep(1.5), webbrowser.open('http://localhost:8999')),
            daemon=True
        ).start()
        
        # Start server with proper string reference
        logger.info("Starting server...")
        import uvicorn.main
        return uvicorn.run("app:app", host="127.0.0.1", port=8999)
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1
    finally:
        if server:
            server.cleanup()
    
    return 0

if __name__ == "__main__":
    load_dotenv()
    exit(main())
