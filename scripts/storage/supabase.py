from supabase import create_client, Client
import logging
from typing import Optional, Dict, Any
import os
import json
import httpx

logger = logging.getLogger(__name__)

DEFAULT_GROUP_IMAGE = "/static/icons/GROUPLOGO.png"  # Changed to use local file

class SupabaseClient:
    def __init__(self, auth_token: str = None, refresh_token: str = None):  # Add refresh_token parameter
        """Initialize Supabase client"""
        self.url = os.getenv('SUPABASE_URL')
        self.key = os.getenv('SUPABASE_KEY')
        self.auth_token = auth_token
        self.refresh_token = refresh_token  # Store refresh token
        
        if not self.url or not self.key:
            raise ValueError("Supabase configuration not found")
        
        # Initialize basic client
        self.client = create_client(self.url, self.key)
        
        # Set session if tokens provided
        if auth_token and refresh_token:
            self.client.auth.set_session(
                access_token=auth_token,
                refresh_token=refresh_token
            )
        
        # Create headers dictionary
        self.headers = {
            'apikey': self.key,
            'Authorization': f'Bearer {auth_token}' if auth_token else f'Bearer {self.key}'
        }
        
        logger.info("Supabase client initialized")

    @classmethod
    async def create_authenticated(cls, email: str, password: str) -> 'SupabaseClient':
        """Create an authenticated client instance"""
        client = cls()
        try:
            auth_response = await client.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            logger.info("Authentication successful")
            return client
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

    async def login(self, email: str, password: str) -> dict:
        """Authenticate user and store access token"""
        try:
            logger.info(f"Attempting login for user: {email}")
            
            auth_response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if not auth_response.session:
                raise ValueError("No session in auth response")
                
            # Store tokens
            self.access_token = auth_response.session.access_token
            self.refresh_token = auth_response.session.refresh_token
            
            # Set session with both tokens
            self.client.auth.set_session(
                access_token=self.access_token,
                refresh_token=self.refresh_token
            )
            
            logger.info("Authentication successful")
            return {
                "token": self.access_token,
                "refresh_token": self.refresh_token,
                "user": {
                    "id": auth_response.session.user.id,
                    "email": auth_response.session.user.email,
                    "last_sign_in": auth_response.session.user.last_sign_in_at
                }
            }
            
        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            raise ValueError(f"Authentication failed: {str(e)}")

    async def verify_session(self, token: str) -> bool:
        """Verify if a session token is valid"""
        try:
            self.client.auth.set_session(token)
            user = await self.client.auth.get_user()
            return bool(user)
        except Exception:
            return False

    async def get_group_info(self, uuid: str) -> Optional[Dict[str, Any]]:
        """Get group information using UUID"""
        try:
            logger.info(f"Looking up group with ID: {uuid}")
            logger.info(f"Using auth headers: {self.headers}")

            # Query by group id with proper headers
            response = self.client.table('groups') \
                .select('id, name, description, type, institution_type, fields, created_by, cover_image, backend_url') \
                .eq('id', uuid) \
                .single() \
                .execute()
            
            logger.info(f"Query response: {json.dumps(response.data if response else 'No response', indent=2)}")
            
            if response.data:
                return response.data

            return None
                
        except Exception as e:
            logger.error(f"Error fetching group info: {e}", exc_info=True)
            return None

    async def list_all_groups(self) -> list:
        """List all available groups"""
        try:
            # Simple direct query
            result = self.client.from_('groups') \
                .select('*') \
                .execute()
            
            logger.info(f"Found {len(result.data)} groups")
            if result.data:
                logger.info(f"First group structure: {json.dumps(result.data[0], indent=2)}")
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Error listing groups: {e}", exc_info=True)
            return []

    async def test_connection(self):
        """Test if the current token is valid"""
        try:
            # Try a simple query to test authentication
            await self.client.auth.get_user()
            return True
        except Exception as e:
            logger.error(f"Supabase connection test failed: {e}")
            raise
