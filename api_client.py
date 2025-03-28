import httpx
from scripts.core.config import BACKEND_CONFIG
import logging

logger = logging.getLogger(__name__)

class AgentAPIClient:
    def __init__(self):
        self.agent_url = BACKEND_CONFIG['AGENT_URL']
        self.api_key = BACKEND_CONFIG['API_KEY']
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-API-Key': self.api_key
        }

    async def register_instance(self, instance_id: str, group_id: str):
        """Register this backend with the agent"""
        try:
            logger.info(f"Registering with agent: {self.agent_url}")
            logger.info(f"Instance ID: {instance_id}")
            logger.info(f"Group ID: {group_id}")
            logger.info(f"Local URL: {BACKEND_CONFIG['LOCAL_URL']}")
            
            async with httpx.AsyncClient(verify=False) as client:  # Allow self-signed certs
                response = await client.post(
                    f"{self.agent_url}/register",
                    json={
                        "instance_id": instance_id,
                        "group_id": group_id,
                        "local_url": BACKEND_CONFIG['LOCAL_URL']
                    },
                    headers=self.headers,
                    timeout=30.0  # Increase timeout
                )
                
                if response.status_code == 403:
                    logger.error("API key validation failed")
                    return None
                
                if not response.is_success:
                    logger.error(f"Registration failed with status {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    return None
                    
                data = response.json()
                logger.info(f"Registration successful: {data}")
                return data
                
        except Exception as e:
            logger.error(f"Registration failed: {e}")
            return None

    async def health_check(self):
        """Check connection to agent"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.agent_url}/health",
                    headers=self.headers
                )
                return response.is_success
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
