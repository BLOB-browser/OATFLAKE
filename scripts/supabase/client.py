from supabase import create_client
import os
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class DomainManager:
    def __init__(self):
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        if not supabase_url or not supabase_key:
            logger.error("Missing Supabase credentials")
            raise ValueError("Supabase credentials not configured")
            
        self.client = create_client(supabase_url, supabase_key)

    async def get_domain(self, instance_id: str) -> Optional[Dict]:
        """Get existing domain for instance"""
        try:
            response = self.client.table('domains').select('*').eq('instance_id', instance_id).execute()
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching domain: {e}")
            return None

    async def update_domain(self, instance_id: str, domain: str, local_url: str) -> bool:
        """Update or create domain entry"""
        try:
            data = {
                'instance_id': instance_id,
                'domain': domain,
                'local_url': local_url,
                'updated_at': 'now()'
            }
            response = self.client.table('domains').upsert(data).execute()
            return bool(response.data)
        except Exception as e:
            logger.error(f"Error updating domain: {e}")
            return False
