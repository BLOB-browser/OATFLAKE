import asyncio
import os
from dotenv import load_dotenv
import logging
from scripts.storage.supabase import SupabaseClient
import json
from postgrest import APIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_supabase_connection():
    client = SupabaseClient()
    test_uuid = "dfe54afa-f0fa-463c-b79f-1c130e91624a"
    
    print("\n=== Supabase Connection Test ===")
    print(f"URL: {client.url}")
    print(f"Key type: {'service_role' if 'service_role' in client.key else 'anon'}")
    
    try:
        # Try a raw query first to check schema
        print("\nChecking table schema...")
        raw_query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'groups';
        """
        
        try:
            schema = client.client.table('groups').select('*').limit(1).execute()
            print("\nTable columns:")
            if schema.data:
                print(json.dumps(list(schema.data[0].keys()), indent=2))
        except Exception as e:
            print(f"Schema query failed: {e}")

        # Try with direct query builder
        print("\nTrying direct query...")
        response = client.client.from_('groups') \
            .select('*') \
            .order('created_at', desc=True) \
            .limit(5) \
            .execute()
        
        print("\nLatest 5 groups:")
        if response.data:
            for group in response.data:
                print(f"\nGroup ID: {group.get('id', 'No ID')}")
                print(f"Name: {group.get('name', 'No name')}")
                print(f"Type: {group.get('institution_type', 'No type')}")
                print("-" * 50)
        else:
            print("No groups found")

        # Try the specific UUID with full debugging
        print(f"\nQuerying specific UUID: {test_uuid}")
        uuid_query = client.client.from_('groups') \
            .select('*') \
            .eq('id', test_uuid) \
            .execute()
        
        print("\nQuery results:")
        print(f"Status: {'Found' if uuid_query.data else 'Not found'}")
        print(f"Data: {json.dumps(uuid_query.data, indent=2)}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        print(f"Error type: {type(e)}")
        print("\nDebug info:")
        print("1. Check Supabase dashboard:")
        print("   - Table structure")
        print("   - RLS policies")
        print("   - API key permissions")
        print(f"2. Full error: {e}")

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(test_supabase_connection())