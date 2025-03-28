from pathlib import Path
import logging
from typing import Dict, List, Optional  # Added Dict import
from datetime import datetime
import json
import asyncio
from .store_types import StoreType, BucketMetadata
from .vector_store_manager import VectorStoreManager
from langchain.schema import Document

logger = logging.getLogger(__name__)

class BucketManager:
    def __init__(self, group_id: str, base_path: Path):
        self.group_id = group_id
        self.base_path = base_path
        self.vector_manager = VectorStoreManager(base_path)
        self.buckets_path = base_path / "buckets"
        self.buckets_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize store metadata
        self.store_info = {
            StoreType.RULES.value: {"name": "Rules", "description": "Group rules and guidelines"},
            StoreType.GOALS.value: {"name": "Goals", "description": "Group goals and objectives"},
            StoreType.DEFINITIONS.value: {"name": "Definitions", "description": "Term definitions"},
            StoreType.CONTENT.value: {"name": "Content", "description": "General content"},
            StoreType.SUMMARY.value: {"name": "Summary", "description": "Content summaries"}
        }

    async def create_or_update_bucket(
        self,
        bucket_type: StoreType,
        documents: List[Document],
        metadata: BucketMetadata
    ) -> bool:
        """Create or update a specific bucket"""
        try:
            # Create store name with type prefix
            store_name = f"{bucket_type.value}_{metadata.name}"
            
            # Add type-specific metadata
            store_metadata = {
                **metadata.dict(),
                **self.store_info[bucket_type.value],
                "bucket_type": bucket_type.value,
                "updated_at": datetime.now().isoformat()
            }
            
            # Create vector store
            success = await self.vector_manager.create_or_update_store(
                store_name=store_name,
                documents=documents,
                metadata=store_metadata
                # group_id parameter not needed
            )
            
            if success:
                # Save bucket metadata
                metadata_path = self.buckets_path / f"{store_name}_meta.json"
                with open(metadata_path, 'w') as f:
                    json.dump(store_metadata, f, indent=2)
                
                # Update summary if needed
                if bucket_type != StoreType.SUMMARY:
                    await self._update_summary()
                    
            return success
            
        except Exception as e:
            logger.error(f"Error creating bucket {bucket_type}: {e}")
            return False

    async def _update_summary(self) -> bool:
        """Update summary store with key information from all buckets"""
        try:
            # Get representative chunks from each store
            all_stores = self.vector_manager.list_stores()
            summary_docs = []
            
            for store in all_stores:
                if not store["name"].startswith("summary_"):
                    chunks = await self.vector_manager.get_representative_chunks(
                        store["name"],
                        num_chunks=3
                    )
                    summary_docs.extend(chunks)
            
            if summary_docs:
                # Create summary store
                return await self.create_or_update_bucket(
                    bucket_type=StoreType.SUMMARY,
                    documents=summary_docs,
                    metadata=BucketMetadata(
                        name="auto_summary",
                        description="Automatically generated summary",
                        topics=["summary"],
                        fields=["all"],
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        source_type="system",
                        source_path="",
                        tags=["summary", "auto"]
                    )
                )
            return True
            
        except Exception as e:
            logger.error(f"Error updating summary: {e}")
            return False

    async def search(
        self,
        query: str,
        bucket_type: Optional[StoreType] = None,
        top_k: int = 5
    ) -> Dict:
        """Search within specific bucket type or across all"""
        try:
            if bucket_type:
                # Search only in stores of specified type
                store_pattern = f"{bucket_type.value}_*"
            else:
                # Try summary first
                summary_results = await self.vector_manager.search(
                    query=query,
                    store_names=["summary_auto_summary"],
                    top_k=top_k
                )
                
                if summary_results and any(summary_results.values()):
                    return summary_results
                    
                # If no good summary matches, search all
                store_pattern = "*"
            
            return await self.vector_manager.search(
                query=query,
                store_names=[store_pattern],
                top_k=top_k
            )
            
        except Exception as e:
            logger.error(f"Error searching buckets: {e}")
            return {}

    def get_bucket_info(self, bucket_type: Optional[StoreType] = None) -> Dict:
        """Get information about buckets"""
        try:
            stores = self.vector_manager.list_stores()
            
            if bucket_type:
                # Filter stores by type
                stores = [s for s in stores if s["name"].startswith(f"{bucket_type.value}_")]
            
            return {
                "total_stores": len(stores),
                "stores": stores,
                "last_updated": max((s["last_updated"] for s in stores), default=None)
            }
            
        except Exception as e:
            logger.error(f"Error getting bucket info: {e}")
            return {}
