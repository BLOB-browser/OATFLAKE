"""
Pinned items manager for search results.

This module allows users to pin and unpin search results, and to retrieve 
the pinned items for use in search and other contexts.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class PinnedItemsManager:
    """
    Manages pinned search results and chunks, providing persistence and retrieval.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the pinned items manager.
        
        Args:
            storage_dir: Directory to store pinned items. If None, defaults to ~/.oatflake/pinned/
        """
        if storage_dir is None:
            self.storage_dir = Path.home() / ".oatflake" / "pinned"
        else:
            self.storage_dir = Path(storage_dir)
            
        # Ensure the storage directory exists
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Store pinned items in memory for fast access
        self.pinned_items: Dict[str, Dict[str, Any]] = {}
        
        # Load existing pinned items
        self._load_pinned_items()
        
        logger.info(f"Initialized PinnedItemsManager with {len(self.pinned_items)} items from {self.storage_dir}")
        
    def _load_pinned_items(self) -> None:
        """Load all pinned items from storage."""
        try:
            pin_file = self.storage_dir / "pinned_items.json"
            if pin_file.exists():
                with open(pin_file, "r") as f:
                    self.pinned_items = json.load(f)
                logger.info(f"Loaded {len(self.pinned_items)} pinned items from {pin_file}")
            else:
                logger.info(f"No pinned items file found at {pin_file}, starting with empty set")
                self.pinned_items = {}
        except Exception as e:
            logger.error(f"Error loading pinned items: {e}")
            self.pinned_items = {}
            
    def _save_pinned_items(self) -> None:
        """Save all pinned items to storage."""
        try:
            pin_file = self.storage_dir / "pinned_items.json"
            with open(pin_file, "w") as f:
                json.dump(self.pinned_items, f, indent=2)
            logger.info(f"Saved {len(self.pinned_items)} pinned items to {pin_file}")
        except Exception as e:
            logger.error(f"Error saving pinned items: {e}")
            
    def pin_item(self, item: Dict[str, Any], item_type: str = "search_result") -> str:
        """
        Pin an item.
        
        Args:
            item: The item to pin (dictionary)
            item_type: Type of item (e.g., "search_result", "chunk")
            
        Returns:
            The ID of the pinned item
        """
        # Generate a unique ID for the item if it doesn't have one
        item_id = item.get("id", str(uuid.uuid4()))
        
        # Add metadata
        item["id"] = item_id
        item["pinned_at"] = datetime.now().isoformat()
        item["item_type"] = item_type
        
        # Store the item
        self.pinned_items[item_id] = item
        
        # Save to disk
        self._save_pinned_items()
        
        logger.info(f"Pinned {item_type} item with ID {item_id}")
        return item_id
        
    def unpin_item(self, item_id: str) -> bool:
        """
        Unpin an item by ID.
        
        Args:
            item_id: ID of the item to unpin
            
        Returns:
            True if the item was unpinned, False if it wasn't found
        """
        if item_id in self.pinned_items:
            item = self.pinned_items.pop(item_id)
            self._save_pinned_items()
            logger.info(f"Unpinned item {item_id} of type {item.get('item_type', 'unknown')}")
            return True
        else:
            logger.warning(f"Attempted to unpin nonexistent item with ID {item_id}")
            return False
            
    def get_pinned_item(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a pinned item by ID.
        
        Args:
            item_id: ID of the item to retrieve
            
        Returns:
            The item if found, None otherwise
        """
        return self.pinned_items.get(item_id)
        
    def get_all_pinned_items(self, item_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all pinned items, optionally filtered by type.
        
        Args:
            item_type: If provided, only return items of this type
            
        Returns:
            List of pinned items
        """
        if item_type is None:
            return list(self.pinned_items.values())
        else:
            return [item for item in self.pinned_items.values() if item.get("item_type") == item_type]
            
    def get_pinned_item_count(self, item_type: Optional[str] = None) -> int:
        """
        Get the count of pinned items, optionally filtered by type.
        
        Args:
            item_type: If provided, only count items of this type
            
        Returns:
            Count of pinned items
        """
        if item_type is None:
            return len(self.pinned_items)
        else:
            return len([item for item in self.pinned_items.values() if item.get("item_type") == item_type])
            
    def clear_all_pinned_items(self) -> int:
        """
        Clear all pinned items.
        
        Returns:
            Number of items cleared
        """
        count = len(self.pinned_items)
        self.pinned_items = {}
        self._save_pinned_items()
        logger.info(f"Cleared all {count} pinned items")
        return count
