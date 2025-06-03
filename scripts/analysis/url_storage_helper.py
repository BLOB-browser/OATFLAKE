#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

def get_discovery_status(url_storage, url: str) -> bool:
    """
    Safe wrapper to get discovery status for a URL - used to avoid errors when method doesn't exist.
    
    Args:
        url_storage: The URL storage manager
        url: The URL to check
        
    Returns:
        True if discovery is completed, False otherwise
    """
    try:
        # Check if method exists
        if hasattr(url_storage, 'get_discovery_status'):
            return url_storage.get_discovery_status(url)
        else:
            # Backward compatibility - always return false to ensure discovery happens
            logger.debug(f"get_discovery_status not implemented, defaulting to False for {url}")
            return False
    except Exception as e:
        logger.warning(f"Error getting discovery status for {url}: {e}")
        # On error, default to False to ensure discovery happens
        return False
        
def set_discovery_status(url_storage, url: str, status: bool) -> None:
    """
    Safe wrapper to set discovery status for a URL - used to avoid errors when method doesn't exist.
    
    Args:
        url_storage: The URL storage manager
        url: The URL to update
        status: The discovery status to set
    """
    try:
        # Only call if method exists
        if hasattr(url_storage, 'set_discovery_status'):
            url_storage.set_discovery_status(url, status)
        else:
            # Log but continue if method doesn't exist
            logger.debug(f"set_discovery_status not implemented, skipping for {url}")
    except Exception as e:
        logger.warning(f"Error setting discovery status for {url}: {e}")
        # Continue execution even if this fails
        
def save_url_resource_relationship(url_storage, url: str, resource_id: str) -> bool:
    """
    Safe wrapper to save URL-resource relationship without raising errors when method doesn't exist.
    
    Args:
        url_storage: The URL storage manager
        url: The URL to associate
        resource_id: The resource ID to associate with the URL
        
    Returns:
        True if relationship was saved, False otherwise
    """
    try:
        # Check if method exists
        if hasattr(url_storage, 'save_url_resource_relationship'):
            url_storage.save_url_resource_relationship(url, resource_id)
            return True
        elif hasattr(url_storage, 'associate_url_with_resource'):
            # Alternative method name
            url_storage.associate_url_with_resource(url, resource_id)
            return True
        else:
            # Method doesn't exist
            logger.debug(f"URL-resource relationship methods not implemented, skipping for {url}")
            return False
    except Exception as e:
        logger.warning(f"Error saving URL-resource relationship for {url}: {e}")
        return False

def save_resource_url(url_storage, resource_id: str, url: str, depth: int = 0) -> bool:
    """
    Safe wrapper to save URL association with a resource without raising errors when method doesn't exist.
    
    Args:
        url_storage: The URL storage manager
        resource_id: The resource ID to associate
        url: The URL to associate with the resource
        depth: The depth level of the URL (0=main, 1=first level, etc.)
        
    Returns:
        True if successfully saved, False otherwise
    """
    try:
        # Check if method exists
        if hasattr(url_storage, 'save_resource_url'):
            url_storage.save_resource_url(resource_id, url, depth)
            return True
        else:
            # Method doesn't exist, try alternative
            logger.debug(f"save_resource_url not implemented, trying alternatives for resource {resource_id}")
            if hasattr(url_storage, 'associate_resource_url'):
                url_storage.associate_resource_url(resource_id, url, depth)
                return True
            return False
    except Exception as e:
        logger.warning(f"Error saving resource URL relationship for {url} ({resource_id}): {e}")
        return False
