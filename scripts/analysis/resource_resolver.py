#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ResourceResolver handles matching URLs to their parent resources.
This is extracted from URLProcessor to make the code more modular.
"""

import logging
import re
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class ResourceResolver:
    """
    Resolves resources for URLs based on various matching strategies.
    This component is responsible for finding the appropriate resource
    for a URL, which is needed for proper processing and context.
    """
    
    def __init__(self):
        """Initialize the ResourceResolver."""
        pass
    
    def resolve_resource_for_url(self, url: str, origin_url: str, resources_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Find the appropriate resource for a URL.
        
        Args:
            url: The URL to find a resource for
            origin_url: The origin URL (where this URL was discovered)
            resources_df: DataFrame containing resources information
            
        Returns:
            Resource dictionary or None if no resource is found
        """
        # Filter resources to only include those with URLs
        resources_with_url = resources_df[resources_df['origin_url'].notna()]
        
        if resources_with_url.empty:
            logger.warning(f"No resources with URLs available")
            return None
        
        # Step 1: Try exact match with origin URL
        for _, row in resources_with_url.iterrows():
            if row['origin_url'] == origin_url:
                resource = row.to_dict()
                logger.info(f"Found exact origin match for URL: {url} -> {origin_url}")
                return resource
        
        # Step 2: Try domain matching
        resource = self._match_by_domain(origin_url, resources_with_url)
        if resource:
            return resource
            
        # Step 3: Try fallback resources
        resource = self._find_fallback_resource(resources_with_url)
        if resource:
            return resource
            
        # If all attempts failed
        logger.warning(f"Could not find any resource for URL: {url}")
        return None
        
    def resolve_resources_by_id(self, resource_ids: List[str], resources_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Find resources matching the provided resource IDs.
        
        Args:
            resource_ids: List of resource IDs to match
            resources_df: DataFrame containing resources information
            
        Returns:
            Dictionary mapping resource IDs to resource dictionaries
        """
        resources_with_url = resources_df[resources_df['origin_url'].notna()]
        resource_mapping = {}
        
        for resource_id in resource_ids:
            if not resource_id:
                continue
                
            # Try to find resource by title match
            resource_found = False
            for _, row in resources_with_url.iterrows():
                title = row.get('title', '').strip()
                if title == resource_id:
                    resource_mapping[resource_id] = row.to_dict()
                    resource_found = True
                    logger.info(f"Found resource for ID: {resource_id}")
                    break
            
            if not resource_found:
                logger.info(f"Could not find exact resource for ID: {resource_id}, using first resource as fallback")
                if not resources_with_url.empty:
                    resource_mapping[resource_id] = resources_with_url.iloc[0].to_dict()
                    
        return resource_mapping
        
    def _match_by_domain(self, url: str, resources_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Match URL to resource based on domain name.
        
        Args:
            url: URL to match
            resources_df: DataFrame containing resources
            
        Returns:
            Matched resource or None
        """
        # Extract main domain from the URL
        origin_domain_match = re.search(r'https?://([^/]+)', url)
        if not origin_domain_match:
            return None
            
        origin_domain = origin_domain_match.group(1)
        logger.info(f"Trying domain match for: {origin_domain}")
        
        # Look for resources with the same domain
        for _, row in resources_df.iterrows():
            resource_url = row['origin_url']
            resource_domain_match = re.search(r'https?://([^/]+)', resource_url)
            if resource_domain_match and resource_domain_match.group(1) == origin_domain:
                resource = row.to_dict()
                logger.info(f"Found domain match for URL: {url} -> resource: {resource_url}")
                return resource
                
        # If exact domain match fails, try base domain matching
        main_domain = origin_domain_match.group(1)
        base_domain = re.sub(r'^[^.]+\.', '', main_domain) if '.' in main_domain else main_domain
        logger.warning(f"Looking for resources matching base domain: {base_domain}")
        
        # Find any resource with this base domain
        for _, row in resources_df.iterrows():
            resource_url = row['origin_url']
            if base_domain in resource_url:
                resource = row.to_dict()
                logger.warning(f"Found base domain match: {base_domain} in {resource_url}")
                return resource
                
        return None
        
    def _find_fallback_resource(self, resources_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Find a fallback resource when no direct match is found.
        
        Args:
            resources_df: DataFrame containing resources
            
        Returns:
            Fallback resource or None
        """
        # First look for fabacademy.org as a fallback (domain-specific logic)
        for _, row in resources_df.iterrows():
            if 'fabacademy.org' in row['origin_url']:
                resource = row.to_dict()
                logger.warning(f"Using fabacademy.org resource as fallback")
                return resource
        
        # Last resort: use the first resource
        if not resources_df.empty:
            resource = resources_df.iloc[0].to_dict()
            logger.warning(f"Using first resource as last-resort fallback")
            return resource
            
        return None
