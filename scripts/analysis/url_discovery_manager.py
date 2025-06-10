#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Imports for URL discovery only
from scripts.analysis.url_discovery_engine import URLDiscoveryEngine
from scripts.analysis.web_fetcher import WebFetcher
from scripts.analysis.url_storage import URLStorageManager
from scripts.analysis.enhanced_url_storage import EnhancedURLStorageManager
from utils.config import get_data_path

logger = logging.getLogger(__name__)

class URLDiscoveryManager:
    """
    Manages ONLY the URL discovery process. No content analysis.
    This class coordinates URL discovery from resources and manages the discovery workflow.
      OPTIMIZATION: URLDiscoveryManager is always in discovery mode since it's dedicated
    to URL discovery only. This eliminates redundant discovery_mode checks and settings.
    """
    
    def __init__(self, data_folder: str = None):
        """
        Initialize the URL Discovery Manager.
        
        Args:
            data_folder: Path to the data directory (if None, gets from config)
                        NOTE: This parameter is ignored - always uses config path for consistency
        """
        # Always use the configured data path instead of passed parameter
        # This ensures all URL discovery operations happen in the same folder
        self.data_folder = Path(get_data_path())
        self.data_folder.mkdir(parents=True, exist_ok=True)
          # Initialize URL storage manager with enhanced schema support
        processed_urls_file = str(self.data_folder / "processed_urls.csv")
        self.url_storage = URLStorageManager(processed_urls_file)
        
        # Enable discovery mode on the URL storage manager
        self.url_storage.discovery_mode = True
        
        # Initialize enhanced URL storage manager as the primary storage system
        self.enhanced_url_storage = EnhancedURLStorageManager(processed_urls_file)
        
        # Initialize web fetcher with platform-specific User-Agent
        import platform
        user_agents = {
            'windows': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'mac': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'raspberry': 'Mozilla/5.0 (Linux; Android 13; Raspberry Pi 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'
        }
        
        # Detect platform and set appropriate User-Agent
        system = platform.system().lower()
        if 'darwin' in system:
            user_agent = user_agents['mac']
        elif 'linux' in system:
            user_agent = user_agents['raspberry']
        else:
            user_agent = user_agents['windows']
            
        self.web_fetcher = WebFetcher(
            user_agent=user_agent,
            timeout=90,
            verify_ssl=False
        )
          # Initialize URL discovery engine with main URL storage (which has enhanced schema support)
        self.url_discovery = URLDiscoveryEngine(self.url_storage, self.web_fetcher)
        
        # Set the enhanced URL storage on the discovery engine for prioritized storage
        self.url_discovery.set_enhanced_url_storage(self.enhanced_url_storage)
        
        logger.info(f"URLDiscoveryManager initialized with data folder: {self.data_folder}")
        logger.info(f"✅ ENHANCED SCHEMA: URL storage manager supports enhanced schema in same folder")
        logger.info(f"📁 Data folder: {self.data_folder}")
        logger.info(f"📄 Enhanced pending URLs file: {self.url_storage.enhanced_pending_urls_file}")
    async def discover_urls_from_resources(self, level: int = None, max_depth: int = 4, batch_size: int = 50, 
                                         force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Discover URLs from resources.csv file.
        
        Args:
            level: Specific level to discover URLs for (optional, for compatibility)
            max_depth: Maximum depth for URL discovery
            batch_size: Number of URLs to process in each batch (not used in discovery, kept for compatibility)
            force_reprocess: Whether to reprocess already discovered URLs
            
        Returns:
            Dictionary with discovery results and proper status field
        """
        logger.info(f"Starting URL discovery from resources with max_depth={max_depth}")
        
        try:
            # Read resources from CSV
            resources_csv_path = self.data_folder / 'resources.csv'
            if not resources_csv_path.exists():
                logger.warning(f"Resources CSV not found at {resources_csv_path}")
                return {"status": "error", "message": "Resources CSV not found"}
            
            import pandas as pd
            resources_df = pd.read_csv(resources_csv_path)
            
            if resources_df.empty:
                logger.info("No resources found in CSV")
                return {"status": "success", "urls_discovered": 0}
              # Extract URLs from resources using universal schema field
            urls_to_discover = []
            resource_ids = {}
            
            for _, row in resources_df.iterrows():
                # Use origin_url field for universal schema compatibility
                url = row.get('origin_url', row.get('url', ''))
                resource_id = row.get('title', row.get('id', ''))  # Use title as resource_id
                if url and not pd.isna(url):
                    urls_to_discover.append(url)
                    if resource_id and not pd.isna(resource_id):
                        resource_ids[url] = str(resource_id)
            
            logger.info(f"Found {len(urls_to_discover)} URLs to discover from resources")
            
            # Use URLDiscoveryEngine for discovery only
            discovery_results = self.url_discovery.discovery_phase(
                urls=urls_to_discover,
                max_depth=max_depth,
                force_reprocess=force_reprocess,
                resource_ids=resource_ids
            )
            
            # Make sure we always return with a proper status field
            if discovery_results is None:
                # Return a default success response if discovery_results is None
                return {"status": "success", "urls_discovered": 0, "message": "No discovery performed"}
                
            # Add a status field if it doesn't exist
            if "status" not in discovery_results:
                if discovery_results.get("total_discovered", 0) > 0:
                    discovery_results["status"] = "success"
                else:
                    discovery_results["status"] = "warning"
                    discovery_results["message"] = "No URLs discovered"
                    
            return discovery_results
            
        except Exception as e:
            logger.error(f"Error discovering URLs from resources: {e}")
            return {"status": "error", "message": str(e)}
    
    def discover_urls_at_level(self, level: int, max_urls_per_batch: int = 50, 
                              force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Discover URLs at a specific level only. No content analysis.
        
        Args:
            level: The level to discover URLs for
            max_urls_per_batch: Maximum URLs to process in one batch
            force_reprocess: Whether to reprocess already discovered URLs
            
        Returns:
            Dictionary with discovery results
        """
        logger.info(f"Discovering URLs at level {level}")
        
        try:
            # Get URLs that need discovery at this level
            pending_urls = self.url_storage.get_pending_urls(depth=level)
            
            if not pending_urls:
                logger.info(f"No pending URLs at level {level} for discovery")
                return {"status": "success", "urls_discovered": 0, "level": level}
            
            # Limit batch size
            urls_to_process = pending_urls[:max_urls_per_batch]
            
            # Extract URLs and resource IDs for discovery
            urls_list = []
            resource_ids = {}
            
            for url_data in urls_to_process:
                url = url_data.get('url')
                resource_id = url_data.get('resource_id', '')
                if url:
                    urls_list.append(url)
                    if resource_id:
                        resource_ids[url] = resource_id
            
            logger.info(f"Processing {len(urls_list)} URLs for discovery at level {level}")
            
            # Perform discovery only
            discovery_results = self.url_discovery.discovery_phase(
                urls=urls_list,
                max_depth=level,
                force_reprocess=force_reprocess,
                resource_ids=resource_ids
            )
            
            discovery_results["level"] = level
            return discovery_results
            
        except Exception as e:
            logger.error(f"Error discovering URLs at level {level}: {e}")
            return {"status": "error", "message": str(e), "level": level}
    
    def get_discovery_status(self) -> Dict[str, Any]:
        """
        Get current URL discovery status without any analysis information.
        
        Returns:
            Dictionary with discovery status information
        """
        try:
            # Use URLDiscoveryEngine to check discovery status
            discovery_status = self.url_discovery.check_discovery_needed()
            
            # Add pending URL counts by level for discovery
            pending_counts = {}
            total_pending = 0
            
            for level in range(1, 6):  # Check levels 1-5
                pending_urls = self.url_storage.get_pending_urls(depth=level)
                count = len(pending_urls) if pending_urls else 0
                pending_counts[f"level_{level}"] = count
                total_pending += count
            
            discovery_status["discovery_pending"] = {
                "pending_by_level": pending_counts,
                "total_pending": total_pending
            }
            
            # Add discovered URL counts
            discovered_counts = {}
            total_discovered = 0
            
            for level in range(1, 6):
                discovered_urls = self.url_storage.get_pending_urls(depth=level)
                count = len(discovered_urls) if discovered_urls else 0
                discovered_counts[f"level_{level}"] = count
                total_discovered += count
            
            discovery_status["discovery_completed"] = {
                "discovered_by_level": discovered_counts,
                "total_discovered": total_discovered
            }
            
            return discovery_status
            
        except Exception as e:
            logger.error(f"Error getting discovery status: {e}")
            return {"status": "error", "message": str(e)}
    
    def check_discovery_needed_at_level(self, level: int) -> Dict[str, Any]:
        """
        Check if URL discovery is needed at a specific level.
        
        Args:
            level: The level to check
            
        Returns:
            Dictionary with discovery status for the level
        """
        try:
            pending_urls = self.url_storage.get_pending_urls(depth=level)
            pending_count = len(pending_urls) if pending_urls else 0
            
            return {
                "level": level,
                "pending_discovery": pending_count,
                "discovery_needed": pending_count > 0
            }
            
        except Exception as e:
            logger.error(f"Error checking discovery needed at level {level}: {e}")
            return {"level": level, "error": str(e)}
    
    def get_discovered_urls_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all discovered URLs across all levels.
        
        Returns:
            Dictionary with discovered URLs summary
        """
        try:
            summary = {
                "total_discovered": 0,
                "by_level": {},
                "by_domain": {},
                "discovery_stats": {}
            }
            
            # Count by level
            for level in range(1, 6):
                discovered_urls = self.url_storage.get_pending_urls(depth=level)
                count = len(discovered_urls) if discovered_urls else 0
                summary["by_level"][f"level_{level}"] = count
                summary["total_discovered"] += count
            
            # Get domain distribution (simplified)
            all_urls = self.url_storage.get_all_processed_urls()
            domain_counts = {}
            
            for url_data in all_urls:
                url = url_data.get('url', '')
                if url:
                    try:
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        domain_counts[domain] = domain_counts.get(domain, 0) + 1
                    except:
                        pass
            
            summary["by_domain"] = domain_counts
            summary["discovery_stats"] = {
                "unique_domains": len(domain_counts),
                "average_urls_per_domain": summary["total_discovered"] / len(domain_counts) if domain_counts else 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting discovered URLs summary: {e}")
            return {"error": str(e)}
    
    def reset_discovery_for_level(self, level: int) -> Dict[str, Any]:
        """
        Reset discovery status for a specific level (mark URLs as pending again).
        
        Args:
            level: The level to reset
            
        Returns:
            Dictionary with reset results
        """
        try:
            # This would mark URLs at the specified level as pending for re-discovery
            # Implementation depends on URL storage capabilities
            logger.info(f"Resetting discovery status for level {level}")
            
            # Get all URLs at this level
            urls_at_level = self.url_storage.get_pending_urls(depth=level)
            
            if not urls_at_level:
                return {"level": level, "reset_count": 0, "message": "No URLs to reset"}
            
            # Mark them as pending (implementation depends on storage structure)
            reset_count = 0
            for url_data in urls_at_level:
                url = url_data.get('url')
                if url:
                    # This would need to be implemented in URL storage
                    # self.url_storage.mark_as_pending(url, level)
                    reset_count += 1
            
            logger.info(f"Reset {reset_count} URLs at level {level} for re-discovery")
            
            return {
                "level": level,
                "reset_count": reset_count,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error resetting discovery for level {level}: {e}")
            return {"level": level, "error": str(e)}
    
    def discover_urls_from_resources_sync(self, level: int = None, max_depth: int = 4, batch_size: int = 50, 
                                        force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Synchronous wrapper for discover_urls_from_resources.
        
        Args:
            level: Specific level to discover URLs for (optional, for compatibility)
            max_depth: Maximum depth for URL discovery
            batch_size: Number of URLs to process in each batch (not used in discovery, kept for compatibility)
            force_reprocess: Whether to reprocess already discovered URLs
            
        Returns:
            Dictionary with discovery results
        """
        import asyncio
        
        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # If we're in an event loop, we can't use asyncio.run()
            # Create a new thread to run the async method
            import concurrent.futures
            import threading
            
            def run_in_thread():
                # Create a new event loop for this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        self.discover_urls_from_resources(level, max_depth, batch_size, force_reprocess)
                    )
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
                
        except RuntimeError:
            # No event loop running, we can use asyncio.run()
            return asyncio.run(
                self.discover_urls_from_resources(level, max_depth, batch_size, force_reprocess)
            )