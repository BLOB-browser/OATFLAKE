#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import from scripts
sys.path.append(str(Path(__file__).parent.parent))

def add_discover_urls_method():
    """
    Add the discover_all_urls_from_resource method to SingleResourceProcessor
    """
    try:
        # File path for SingleResourceProcessor
        file_path = Path(__file__).parent.parent / "scripts" / "analysis" / "single_resource_processor.py"
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return
            
        # Read the current file content
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Check if the method already exists
        if "def discover_all_urls_from_resource" in content:
            logger.info("Method already exists, no changes needed.")
            return
            
        # Find where to insert the new method
        # Look for the last method in the class
        last_method_pos = content.rfind("    def ")
        if last_method_pos == -1:
            logger.error("Could not find a method in the class")
            return
            
        # Find the end of the last method
        next_method_pos = content.find("    def ", last_method_pos + 1)
        if next_method_pos == -1:
            # This is the last method in the file
            insertion_point = len(content)
        else:
            # Insert before the next method
            insertion_point = next_method_pos
            
        # New method to add
        new_method = """
    def discover_all_urls_from_resource(self, resource: Dict[str, Any], resource_id: str, max_depth: int = 4) -> Dict[str, Any]:
        """
        Discover all URLs from a resource without analyzing them.
        This is used to populate the pending_urls queue with all URLs before starting analysis.
        
        Args:
            resource: The resource dictionary
            resource_id: A string identifier for the resource
            max_depth: Maximum URL discovery depth
            
        Returns:
            Dictionary with discovery results
        """
        logger.info(f"[{resource_id}] Discovering URLs from resource (max_depth={max_depth})")
        
        result = {
            "success": False,
            "total_urls": 0,
            "urls_by_level": {}
        }
        
        try:
            resource_url = resource.get('url', '')
            resource_title = resource.get('title', 'Unnamed')
            
            if not resource_url:
                logger.warning(f"[{resource_id}] Resource has no URL")
                return result
                
            # Use force_reprocess=True to ensure we rediscover all URLs even if we've seen the page before
            # But we'll still filter out URLs that are already processed when saving to pending
            success, page_data = self.content_fetcher.fetch_content(
                resource_url, 
                max_depth=max_depth,
                process_by_level=True,
                force_reprocess=True,  # Force reprocessing to ensure we discover all URLs
                discover_only=True     # Only discover URLs, don't process content
            )
            
            if not success:
                error_msg = page_data.get("error", "Unknown error")
                logger.error(f"[{resource_id}] Failed to fetch content for URL discovery: {error_msg}")
                result["error"] = f"Failed to fetch content: {error_msg}"
                return result
                
            # Extract URL counts by level
            urls_by_level = page_data.get("urls_by_level", {})
            total_urls = 0
            
            for level, urls in urls_by_level.items():
                level_count = len(urls)
                result["urls_by_level"][level] = level_count
                total_urls += level_count
                logger.info(f"[{resource_id}] Discovered {level_count} URLs at level {level}")
                
            result["total_urls"] = total_urls
            result["success"] = True
            
            logger.info(f"[{resource_id}] Total URLs discovered: {total_urls}")
            
            return result
            
        except Exception as e:
            logger.error(f"[{resource_id}] Error during URL discovery: {e}", exc_info=True)
            result["error"] = str(e)
            return result
"""
        
        # Insert the new method
        updated_content = content[:insertion_point] + new_method + content[insertion_point:]
        
        # Write the updated content back to the file
        with open(file_path, 'w') as f:
            f.write(updated_content)
            
        logger.info(f"Added discover_all_urls_from_resource method to {file_path}")
        
    except Exception as e:
        logger.error(f"Error adding method: {e}", exc_info=True)
        
if __name__ == "__main__":
    add_discover_urls_method()
    print("Done! Now modify the ContentFetcher class to add the discover_only parameter to fetch_content")
