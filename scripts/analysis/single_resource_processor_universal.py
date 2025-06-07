import logging
from typing import Dict, Any, List
from datetime import datetime

from scripts.analysis.universal_analysis_llm import UniversalAnalysisLLM
from scripts.analysis.content_analyzer import ContentAnalyzer
from scripts.analysis.data_saver import DataSaver
from scripts.analysis.interruptible_llm import is_interrupt_requested, clear_interrupt

logger = logging.getLogger(__name__)

class SingleResourceProcessorUniversal:
    """
    Processes a single resource using a universal LLM for all content extraction.
    
    This implementation is ANALYSIS-ONLY and does not handle URL discovery.
    URL discovery is handled separately by the URLDiscoveryManager in the Knowledge Orchestrator.
    
    This class provides two analysis methods:
    1. process_resource() - Analyzes the main URL content for a resource
    2. process_specific_url() - Analyzes individual URLs that were previously discovered
    
    Both methods use the universal LLM to extract structured information from content
    without performing any URL discovery or crawling operations.
    """
    
    def __init__(self, data_folder: str = None):
      from utils.config import get_data_path
      self.data_folder = data_folder or get_data_path()
      # Explicitly load settings from analysis-model-settings.json
      import os
      import json
      settings_path = os.path.join(os.path.dirname(__file__), "..", "settings", "analysis-model-settings.json")
      
      # Check if the settings file exists
      if not os.path.exists(settings_path):
          logger.warning(f"Analysis model settings file not found at: {settings_path}")
          # Fallback to default initialization with data folder
          self.universal_llm = UniversalAnalysisLLM(data_folder=self.data_folder)
      else:
          # Load settings and explicitly pass the model
          with open(settings_path, 'r') as f:
              analysis_settings = json.load(f)
          
          # Get the correct model based on provider
          if analysis_settings["provider"] == "openrouter":
              model = analysis_settings["openrouter_model"]
              logger.info(f"Using analysis OpenRouter model: {model}")
          else:
              model = analysis_settings["model_name"]
              logger.info(f"Using analysis Ollama model: {model}")
          
          # Initialize with explicit model and data folder
          self.universal_llm = UniversalAnalysisLLM(model=model, data_folder=self.data_folder)
      
      self.content_analyzer = ContentAnalyzer()
      self.data_saver = DataSaver(data_folder)
      
      # Initialize URL storage for the content analyzer if it doesn't have one
      if self.content_analyzer.url_storage is None:
          from scripts.analysis.url_storage import URLStorageManager
          import os
          processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
          self.content_analyzer.url_storage = URLStorageManager(processed_urls_file)
      
      self.url_storage = self.content_analyzer.url_storage

      # Set up temporary storage for content and vector store
      from scripts.storage.temporary_storage_service import TemporaryStorageService
      from scripts.storage.content_storage_service import ContentStorageService
      self.temp_storage = TemporaryStorageService(self.data_folder)
      self.content_storage = ContentStorageService(self.data_folder)
      self.vector_generation_needed = False

    def process_resource(
        self,
        resource: Dict,
        resource_id: str,
        idx: int = None,
        csv_path: str = None,
        on_subpage_processed = None,
        process_by_level: bool = True,
        max_depth: int = 4
    ) -> Dict[str, Any]:
        """
        Analyze the main URL content for a resource using the universal LLM.
        
        This method is ANALYSIS-ONLY and does not perform URL discovery.
        It fetches content from the main resource URL and extracts structured information
        using the universal LLM for all configured content types.
        
        Args:
            resource: Resource dictionary containing metadata and main URL
            resource_id: String identifier for the resource
            idx: Optional index in the resources CSV file (unused)
            csv_path: Optional path to resources CSV file (unused)
            on_subpage_processed: Optional callback function (unused in universal processor)
            process_by_level: Optional flag for level-based processing (unused in universal processor)
            max_depth: Optional maximum crawl depth (unused in universal processor)
            
        Returns:
            Dictionary with analysis results
        """
        start_time = datetime.now()
        result = {"resource_id": resource_id, "universal_results": {}, "errors": []}

        try:
            # Use origin_url for universal schema compatibility, fallback to url if not available
            main_url = resource.get("origin_url", resource.get("url"))
            resource_id = resource.get("title", "")
            logger.info(f"[{resource_id}] Starting universal analysis for {main_url}")

            # Fetch main page content without discovering additional links
            main_page_result = self.content_analyzer.fetch_content_for_analysis(main_url, resource_id=resource_id)
            if not main_page_result[0]:  # Check success flag
                logger.warning(f"[{resource_id}] No content fetched for {main_url}")
                result["errors"].append("No content fetched")
                return result
            
            main_page_text = main_page_result[1].get('main', '') if main_page_result[1] else ''
            
            if not main_page_text:
                logger.warning(f"[{resource_id}] Empty content fetched for {main_url}")
                result["errors"].append("Empty content fetched")
                return result

            # Universal extraction for all configured content types
            for content_type in self.universal_llm.task_config.keys():
                logger.info(f"[{resource_id}] Universal analysis for content type: {content_type}")
                items = self.perform_universal_analysis(
                    title=resource_id,
                    url=main_url,
                    content=main_page_text,
                    content_type=content_type
                )
                
                # If we have results, ensure required fields and save them
                if items:
                    logger.info(f"[{resource_id}] Universal analysis for URL {main_url} ({content_type}): extracted {len(items)} items")
                    
                    # Ensure all required fields are present ONLY for items we're actually going to save
                    items = self._ensure_required_fields(items, content_type)
                    
                    # Save items to CSV here - this is where we actually persist the data
                    self.universal_llm.save_to_universal_csv(items)
                else:
                    logger.info(f"[{resource_id}] Universal analysis for URL {main_url} ({content_type}): no items extracted")
                
                result["universal_results"][content_type] = items

            # Mark main URL as processed
            self.url_storage.save_processed_url(main_url, resource_id=resource_id)
            logger.info(f"[{resource_id}] Marked {main_url} as processed.")

        except Exception as e:
            logger.error(f"[{resource_id}] Error in universal processing: {e}")
            result["errors"].append(str(e))

        return result
    
    def process_specific_url(self, url: str, origin_url: str, resource: Dict, depth: int) -> Dict[str, Any]:
        """
        Analyze content from a specific URL using the universal LLM.
        
        This method is used by the Knowledge Orchestrator to analyze content from URLs
        that were previously discovered and queued for processing at specific depth levels.
        
        This method is ANALYSIS-ONLY and does not perform URL discovery.
        
        Args:
            url: The specific URL to analyze (a previously discovered URL)
            origin_url: The original resource URL that this URL was found from (preferred for universal schema)
            resource: The resource dictionary containing metadata
            depth: The depth level of this URL
            
        Returns:
            Dictionary with analysis results and processing status
        """
        start_time = datetime.now()
        
        # Reset interruption state
        clear_interrupt()
        
        # Get the resource title and create a logging ID
        resource_id = resource.get('title', 'Unnamed') if resource else 'Unnamed'
        
        # Check if we have an actual resource ID from pending URLs
        pending_resource_id = ""
        pending_url_data = None
        if hasattr(self.url_storage, "_pending_urls_cache") and url in self.url_storage._pending_urls_cache:
            pending_url_data = self.url_storage._pending_urls_cache[url]
            pending_resource_id = pending_url_data.get("resource_id", "")
            
        # If we don't have a pending resource ID, try to get it from resource_urls mapping
        if not pending_resource_id:
            pending_resource_id = self.url_storage.get_resource_id_for_url(url)
            
        # Determine the final resource ID for the result
        final_resource_id = pending_resource_id if pending_resource_id else resource_id
            
        # If we have a pending resource ID, use it to update the resource title
        if pending_resource_id:
            resource_id = pending_resource_id
            # Update resource dict too if it exists
            if resource:
                resource['title'] = pending_resource_id
                logger.info(f"Updated resource title to '{pending_resource_id}' from resource_id mapping")
            else:
                logger.info(f"Using resource_id '{pending_resource_id}' from pending URLs (resource dict is None)")
            
        # Create a composite ID for logging purposes only
        logging_resource_id = f"{resource_id}::{url}"
        
        # Debug logging - improved format to avoid confusion
        logger.info(f"[Resource: {logging_resource_id}] Processing URL: {url} (origin: {origin_url}) at level {depth}")
        if pending_resource_id:
            logger.info(f"[Resource: {logging_resource_id}] URL {url} is associated with resource: {pending_resource_id}")
            
        url_is_processed = self.url_storage.url_is_processed(url)
        logger.info(f"[Resource: {logging_resource_id}] URL {url} already in processed list: {url_is_processed}")
        
        # Skip if already processed
        if url_is_processed:
            logger.info(f"[Resource: {logging_resource_id}] URL {url} already processed, skipping")
            # Ensure it's removed from pending
            self.url_storage.remove_pending_url(url)
            return {
                "success": True, 
                "url": url,
                "origin_url": origin_url,
                "depth": depth,
                "resource_id": final_resource_id,
                "status": "skipped_processed",
                "message": "URL already processed"
            }
        
        # Initialize result structure
        result = {
            "success": False,
            "url": url,
            "origin_url": origin_url,
            "depth": depth,
            "resource_id": final_resource_id,
            "universal_results": {},
            "error": None
        }
        
        try:
            # Get current attempt count for this URL before processing
            attempt_count = 0
            try:
                # Get attempt count from pending URLs cache
                if hasattr(self.url_storage, "_pending_urls_cache") and url in self.url_storage._pending_urls_cache:
                    attempt_count = self.url_storage._pending_urls_cache[url].get("attempt_count", 0)
                    logger.info(f"[Resource: {logging_resource_id}] Found attempt count in cache for URL {url}: {attempt_count}")
                else:
                    logger.info(f"[Resource: {logging_resource_id}] URL {url} not in pending cache, using attempt count of 0")
            except Exception as e:
                logger.error(f"[Resource: {logging_resource_id}] Error getting attempt count for URL {url}: {e}")
                attempt_count = 0

            # Fetch content without discovering additional URLs
            logger.info(f"[Resource: {logging_resource_id}] Fetching content from URL: {url}")
            content_result = self.content_analyzer.fetch_content_for_analysis(url, resource_id=final_resource_id)
            
            # Handle different return types from fetch_content
            if isinstance(content_result, tuple) and len(content_result) == 2:
                success, page_data = content_result
                if not success:
                    logger.warning(f"[Resource: {logging_resource_id}] Content fetch failed for URL: {url}")
                    result["error"] = page_data.get("error", "Failed to fetch page content")
                    
                    # Handle fetch failure with retry logic
                    if attempt_count >= 2:  # After 3 attempts, mark as processed with error
                        logger.warning(f"[Resource: {logging_resource_id}] URL {url} failed to fetch content after {attempt_count+1} attempts, marking as processed with error")
                        self.url_storage.save_processed_url(url, error=True, resource_id=final_resource_id)
                        self.url_storage.remove_pending_url(url)
                        result["success"] = False
                    else:
                        # Increment attempt count and keep in pending queue for retry
                        logger.info(f"[Resource: {logging_resource_id}] Content fetch failed for URL {url} (attempt {attempt_count+1}), incrementing attempt count")
                        self.url_storage.increment_url_attempt(url)
                        result["success"] = False
                    
                    return result
                page_text = page_data.get("main", "")
            elif isinstance(content_result, dict):
                page_text = content_result.get("main", "")
            else:
                page_text = content_result
            
            if not page_text:
                logger.warning(f"[Resource: {logging_resource_id}] No content fetched for URL: {url}")
                result["error"] = "Failed to fetch page content"
                
                # Handle empty content with retry logic
                if attempt_count >= 2:  # After 3 attempts, mark as processed with error
                    logger.warning(f"[Resource: {logging_resource_id}] URL {url} returned empty content after {attempt_count+1} attempts, marking as processed with error")
                    self.url_storage.save_processed_url(url, error=True, resource_id=final_resource_id)
                    self.url_storage.remove_pending_url(url)
                    result["success"] = False
                else:
                    # Increment attempt count and keep in pending queue for retry
                    logger.info(f"[Resource: {logging_resource_id}] Empty content for URL {url} (attempt {attempt_count+1}), incrementing attempt count")
                    self.url_storage.increment_url_attempt(url)
                    result["success"] = False
                
                return result
            
            # Page key for logging
            page_key = url.rsplit('/', 1)[-1] or f'level{depth}_page'
            combined_title = f"{logging_resource_id} - {page_key}"
            
            # Track successful analysis
            success_count = 0
            
            # Universal extraction for all configured content types
            for content_type in self.universal_llm.task_config.keys():
                logger.info(f"[Resource: {logging_resource_id}] Universal analysis for URL {url}, content type: {content_type}")
                
                try:
                    # Perform analysis with our method
                    items = self.perform_universal_analysis(
                        title=combined_title,
                        url=url,
                        content=page_text,
                        content_type=content_type,
                        origin_url=origin_url
                    )
                    
                    # Check if items is a list and not empty
                    if not isinstance(items, list):
                        items = []
                        logger.warning(f"[Resource: {logging_resource_id}] Universal analysis returned non-list result for URL {url}, content type: {content_type}")
                    
                    # Count as success if analysis completed without error, even if no items were extracted
                    # This prevents repeated attempts when the LLM doesn't find anything to extract
                    success_count += 1
                    
                    # If we have results, ensure required fields and save them
                    if items:
                        logger.info(f"[Resource: {logging_resource_id}] Universal analysis for URL {url} ({content_type}): extracted {len(items)} items")
                        
                        # Ensure all required fields are present ONLY for items we're actually going to save
                        items = self._ensure_required_fields(items, content_type)
                        
                        logger.info(f"[Resource: {logging_resource_id}] Saving {len(items)} {content_type} items to CSV for URL {url}")
                        # Save to CSV here - this is the ONLY place where these items are saved
                        self.universal_llm.save_to_universal_csv(items, content_type)
                    else:
                        logger.info(f"[Resource: {logging_resource_id}] Universal analysis for URL {url} ({content_type}): no items extracted")
                    
                    result["universal_results"][content_type] = items
                except Exception as e:
                    logger.error(f"[Resource: {logging_resource_id}] Error in {content_type} analysis for URL {url}: {str(e)}")
                    # Continue with other content types
            
            # Decide whether to mark as processed
            if success_count > 0:
                # We have data, mark as processed
                logger.info(f"[Resource: {logging_resource_id}] Marking URL {url} as processed after successful extraction")
                self.url_storage.save_processed_url(url, error=False, resource_id=final_resource_id)
                self.url_storage.remove_pending_url(url)
                result["success"] = True
            elif attempt_count >= 2:  # After 3 attempts, force mark as processed
                logger.warning(f"[Resource: {logging_resource_id}] URL {url} failed to yield data after {attempt_count+1} attempts, marking as processed with error")
                self.url_storage.save_processed_url(url, error=True, resource_id=final_resource_id)
                self.url_storage.remove_pending_url(url)
                result["success"] = False
            else:
                # Increment attempt count and keep in pending queue
                logger.info(f"[Resource: {logging_resource_id}] No data extracted for URL {url} (attempt {attempt_count+1}), not marking as processed yet")
                self.url_storage.increment_url_attempt(url)
                result["success"] = False
            
            # Add processing time
            result["processing_time_seconds"] = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[Resource: {logging_resource_id}] Error processing URL {url}: {error_msg}")
            result["error"] = error_msg
            
            # Increment attempt on error
            self.url_storage.increment_url_attempt(url)
        
        return result

    def perform_universal_analysis(self, title: str, url: str, content: str, content_type: str, origin_url: str = None) -> List[Dict[str, Any]]:
        """
        Extract structured information from content using the universal LLM.
        
        This method performs content analysis only - it does not save results to CSV.
        The calling method is responsible for saving the extracted items.
        
        Args:
            title: Title of the content being analyzed
            url: URL of the content being analyzed
            content: The text content to analyze
            content_type: The type of content to extract (e.g., 'definitions', 'methods')
            origin_url: Optional original URL if this is a subpage
            
        Returns:
            List of extracted items of the specified content type
        """
        try:
            # Use direct analysis since content is already pre-limited by HTML extractor (2000-4000 chars)
            # This avoids unnecessary chunking overhead that creates many tiny chunks
            items = self.universal_llm.analyze_content(
                title=title,
                url=url,
                content=content,
                content_type=content_type
            )
            
            # Ensure all items have origin_url field
            if items:
                for item in items:
                    # Use origin_url if provided, otherwise use the current URL
                    # This ensures all items have origin_url set properly for universal schema
                    if origin_url:
                        item['origin_url'] = origin_url
                    else:
                        item['origin_url'] = url
                    
                    # Ensure both fields are present for backward compatibility
                    if 'url' not in item:
                        item['url'] = item['origin_url']
            
            # Mark that vector generation will be needed if we found items
            if items:
                self.vector_generation_needed = True
            
            return items
        except Exception as e:
            logger.error(f"Error performing universal analysis for {content_type}: {e}")
            return []
            
    def _ensure_required_fields(self, items: List[Dict], content_type: str) -> List[Dict]:
        """
        Helper method to ensure all items have the required fields for universal table structure.
        
        Args:
            items: List of extracted items
            content_type: Type of content being processed
            
        Returns:
            List of items with all required fields added if missing
        """
        if not items:
            return []
            
        required_fields = ["title", "description", "purpose", "tags", "location", "origin_url"]
        enriched_items = []
        
        for item in items:
            if not isinstance(item, dict):
                continue
                
            # Add missing fields with default values
            if "title" not in item or not item["title"]:
                item["title"] = f"Untitled {content_type}"
                
            if "description" not in item or not item["description"]:
                item["description"] = f"Auto-generated {content_type} from content analysis"
                
            if "purpose" not in item or not item["purpose"]:
                item["purpose"] = f"To document {content_type} information"
                
            if "tags" not in item or not item["tags"]:
                item["tags"] = [content_type, "auto-tagged"]
            elif not isinstance(item["tags"], list):
                item["tags"] = [str(item["tags"]).lower()]
                
            # Match schema - location is Optional[str] with default empty string
            if "location" not in item:
                item["location"] = ""
                
            # Ensure backward compatibility with both url and origin_url fields
            if "origin_url" not in item and "url" in item:
                item["origin_url"] = item["url"]
            elif "url" not in item and "origin_url" in item:
                item["url"] = item["origin_url"]
                
            enriched_items.append(item)
            
        return enriched_items