#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
import os
import csv
from typing import Dict, List, Any, Callable, Optional, Tuple
import tempfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Import analyzer modules
from scripts.analysis.llm_analyzer import LLMAnalyzer
from scripts.analysis.resource_llm import ResourceLLM
from scripts.analysis.extraction_utils import ExtractionUtils
from scripts.analysis.content_fetcher import ContentFetcher
from scripts.analysis.method_llm import MethodLLM
from scripts.analysis.goal_llm import GoalLLM
from scripts.analysis.data_saver import DataSaver
from scripts.analysis.interruptible_llm import is_interrupt_requested, clear_interrupt
# Import HTML extraction functions
from scripts.analysis.html_extractor import extract_text, extract_page_texts

class SingleResourceProcessor:
    """Processes a single resource to extract structured data"""
    
    def __init__(self, data_folder: str = None):
        """Initialize SingleResourceProcessor
        
        Args:
            data_folder: Path to the data directory
        """
        logger.info(f"SingleResourceProcessor initialized with data folder: {data_folder}")
        from utils.config import get_data_path
        self.data_folder = data_folder or get_data_path()
        
        # Initialize content fetcher for URL processing
        self.content_fetcher = ContentFetcher()
        
        # Set up temporary storage for content
        from scripts.storage.temporary_storage_service import TemporaryStorageService
        from scripts.storage.content_storage_service import ContentStorageService
        self.temp_storage = TemporaryStorageService(self.data_folder)
        self.content_storage = ContentStorageService(self.data_folder)
        
        # Track if vector generation is needed
        self.vector_generation_needed = False
        
        # Initialize analyzer components
        self.extraction_utils = ExtractionUtils()
        self.method_llm = MethodLLM()
        self.data_saver = DataSaver(self.data_folder)
    
    def process_resource(self, resource: Dict, resource_id: str, idx: int = None, csv_path: str = None,
                      on_subpage_processed: Callable = None, process_by_level: bool = True, 
                      max_depth: int = 4, current_level: int = None, discovery_only: bool = False) -> Dict[str, Any]:
        """
        Process a single resource through content fetching, analysis and storage.
        
        Args:
            resource: Dictionary with resource data
            resource_id: String identifier for logging
            idx: Index in the CSV file
            csv_path: Path to the CSV file for saving updates
            on_subpage_processed: Optional callback function to call after each subpage is processed
            process_by_level: If True, processes URLs strictly level by level (all level 1 URLs before level 2)
            max_depth: Maximum crawl depth (1=just main page links, 2=two levels, 3=three levels, etc.)
            current_level: If specified, only process URLs at this specific level (for level-based processing)
            discovery_only: If True, only discover URLs without analyzing content
            
        Returns:
            Dictionary with processing results
        """
        start_time = datetime.now()
        
        # Reset interruption state at start of processing each resource
        clear_interrupt()
        
        # Initialize result structure
        result = {
            "success": False,
            "resource": resource,
            "definitions": [],
            "projects": [],
            "methods": [],
            "error": None
        }
        
        try:
            resource_title = resource.get('title', 'Unnamed')
            resource_url = resource.get('url', '')
            
            # Validate URL before proceeding
            if not resource_url or not resource_url.startswith(('http://', 'https://')):
                logger.error(f"[{resource_id}] Invalid URL format: {resource_url}")
                result["error"] = f"Invalid URL format: {resource_url}"
                return result
            
            # STEP 1: Fetch main page content
            logger.info(f"[{resource_id}] {'Discovering URLs from' if discovery_only else 'Fetching content from'} {resource_url}")
            
            # Create temp file for storing content
            resource_slug = resource_title.replace(" ", "_").lower()[:30]
            temp_path = self.temp_storage.create_temp_file(prefix=resource_slug)
            
            # If only processing a specific level, skip the initial fetch if it's not level 0
            if current_level is not None and current_level > 0:
                logger.info(f"[{resource_id}] Skipping main page fetch, only processing level {current_level}")
                
                # Mark the resource as partially processed in the CSV
                if idx is not None and csv_path is not None:
                    # Update the CSV field to indicate partial processing
                    from scripts.services.storage import DataSaver as ServiceDataSaver
                    service_saver = ServiceDataSaver()
                    resource_copy = resource.copy()
                    resource_copy['partial_processing'] = True
                    resource_copy['level_processed'] = current_level
                    # The method is called 'save_resource' in the services DataSaver, not 'save_single_resource'
                    service_saver.save_resource(resource_copy, csv_path, idx)
                
                # Return early with empty but successful result
                result["success"] = True
                return result
            
            # Fetch content with the specified depth level
            # Always discover URLs, even from processed resources
            success, page_data = self.content_fetcher.fetch_content(
                resource_url, 
                max_depth=max_depth,
                process_by_level=process_by_level
            )
            
            if not success:
                error_msg = page_data.get("error", "Unknown error")
                logger.error(f"[{resource_id}] Failed to fetch content: {error_msg}")
                result["error"] = f"Failed to fetch content: {error_msg}"
                return result
            
            # Extract main page HTML and get the list of additional URLs
            main_html = page_data.get("main", "")
            
            if not main_html:
                logger.error(f"[{resource_id}] Failed to extract main HTML content")
                result["error"] = "Failed to extract main HTML content"
                # Don't mark URL as processed since extraction failed
                return result
            
            headers = page_data.get("headers", {})
            visited_urls = page_data.get("visited_urls", set())
            
            # Get additional URLs by level
            urls_by_level = page_data.get("urls_by_level", {})
            level1_urls = urls_by_level.get(1, [])
            additional_urls = page_data.get("additional_urls", [])
            
            # Log discoveries by level
            for level, urls in urls_by_level.items():
                logger.info(f"[{resource_id}] Discovered {len(urls)} URLs at level {level}")
            
            # If discovery_only mode, return early after logging URL counts
            if discovery_only:
                logger.info(f"[{resource_id}] URL discovery completed: found {sum(len(urls) for urls in urls_by_level.values())} URLs across all levels")
                # Don't mark main URL as processed - we've only discovered URLs
                result["success"] = True
                result["discovered_urls_count"] = sum(len(urls) for urls in urls_by_level.values())
                result["urls_by_level"] = {level: len(urls) for level, urls in urls_by_level.items()}
                return result
            
            # Extract text content from main HTML
            main_page_text = extract_text(main_html)
            
            # Initialize tracking structures
            page_results = {}
            
            # Extract LLM results from main page text
            try:
                # STEP 2: Extract definition and content classification
                try:
                    main_definitions = self.extraction_utils.extract_definitions(
                        title=resource_title, 
                        url=resource_url, 
                        content=main_page_text
                    )
                    
                    # Log definition extraction results
                    definitions_found = len(main_definitions)
                    logger.info(f"[{resource_id}] Extracted {definitions_found} definition(s) from main page")
                    
                    # Save definitions immediately after extraction
                    if main_definitions:
                        logger.info(f"[{resource_id}] SAVING DATA: Saving {len(main_definitions)} definitions to CSV immediately")
                        self.data_saver.save_definitions(main_definitions)
                        self.vector_generation_needed = True
                except Exception as e:
                    logger.error(f"[{resource_id}] Error extracting or saving definitions: {e}")
                    main_definitions = []
                
                # STEP 3: Extract project information
                try:
                    main_projects = self.extraction_utils.identify_projects(
                        title=resource_title, 
                        url=resource_url, 
                        content=main_page_text
                    )
                    
                    # Log project extraction results
                    projects_found = len(main_projects)
                    logger.info(f"[{resource_id}] Identified {projects_found} project(s) from main page")
                    
                    # Save projects immediately after extraction
                    if main_projects:
                        logger.info(f"[{resource_id}] SAVING DATA: Saving {len(main_projects)} projects to CSV immediately")
                        for idx, proj in enumerate(main_projects):
                            logger.info(f"[{resource_id}] SAVING PROJECT #{idx+1}: {proj.get('title', 'Untitled')} with tags: {proj.get('tags', [])}")
                        self.data_saver.save_projects(main_projects)
                        self.vector_generation_needed = True
                except Exception as e:
                    logger.error(f"[{resource_id}] Error extracting or saving projects: {e}")
                    main_projects = []
                
                # STEP 4: Extract methods
                try:
                    main_methods = self.method_llm.extract_methods(
                        title=resource_title, 
                        url=resource_url, 
                        content=main_page_text
                    )
                    
                    # Log method extraction results
                    methods_found = len(main_methods)
                    logger.info(f"[{resource_id}] Extracted {methods_found} method(s) from main page")
                    
                    # Save methods immediately after extraction
                    if main_methods:
                        logger.info(f"[{resource_id}] SAVING DATA: Saving {len(main_methods)} methods to CSV immediately")
                        for idx, method in enumerate(main_methods):
                            logger.info(f"[{resource_id}] SAVING METHOD #{idx+1}: {method.get('title', 'Untitled')} with {len(method.get('steps', []))} steps")
                        self.data_saver.save_methods(main_methods)
                        self.vector_generation_needed = True
                except Exception as e:
                    logger.error(f"[{resource_id}] Error extracting or saving methods: {e}")
                    main_methods = []
                
                # Check for cancellation before finalizing
                if is_interrupt_requested():
                    logger.info(f"[{resource_id}] Resource processing cancelled during extraction")
                    return result
            
            except KeyboardInterrupt:
                logger.warning(f"[{resource_id}] Processing interrupted by keyboard interrupt")
                return result
            
            # Store results and add source information
            page_results['main'] = {
                "definitions": main_definitions,
                "projects": main_projects,
                "methods": main_methods,
                "url": resource_url,
                "content": main_page_text[:500] + "..." if len(main_page_text) > 500 else main_page_text
            }
            
            # Update overall result with main page extraction
            result["definitions"].extend(main_definitions)
            result["projects"].extend(main_projects)
            result["methods"].extend(main_methods)
            
            # Cache the content to the temporary file for vector building later
            content_obj = {
                "source": resource_url,
                "title": resource_title,
                "content": main_page_text,
                "extracted_at": datetime.now().isoformat(),
                "resource_id": resource_id,
                "page_type": "main"
            }
            self.temp_storage.append_to_file(temp_path, json.dumps(content_obj) + "\n")
            
            # Call the callback if provided, e.g., for incremental vector generation
            if on_subpage_processed:
                on_subpage_processed()
            
            # Skip additional pages if only processing level 0 or if we're in discovery_only mode
            if (current_level is not None and current_level == 0) or discovery_only:
                logger.info(f"[{resource_id}] {'Discovery only mode, skipping content analysis' if discovery_only else 'Only processing level 0, skipping additional pages'}")
                result["success"] = True
                return result
            
            # Process additional pages if any
            if additional_urls:
                logger.info(f"[{resource_id}] Processing {len(additional_urls)} additional pages")
                
                # If we're only processing a specific level, filter URLs to that level
                if current_level is not None:
                    # Get URLs from the specific level
                    level_urls = urls_by_level.get(current_level, [])
                    logger.info(f"[{resource_id}] Filtering to process only {len(level_urls)} URLs at level {current_level}")
                    
                    # Only process URLs at the specified level
                    additional_urls = level_urls
                
                if not additional_urls:
                    logger.info(f"[{resource_id}] No additional URLs to process at this level")
                    result["success"] = True
                    return result
                
                # Collections for storing page content
                additional_pages_html = {}
                
                # Fetch each additional page
                for idx, additional_url in enumerate(additional_urls):
                    # Check for interruption
                    if is_interrupt_requested():
                        logger.info(f"[{resource_id}] Processing interrupted during additional page fetching")
                        break
                        
                    try:
                        # Check if the additional URL has already been processed
                        if self.content_fetcher.url_storage.url_is_processed(additional_url):
                            logger.info(f"[{resource_id}] Subpage URL {additional_url} already processed, skipping")
                            continue
                            
                        logger.info(f"[{resource_id}] Fetching page {idx+1}/{len(additional_urls)}: {additional_url}")
                        # Log the exact URL being fetched
                        logger.info(f"[{resource_id}] Fetching subpage using fetch_additional_page method: {additional_url}")
                        success, html_content = self.content_fetcher.fetch_additional_page(additional_url, headers)
                        
                        if not success or not html_content:
                            logger.warning(f"[{resource_id}] Failed to fetch {additional_url}, skipping")
                            continue
                        
                        # Use page name as key (simplified URL)
                        page_name = additional_url.rsplit('/', 1)[-1] or f'page{idx+1}'
                        additional_pages_html[page_name] = html_content
                        
                    except Exception as e:
                        logger.error(f"[{resource_id}] Error fetching page {additional_url}: {e}")
                        continue
                
                # Process all pages at once using extract_page_texts
                if additional_pages_html:
                    logger.info(f"[{resource_id}] Extracting text from {len(additional_pages_html)} pages")
                    extracted_pages = extract_page_texts(additional_pages_html, 2000)
                    
                    # Process each extracted page
                    for page_key, page_text in extracted_pages.items():
                        # Check for interruption
                        if is_interrupt_requested():
                            logger.info(f"[{resource_id}] Processing interrupted during page analysis")
                            break
                            
                        try:
                            # Get the URL for this page
                            page_url = next((url for url in additional_urls if page_key in url), page_key)
                            
                            # STEP 2: Extract definitions
                            page_definitions = self.extraction_utils.extract_definitions(
                                title=f"{resource_title} - {page_key}", 
                                url=page_url, 
                                content=page_text
                            )
                            
                            # STEP 3: Extract project information
                            page_projects = self.extraction_utils.identify_projects(
                                title=f"{resource_title} - {page_key}", 
                                url=page_url, 
                                content=page_text
                            )
                            
                            # STEP 4: Extract methods
                            page_methods = self.method_llm.extract_methods(
                                title=f"{resource_title} - {page_key}", 
                                url=page_url, 
                                content=page_text
                            )
                            
                            # Log extraction results
                            logger.info(f"[{resource_id}] Extracted {len(page_definitions)} definition(s), {len(page_projects)} project(s), {len(page_methods)} method(s) from {page_key}")
                            
                            # Update overall results
                            result["definitions"].extend(page_definitions)
                            result["projects"].extend(page_projects)
                            result["methods"].extend(page_methods)
                            
                            # Store page results
                            page_results[page_key] = {
                                "definitions": page_definitions,
                                "projects": page_projects,
                                "methods": page_methods,
                                "url": page_url,
                                "content": page_text[:500] + "..." if len(page_text) > 500 else page_text
                            }
                            
                            # Save to storage if we have results
                            if page_definitions or page_projects or page_methods:
                                self.vector_generation_needed = True
                                
                                # Save to CSV using DataSaver with detailed logging
                                logger.info(f"[{resource_id}] SAVING DATA FROM {page_key}: Saving {len(page_definitions)} definitions to CSV")
                                self.data_saver.save_definitions(page_definitions)
                                
                                logger.info(f"[{resource_id}] SAVING DATA FROM {page_key}: Saving {len(page_projects)} projects to CSV")
                                if page_projects:
                                    for idx, proj in enumerate(page_projects):
                                        logger.info(f"[{resource_id}] SAVING PROJECT #{idx+1} FROM {page_key}: {proj.get('title', 'Untitled')} with tags: {proj.get('tags', [])}")
                                self.data_saver.save_projects(page_projects)
                                
                                logger.info(f"[{resource_id}] SAVING DATA FROM {page_key}: Saving {len(page_methods)} methods to CSV")
                                if page_methods:
                                    for idx, method in enumerate(page_methods):
                                        logger.info(f"[{resource_id}] SAVING METHOD #{idx+1} FROM {page_key}: {method.get('title', 'Untitled')} with {len(method.get('steps', []))} steps")
                                self.data_saver.save_methods(page_methods)
                                
                                # Cache the content to the temporary file for vector building later
                                content_obj = {
                                    "source": page_url,
                                    "title": f"{resource_title} - {page_key}",
                                    "content": page_text,
                                    "extracted_at": datetime.now().isoformat(),
                                    "resource_id": resource_id,
                                    "page_type": "subpage"
                                }
                                self.temp_storage.append_to_file(temp_path, json.dumps(content_obj) + "\n")
                                
                                # Only mark URL as processed if we actually extracted meaningful data
                                # This ensures URLs that didn't yield useful content will be retried
                                if page_definitions or page_projects or page_methods:
                                    # Determine depth level based on which level this URL was from
                                    subpage_depth = 1  # Default to level 1
                                    for level, urls in urls_by_level.items():
                                        if page_url in urls:
                                            subpage_depth = int(level)
                                            break
                                            
                                    logger.info(f"[{resource_id}] Marking subpage URL as processed at depth {subpage_depth}: {page_url}")
                                    self.content_fetcher.mark_url_as_processed(page_url, depth=subpage_depth, origin=resource_url)
                                else:
                                    logger.info(f"[{resource_id}] No data extracted from {page_url}, not marking as processed yet")
                            
                            # Call the callback after each page if provided
                            if on_subpage_processed:
                                on_subpage_processed()
                                
                        except Exception as e:
                            logger.error(f"[{resource_id}] Error processing page {page_key}: {e}")
                            continue
            
            # Extract resource_id_str from resource for tracking
            resource_id_str = resource.get('title', '') or resource_url.split('//')[-1].replace('/', '_')
            resource_id_str = resource_id_str.strip()
            
            # Mark URL as processed if we extracted any meaningful data 
            # (definitions, projects, or methods) or if we've tried too many times
            
            # Get current attempt count from url_storage
            attempt_count = 0
            try:
                # Check if this URL already has attempt data
                pending_urls = self.content_fetcher.url_storage.get_pending_urls()
                for pending_url in pending_urls:
                    if pending_url.get('url') == resource_url:
                        attempt_count = pending_url.get('attempt_count', 0)
                        break
            except Exception as e:
                logger.error(f"[{resource_id}] Error getting attempt count: {e}")
                attempt_count = 0
                
            # Check if we got any data or if we've tried too many times
            if main_definitions or main_projects or main_methods:
                # We have data, mark as processed
                logger.info(f"[{resource_id}] Marking main URL as processed after successful extraction: {resource_url}")
                self.content_fetcher.mark_url_as_processed(resource_url, depth=0, origin="")
                # Also remove from pending list
                self.content_fetcher.url_storage.remove_pending_url(resource_url)
                
                # Associate URL with resource_id if we have one
                if resource_id_str:
                    logger.info(f"[{resource_id}] Associating resource URL with resource ID {resource_id_str}: {resource_url}")
                    self.url_storage.save_resource_url(resource_id_str, resource_url, depth=0)
            elif attempt_count >= 2:  # After 3 attempts (0, 1, 2), force mark as processed
                # Failed 3 times, force mark as processed to stop trying
                logger.warning(f"[{resource_id}] Main URL failed to yield data after {attempt_count+1} attempts, force marking as processed: {resource_url}")
                self.content_fetcher.mark_url_as_processed(resource_url, depth=0, origin="")
                # Also remove from pending list
                self.content_fetcher.url_storage.remove_pending_url(resource_url)
                
                # Still associate with resource_id even if no data was extracted
                if resource_id_str:
                    logger.info(f"[{resource_id}] Associating resource URL with resource ID {resource_id_str} (no data extracted): {resource_url}")
                    self.url_storage.save_resource_url(resource_id_str, resource_url, depth=0)
            else:
                # Still have attempts left, increment attempt count and keep in pending
                # But still associate with resource_id for future tracking
                if resource_id_str:
                    self.content_fetcher.url_storage.save_pending_url(
                        url=resource_url,
                        depth=0,
                        origin="",
                        attempt_count=attempt_count + 1,
                        resource_id=resource_id_str
                    )
                logger.info(f"[{resource_id}] No data extracted from main URL (attempt {attempt_count+1}), keeping in pending queue: {resource_url}")
                self.content_fetcher.url_storage.save_pending_url(resource_url, depth=0, origin="", attempt_count=attempt_count+1)
            
            # For the resource entry in CSV, we need to be careful about setting analysis_completed
            if idx is not None and csv_path is not None:
                # Update the CSV file
                from scripts.services.storage import DataSaver as ServiceDataSaver
                from scripts.analysis.url_storage import URLStorageManager
                
                # Initialize the service saver
                service_saver = ServiceDataSaver()
                resource_copy = resource.copy()
                
                # Get pending URLs to check if all levels are complete
                url_storage = self.content_fetcher.url_storage
                
                # Check if there are any pending URLs left for this resource
                pending_urls = []
                try:
                    # We only want to check pending URLs relevant to this resource
                    for level in range(1, max_depth + 1):
                        pending_for_level = url_storage.get_pending_urls(depth=level)
                        # Filter to only keep URLs that originated from this resource
                        pending_for_resource = [u for u in pending_for_level if u.get('origin') == resource_url]
                        pending_urls.extend(pending_for_resource)
                        
                    # Check if all 4 levels have been processed before marking as completed
                    # Count how many levels have been fully processed
                    processed_levels = []
                    missing_levels = []
                    for level in range(1, 5):  # Check levels 1-4
                        level_pending = [u for u in pending_urls if int(u.get('depth', 0)) == level]
                        if not level_pending:
                            processed_levels.append(level)
                        else:
                            missing_levels.append(level)
                    
                    # Only mark as completed if ALL 4 levels are processed
                    if len(processed_levels) == 4:
                        logger.info(f"[{resource_id}] All 4 levels fully processed, marking resource as fully analyzed")
                        resource_copy['analysis_completed'] = True
                        resource_copy['partial_processing'] = False
                    else:
                        # Not all levels processed yet, mark as partial processing
                        logger.info(f"[{resource_id}] Not all levels processed yet. Processed: {processed_levels}, Missing: {missing_levels}")
                        logger.info(f"[{resource_id}] Still have {len(pending_urls)} pending URLs across all levels, marking as partial processing")
                        resource_copy['analysis_completed'] = False
                        resource_copy['partial_processing'] = True
                        resource_copy['pending_levels'] = str(missing_levels)
                        resource_copy['pending_urls_count'] = len(pending_urls)
                        # We've already set pending_levels from the missing_levels list
                        
                except Exception as e:
                    logger.error(f"[{resource_id}] Error checking pending URLs: {e}")
                    # Default to not completed in case of error
                    resource_copy['analysis_completed'] = False
                    resource_copy['partial_processing'] = True
                
                # The method is called 'save_resource' in the services DataSaver, not 'save_single_resource'
                service_saver.save_resource(resource_copy, csv_path, idx)
            
            # Summary of results
            total_definitions = len(result["definitions"])
            total_projects = len(result["projects"])
            total_methods = len(result["methods"])
            
            logger.info(f"[{resource_id}] Processing complete: Found {total_definitions} definition(s), {total_projects} project(s), {total_methods} method(s)")
            
            # Mark the operation as successful
            result["success"] = True
            result["processing_time_seconds"] = (datetime.now() - start_time).total_seconds()
            
            return result
            
        except Exception as e:
            logger.error(f"[{resource_id}] Error processing resource: {e}")
            result["error"] = str(e)
            return result
    
    def process_specific_url(self, url: str, origin_url: str, resource: Dict, depth: int) -> Dict[str, Any]:
        """
        Process a specific URL for a resource.
        This is used for level-based processing where we process all URLs at a given depth level.
        
        Args:
            url: The specific URL to process
            origin_url: The original resource URL that this URL was found from
            resource: The resource dictionary containing metadata
            depth: The depth level of this URL
            
        Returns:
            Dictionary with processing results
        """
        start_time = datetime.now()
        
        # Reset interruption state
        clear_interrupt()
        
        # Create a resource ID for logging
        resource_title = resource.get('title', 'Unnamed')
        resource_id = f"{resource_title}::{url}"
        
        # Enhanced debug logging for URL processing 
        logger.info(f"[{resource_id}] Processing URL at level {depth}: {url}")
        url_is_processed = self.content_fetcher.url_storage.url_is_processed(url)
        logger.info(f"[{resource_id}] URL in processed_urls.csv: {url_is_processed}")
        
        # Always check if URL is already in processed list and skip if it is
        if url_is_processed:
            # Log detailed status about this URL
            logger.info(f"[{resource_id}] URL found in processed_urls.csv: {url}")
            
            # But let's check if it's ACTUALLY been analyzed by checking for method/definition files
            import os
            import pandas as pd
            from pathlib import Path
            data_path = Path(self.data_folder)
            
            # Check if we have any definitions, methods, or projects from this URL
            definitions_analyzed = False
            methods_analyzed = False
            projects_analyzed = False
            
            # Check definitions
            definitions_csv = data_path / "definitions.csv"
            if definitions_csv.exists():
                try:
                    defs_df = pd.read_csv(definitions_csv)
                    if 'source' in defs_df.columns:
                        if any(defs_df['source'] == url):
                            definitions_analyzed = True
                            logger.info(f"[{resource_id}] Found definitions from this URL")
                except Exception as e:
                    logger.error(f"[{resource_id}] Error checking definitions.csv: {e}")
            
            # Check methods
            methods_csv = data_path / "methods.csv"
            if methods_csv.exists():
                try:
                    methods_df = pd.read_csv(methods_csv)
                    if 'source' in methods_df.columns:
                        if any(methods_df['source'] == url):
                            methods_analyzed = True
                            logger.info(f"[{resource_id}] Found methods from this URL")
                except Exception as e:
                    logger.error(f"[{resource_id}] Error checking methods.csv: {e}")
                    
            # Check projects
            projects_csv = data_path / "projects.csv"
            if projects_csv.exists():
                try:
                    projects_df = pd.read_csv(projects_csv)
                    if 'source' in projects_df.columns:
                        if any(projects_df['source'] == url):
                            projects_analyzed = True
                            logger.info(f"[{resource_id}] Found projects from this URL")
                except Exception as e:
                    logger.error(f"[{resource_id}] Error checking projects.csv: {e}")
            
            # If we have actual analysis data, skip processing
            if definitions_analyzed or methods_analyzed or projects_analyzed:
                logger.info(f"[{resource_id}] URL has analysis data - skipping: {url}")
                # Remove from pending_urls.csv to clean up
                self.content_fetcher.url_storage.remove_pending_url(url)
                return {
                    "success": True, 
                    "url": url,
                    "origin_url": origin_url,
                    "depth": depth,
                    "status": "skipped_processed",
                    "message": "URL already processed with analysis data"
                }
            else:
                # URL is in processed list but has no data - we'll reprocess it
                logger.warning(f"[{resource_id}] URL was in processed_urls.csv but has NO ANALYSIS DATA - reprocessing: {url}")
                # Continue with processing since the URL doesn't have real analysis data
        
        # Initialize result structure
        result = {
            "success": False,
            "url": url,
            "origin_url": origin_url,
            "depth": depth,
            "definitions": [],
            "projects": [],
            "methods": [],
            "error": None
        }
        
        try:
            # Create temp file for storing content
            resource_slug = resource_title.replace(" ", "_").lower()[:30]
            temp_path = self.temp_storage.create_temp_file(prefix=f"{resource_slug}_level{depth}")
            
            # When processing any URL, we just want to get its content without additional discovery
            # The discovery phase has already happened separately at the beginning
            logger.info(f"[{resource_id}] Processing level {depth} URL: {url} without extra discovery")
            
            # Create headers using WebFetcher
            headers = self.content_fetcher.web_fetcher.create_headers()
            
            # Directly fetch this URL's content without discovering additional URLs
            success, html_content = self.content_fetcher.web_fetcher.fetch_page(url, headers)
            
            # Check if the fetch was successful
            if not success or not html_content:
                logger.error(f"[{resource_id}] Failed to fetch URL: {url}")
                result["error"] = "Failed to fetch URL content"
                return result
                
            # Extract text content from HTML
            page_text = extract_text(html_content)
            
            # Process the extracted text
            try:
                # Use a combined page title
                page_key = url.rsplit('/', 1)[-1] or f'level{depth}_page'
                combined_title = f"{resource_title} - {page_key}"
                
                # STEP 1: Extract definitions
                page_definitions = self.extraction_utils.extract_definitions(
                    title=combined_title, 
                    url=url, 
                    content=page_text
                )
                
                # STEP 2: Extract project information
                page_projects = self.extraction_utils.identify_projects(
                    title=combined_title, 
                    url=url, 
                    content=page_text
                )
                
                # STEP 3: Extract methods
                page_methods = self.method_llm.extract_methods(
                    title=combined_title, 
                    url=url, 
                    content=page_text
                )
                
                # Log extraction results
                logger.info(f"[{resource_id}] Extracted {len(page_definitions)} definition(s), {len(page_projects)} project(s), {len(page_methods)} method(s)")
                
                # Update results
                result["definitions"] = page_definitions
                result["projects"] = page_projects
                result["methods"] = page_methods
                
                # Save to storage if we have results
                if page_definitions or page_projects or page_methods:
                    self.vector_generation_needed = True
                    
                    # Save to CSV using DataSaver with detailed logging
                    logger.info(f"[{resource_id}] SAVING DATA FROM SPECIFIC URL: Saving {len(page_definitions)} definitions to CSV")
                    self.data_saver.save_definitions(page_definitions)
                    
                    logger.info(f"[{resource_id}] SAVING DATA FROM SPECIFIC URL: Saving {len(page_projects)} projects to CSV")
                    if page_projects:
                        for idx, proj in enumerate(page_projects):
                            logger.info(f"[{resource_id}] SAVING PROJECT #{idx+1} FROM SPECIFIC URL: {proj.get('title', 'Untitled')} with tags: {proj.get('tags', [])}")
                    self.data_saver.save_projects(page_projects)
                    
                    logger.info(f"[{resource_id}] SAVING DATA FROM SPECIFIC URL: Saving {len(page_methods)} methods to CSV")
                    if page_methods:
                        for idx, method in enumerate(page_methods):
                            logger.info(f"[{resource_id}] SAVING METHOD #{idx+1} FROM SPECIFIC URL: {method.get('title', 'Untitled')} with {len(method.get('steps', []))} steps")
                    self.data_saver.save_methods(page_methods)
                    
                    # Cache the content to the temporary file for vector building later
                    content_obj = {
                        "source": url,
                        "title": combined_title,
                        "content": page_text,
                        "extracted_at": datetime.now().isoformat(),
                        "resource_id": resource_id,
                        "page_type": f"level{depth}"
                    }
                    self.temp_storage.append_to_file(temp_path, json.dumps(content_obj) + "\n")
                
                # Mark the URL as processed using ContentFetcher's method
                # Only mark as processed if we actually extracted meaningful data
                # Get current attempt count for this URL from url_storage
                attempt_count = 0
                try:
                    # Check if this URL already has attempt data
                    pending_urls = self.content_fetcher.url_storage.get_pending_urls()
                    for pending_url in pending_urls:
                        if pending_url.get('url') == url:
                            attempt_count = pending_url.get('attempt_count', 0)
                            break
                except Exception as e:
                    logger.error(f"[{resource_id}] Error getting attempt count: {e}")
                    attempt_count = 0
                
                # Check if we got any data or if we've tried too many times
                if page_definitions or page_projects or page_methods:
                    # We have data, mark as processed
                    logger.info(f"[{resource_id}] Marking URL as processed after successful extraction: {url}")
                    self.content_fetcher.mark_url_as_processed(url, depth=depth, origin=origin_url)
                    # Also make sure to remove from pending list after successful processing
                    self.content_fetcher.url_storage.remove_pending_url(url)
                    successful_extraction = True
                elif attempt_count >= 2:  # After 3 attempts (0, 1, 2), force mark as processed
                    # Failed 3 times, force mark as processed to stop trying
                    logger.warning(f"[{resource_id}] URL failed to yield data after {attempt_count+1} attempts, force marking as processed: {url}")
                    self.content_fetcher.mark_url_as_processed(url, depth=depth, origin=origin_url)
                    # Also remove from pending list 
                    self.content_fetcher.url_storage.remove_pending_url(url)
                    successful_extraction = False
                else:
                    # Still have attempts left, increment attempt count and keep in pending
                    logger.info(f"[{resource_id}] No data extracted from URL (attempt {attempt_count+1}), keeping in pending queue: {url}")
                    self.content_fetcher.url_storage.save_pending_url(url, depth=depth, origin=origin_url, attempt_count=attempt_count+1)
                    successful_extraction = False
                
                # Only update the resource status if we've marked the URL as processed
                if successful_extraction or attempt_count >= 2:
                    # Check if this was the last URL for this resource and update the resource status in the CSV if needed
                    try:
                        # Get the original resource's CSV path and index
                        from scripts.services.storage import DataSaver as ServiceDataSaver
                        
                        # Get all resources
                        data_path = Path(self.data_folder)
                        csv_path = data_path / "resources.csv"
                        
                        if csv_path.exists():
                            # Find the resource by URL
                            import pandas as pd
                            resources_df = pd.read_csv(csv_path)
                            resource_row = resources_df[resources_df['url'] == origin_url]
                            
                            if not resource_row.empty:
                                idx = resource_row.index[0]
                                
                                # Check if we have any pending URLs left for this resource
                                url_storage = self.content_fetcher.url_storage
                                pending_urls = []
                                
                                # We only want to check pending URLs relevant to this resource
                                for level in range(1, 5):  # Check levels 1-4
                                    pending_for_level = url_storage.get_pending_urls(depth=level)
                                    # Filter to only keep URLs that originated from this resource
                                    pending_for_resource = [u for u in pending_for_level if u.get('origin') == origin_url]
                                    pending_urls.extend(pending_for_resource)
                                
                                # Update resource status
                                service_saver = ServiceDataSaver()
                                resource_copy = resource.copy()
                                
                                # Check if all 4 levels have been processed before marking as completed
                                # Count how many levels have been fully processed
                                processed_levels = []
                                missing_levels = []
                                for level in range(1, 5):  # Check levels 1-4
                                    level_pending = [u for u in pending_urls if int(u.get('depth', 0)) == level]
                                    if not level_pending:
                                        processed_levels.append(level)
                                    else:
                                        missing_levels.append(level)
                                
                                # Only mark as completed if ALL 4 levels are processed
                                if len(processed_levels) == 4:
                                    logger.info(f"[{resource_id}] All 4 levels fully processed, marking resource as fully analyzed")
                                    resource_copy['analysis_completed'] = True
                                    resource_copy['partial_processing'] = False
                                else:
                                    logger.info(f"[{resource_id}] Not all levels processed yet. Processed: {processed_levels}, Missing: {missing_levels}")
                                    logger.info(f"[{resource_id}] Still have {len(pending_urls)} pending URLs, keeping as partial processing")
                                    resource_copy['analysis_completed'] = False
                                    resource_copy['partial_processing'] = True
                                    resource_copy['pending_urls_count'] = len(pending_urls)
                                    resource_copy['pending_levels'] = str(missing_levels)
                                
                                # Update the CSV
                                service_saver.save_resource(resource_copy, str(csv_path), idx)
                    except Exception as e:
                        logger.error(f"[{resource_id}] Error updating resource status after URL processing: {e}")
                else:
                    logger.info(f"[{resource_id}] No data extracted from URL, not marking as processed yet")
                
                # Mark the operation as successful
                result["success"] = True
                result["processing_time_seconds"] = (datetime.now() - start_time).total_seconds()
                
            except Exception as e:
                logger.error(f"[{resource_id}] Error processing URL content: {e}")
                result["error"] = str(e)
                
        except Exception as e:
            logger.error(f"[{resource_id}] Error processing URL: {e}")
            result["error"] = str(e)
            
        return result