#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Import interruptible handling
from scripts.analysis.interruptible_llm import is_interrupt_requested, clear_interrupt
# Import HTML extraction functions
from scripts.analysis.html_extractor import extract_text, extract_page_texts

class SingleResourceProcessor:
    """
    Processes a single resource through content fetching, analysis and storage.
    This class handles the detailed processing of individual resource entries.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the SingleResourceProcessor.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        
        # Import required services
        from scripts.analysis.content_fetcher import ContentFetcher
        from scripts.analysis.llm_analyzer import LLMAnalyzer
        from scripts.analysis.method_llm import MethodLLM
        from scripts.storage.content_storage_service import ContentStorageService
        from scripts.storage.temporary_storage_service import TemporaryStorageService
        from scripts.services.storage import DataSaver
        
        # Initialize services
        self.content_fetcher = ContentFetcher()
        self.llm_analyzer = LLMAnalyzer()
        self.method_llm = MethodLLM()
        self.content_storage = ContentStorageService(data_folder)
        self.temp_storage = TemporaryStorageService(data_folder)
        self.data_saver = DataSaver()
        
        # Track whether vector generation is needed
        self.vector_generation_needed = False
    
    def process_resource(self, resource: Dict, resource_id: str, idx: int, csv_path: str, on_subpage_processed=None) -> Dict[str, Any]:
        """
        Process a single resource through content fetching, analysis and storage.
        
        Args:
            resource: The resource dictionary 
            resource_id: String identifier for logging
            idx: Index in the CSV file
            csv_path: Path to the CSV file for saving updates
            on_subpage_processed: Optional callback function to call after each subpage is processed
            
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
            
            # STEP 1: Fetch main page content
            logger.info(f"[{resource_id}] Fetching content from {resource_url}")
            
            # Create temp file for storing content
            resource_slug = resource_title.replace(" ", "_").lower()[:30]
            temp_path = self.temp_storage.create_temp_file(prefix=resource_slug)
            
            # Fetch main page and extract links
            success, main_data = self.content_fetcher.get_main_page_with_links(resource_url, max_depth=2)
            
            if not success or "error" in main_data:
                error_msg = main_data.get("error", "Unknown error")
                logger.warning(f"[{resource_id}] Failed to fetch {resource_url}: {error_msg}")
                result["error"] = f"Fetch error: {error_msg}"
                return result
            
            # Get content from main page
            main_html = main_data["main_html"]
            additional_urls = main_data["additional_urls"]
            
            # Don't limit additional pages - process all of them
            logger.info(f"[{resource_id}] Processing all {len(additional_urls)} additional pages without limitation")
            
            # Extract text from main page using html_extractor
            main_page_text = extract_text(main_html, 2000)
            if not main_page_text:
                logger.warning(f"[{resource_id}] Failed to extract text from {resource_url}")
                result["error"] = "Text extraction failed"
                return result
                
            # Initialize collectors for extracted information
            all_definitions = []
            all_projects = []
            all_methods = []
            page_results = {}
            
            # Write main page to temp file
            try:
                content_to_write = f"MAIN PAGE: {resource_title}\n\n{main_page_text}"
                total_bytes = self.temp_storage.write_to_file(temp_path, content_to_write, mode="w")
                logger.info(f"[{resource_id}] Wrote main page ({total_bytes/1024:.1f} KB) to temporary file")
            except Exception as e:
                logger.error(f"[{resource_id}] Error writing to temp file: {e}")
            
            # Store content for later vector generation (instead of immediate embedding)
            self.content_storage.store_original_content(resource_title, resource_url, main_page_text, resource_id)
            self.vector_generation_needed = True  # Flag that we'll need vector generation at the end
            
            # STEP 2: Process main page
            logger.info(f"[{resource_id}] Analyzing main page")
            
            # Check for cancellation before extraction starts
            if is_interrupt_requested():
                logger.info(f"[{resource_id}] Resource processing cancelled")
                return result
            
            # Extract definitions, projects, and methods
            try:
                # Extract definitions if requested
                main_definitions = self.llm_analyzer.extract_definitions(
                    f"{resource_title} - main", resource_url, main_page_text
                )
                if main_definitions and not is_interrupt_requested():
                    for definition in main_definitions:
                        definition['source'] = resource_url
                        definition['resource_url'] = resource_url
                    all_definitions.extend(main_definitions)
                    logger.info(f"[{resource_id}] Found {len(main_definitions)} definitions on main page")
                    # Save to CSV using DataSaver instead of function call
                    self.data_saver.save_definition(main_definitions)
                
                # Identify projects if requested
                main_projects = self.llm_analyzer.identify_projects(
                    f"{resource_title} - main", resource_url, main_page_text
                )
                if main_projects and not is_interrupt_requested():
                    for project in main_projects:
                        # Only set source if not already set by the extractor
                        if 'source' not in project:
                            project['source'] = resource_url
                        # Always set resource_url (main page URL)
                        project['resource_url'] = resource_url
                        # Ensure origin_title is set if not already
                        if 'origin_title' not in project:
                            project['origin_title'] = resource_title
                    all_projects.extend(main_projects)
                    logger.info(f"[{resource_id}] Found {len(main_projects)} projects on main page")
                    
                    # Save to CSV using DataSaver
                    self.data_saver.save_project(main_projects)
                
                # Extract methods if requested
                main_methods = self.method_llm.extract_methods(
                    f"{resource_title} - main", resource_url, main_page_text
                )
                if main_methods and not is_interrupt_requested():
                    for method in main_methods:
                        # Only set source if not already set by the extractor
                        if 'source' not in method:
                            method['source'] = resource_url
                        # Always set resource_url (main page URL)
                        method['resource_url'] = resource_url
                        # Ensure origin_title is set if not already
                        if 'origin_title' not in method:
                            method['origin_title'] = resource_title
                    all_methods.extend(main_methods)
                    logger.info(f"[{resource_id}] Found {len(main_methods)} methods on main page")
                    
                    # Save to CSV using DataSaver
                    self.data_saver.save_method(main_methods)
                
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
            
            # STEP 3: Process additional pages
            logger.info(f"[{resource_id}] Processing {len(additional_urls)} additional pages")
            headers = main_data.get("headers", {})
            
            # Create a dictionary to store all HTML content for batch processing
            additional_pages_html = {}
            
            # Track subpage count for the result
            subpage_count = len(additional_urls)
            
            # First, fetch all pages
            for idx, additional_url in enumerate(additional_urls):
                try:
                    # Fetch page content
                    logger.info(f"[{resource_id}] Fetching page {idx+1}/{len(additional_urls)}: {additional_url}")
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
                for page_name, page_text in extracted_pages.items():
                    additional_url = additional_urls[list(additional_pages_html.keys()).index(page_name)] if page_name in list(additional_pages_html.keys()) else "unknown"
                    
                    # Append to temp file
                    try:
                        content_to_append = f"\n\n--- PAGE: {page_name} ---\n\n{page_text}"
                        page_bytes = self.temp_storage.append_to_file(temp_path, content_to_append)
                        logger.info(f"[{resource_id}] Wrote page {page_name} ({page_bytes/1024:.1f} KB) to temp file")
                    except Exception as e:
                        logger.error(f"[{resource_id}] Error appending to temp file: {e}")
                    
                    # Process page content
                    page_definitions = self.llm_analyzer.extract_definitions(
                        f"{resource_title} - {page_name}", additional_url, page_text
                    )
                    page_projects = self.llm_analyzer.identify_projects(
                        f"{resource_title} - {page_name}", additional_url, page_text
                    )
                    page_methods = self.method_llm.extract_methods(
                        f"{resource_title} - {page_name}", additional_url, page_text
                    )
                    
                    # Store results for this page
                    page_results[page_name] = {
                        "definitions": page_definitions,
                        "projects": page_projects,
                        "methods": page_methods,
                        "url": additional_url,
                        "content": page_text[:500] + "..." if len(page_text) > 500 else page_text
                    }
                    
                    # Process and add source information
                    if page_definitions:
                        for definition in page_definitions:
                            # Only set source if not already set by the extractor
                            if 'source' not in definition:
                                definition['source'] = additional_url
                            # Always set resource_url (main page URL)
                            definition['resource_url'] = resource_url
                            # Ensure origin_title is set if not already
                            if 'origin_title' not in definition:
                                definition['origin_title'] = f"{resource_title} - {page_name}"
                        all_definitions.extend(page_definitions)
                        # Use DataSaver
                        self.data_saver.save_definition(page_definitions)
                        logger.info(f"[{resource_id}] Found {len(page_definitions)} definitions on page '{page_name}'")
                    
                    # Process page-specific projects
                    if page_projects:
                        for project in page_projects:
                            # Only set source if not already set by the extractor
                            if 'source' not in project:
                                project['source'] = additional_url
                            # Always set resource_url (main page URL)
                            project['resource_url'] = resource_url
                            # Ensure origin_title is set if not already  
                            if 'origin_title' not in project:
                                project['origin_title'] = f"{resource_title} - {page_name}"
                        all_projects.extend(page_projects)
                        # Use DataSaver
                        self.data_saver.save_project(page_projects)
                        logger.info(f"[{resource_id}] Found {len(page_projects)} projects on page '{page_name}'")
                    
                    # Process page-specific methods
                    if page_methods:
                        for method in page_methods:
                            # Only set source if not already set by the extractor
                            if 'source' not in method:
                                method['source'] = additional_url
                            # Always set resource_url (main page URL)
                            method['resource_url'] = resource_url
                            # Ensure origin_title is set if not already
                            if 'origin_title' not in method:
                                method['origin_title'] = f"{resource_title} - {page_name}"
                        all_methods.extend(page_methods)
                        # Use DataSaver
                        self.data_saver.save_method(page_methods)
                        logger.info(f"[{resource_id}] Found {len(page_methods)} methods on page '{page_name}'")
                    
                    # Store page content for later vector generation
                    self.content_storage.store_original_content(
                        f"{resource_title} - {page_name}", 
                        additional_url, 
                        page_text, 
                        f"{resource_id} - {page_name}"
                    )
                    
                    # Mark this URL as processed immediately after successful analysis
                    self.content_fetcher.mark_url_as_processed(additional_url, depth=1, origin=resource_url)
                    logger.info(f"[{resource_id}] Marked subpage as processed: {additional_url}")
                    
                    # Call the callback after each subpage is processed
                    if on_subpage_processed is not None:
                        on_subpage_processed()
            
            # STEP 4: Store analysis results
            logger.info(f"[{resource_id}] Completed processing all pages")
            
            # Initialize analysis_results if needed
            if 'analysis_results' not in resource:
                resource['analysis_results'] = {}
            
            # Ensure analysis_results is a dictionary
            if isinstance(resource['analysis_results'], str):
                try:
                    resource['analysis_results'] = json.loads(resource['analysis_results'])
                except:
                    resource['analysis_results'] = {}
            
            if not isinstance(resource['analysis_results'], dict):
                resource['analysis_results'] = {}
                    
            # Store all the page results in the resource
            resource['analysis_results']['pages'] = page_results
            resource['analysis_results']['definitions'] = all_definitions
            resource['analysis_results']['projects'] = all_projects
            resource['analysis_results']['methods'] = all_methods
            
            # STEP 5: Generate description if needed
            if not resource.get('description') or len(resource.get('description', '').strip()) < 10:
                logger.info(f"[{resource_id}] Generating description")
                
                # Read from temp file
                try:
                    combined_text = self.temp_storage.read_from_file(temp_path, max_size=10000)
                    if not combined_text:
                        combined_text = main_page_text
                except Exception as e:
                    logger.error(f"[{resource_id}] Error reading from temp file: {e}")
                    combined_text = main_page_text
                
                # Generate description
                description = self.llm_analyzer.generate_description(
                    resource_title, resource_url, combined_text
                )
                
                resource['description'] = description
                logger.info(f"[{resource_id}] Generated description: {len(description)} chars")
            
            # STEP 6: Generate tags if needed
            if not resource.get('tags') or not isinstance(resource.get('tags'), list) or len(resource.get('tags')) < 3:
                logger.info(f"[{resource_id}] Generating tags")
                
                # Prepare existing tags
                existing_tags = []
                if resource.get('tags'):
                    if isinstance(resource.get('tags'), str):
                        try:
                            existing_tags = json.loads(resource['tags'].replace("'", '"'))
                        except:
                            pass
                    elif isinstance(resource.get('tags'), list):
                        existing_tags = resource.get('tags')
                
                # Create context for tag generation
                all_terms = [d.get('term', '').lower() for d in all_definitions 
                            if isinstance(d, dict) and 'term' in d]
                all_project_titles = [p.get('title', '') for p in all_projects 
                                     if isinstance(p, dict) and 'title' in p]
                
                tag_context = f"""
                Resource Title: {resource_title}
                URL: {resource_url}
                
                Description: {resource.get('description', '')}
                
                Key terms: {', '.join(all_terms[:20])}
                Projects: {', '.join(all_project_titles[:5])}
                """
                
                # Generate tags
                tags = self.llm_analyzer.generate_tags(
                    resource_title, resource_url, tag_context,
                    resource.get('description', ''), existing_tags
                )
                    
                resource['tags'] = tags
                logger.info(f"[{resource_id}] Generated tags: {tags}")
            
            # STEP 7: Process batches for vector storage
            logger.info(f"[{resource_id}] Processing content in batches for later vector generation")
            
            try:
                # Check if temp file exists
                if os.path.exists(temp_path):
                    file_size = os.path.getsize(temp_path)
                    logger.info(f"[{resource_id}] Processing temp file: {file_size/1024/1024:.2f} MB")
                    
                    # Process in batches
                    batch_size = 32 * 1024  # 32KB per batch
                    
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        # Get file size
                        f.seek(0, os.SEEK_END)
                        total_size = f.tell()
                        f.seek(0)
                        
                        batch_number = 1
                        total_processed = 0
                        
                        # Process each batch
                        while True:
                            batch_text = f.read(batch_size)
                            if not batch_text:
                                break
                                
                            # Track progress
                            total_processed += len(batch_text)
                            progress = (total_processed / total_size) * 100
                            
                            # Store batch for later vector generation
                            batch_title = f"{resource_title} - Batch {batch_number}"
                            batch_metadata = {
                                "resource_title": resource_title,
                                "resource_url": resource_url,
                                "batch_number": batch_number,
                                "progress_percent": progress,
                                "definitions_count": len(all_definitions),
                                "projects_count": len(all_projects),
                                "methods_count": len(all_methods)
                            }
                            
                            # Store content batch - this just stores it for later, no vector generation now
                            self.content_storage.store_content_batch(
                                batch_title, resource_url, batch_text,
                                all_definitions, all_projects, all_methods, batch_metadata
                            )
                            
                            logger.info(f"[{resource_id}] Processed batch {batch_number} ({progress:.1f}%)")
                            batch_number += 1
                            
                    logger.info(f"[{resource_id}] Completed processing {batch_number-1} batches")
                    
                    # Delete temp file
                    try:
                        os.unlink(temp_path)
                        logger.info(f"[{resource_id}] Deleted temporary file")
                    except Exception as e:
                        logger.warning(f"[{resource_id}] Failed to delete temporary file: {e}")
            
            except Exception as e:
                logger.error(f"[{resource_id}] Error processing batches: {e}")
                # If batch processing failed, store everything at once
                try:
                    combined_text = self.temp_storage.read_from_file(temp_path)
                    if combined_text:
                        self.content_storage.store_content_with_analysis(
                            resource_title, resource_url, combined_text,
                            all_definitions, all_projects, all_methods
                        )
                        logger.info(f"[{resource_id}] Stored combined content as fallback")
                except Exception as fallback_error:
                    logger.error(f"[{resource_id}] Fallback storage failed: {fallback_error}")
            
            # Check if processing was at least partially successful
            # We'll mark with appropriate status rather than just boolean:
            # - "completed": Successfully processed with data extracted or pages processed
            # - "failed": Complete failure to process
            has_extracted_data = bool(all_definitions or all_projects or all_methods)
            has_processed_pages = bool(page_results) and any(
                page.get('content') for page in page_results.values() if isinstance(page, dict)
            )
            
            # Set analysis status based on processing results
            if has_extracted_data or has_processed_pages:
                resource['analysis_completed'] = True
                resource['analysis_status'] = "completed"
            else:
                resource['analysis_completed'] = False
                resource['analysis_status'] = "failed"
            
            logger.info(f"[{resource_id}] Setting analysis status to '{resource.get('analysis_status', 'unknown')}' " +
                        f"(analysis_completed={resource['analysis_completed']}, " +
                        f"data extracted: {has_extracted_data}, pages processed: {has_processed_pages})")

            # Use DataSaver for saving resource - ensure idx is passed properly
            success = self.data_saver.save_resource(resource, csv_path, idx)
            if not success:
                logger.warning(f"[{resource_id}] Failed to save resource to CSV")
            
            # Update result
            result["success"] = True
            result["resource"] = resource
            result["definitions"] = all_definitions
            result["projects"] = all_projects
            result["methods"] = all_methods
            
            # Update result with subpage count
            result["subpage_count"] = subpage_count
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ [{resource_id}] Processing completed in {duration:.2f}s: {len(all_definitions)} definitions, {len(all_projects)} projects, {len(all_methods)} methods")
            
            return result
            
        except Exception as e:
            logger.error(f"[{resource_id}] Error processing resource: {e}", exc_info=True)
            result["error"] = str(e)
            return result
    
    def save_single_resource(self, resource, csv_path, idx):
        """Save a single resource to the CSV file using DataSaver."""
        return self.data_saver.save_resource(resource, csv_path, idx)
