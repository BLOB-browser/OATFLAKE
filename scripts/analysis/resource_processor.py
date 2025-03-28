#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Fix imports to use absolute imports instead of relative ones
from scripts.analysis.content_fetcher import ContentFetcher
from scripts.analysis.llm_analyzer import LLMAnalyzer
from scripts.analysis.method_llm import MethodLLM
# Import DataSaver class directly
from scripts.services.storage import DataSaver

logger = logging.getLogger(__name__)

class ResourceProcessor:
    """
    Handles processing of individual resources.
    Extracts content, definitions, projects, and methods.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the resource processor.
        
        Args:
            data_folder: Path to data directory
        """
        self.data_folder = data_folder
        
        # Import required services
        from scripts.storage.content_storage_service import ContentStorageService
        from scripts.storage.temporary_storage_service import TemporaryStorageService
        
        # Initialize services
        self.content_fetcher = ContentFetcher()
        self.llm_analyzer = LLMAnalyzer()
        self.method_llm = MethodLLM()
        
        # Initialize DataSaver
        self.data_saver = DataSaver()
        
        self.content_storage = ContentStorageService(data_folder)
        self.temp_storage = TemporaryStorageService(data_folder)
        
    def process_resource(self, resource: Dict, resource_id: str, idx: int, csv_path: str) -> Dict[str, Any]:
        # Process a single resource through content fetching, analysis and storage
        
        # STEP 1: Fetch main page content
        # STEP 2: Process main page and additional pages
        # STEP 3: Store analysis results
        # STEP 4: Generate description if needed
        # STEP 5: Generate tags if needed
        # STEP 6: Process batches for vector storage
        
        start_time = datetime.now()
        
        # Initialize result structure
        result = {
            "success": False,
            "resource": resource,
            "definitions": [],
            "projects": [],
            "methods": [],
            "error": None,
            "vector_needed": False
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
            success, main_data = self.content_fetcher.get_main_page_with_links(resource_url, go_deeper=True)
            
            if not success or "error" in main_data:
                error_msg = main_data.get("error", "Unknown error")
                logger.warning(f"[{resource_id}] Failed to fetch {resource_url}: {error_msg}")
                result["error"] = f"Fetch error: {error_msg}"
                return result
            
            # STEP 2: Process main page and additional pages
            all_definitions, all_projects, all_methods, page_results = self._process_pages(
                resource_title, resource_url, resource_id, main_data, temp_path
            )
            
            # STEP 3: Store analysis results
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
            
            # STEP 4: Generate description if needed
            if not resource.get('description') or len(resource.get('description', '').strip()) < 10:
                description = self._generate_description(resource_title, resource_url, temp_path, main_data["main_html"])
                resource['description'] = description
                logger.info(f"[{resource_id}] Generated description: {len(description)} chars")
            
            # STEP 5: Generate tags if needed
            if not resource.get('tags') or not isinstance(resource.get('tags'), list) or len(resource.get('tags')) < 3:
                tags = self._generate_tags(resource_title, resource_url, resource, all_definitions, all_projects)
                resource['tags'] = tags
                logger.info(f"[{resource_id}] Generated tags: {tags}")
            
            # STEP 6: Process batches for vector storage
            result["vector_needed"] = self._process_batches(
                resource_title, resource_url, temp_path, 
                all_definitions, all_projects, all_methods
            )
            
            # Mark resource as completed - fixed logic to ensure it's only true when we actually found data
            has_extracted_data = bool(all_definitions or all_projects or all_methods)
            resource['analysis_completed'] = has_extracted_data
            
            # Save the resource with DataSaver instead of the custom method
            self.data_saver.save_resource(resource, csv_path, idx)
            
            # Update result
            result["success"] = True
            result["resource"] = resource
            result["definitions"] = all_definitions
            result["projects"] = all_projects
            result["methods"] = all_methods
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ [{resource_id}] Processing completed in {duration:.2f}s: {len(all_definitions)} definitions, {len(all_projects)} projects, {len(all_methods)} methods")
            logger.info(f"✓ [{resource_id}] Analysis completed flag set to: {has_extracted_data}")
            
            return result
            
        except Exception as e:
            logger.error(f"[{resource_id}] Error processing resource: {e}", exc_info=True)
            result["error"] = str(e)
            return result
            
    def _process_pages(self, resource_title, resource_url, resource_id, main_data, temp_path):
        """Process main page and additional pages to extract information."""
        # Extract content from main page
        main_html = main_data["main_html"]
        additional_urls = main_data["additional_urls"]
        
        # Limit additional pages
        max_additional_pages = 15
        if len(additional_urls) > max_additional_pages:
            logger.info(f"Limiting from {len(additional_urls)} to {max_additional_pages} additional pages")
            additional_urls = additional_urls[:max_additional_pages]
        
        # Extract text from main page
        main_page_text = self.content_fetcher.extract_text(main_html, 2000)
        if not main_page_text:
            raise ValueError(f"Failed to extract text from {resource_url}")
        
        # Initialize collectors for extracted information
        all_definitions = []
        all_projects = []
        all_methods = []
        page_results = {}
        
        # Process main page
        self._process_main_page(
            resource_title, resource_url, resource_id, main_page_text, temp_path,
            all_definitions, all_projects, all_methods, page_results
        )
        
        # Process additional pages
        self._process_additional_pages(
            resource_title, resource_url, resource_id, additional_urls, main_data.get("headers", {}), 
            temp_path, all_definitions, all_projects, all_methods, page_results
        )
        
        return all_definitions, all_projects, all_methods, page_results
    
    def _process_main_page(self, resource_title, resource_url, resource_id, main_page_text, 
                          temp_path, all_definitions, all_projects, all_methods, page_results):
        """Process the main page of a resource."""
        # Write main page to temp file
        try:
            content_to_write = f"MAIN PAGE: {resource_title}\n\n{main_page_text}"
            total_bytes = self.temp_storage.write_to_file(temp_path, content_to_write, mode="w")
            logger.info(f"[{resource_id}] Wrote main page ({total_bytes/1024:.1f} KB) to temporary file")
        except Exception as e:
            logger.error(f"[{resource_id}] Error writing to temp file: {e}")
        
        # Store content for later vector generation
        self.content_storage.store_original_content(resource_title, resource_url, main_page_text, resource_id)
        
        # Extract definitions, projects, and methods
        logger.info(f"[{resource_id}] Analyzing main page")
        main_definitions = self.llm_analyzer.extract_definitions(
            f"{resource_title} - main", resource_url, main_page_text
        )
        
        main_projects = self.llm_analyzer.identify_projects(
            f"{resource_title} - main", resource_url, main_page_text
        )
        
        main_methods = self.method_llm.extract_methods(
            f"{resource_title} - main", resource_url, main_page_text
        )
        
        # Store results and add source information
        page_results['main'] = {
            "definitions": main_definitions,
            "projects": main_projects,
            "methods": main_methods,
            "url": resource_url,
            "content": main_page_text[:500] + "..." if len(main_page_text) > 500 else main_page_text
        }
        
        # Process and store main page definitions
        if main_definitions:
            for definition in main_definitions:
                definition['source'] = resource_url
                definition['resource_url'] = resource_url
            all_definitions.extend(main_definitions)
            logger.info(f"[{resource_id}] Found {len(main_definitions)} definitions on main page")
            
            # Use DataSaver instance instead of direct function call
            self.data_saver.save_definition(main_definitions)
        
        # Process and store main page projects
        if main_projects:
            for project in main_projects:
                project['source'] = resource_url
                project['resource_url'] = resource_url
            all_projects.extend(main_projects)
            logger.info(f"[{resource_id}] Found {len(main_projects)} projects on main page")
            
            # Use DataSaver instance
            self.data_saver.save_project(main_projects)
        
        # Process and store main page methods
        if main_methods:
            for method in main_methods:
                method['source'] = resource_url
                method['resource_url'] = resource_url
            all_methods.extend(main_methods)
            logger.info(f"[{resource_id}] Found {len(main_methods)} methods on main page")
            
            # Use DataSaver instance
            self.data_saver.save_method(main_methods)
    
    def _process_additional_pages(self, resource_title, resource_url, resource_id, additional_urls, headers,
                                temp_path, all_definitions, all_projects, all_methods, page_results):
        """Process additional pages from a resource."""
        logger.info(f"[{resource_id}] Processing {len(additional_urls)} additional pages")
        
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
                
                # Extract text from page
                page_text = self.content_fetcher.extract_text(html_content, 2000)
                if not page_text:
                    logger.warning(f"[{resource_id}] Failed to extract text from {additional_url}, skipping")
                    continue
                
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
                
                # Process and save extracted data
                if page_definitions:
                    for definition in page_definitions:
                        definition['source'] = additional_url
                        definition['resource_url'] = resource_url
                    all_definitions.extend(page_definitions)
                    self.data_saver.save_definition(page_definitions)
                    logger.info(f"[{resource_id}] Found {len(page_definitions)} definitions on page '{page_name}'")
                
                if page_projects:
                    for project in page_projects:
                        project['source'] = additional_url
                        project['resource_url'] = resource_url
                    all_projects.extend(page_projects)
                    self.data_saver.save_project(page_projects)
                    logger.info(f"[{resource_id}] Found {len(page_projects)} projects on page '{page_name}'")
                
                if page_methods:
                    for method in page_methods:
                        method['source'] = additional_url
                        method['resource_url'] = resource_url
                    all_methods.extend(page_methods)
                    self.data_saver.save_method(page_methods)
                    logger.info(f"[{resource_id}] Found {len(page_methods)} methods on page '{page_name}'")
                
                # Store page content for later vector generation
                self.content_storage.store_original_content(
                    f"{resource_title} - {page_name}", 
                    additional_url, 
                    page_text, 
                    f"{resource_id} - {page_name}"
                )
                
            except Exception as e:
                logger.error(f"[{resource_id}] Error processing page {additional_url}: {e}")
                continue
    
    def _generate_description(self, resource_title, resource_url, temp_path, main_html):
        """Generate a description for the resource."""
        try:
            # Read from temp file
            combined_text = self.temp_storage.read_from_file(temp_path, max_size=10000)
            if not combined_text:
                combined_text = self.content_fetcher.extract_text(main_html, 2000)
        except Exception as e:
            logger.error(f"Error reading from temp file: {e}")
            combined_text = self.content_fetcher.extract_text(main_html, 2000)
        
        # Generate description
        return self.llm_analyzer.generate_description(resource_title, resource_url, combined_text)
    
    def _generate_tags(self, resource_title, resource_url, resource, all_definitions, all_projects):
        """Generate tags for the resource."""
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
        return self.llm_analyzer.generate_tags(
            resource_title, resource_url, tag_context,
            resource.get('description', ''), existing_tags
        )
    
    def _process_batches(self, resource_title, resource_url, temp_path, all_definitions, all_projects, all_methods):
        """Process content in batches for vector storage."""
        try:
            # Check if temp file exists
            if not os.path.exists(temp_path):
                logger.warning("Temporary file not found, skipping batch processing")
                return False
                
            file_size = os.path.getsize(temp_path)
            logger.info(f"Processing temp file: {file_size/1024/1024:.2f} MB")
            
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
                    
                    logger.info(f"Processed batch {batch_number} ({progress:.1f}%)")
                    batch_number += 1
                    
            logger.info(f"Completed processing {batch_number-1} batches")
            
            # Delete temp file
            try:
                os.unlink(temp_path)
                logger.info(f"Deleted temporary file")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")
                
            return True
        
        except Exception as e:
            logger.error(f"Error processing batches: {e}")
            # If batch processing failed, store everything at once
            try:
                combined_text = self.temp_storage.read_from_file(temp_path)
                if combined_text:
                    self.content_storage.store_content_with_analysis(
                        resource_title, resource_url, combined_text,
                        all_definitions, all_projects, all_methods
                    )
                    logger.info(f"Stored combined content as fallback")
                    return True
            except Exception as fallback_error:
                logger.error(f"Fallback storage failed: {fallback_error}")
                
            return False