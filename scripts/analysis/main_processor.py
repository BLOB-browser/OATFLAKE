#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
import pandas as pd
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Class-level cancellation flag
_cancel_processing = False

# Import interruptible handling
from scripts.analysis.interruptible_llm import is_interrupt_requested, clear_interrupt

class MainProcessor:
    """
    Simplified main processor that coordinates content fetching, analysis, and storage.
    Vectors are only generated at the end of processing all resources.
    """
    
    @classmethod
    def reset_cancellation(cls):
        """Reset the cancellation flag"""
        cls._cancel_processing = False
    
    @classmethod
    def check_cancellation(cls):
        """Check if processing should be cancelled"""
        return cls._cancel_processing or is_interrupt_requested()
    
    def __init__(self, data_folder: str):
        """
        Initialize the main processor.
        
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
        # Use DataSaver class instead of individual functions
        from scripts.services.storage import DataSaver
        
        # Initialize services
        self.content_fetcher = ContentFetcher()
        self.llm_analyzer = LLMAnalyzer()
        self.method_llm = MethodLLM()
        self.content_storage = ContentStorageService(data_folder)
        self.temp_storage = TemporaryStorageService(data_folder)
        
        # Use DataSaver instance instead of individual functions
        self.data_saver = DataSaver()
        
        # Track processing state
        self.vector_generation_needed = False
        
        # Reset cancellation flag when creating a new instance
        MainProcessor.reset_cancellation()
        
        logger.info(f"MainProcessor initialized with data folder: {data_folder}")
    
    def process_resources(self, csv_path: str = None, max_resources: int = None, force_reanalysis: bool = False, skip_vector_generation: bool = False) -> Dict[str, Any]:
        # Process resources from a CSV file
        
        # Reset cancellation flag when starting new processing
        MainProcessor.reset_cancellation()
        
        # Set default CSV path if not provided
        if csv_path is None:
            csv_path = os.path.join(self.data_folder, 'resources.csv')
        
        if not os.path.exists(csv_path):
            logger.error(f"Resources file not found: {csv_path}")
            return {
                "status": "error",
                "error": "Resources file not found",
                "resources_processed": 0
            }
        
        # Statistics to track
        stats = {
            "resources_processed": 0,
            "success_count": 0,
            "error_count": 0,
            "definitions_found": 0,
            "projects_found": 0,
            "methods_found": 0,
            "start_time": datetime.now()
        }
        
        try:
            # Read resources CSV file
            df = pd.read_csv(csv_path)
            logger.info(f"Read {len(df)} resources from {csv_path}")
            
            # Prepare analysis_completed column
            if 'analysis_completed' not in df.columns:
                df['analysis_completed'] = False
                logger.info("Added analysis_completed column to resources")
            
            # Convert string representation of boolean values if needed
            if df['analysis_completed'].dtype == 'object':
                df['analysis_completed'] = df['analysis_completed'].map({
                    'True': True, 'true': True, 'TRUE': True, 'T': True, '1': True, 1: True,
                    'False': False, 'false': False, 'FALSE': False, 'F': False, '0': False, 0: False,
                    None: False, pd.NA: False
                }).fillna(False).astype(bool)
                
                logger.info("Converted string representation of boolean values")
            
            # Save the DataFrame with properly typed column
            df.to_csv(csv_path, index=False)
            
            # Filter resources based on analysis_completed unless force_reanalysis is True
            if not force_reanalysis:
                unprocessed_mask = ~df['analysis_completed'].fillna(False).astype(bool)
                unprocessed_count = unprocessed_mask.sum()
                
                if unprocessed_count < len(df):
                    logger.info(f"Filtering out {len(df) - unprocessed_count} already processed resources")
                    df = df[unprocessed_mask]
            
            # Limit resources if specified
            if max_resources is not None and len(df) > max_resources:
                df = df.head(max_resources)
                logger.info(f"Limited to {len(df)} resources as requested")
            
            if len(df) == 0:
                logger.info("No resources to process after filtering")
                return {
                    "status": "completed",
                    "resources_processed": 0,
                    "message": "No resources needed processing"
                }
            
            # Create collectors for processed items
            processed_resources = []
            all_definitions = []
            all_projects = []
            all_methods = []
            
            # Process each resource
            for idx, row in df.iterrows():
                resource = row.to_dict()
                resource_id = f"#{idx+1}/{len(df)}"
                
                # Check for cancellation
                if MainProcessor.check_cancellation():
                    logger.info("Resource processing cancelled by external request")
                    break
                
                logger.info(f"Processing resource {resource_id}: {resource.get('title', 'Unnamed')}")
                
                try:
                    # Process this resource
                    result = self.process_single_resource(resource, resource_id, idx, csv_path)
                    
                    # Update statistics
                    stats["resources_processed"] += 1
                    if result["success"]:
                        stats["success_count"] += 1
                        stats["definitions_found"] += len(result.get("definitions", []))
                        stats["projects_found"] += len(result.get("projects", []))
                        stats["methods_found"] += len(result.get("methods", []))
                        
                        # Store the processed resource and its extracted data
                        processed_resources.append(result["resource"])
                        all_definitions.extend(result.get("definitions", []))
                        all_projects.extend(result.get("projects", []))
                        all_methods.extend(result.get("methods", []))
                    else:
                        stats["error_count"] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing resource {resource_id}: {e}")
                    stats["error_count"] += 1
                
                # Check for cancellation again after processing each resource
                if MainProcessor.check_cancellation():
                    logger.info("Resource processing cancelled after processing resource")
                    break
            
            # Calculate duration
            stats["duration_seconds"] = (datetime.now() - stats["start_time"]).total_seconds()
            stats["status"] = "completed"
            
            # Include processed items in the stats
            stats["processed_resources"] = processed_resources
            stats["definitions"] = all_definitions
            stats["projects"] = all_projects
            stats["methods"] = all_methods
            
            # Generate vector stores if needed and not explicitly skipped
            if self.vector_generation_needed and not skip_vector_generation:
                logger.info("Starting vector generation after all resources were processed")
                
                # Get all content paths that need processing
                from scripts.analysis.vector_generator import VectorGenerator
                import asyncio
                
                # Get all content files from the temporary storage path
                content_paths = list(self.content_storage.temp_storage_path.glob("*.jsonl"))
                logger.info(f"Found {len(content_paths)} content files for vector generation")
                
                # Use VectorGenerator instead of internal method
                vector_generator = VectorGenerator(self.data_folder)
                
                # Handle async vector generation
                try:
                    # Run the vector generation
                    vector_stats = asyncio.run(vector_generator.generate_vector_stores(content_paths))
                    stats["vector_stats"] = vector_stats
                    logger.info(f"Vector generation complete: {vector_stats}")
                except Exception as ve:
                    logger.error(f"Error during vector generation: {ve}")
                    stats["vector_error"] = str(ve)
                
                # Add comprehensive cleanup after vector generation (whether successful or not)
                self._cleanup_temporary_files()
            elif skip_vector_generation and self.vector_generation_needed:
                logger.info("Vector generation needed but explicitly skipped - will be handled at a higher level")
                stats["vector_generation_skipped"] = True
            
            return stats
            
        except Exception as e:
            logger.error(f"Error during resource processing: {e}")
            stats["status"] = "error"
            stats["error"] = str(e)
            stats["duration_seconds"] = (datetime.now() - stats["start_time"]).total_seconds()
            return stats
    
    def process_single_resource(self, resource: Dict, resource_id: str, idx: int, csv_path: str) -> Dict[str, Any]:
        """
        Process a single resource through content fetching, analysis and storage.
        
        Args:
            resource: The resource dictionary 
            resource_id: String identifier for logging
            idx: Index in the CSV file
            csv_path: Path to the CSV file for saving updates
            
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
            success, main_data = self.content_fetcher.get_main_page_with_links(resource_url, go_deeper=True)
            
            if not success or "error" in main_data:
                error_msg = main_data.get("error", "Unknown error")
                logger.warning(f"[{resource_id}] Failed to fetch {resource_url}: {error_msg}")
                result["error"] = f"Fetch error: {error_msg}"
                return result
            
            # Get content from main page
            main_html = main_data["main_html"]
            additional_urls = main_data["additional_urls"]
            
            # Limit additional pages
            max_additional_pages = 15
            if len(additional_urls) > max_additional_pages:
                logger.info(f"[{resource_id}] Limiting from {len(additional_urls)} to {max_additional_pages} additional pages")
                additional_urls = additional_urls[:max_additional_pages]
            
            # Extract text from main page
            main_page_text = self.content_fetcher.extract_text(main_html, 2000)
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
            if MainProcessor.check_cancellation():
                logger.info(f"[{resource_id}] Resource processing cancelled")
                return result
            
            # Extract definitions, projects, and methods
            try:
                # Extract definitions if requested
                main_definitions = self.llm_analyzer.extract_definitions(
                    f"{resource_title} - main", resource_url, main_page_text
                )
                if main_definitions and not MainProcessor.check_cancellation():
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
                if main_projects and not MainProcessor.check_cancellation():
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
                if main_methods and not MainProcessor.check_cancellation():
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
                if MainProcessor.check_cancellation():
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
                    
                except Exception as e:
                    logger.error(f"[{resource_id}] Error processing page {additional_url}: {e}")
                    continue
            
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
            
            # Mark as completed
            has_extracted_data = bool(all_definitions or all_projects or all_methods)
            resource['analysis_completed'] = has_extracted_data
            logger.info(f"[{resource_id}] Setting analysis_completed flag to: {has_extracted_data}")

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
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ [{resource_id}] Processing completed in {duration:.2f}s: {len(all_definitions)} definitions, {len(all_projects)} projects, {len(all_methods)} methods")
            
            return result
            
        except Exception as e:
            logger.error(f"[{resource_id}] Error processing resource: {e}", exc_info=True)
            result["error"] = str(e)
            return result
    
    # Replace save_single_resource with a method that uses data_saver
    def save_single_resource(self, resource, csv_path, idx):
        """Save a single resource to the CSV file using DataSaver."""
        return self.data_saver.save_resource(resource, csv_path, idx)

    async def generate_vector_stores(self) -> Dict[str, Any]:
        """
        DEPRECATED: Use VectorGenerator.generate_vector_stores() instead.
        This method is kept for backward compatibility.
        """
        logger.warning("Using deprecated generate_vector_stores method. Consider using VectorGenerator directly")
        
        from scripts.analysis.vector_generator import VectorGenerator
        
        # Get all content files from the temporary storage path
        content_paths = list(self.content_storage.temp_storage_path.glob("*.jsonl"))
        
        # Use VectorGenerator
        vector_generator = VectorGenerator(self.data_folder)
        return await vector_generator.generate_vector_stores(content_paths)
    
    def _cleanup_temporary_files(self):
        """Clean up all temporary files and directories after processing."""
        try:
            logger.info("Performing comprehensive cleanup of temporary files")
            
            # 1. Clean up any remaining resource temp files
            temp_files = list(Path(self.temp_storage.temp_dir).glob("*"))
            if temp_files:
                logger.info(f"Cleaning up {len(temp_files)} temporary resource files")
                for temp_file in temp_files:
                    try:
                        if (temp_file.is_file()):
                            temp_file.unlink()
                        elif (temp_file.is_dir()):
                            import shutil
                            shutil.rmtree(temp_file, ignore_errors=True)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
            
            # 2. Clean up any remaining JSONL files in multiple locations
            # 2a. Clean JSONL files in content storage temp path
            jsonl_files = list(self.content_storage.temp_storage_path.glob("*.jsonl"))
            if jsonl_files:
                logger.info(f"Cleaning up {len(jsonl_files)} remaining JSONL files in content storage")
                for jsonl_file in jsonl_files:
                    try:
                        jsonl_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove JSONL file {jsonl_file}: {e}")
            
            # 2b. Clean JSONL files in general temp directory
            temp_dir = Path(self.data_folder) / "temp"
            if temp_dir.exists():
                temp_jsonl_files = list(temp_dir.glob("*.jsonl"))
                if temp_jsonl_files:
                    logger.info(f"Cleaning up {len(temp_jsonl_files)} remaining JSONL files in temp directory")
                    for jsonl_file in temp_jsonl_files:
                        try:
                            jsonl_file.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to remove JSONL file {jsonl_file}: {e}")
            
            # 3. Log completion of cleanup
            logger.info("Temporary file cleanup completed")
        except Exception as e:
            logger.error(f"Error during temporary file cleanup: {e}")
