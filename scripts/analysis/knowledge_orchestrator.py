#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class KnowledgeOrchestrator:
    """
    Orchestrates the entire knowledge processing workflow by coordinating all steps.
    This is the main entry point for the knowledge processing pipeline.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the knowledge orchestrator.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        self.processing_active = False
        self.cancel_requested = False
        # Removed force_url_fetch flag  # Default to not forcing URL fetching
        
    def cancel_processing(self) -> Dict[str, Any]:
        """
        Cancel any active processing.
        
        Returns:
            Status dictionary
        """
        if not self.processing_active:
            return {
                "status": "success",
                "message": "No active processing to cancel"
            }
        
        self.cancel_requested = True
        logger.info("Cancellation requested for knowledge processing")
        
        # Also cancel processing in underlying components
        from scripts.analysis.interruptible_llm import request_interrupt
        request_interrupt()
        
        return {
            "status": "success",
            "message": "Cancellation request sent"
        }
    
    async def process_knowledge(self, 
                               request_app_state=None,
                               skip_markdown_scraping: bool = True,
                               analyze_resources: bool = True,
                               analyze_all_resources: bool = False,
                               batch_size: int = 5,
                               resource_limit: Optional[int] = None,
                               force_update: bool = False,
                               skip_vector_generation: bool = False,
                               check_unanalyzed: bool = True,
                               skip_questions: bool = False,
                               skip_goals: bool = False,
                               max_depth: int = 4,
                               force_url_fetch: bool = False,
                               process_level: Optional[int] = None,
                               auto_advance_level: bool = False,
                               continue_until_end: bool = False) -> Dict[str, Any]:
        """
        Process all knowledge base files and generate embeddings.
        
        Args:
            request_app_state: The FastAPI request.app.state (for ollama client)
            skip_markdown_scraping: If True, don't scrape web content from markdown links
            analyze_resources: If True, analyze resources with LLM
            analyze_all_resources: If True, analyze all resources even if already analyzed
            batch_size: Number of resources to process at once
            resource_limit: Maximum number of resources to process
            force_update: If True, forces a full update regardless of changes
            skip_vector_generation: If True, skip vector generation in this step
            check_unanalyzed: If True, always processes resources that haven't been analyzed yet
            skip_questions: If True, skip question generation step
            skip_goals: If True, skip goal extraction step
            max_depth: Maximum depth for crawling
            force_url_fetch: If True, forces refetching of URLs even if already processed
            process_level: If specified, only process URLs at this specific level
            auto_advance_level: If True, automatically advance to the next level after processing
            continue_until_end: If True, continue processing all levels until completion or cancellation
        
        Returns:
            Dictionary with processing results
        """
        # Set processing flags
        self.processing_active = True
        self.cancel_requested = False
        self.force_url_fetch = force_url_fetch  # Store the parameter in the class
        
        # Note: When no changes and no unanalyzed resources are detected, but also no pending URLs exist,
        # we'll automatically enable force_url_fetch temporarily to ensure URLs are processed again.
        # This ensures that even in the "no changes" scenario, URLs will still be refetched periodically.
        
        logger.info(f"Starting comprehensive knowledge base processing with force_update={force_update}, force_url_fetch={force_url_fetch}")
        
        try:
            # Initialize result object
            result = {
                "vector_generation": {},
                "content_processing": {"status": "skipped"}
            }
            
            # STEP 0: Check for changes to determine what processing is needed
            from scripts.analysis.change_detector import ChangeDetector
            change_detector = ChangeDetector(self.data_folder)
            change_result = change_detector.determine_processing_needs(force_update, check_unanalyzed)
            
            process_all_steps = change_result["process_all_steps"] 
            unanalyzed_resources_exist = change_result["unanalyzed_resources_exist"]
            
            # Update result with processing status
            result["content_processing"] = {"status": "pending" if process_all_steps else "skipped"}
            result["change_detection"] = change_result
            
            # Check for cancellation
            if self.cancel_requested:
                logger.info("Knowledge processing cancelled before starting")
                self.processing_active = False
                return {
                    "status": "cancelled",
                    "message": "Knowledge processing cancelled before starting",
                    "data": result
                }
            
            # Always check all resources for URL discovery before beginning analysis
            from scripts.analysis.url_storage import URLStorageManager
            processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
            url_storage = URLStorageManager(processed_urls_file)
            
            # Get counts of pending URLs by level
            pending_urls_count = 0
            pending_by_level = {}
            for level in range(1, max_depth + 1):
                pending_urls = url_storage.get_pending_urls(depth=level)
                pending_by_level[level] = len(pending_urls) if pending_urls else 0
                if pending_urls:
                    pending_urls_count += len(pending_urls)
            
            # Log the pending URL counts by level
            if pending_urls_count > 0:
                level_info = ", ".join([f"level {l}: {c}" for l, c in sorted(pending_by_level.items()) if c > 0])
                logger.info(f"Found {pending_urls_count} total pending URLs: {level_info}")
            else:
                logger.info("No pending URLs found in any level. Will trigger URL discovery on all resources.")
                # Force enable URL discovery when no pending URLs exist
                # This ensures we always have URLs to process even without explicit force_url_fetch
                self.force_url_fetch = True
            
            # Only process content if changes detected, force_update is True, or unanalyzed resources exist
            if process_all_steps or unanalyzed_resources_exist:
                # STEP 1: Process PDFs and Methods first (highest priority content)
                if self.cancel_requested:
                    return {"status": "cancelled", "message": "Processing cancelled", "data": result}
                
                from scripts.analysis.critical_content_processor import CriticalContentProcessor
                critical_processor = CriticalContentProcessor(self.data_folder)
                critical_content_result = await critical_processor.process_critical_content()
                result["critical_content"] = critical_content_result
                
                # STEP 2: Process markdown files to extract resources
                if self.cancel_requested:
                    return {"status": "cancelled", "message": "Processing cancelled", "data": result}
                
                from scripts.analysis.markdown_processor_step import MarkdownProcessingStep
                markdown_processor = MarkdownProcessingStep(self.data_folder)
                markdown_files = markdown_processor.get_markdown_files()
                
                if markdown_files:
                    # Use "default" for group_id since we don't store in group-specific folders
                    markdown_result = await markdown_processor.process_markdown_files(
                        skip_scraping=skip_markdown_scraping,
                        group_id="default"
                    )
                    result["markdown_processing"] = markdown_result
                else:
                    logger.info("No markdown files found to process")
                    result["markdown_processing"] = {"status": "skipped", "data_extracted": {}}
                
                # STEP 3: Analyze resources
                if self.cancel_requested:
                    return {"status": "cancelled", "message": "Processing cancelled", "data": result}
                
                if analyze_resources:
                    from scripts.analysis.resource_analyzer_step import ResourceAnalyzerStep
                    resource_analyzer = ResourceAnalyzerStep(self.data_folder)
                    resource_analysis_result = await resource_analyzer.analyze_resources(
                        analyze_all=analyze_all_resources,
                        batch_size=batch_size,
                        limit=resource_limit,
                        force_update=force_update
                    )
                    result["resource_analysis"] = resource_analysis_result
                else:
                    logger.info("Resource analysis disabled by parameter, skipping")
                    result["resource_analysis"] = {"status": "skipped", "reason": "analyze_resources=False"}
                
                # Only continue with other steps if full processing is needed
                if process_all_steps:
                    # STEP 4: Process remaining knowledge base documents
                    if self.cancel_requested:
                        return {"status": "cancelled", "message": "Processing cancelled", "data": result}
                    
                    from scripts.analysis.knowledge_base_processor import KnowledgeBaseProcessor
                    kb_processor = KnowledgeBaseProcessor(self.data_folder)
                    knowledge_result = await kb_processor.process_remaining_knowledge(force_update=force_update)
                    result.update(knowledge_result)
                
                # After processing all content, update result
                result["content_processing"] = {"status": "success"}
            
            # STEP 4: Process resources with clear separation of discovery and analysis phases
            if not self.cancel_requested:
                # Import required components
                from scripts.analysis.single_resource_processor import SingleResourceProcessor
                from scripts.analysis.url_storage import URLStorageManager
                from scripts.analysis.content_fetcher import ContentFetcher
                import pandas as pd
                
                # Initialize components
                single_processor = SingleResourceProcessor(self.data_folder)
                content_fetcher = ContentFetcher()
                resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
                processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
                url_storage = URLStorageManager(processed_urls_file)
                
                # If force_url_fetch is enabled, set it in the SingleResourceProcessor
                if self.force_url_fetch:
                    logger.info(f"Setting force_fetch=True in SingleResourceProcessor")
                    single_processor.force_fetch = True
                
                # Initialize results
                processed_resources = 0
                processed_urls_count = 0
                skipped_resources = 0
                success_count = 0
                error_count = 0
                discovered_urls = 0
                
                try:
                    # Load resources from CSV
                    if os.path.exists(resources_csv_path):
                        resources_df = pd.read_csv(resources_csv_path)
                        
                        # Only process resources that have a URL
                        resources_with_url = resources_df[resources_df['url'].notna()]
                        
                        # PHASE 1: URL DISCOVERY (only if needed)
                        # Check if we need to do discovery by checking pending URLs
                        pending_urls_by_level = {}
                        total_pending_urls = 0
                        
                        # Count pending URLs at each level
                        for level in range(1, max_depth + 1):
                            pending_urls = url_storage.get_pending_urls(depth=level)
                            pending_urls_by_level[level] = len(pending_urls)
                            total_pending_urls += len(pending_urls)
                        
                        # Count total pending URLs across all levels
                        total_pending_count = 0
                        pending_by_level = {}
                        
                        for level in range(1, max_depth + 1):
                            pending_urls = url_storage.get_pending_urls(depth=level)
                            count = len(pending_urls) if pending_urls else 0
                            pending_by_level[level] = count
                            total_pending_count += count
                        
                        # First check if any resources are not fully analyzed
                        unanalyzed_resources_exists = False
                        for _, row in resources_with_url.iterrows():
                            resource = row.to_dict()
                            if not resource.get('analysis_completed', False):
                                unanalyzed_resources_exists = True
                                break
                                
                        if not unanalyzed_resources_exists:
                            # All resources are already fully analyzed - nothing more to do
                            logger.info(f"STEP 4A: ALL DONE - All resources are fully analyzed at all 4 levels. Nothing more to process.")
                            # Skip both discovery and analysis phases
                            return {
                                "status": "success",
                                "message": "All resources are fully analyzed at all 4 levels",
                                "data": {"analysis_status": "complete"}
                            }
                
                        # Check if we have any pending URLs
                        if total_pending_count > 0:
                            # We have pending URLs, process them before doing any discovery
                            level_counts = ", ".join([f"level {l}: {c}" for l, c in pending_by_level.items() if c > 0])
                            logger.info(f"STEP 4A: SKIPPING DISCOVERY - Found {total_pending_count} pending URLs ({level_counts}), processing these first")
                            
                            # Skip discovery phase entirely and jump directly to URL processing
                            logger.info(f"Proceeding directly to analysis phase to process {total_pending_count} pending URLs")
                            # We'll skip the discovery section and go directly to processing
                            discovered_urls = 0
                        else:
                            # No pending URLs, run discovery to find new ones at deeper levels
                            logger.info(f"STEP 4A: DISCOVERY PHASE - No pending URLs found, running discovery for all resources up to level {max_depth}")
                            
                            # Process each resource for discovery only
                            for idx, row in resources_with_url.iterrows():
                                # Check for cancellation
                                if self.cancel_requested:
                                    logger.info("Discovery phase cancelled")
                                    break
                                    
                                resource = row.to_dict()
                                resource_url = resource.get('url', '')
                                resource_id = f"{idx+1}/{len(resources_with_url)}"
                                
                                logger.info(f"[{resource_id}] Discovering URLs for resource: {resource.get('title', 'Unnamed')} - {resource_url}")
                                
                                # IMPORTANT: This is STRICTLY discovery only - no analysis should happen
                                # We want to discover URLs at all levels but never analyze content
                                logger.info(f"[{resource_id}] RUNNING STRICTLY DISCOVERY-ONLY MODE - Finding new URLs at deeper levels")
                                
                                # CRITICAL: We must set force_reprocess=True to ensure we explore deeper URLs
                                # even when the main URL has been processed!
                                # We won't reanalyze processed URLs, but we'll use them to find deeper URLs
                                success, discovery_result = content_fetcher.fetch_content(
                                    url=resource_url, 
                                    max_depth=max_depth,
                                    discovery_only=True,  # This flag ensures we only do discovery
                                    force_reprocess=True  # We MUST force discovery to find deeper URLs
                                )
                                
                                if success:
                                    discovered_urls += 1
                            
                            # Recount pending URLs after discovery
                            pending_urls_by_level = {}
                            total_pending_urls = 0
                            for level in range(1, max_depth + 1):
                                pending_urls = url_storage.get_pending_urls(depth=level)
                                pending_urls_by_level[level] = len(pending_urls)
                                total_pending_urls += len(pending_urls)
                            
                            logger.info(f"Discovery phase complete: {discovered_urls} resources processed, found {total_pending_urls} pending URLs across all levels")
                            
                            # Log details about pending URLs by level after discovery
                            for level, count in pending_urls_by_level.items():
                                if count > 0:
                                    logger.info(f"  Level {level}: {count} pending URLs")
                        
                        # PHASE 2: CONTENT ANALYSIS
                        # Make sure discovery is fully complete before starting analysis
                        logger.info(f"===================================================")
                        logger.info(f"DISCOVERY PHASE COMPLETE - STARTING ANALYSIS PHASE")
                        logger.info(f"===================================================")
                        
                        # Check if we should process a specific level
                        if process_level is not None and process_level > 0:
                            logger.info(f"STEP 4B: ANALYSIS PHASE - PROCESSING ONLY LEVEL {process_level} URLS")
                            
                            # Use the enhanced URL processor for level-specific processing
                            from scripts.analysis.orchestration.url_processor import URLProcessor
                            url_processor = URLProcessor(self.data_folder)
                            
                            # Set the cancellation flag in the URL processor if needed
                            url_processor.cancel_requested = self.cancel_requested
                            
                            logger.info(f"Using URLProcessor to process URLs at level {process_level} with force_fetch={self.force_url_fetch}")
                            
                            # Process only the specified level
                            level_result = url_processor.process_urls_at_level(
                                level=process_level,
                                batch_size=batch_size,
                                force_fetch=self.force_url_fetch
                            )
                            
                            # Log the results
                            processed_urls_count = level_result.get('processed_urls', 0)
                            success_count = level_result.get('success_count', 0)
                            error_count = level_result.get('error_count', 0)
                            
                            logger.info(f"Level {process_level} processing complete: {processed_urls_count} URLs processed, {success_count} successful, {error_count} errors")
                                
                        else:
                            # Process level 0 (main resource URLs)
                            logger.info(f"STEP 4B: ANALYSIS PHASE - PROCESSING MAIN RESOURCE PAGES (LEVEL 0)")
                            
                            # Process each resource
                            for idx, row in resources_with_url.iterrows():
                                # Check for cancellation
                                if self.cancel_requested:
                                    logger.info("Resource processing cancelled")
                                    break
                                    
                                resource = row.to_dict()
                                resource_url = resource.get('url', '')
                                resource_id = f"{idx+1}/{len(resources_with_url)}"
                                
                                # Check if the resource has already been fully processed
                                if resource.get('analysis_completed', False):
                                    logger.info(f"[{resource_id}] Skipping already processed resource: {resource.get('title', 'Unnamed')}")
                                    skipped_resources += 1
                                    continue
                                
                                # Check if the main URL has already been processed in processed_urls.csv
                                if url_storage.url_is_processed(resource_url):
                                    logger.info(f"[{resource_id}] Main URL already in processed_urls.csv, skipping analysis: {resource_url}")
                                    skipped_resources += 1
                                    continue
                                    
                                logger.info(f"[{resource_id}] Processing resource: {resource.get('title', 'Unnamed')} - {resource_url}")
                                
                                # Process only the main resource page (level 0)
                                resource_result = single_processor.process_resource(
                                    resource=resource,
                                    resource_id=resource_id,
                                    idx=idx,
                                    csv_path=resources_csv_path,
                                    max_depth=0,  # Only process level 0
                                    process_by_level=True  # Enable level-based processing
                                )
                                
                                # Update counts
                                processed_resources += 1
                                
                                if resource_result.get('success', False):
                                    success_count += 1
                                else:
                                    error_count += 1
                                    logger.error(f"[{resource_id}] Error processing resource: {resource_result.get('error', 'Unknown error')}")
                                
                                # Check if vector generation is needed
                                if single_processor.vector_generation_needed:
                                    logger.info(f"[{resource_id}] Vector generation needed for this resource")
                                    
                            logger.info(f"Main resource processing complete: {processed_resources} processed, {skipped_resources} skipped, {success_count} successful, {error_count} errors")
                            
                            # Suggest to continue with level-based processing
                            logger.info(f"To continue processing deeper levels, run with process_level=1, then process_level=2, etc.")
                    else:
                        logger.warning(f"Resources file not found: {resources_csv_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing resources: {e}")
                    error_count += 1
                
                # After processing all resources directly, check if there are pending URLs at different levels
                from scripts.analysis.url_storage import URLStorageManager
                processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
                url_storage = URLStorageManager(processed_urls_file)
                
                # Get counts of pending URLs by level
                pending_by_level = {}
                for level in range(1, max_depth + 1):
                    pending_urls = url_storage.get_pending_urls(depth=level)
                    if pending_urls:
                        pending_by_level[level] = len(pending_urls)
                
                # Log the pending URL counts by level
                if pending_by_level:
                    level_info = ", ".join([f"level {l}: {c}" for l, c in sorted(pending_by_level.items())])
                    total_pending = sum(pending_by_level.values())
                    logger.info(f"Found {total_pending} total pending URLs: {level_info}")
                    
                    # Process URLs level by level, starting from the lowest level with pending URLs
                    next_level_to_process = None
                    for level in range(1, max_depth + 1):
                        if level in pending_by_level and pending_by_level[level] > 0:
                            next_level_to_process = level
                            break
                            
                    if next_level_to_process:
                        logger.info(f"Processing pending URLs at level {next_level_to_process}")
                        
                        # Use the enhanced URL processor to handle pending URLs more efficiently
                        from scripts.analysis.orchestration.url_processor import URLProcessor
                        url_processor = URLProcessor(self.data_folder)
                        
                        # Set the cancellation flag in the URL processor if needed
                        url_processor.cancel_requested = self.cancel_requested
                        
                        logger.info(f"Using URLProcessor to process URLs by level with force_fetch={self.force_url_fetch}")
                        analysis_results = url_processor.process_urls_by_levels(
                            max_depth=max_depth,
                            batch_size=batch_size,
                            force_fetch=self.force_url_fetch
                        )
                        
                        # Log the results
                        logger.info(f"URLProcessor results: {analysis_results}")
                        logger.info(f"Total URLs processed: {analysis_results.get('total_processed', 0)}")
                        logger.info(f"Successful: {analysis_results.get('successful', 0)}")
                        logger.info(f"Failed: {analysis_results.get('failed', 0)}")
                        
                        # Process results by level for detailed logging
                        for level, count in analysis_results.get("processed_by_level", {}).items():
                            if count > 0:
                                logger.info(f"Processed {count} URLs at level {level}")
                    else:
                        logger.info("No pending URLs found that need processing")
                else:
                    # No pending URLs found - check if we need to trigger URL discovery
                    # Calculate how many resources still need analysis
                    unanalyzed_resources_count = 0
                    if os.path.exists(resources_csv_path):
                        resources_df = pd.read_csv(resources_csv_path)
                        resources_with_url = resources_df[resources_df['url'].notna()]
                        unanalyzed_mask = ~resources_df['analysis_completed'].fillna(False).astype(bool)
                        unanalyzed_resources_with_url = resources_with_url[unanalyzed_mask]
                        unanalyzed_resources_count = len(unanalyzed_resources_with_url)
                    
                    logger.info(f"No pending URLs found at any level with {unanalyzed_resources_count} unanalyzed resources")
                    
                    if unanalyzed_resources_count > 0 or self.force_url_fetch:
                        # Temporarily enable force_fetch to discover new URLs even for already processed resources
                        original_force_fetch = self.force_url_fetch
                        self.force_url_fetch = True
                        logger.info("Temporarily enabling force_url_fetch to trigger URL discovery on all resources")
                        
                        # Process all resources to discover URLs at level 1
                        level_result = self.process_urls_at_level(level=1, force_fetch=True)
                        logger.info(f"URL discovery result: {level_result}")
                        
                        # Restore original force_fetch setting
                        self.force_url_fetch = original_force_fetch
                        
                        # Check again for pending URLs after discovery
                        for level in range(1, max_depth + 1):
                            pending_urls = url_storage.get_pending_urls(depth=level)
                            if pending_urls:
                                pending_by_level[level] = len(pending_urls)
                        
                        if pending_by_level:
                            level_info = ", ".join([f"level {l}: {c}" for l, c in sorted(pending_by_level.items())])
                            total_pending = sum(pending_by_level.values())
                            logger.info(f"After discovery, found {total_pending} total pending URLs: {level_info}")
                    else:
                        logger.info("No unanalyzed resources and no pending URLs - all processing complete")
                    
                    # Find the lowest level with pending URLs
                    next_level = min(pending_by_level.keys()) if pending_by_level else None
                    
                    if next_level and next_level > 1:
                        logger.info(f"All level 1 URLs processed, now processing level {next_level} URLs")
                        
                        # Process pending URLs at this level
                        processed_at_level = 0
                        pending_at_level = pending_by_level[next_level]
                        
                        # Get all pending URLs for this level
                        pending_urls = url_storage.get_pending_urls(depth=next_level)
                        
                        # Group URLs by their origin (the URL that led to them)
                        urls_by_origin = {}
                        for pending_url_data in pending_urls:
                            url = pending_url_data.get('url')
                            origin = pending_url_data.get('origin', '')
                            
                            if origin not in urls_by_origin:
                                urls_by_origin[origin] = []
                            
                            urls_by_origin[origin].append(pending_url_data)
                        
                        logger.info(f"Processing {pending_at_level} URLs at level {next_level} (from {len(urls_by_origin)} different origin URLs)")
                        
                        # Process URLs for each origin
                        for origin_url, urls_for_origin in urls_by_origin.items():
                            # Find the resource that this origin URL belongs to
                            resource_for_origin = None
                            for _, row in resources_with_url.iterrows():
                                if row['url'] == origin_url:
                                    resource_for_origin = row.to_dict()
                                    break
                                    
                            if resource_for_origin:
                                logger.info(f"Processing {len(urls_for_origin)} URLs at level {next_level} from origin: {origin_url}")
                                
                                # Process each URL using the specific_url_processor in SingleResourceProcessor
                                for pending_url_data in urls_for_origin:
                                    url_to_process = pending_url_data.get('url')
                                    
                                    # Process this URL
                                    try:
                                        url_result = single_processor.process_specific_url(
                                            url=url_to_process,
                                            origin_url=origin_url,
                                            resource=resource_for_origin,
                                            depth=next_level
                                        )
                                        
                                        processed_at_level += 1
                                        
                                        if url_result.get('success', False):
                                            success_count += 1
                                        else:
                                            error_count += 1
                                            logger.error(f"Error processing URL at level {next_level}: {url_to_process}")
                                    except Exception as url_error:
                                        processed_at_level += 1
                                        error_count += 1
                                        logger.error(f"Exception processing URL at level {next_level}: {url_to_process} - Error: {url_error}")
                                        # Continue to next URL even after an exception
                                        logger.error(f"Error processing URL at level {next_level}: {url_to_process}")
                                                     # Show progress
                                    if processed_at_level % 10 == 0:
                                        logger.info(f"Processed {processed_at_level}/{pending_at_level} URLs at level {next_level}")
                            else:
                                logger.warning(f"Could not find resource for origin URL: {origin_url}")
                        
                        logger.info(f"Completed processing level {next_level} URLs: {processed_at_level} processed, {success_count} successful, {error_count} errors")
                        
                        # Update counters
                        processed_resources += processed_at_level
                    else:
                        logger.info("No pending URLs found at any level - all levels have been processed")
                
                # Store processing results
                result["url_processing"] = {
                    "resources_processed": processed_resources,
                    "resources_skipped": skipped_resources,
                    "success_count": success_count,
                    "error_count": error_count,
                    "force_fetch_enabled": self.force_url_fetch,
                    "pending_urls_by_level": pending_by_level
                }
                
                # Reset force_url_fetch if it was temporarily enabled
                if self.force_url_fetch != force_url_fetch:
                    logger.info("Resetting force_url_fetch to original value")
                    self.force_url_fetch = force_url_fetch
            
            # STEP 5: Generate vector stores if not cancelled
            if not skip_vector_generation and not self.cancel_requested:
                from scripts.analysis.vector_store_generator import VectorStoreGenerator
                vector_generator = VectorStoreGenerator(self.data_folder)
                
                # Check for any remaining pending URLs before generating vectors
                remaining_pending_count = self._check_for_remaining_pending_urls(max_depth=max_depth)
                if remaining_pending_count > 0:
                    logger.warning(f"Skipping vector generation due to {remaining_pending_count} remaining pending URLs")
                    result["vector_generation"] = {"status": "skipped", "reason": "remaining_pending_urls"}
                else:
                    vector_result = await vector_generator.generate_vector_stores(force_update=force_update)
                    result["vector_generation"] = vector_result
                    
                    # Also clean up temporary files
                    vector_generator.cleanup_temporary_files()
            elif self.cancel_requested:
                logger.info("Vector generation skipped due to cancellation")
                result["vector_generation"] = {"status": "cancelled"}
            else:
                logger.info("Vector generation explicitly skipped")
                result["vector_generation"] = {"status": "skipped", "reason": "explicitly skipped"}
            
            # STEP 6: Extract goals from vector stores (unless skipped)
            if skip_goals:
                logger.info("STEP 6: SKIPPING GOAL EXTRACTION (skip_goals=True)")
                result["goals"] = {
                    "status": "skipped",
                    "message": "Goal extraction explicitly skipped"
                }
            else:
                from scripts.analysis.goal_extractor_step import GoalExtractorStep
                goal_extractor = GoalExtractorStep(self.data_folder)
                goals_result = await goal_extractor.extract_goals(
                    ollama_client=request_app_state.ollama_client if request_app_state else None
                )
                result["goals"] = goals_result
            
            # STEP 7: Generate questions from processed knowledge (unless skipped)
            if skip_questions:
                logger.info("STEP 7: SKIPPING QUESTION GENERATION (skip_questions=True)")
                result["questions"] = {
                    "status": "skipped",
                    "message": "Question generation explicitly skipped"
                }
            else:
                from scripts.analysis.question_generator_step import QuestionGeneratorStep
                question_generator = QuestionGeneratorStep(self.data_folder)
                questions_result = await question_generator.generate_questions()
                result["questions"] = questions_result
            
            # Save the file state with updated timestamps
            change_detector.save_file_state()
            
            # Include whether this was a force update
            result["force_update"] = force_update
            
            # Auto-advance level logic
            result["current_level"] = process_level
            
            # If auto_advance_level is enabled, determine the next level to process
            if auto_advance_level and process_level is not None:
                next_level = process_level + 1
                
                # If we've reached max_depth, go back to discovery (level 0)
                if next_level > max_depth:
                    next_level = 0
                    
                result["next_level"] = next_level
                logger.info(f"Auto-advancing level: Current level {process_level} -> Next level {next_level}")
                
                # Update the current_process_level in config.json
                try:
                    from scripts.services.training_scheduler import update_process_level
                    if update_process_level(next_level):
                        logger.info(f"Successfully updated current_process_level to {next_level} in config.json")
                    else:
                        logger.warning(f"Failed to update current_process_level in config.json")
                except Exception as e:
                    logger.error(f"Error updating current_process_level in config.json: {e}")
            
            # Reset processing flag temporarily
            self.processing_active = False
            
            # Check if we should continue processing with the next level
            if continue_until_end and auto_advance_level and process_level is not None and not self.cancel_requested:
                next_level = process_level + 1
                
                # If we've reached max_depth, go back to discovery (level 0)
                if next_level > max_depth:
                    next_level = 0
                
                # Reset if we're doing vector generation, questions, and goals this time
                # We want to do these steps only at the end of a full cycle
                do_vectors_questions_goals = next_level == 0
                
                logger.info(f"Continuing to next level: {next_level} (vectors/questions/goals: {do_vectors_questions_goals})")
                
                # Recursively call the process_knowledge method with the next level
                # We use the same parameters except for changing the level and possibly enabling vector generation
                recursive_result = await self.process_knowledge(
                    request_app_state=request_app_state,
                    skip_markdown_scraping=skip_markdown_scraping,
                    analyze_resources=analyze_resources,
                    analyze_all_resources=analyze_all_resources,
                    batch_size=batch_size,
                    resource_limit=resource_limit,
                    force_update=force_update,
                    skip_vector_generation=not do_vectors_questions_goals,  # Only generate vectors on cycle completion
                    check_unanalyzed=check_unanalyzed,
                    skip_questions=not do_vectors_questions_goals,  # Only generate questions on cycle completion
                    skip_goals=not do_vectors_questions_goals,  # Only extract goals on cycle completion
                    max_depth=max_depth,
                    force_url_fetch=force_url_fetch,
                    process_level=next_level,
                    auto_advance_level=auto_advance_level,
                    continue_until_end=continue_until_end
                )
                
                # Combine results from this level and subsequent levels
                recursive_result["data"]["previous_levels"] = recursive_result["data"].get("previous_levels", [])
                recursive_result["data"]["previous_levels"].append(result)
                
                return recursive_result
            
            logger.info(f"Knowledge base processing completed successfully")
            
            return {
                "status": "success",
                "message": "Knowledge base processed successfully",
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Error processing knowledge base: {e}", exc_info=True)
            self.processing_active = False
            return {
                "status": "error",
                "message": str(e),
                "data": {}
            }
    
    def check_url_processing_state(self) -> Dict[str, Any]:
        """
        Check the current state of resource processing to identify potential issues.
        
        Returns:
            Dictionary with diagnostic information
        """
        try:
            import pandas as pd
            import os
            
            # Get resources CSV path
            resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
            processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
            
            if not os.path.exists(resources_csv_path):
                return {
                    "status": "error",
                    "error": f"Resources file not found: {resources_csv_path}"
                }
                
            # Load resources
            resources_df = pd.read_csv(resources_csv_path)
            
            # Get URL statistics
            from scripts.analysis.url_storage import URLStorageManager
            url_storage = URLStorageManager(processed_urls_file)
            
            # Get pending URLs for all levels
            pending_urls_by_level = {}
            total_pending_urls = 0
            for level in range(1, 5):
                pending_urls = url_storage.get_pending_urls(depth=level)
                pending_urls_by_level[level] = len(pending_urls)
                total_pending_urls += len(pending_urls)
            
            # Get overall resource statistics
            total_resources = len(resources_df)
            resources_with_url = resources_df['url'].notna().sum()
            analyzed_resources = resources_df['analysis_completed'].fillna(False).sum()
            unanalyzed_resources = resources_with_url - analyzed_resources
            
            # Check if we need to trigger URL discovery
            # We need to discover URLs when there are no pending URLs but still unanalyzed resources
            discovery_needed = (total_pending_urls == 0) and (unanalyzed_resources > 0)
            
            # Log detailed information about the state
            logger.info(f"URL Processing State Check:")
            logger.info(f"  Total resources: {total_resources}")
            logger.info(f"  Resources with URL: {int(resources_with_url)}")
            logger.info(f"  Analyzed resources: {int(analyzed_resources)}")
            logger.info(f"  Unanalyzed resources: {int(unanalyzed_resources)}")
            logger.info(f"  Total pending URLs: {total_pending_urls}")
            
            # Log pending URLs by level
            for level, count in pending_urls_by_level.items():
                if count > 0:
                    logger.info(f"  Level {level}: {count} pending URLs")
            
            # Log discovery status
            if discovery_needed:
                logger.info(f"  Discovery needed: YES - No pending URLs but {unanalyzed_resources} unanalyzed resources")
            else:
                if total_pending_urls > 0:
                    logger.info(f"  Discovery needed: NO - {total_pending_urls} pending URLs already exist")
                else:
                    logger.info(f"  Discovery needed: NO - All resources are fully analyzed")
            
            result = {
                "total_resources": total_resources,
                "resources_with_url": int(resources_with_url),
                "analyzed_resources": int(analyzed_resources),
                "unanalyzed_resources": int(unanalyzed_resources),
                "total_pending_urls": total_pending_urls,
                "pending_urls_by_level": pending_urls_by_level,
                "discovery_needed": discovery_needed,
                "status": "success"
            }
            
            # If discovery is needed and there are resources with URLs, suggest triggering discovery
            if discovery_needed and resources_with_url > 0:
                result["recommendation"] = "No pending URLs found but there are unanalyzed resources. Consider triggering URL discovery by running with force_url_fetch=True"
                logger.info("RECOMMENDATION: Run with force_url_fetch=True to trigger URL discovery")
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking resource processing state: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def process_urls_at_level(self, level: int, batch_size: int = 50, force_fetch: bool = False) -> Dict[str, Any]:
        """
        Process all pending URLs at a specific level.
        
        Args:
            level: The depth level to process (1=first level, 2=second level, etc.)
            batch_size: Maximum number of URLs to process in one batch
            force_fetch: If True, force reprocessing of already processed URLs
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing all pending URLs at level {level}")
        logger.info(f"Force fetch enabled: {force_fetch}")
        
        # Initialize components
        from scripts.analysis.single_resource_processor import SingleResourceProcessor
        from scripts.analysis.url_storage import URLStorageManager
        import pandas as pd
        import os
        import csv
        
        single_processor = SingleResourceProcessor(self.data_folder)
        resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
        processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)
        
        # Set force_fetch if enabled - ensure it's propagated to the SingleResourceProcessor
        if force_fetch:
            logger.info(f"Setting force_fetch=True in SingleResourceProcessor for level {level}")
            single_processor.force_fetch = True
            # Warn if we're going to reprocess already processed URLs
            logger.warning("Force fetch enabled: Will process URLs even if already in processed_urls.csv")
        else:
            # Explicitly set to False to ensure consistent behavior
            single_processor.force_fetch = False
            logger.info("Force fetch disabled: Will skip URLs that are already in processed_urls.csv")
        
        # Get all pending URLs at this level
        pending_urls = url_storage.get_pending_urls(depth=level)
        
        if not pending_urls:
            logger.info(f"No pending URLs found at level {level}")
            
            # When there are no pending URLs at this level, we should trigger URL discovery
            # by processing all resources with force_fetch enabled to discover new URLs
            if level == 1:  # Only do this for level 1, as deeper levels depend on level 1 discoveries
                logger.info(f"No pending URLs at level {level}, triggering URL discovery on all resources")
                # Load resources to trigger discovery
                if os.path.exists(resources_csv_path):
                    resources_df = pd.read_csv(resources_csv_path)
                    
                    # Only process resources that have a URL
                    resources_with_url = resources_df[resources_df['url'].notna()]
                    
                    # Process every resource to trigger URL discovery
                    # We'll do this by forcing URL fetching on each resource
                    processed_resources = 0
                    for idx, row in resources_with_url.iterrows():
                        resource = row.to_dict()
                        resource_url = resource.get('url', '')
                        
                        if not resource_url:
                            continue
                            
                        logger.info(f"Triggering URL discovery for resource: {resource.get('title', 'Unnamed')} - {resource_url}")
                        
                        # Process the resource to discover URLs
                        result = single_processor.process_resource(
                            resource=resource,
                            resource_id=f"{idx+1}/{len(resources_with_url)}",
                            idx=idx,
                            csv_path=resources_csv_path,
                            max_depth=4,  # Use maximum depth for discovery
                            process_by_level=True  # Ensure we use level-based processing for proper URL discovery
                        )
                        
                        processed_resources += 1
                    
                    # After discovery, check again for pending URLs
                    pending_urls = url_storage.get_pending_urls(depth=level)
                    
                    if not pending_urls:
                        logger.info(f"Still no pending URLs found at level {level} after discovery")
                        return {"status": "skipped", "reason": "no_pending_urls_after_discovery", "resources_processed": processed_resources}
                    else:
                        logger.info(f"Found {len(pending_urls)} pending URLs at level {level} after discovery")
                else:
                    logger.warning(f"Resources file not found: {resources_csv_path}")
                    return {"status": "error", "error": "resources_file_not_found"}
            else:
                return {"status": "skipped", "reason": "no_pending_urls"}
        
        # Initialize counters
        total_urls = len(pending_urls)
        processed_urls_count = 0
        success_count = 0
        error_count = 0
        
        try:
            # Load resources
            if not os.path.exists(resources_csv_path):
                logger.error(f"Resources file not found: {resources_csv_path}")
                return {"status": "error", "error": "resources_file_not_found"}
                
            resources_df = pd.read_csv(resources_csv_path)
            resources_with_url = resources_df[resources_df['url'].notna()]
            
            # Group URLs by origin
            urls_by_origin = {}
            for pending_url_data in pending_urls:
                url = pending_url_data.get('url')
                origin = pending_url_data.get('origin', '')
                
                if origin not in urls_by_origin:
                    urls_by_origin[origin] = []
                
                urls_by_origin[origin].append(pending_url_data)
            
            logger.info(f"Processing {total_urls} URLs at level {level} from {len(urls_by_origin)} origin URLs")
            
            # Process URLs in batches, grouping by origin
            for origin_url, urls_for_origin in urls_by_origin.items():
                # Check for cancellation
                if self.cancel_requested:
                    logger.info(f"URL processing at level {level} cancelled")
                    break
                
                # Find the resource this origin URL belongs to
                resource_for_origin = None
                for _, row in resources_with_url.iterrows():
                    if row['url'] == origin_url:
                        resource_for_origin = row.to_dict()
                        break
                
                if resource_for_origin:
                    logger.info(f"Processing {len(urls_for_origin)} URLs at level {level} from origin: {origin_url}")
                    origin_resource_title = resource_for_origin.get('title', 'Unnamed')
                    
                    # Process each URL using the specific_url_processor
                    for idx, pending_url_data in enumerate(urls_for_origin):
                        # Check for cancellation
                        if self.cancel_requested:
                            logger.info(f"URL processing at level {level} cancelled")
                            break
                        
                        url_to_process = pending_url_data.get('url')
                        
                        # Process this URL
                        try:
                            url_result = single_processor.process_specific_url(
                                url=url_to_process,
                                origin_url=origin_url,
                                resource=resource_for_origin,
                                depth=level
                            )
                            
                            processed_urls_count += 1
                            
                            if url_result.get('success', False):
                                success_count += 1
                            else:
                                error_count += 1
                                logger.error(f"Error processing URL at level {level}: {url_to_process}")
                        except Exception as url_error:
                            processed_urls_count += 1
                            error_count += 1
                            logger.error(f"Exception processing URL at level {level}: {url_to_process} - Error: {url_error}")
                            # Also remove from pending URLs to avoid getting stuck
                            try:
                                url_storage.remove_pending_url(url_to_process)
                                logger.info(f"Removed problematic URL from pending queue: {url_to_process}")
                            except Exception as rm_error:
                                logger.error(f"Error removing URL from pending queue: {rm_error}")
                            # Continue to next URL even after an exception
                        
                        # Show progress every 10 URLs
                        if processed_urls_count % 10 == 0 or processed_urls_count == total_urls:
                            logger.info(f"Progress: {processed_urls_count}/{total_urls} URLs ({int(processed_urls_count/total_urls*100)}%)")
                else:
                    logger.warning(f"Could not find resource for origin URL: {origin_url}")
                    # Skip these URLs since we can't find their origin resource
                    processed_urls_count += len(urls_for_origin)
                    error_count += len(urls_for_origin)
                    
            logger.info(f"Level {level} processing complete: {processed_urls_count} URLs processed, {success_count} successful, {error_count} errors")
            
            return {
                "status": "success",
                "level": level,
                "total_urls": total_urls,
                "processed_urls": processed_urls_count,
                "success_count": success_count,
                "error_count": error_count
            }
            
        except Exception as e:
            logger.error(f"Error processing URLs at level {level}: {e}", exc_info=True)
            return {
                "status": "error",
                "level": level,
                "error": str(e),
                "processed_urls": processed_urls_count,
                "success_count": success_count,
                "error_count": error_count
            }
    
    async def process_knowledge_in_phases(self, 
                                    request_app_state=None,
                                    max_depth: int = 4,
                                    force_url_fetch: bool = False,
                                    batch_size: int = 50) -> Dict[str, Any]:
        """
        Process knowledge base in strictly separated phases:
        1. Discovery phase - Find all URLs from all resources at all levels
        2. Analysis phase - Process content from discovered URLs
        
        Args:
            request_app_state: The FastAPI request.app.state
            max_depth: Maximum depth for crawling
            force_url_fetch: If True, forces refetching of URLs even if already processed
            batch_size: Number of URLs to process in each batch
        
        Returns:
            Dictionary with processing results
        """
        logger.info("==================================================")
        logger.info("STARTING KNOWLEDGE PROCESSING IN STRICT PHASES:")
        logger.info("1. DISCOVERY PHASE - Find all URLs at all levels")
        logger.info("2. ANALYSIS PHASE - Process URLs level by level")
        logger.info("==================================================")
        logger.info(f"Max depth: {max_depth}, Force URL fetch: {force_url_fetch}")
        
        # Set processing flags
        self.processing_active = True
        self.cancel_requested = False
        # Removed: self.force_url_fetch = force_url_fetch
        
        try:
            # Initialize result object
            result = {
                "discovery_phase": {},
                "analysis_phase": {},
                "vector_generation": {}
            }
            
            # PHASE 1: Get all resources with URLs
            import pandas as pd
            import os
            
            resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
            
            if not os.path.exists(resources_csv_path):
                logger.error(f"Resources file not found: {resources_csv_path}")
                return {
                    "status": "error",
                    "message": "Resources file not found",
                    "data": {}
                }
                
            # Load resources
            resources_df = pd.read_csv(resources_csv_path)
            resources_with_url = resources_df[resources_df['url'].notna()]
            
            if len(resources_with_url) == 0:
                logger.warning("No resources with URLs found")
                return {
                    "status": "warning",
                    "message": "No resources with URLs found",
                    "data": {}
                }
                
            # Create ContentFetcher for discovery phase
            from scripts.analysis.content_fetcher import ContentFetcher
            content_fetcher = ContentFetcher(timeout=120)  # Increased timeout for discovery
            
            # PHASE 1.5: Check for existing unprocessed URLs
            from scripts.analysis.url_storage import URLStorageManager
            processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
            url_storage = URLStorageManager(processed_urls_file)
            
            # Log summary of state before starting
            state_result = self.check_url_processing_state()
            
            # Add debugging to check processed vs pending URLs
            logger.warning("================================")
            logger.warning("CHECKING PROCESSED VS PENDING URLS")
            logger.warning("================================")
            
            # Check how many URLs are already in processed_urls.csv
            processed_count = len(url_storage.get_processed_urls())
            logger.warning(f"Found {processed_count} URLs in processed_urls.csv")
            
            # Check pending URLs directly from the file (not filtered)
            pending_csv_path = os.path.join(os.path.dirname(processed_urls_file), "pending_urls.csv") 
            if os.path.exists(pending_csv_path):
                try:
                    with open(pending_csv_path, 'r', encoding='utf-8') as f:
                        import csv
                        reader = csv.reader(f)
                        next(reader, None)  # Skip header
                        
                        # Count total lines and URLs already in processed
                        total_lines = 0
                        already_processed = 0
                        by_level = {}
                        for row in reader:
                            if len(row) >= 2:
                                url = row[0]
                                depth = int(row[1]) if row[1].isdigit() else 0
                                total_lines += 1
                                
                                # Track by level
                                if depth not in by_level:
                                    by_level[depth] = {"total": 0, "processed": 0}
                                by_level[depth]["total"] += 1
                                
                                # Check if in processed list
                                if url_storage.url_is_processed(url):
                                    already_processed += 1
                                    by_level[depth]["processed"] += 1
                        
                        logger.warning(f"Raw pending_urls.csv file has {total_lines} total URLs")
                        logger.warning(f"Of which {already_processed} are already in processed_urls.csv")
                        logger.warning(f"This means {total_lines - already_processed} should actually be processed")
                        
                        # Log by level
                        for level, stats in sorted(by_level.items()):
                            if stats["total"] > 0:
                                logger.warning(f"Level {level}: {stats['total']} total URLs, {stats['processed']} already processed")
                except Exception as e:
                    logger.error(f"Error checking pending_urls.csv: {e}")
            
            logger.warning("================================")
            
            # PHASE 2: URL DISCOVERY PHASE
            # ==========================
            logger.info("==================================================")
            logger.info("STARTING DISCOVERY PHASE")
            logger.info("==================================================")
            logger.info(f"Finding all URLs from {len(resources_with_url)} resources at all levels (1-{max_depth})")
            logger.info(f"This phase will ONLY discover URLs, NO content analysis will be performed")
            
            # Get all URLs from resources
            all_urls = resources_with_url['url'].tolist()
            
            # IMPORTANT: Set discovery_only_mode to True for this phase
            # This ensures we only discover URLs without analyzing them
            content_fetcher.discovery_only_mode = True
            
            # Run discovery phase (STRICTLY URL discovery only, no content analysis)
            discovery_result = content_fetcher.discovery_phase(
                urls=all_urls,
                max_depth=max_depth,
                force_reprocess=force_url_fetch
            )
            
            result["discovery_phase"] = discovery_result
            logger.info(f"Discovery phase completed: {discovery_result['total_discovered']} total URLs discovered across all levels")
            
            # Log number of URLs discovered at each level
            if "discovered_by_level" in discovery_result:
                logger.info("URLs discovered by level:")
                for level, count in sorted(discovery_result["discovered_by_level"].items()):
                    logger.info(f"  Level {level}: {count} URLs discovered")
                
            # DEBUGGING: Check for any URLs that might be stuck in processed_urls.csv without analysis
            logger.warning("================================")
            logger.warning("DEBUGGING: CHECKING FOR 'ZOMBIE' URLS")
            logger.warning("URLs that are in processed_urls.csv but might not have been analyzed")
            logger.warning("================================")
            
            # If force_fetch is set, we'll attempt to analyze ALL URLs even if they're in processed_urls.csv
            if force_url_fetch:
                logger.warning(f"Force fetch is enabled - will attempt to reprocess ALL URLs in pending_urls.csv regardless of processed status")
            
            # Check for cancellation
            if self.cancel_requested:
                logger.info("Knowledge processing cancelled after discovery phase")
                self.processing_active = False
                return {
                    "status": "cancelled",
                    "message": "Knowledge processing cancelled after discovery phase",
                    "data": result
                }
                
            # PHASE 3: ANALYSIS PHASE
            # ======================
            logger.info("==================================================")
            logger.info("STARTING ANALYSIS PHASE")
            logger.info("==================================================")
            logger.info("All URL discovery is now complete - starting content analysis")
            
            # Recheck pending URLs after discovery
            pending_by_level = {}
            total_pending_urls = 0
            for level in range(1, max_depth + 1):
                pending_urls = url_storage.get_pending_urls(depth=level)
                pending_by_level[level] = len(pending_urls) if pending_urls else 0
                total_pending_urls += pending_by_level[level]
                
            logger.info(f"Found {total_pending_urls} total pending URLs to analyze across all levels:")
            for level, count in sorted(pending_by_level.items()):
                if count > 0:
                    logger.info(f"  Level {level}: {count} pending URLs")
            
            # Process each level sequentially
            analysis_results = {
                "processed_by_level": {},
                "total_processed": 0,
                "successful": 0,
                "failed": 0
            }
            
            # Reset discovery_only_mode to False for analysis phase
            content_fetcher.discovery_only_mode = False
            
            # Use the URLProcessor to process URLs by level
            from scripts.analysis.orchestration.url_processor import URLProcessor
            url_processor = URLProcessor(self.data_folder)
            
            # Set the cancellation flag in the URL processor if needed
            url_processor.cancel_requested = self.cancel_requested
            
            # Process all URLs by level using the specialized processor
            analysis_results = url_processor.process_urls_by_levels(
                max_depth=max_depth,
                batch_size=batch_size,
                force_fetch=force_url_fetch
            )
            
            # Check if the processing was successful
            if analysis_results["total_processed"] == 0:
                logger.warning("URLProcessor didn't process any URLs. This might indicate an issue.")
            else:
                logger.info(f"URLProcessor successfully processed {analysis_results['total_processed']} URLs")
                logger.info(f"Successful: {analysis_results['successful']}, Failed: {analysis_results['failed']}")
                
                # Log detailed results by level
                for level, count in sorted(analysis_results.get("processed_by_level", {}).items()):
                    logger.info(f"  Level {level}: {count} URLs processed")
            
            # Store the analysis results in the overall result
            result["analysis_phase"] = analysis_results
            
            # Verify that analysis phase actually processed URLs before proceeding to vector generation
            # If no URLs were processed (zero total_processed), log a warning but continue
            if analysis_results["total_processed"] == 0:
                logger.warning("==================================================")
                logger.warning("WARNING: Analysis phase did not process any URLs!")
                logger.warning("This indicates one of two issues:")
                logger.warning("1. All pending URLs were already in processed_urls.csv")
                logger.warning("2. There were no valid pending URLs to process")
                logger.warning("==================================================")
                
                # Double-check for pending URLs that weren't processed
                remaining_pending_urls = 0
                for level in range(1, max_depth + 1):
                    level_pending = url_storage.get_pending_urls(depth=level)
                    if level_pending:
                        remaining_pending_urls += len(level_pending)
                
                if remaining_pending_urls > 0:
                    logger.warning(f"There are still {remaining_pending_urls} pending URLs that weren't processed!")
                    logger.warning("These URLs should be processed before vector generation.")
                    logger.warning("Attempting to force process these remaining URLs...")
                    
                    # Attempt to force process the remaining URLs
                    for level in range(1, max_depth + 1):
                        pending_urls = url_storage.get_pending_urls(depth=level)
                        if pending_urls:
                            logger.warning(f"Force processing {len(pending_urls)} remaining URLs at level {level}")
                            # Set force_fetch to True to ensure processing
                            level_result = self.process_urls_at_level(
                                level=level,
                                batch_size=batch_size,
                                force_fetch=True  # Force processing
                            )
                            
                            # Log results of forced processing
                            logger.warning(f"Forced processing at level {level} results: {level_result}")
                            
                            # Update our analysis results
                            processed = level_result.get('processed_urls', 0) 
                            analysis_results["processed_by_level"][level] = processed
                            analysis_results["total_processed"] += processed
                            analysis_results["successful"] += level_result.get('success_count', 0)
                            analysis_results["failed"] += level_result.get('error_count', 0)
                else:
                    # Check pending_urls.csv directly for entries that might be already processed
                    pending_csv_path = os.path.join(os.path.dirname(processed_urls_file), "pending_urls.csv")
                    total_entries = 0
                    already_processed = 0
                    
                    if os.path.exists(pending_csv_path):
                        try:
                            logger.warning("Checking raw pending_urls.csv entries against processed_urls.csv")
                            with open(pending_csv_path, 'r', encoding='utf-8') as f:
                                reader = csv.reader(f)
                                next(reader, None)  # Skip header
                                for row in reader:
                                    if len(row) >= 1:
                                        url = row[0]
                                        total_entries += 1
                                        if url_storage.url_is_processed(url):
                                            already_processed += 1
                            
                            logger.warning(f"Found {total_entries} entries in pending_urls.csv, with {already_processed} already in processed_urls.csv")
                            logger.warning(f"This means all URLs in pending_urls.csv have been processed already")
                        except Exception as e:
                            logger.error(f"Error checking pending_urls.csv: {e}")
                    
                    # If all pending URLs have already been processed, let's proceed with discovery again
                    if already_processed == total_entries and total_entries > 0:
                        logger.warning("Since all pending URLs have already been processed, attempting to discover NEW URLs...")
                        
                        # Run discovery with force_reprocess=True to find new URLs
                        try:
                            # This should be a fresh discovery to identify new URLs
                            all_urls = resources_with_url['url'].tolist()
                            content_fetcher.discovery_only_mode = True
                            
                            # Force discover from a subset of main URLs
                            if len(all_urls) > 3:
                                urls_to_rediscover = all_urls[:3]  # Only check first 3 URLs
                                logger.warning(f"Running forced discovery on {len(urls_to_rediscover)} main URLs to find new URLs")
                            else:
                                urls_to_rediscover = all_urls
                            
                            discovery_result = content_fetcher.discovery_phase(
                                urls=urls_to_rediscover,
                                max_depth=max_depth,
                                force_reprocess=True  # Force rediscovery
                            )
                            
                            logger.warning(f"Forced discovery results: {discovery_result}")
                            
                            # Reset discovery mode
                            content_fetcher.discovery_only_mode = False
                            
                            # Check again for new pending URLs
                            new_pending_count = 0
                            for level in range(1, max_depth + 1):
                                level_pending = url_storage.get_pending_urls(depth=level)
                                if level_pending:
                                    new_pending_count += len(level_pending)
                            
                            if new_pending_count > 0:
                                logger.warning(f"Found {new_pending_count} new pending URLs after forced discovery!")
                                logger.warning("Attempting to process these new URLs...")
                                
                                # Process these new URLs
                                for level in range(1, max_depth + 1):
                                    pending_urls = url_storage.get_pending_urls(depth=level)
                                    if pending_urls:
                                        logger.warning(f"Processing {len(pending_urls)} new URLs at level {level}")
                                        level_result = self.process_urls_at_level(
                                            level=level,
                                            batch_size=batch_size,
                                            force_fetch=False  # Don't force processing
                                        )
                                        
                                        # Update analysis results
                                        processed = level_result.get('processed_urls', 0)
                                        analysis_results["processed_by_level"][level] = processed
                                        analysis_results["total_processed"] += processed
                                        analysis_results["successful"] += level_result.get('success_count', 0)
                                        analysis_results["failed"] += level_result.get('error_count', 0)
                            else:
                                logger.warning("No new pending URLs found after forced discovery. All URLs have been processed.")
                                
                        except Exception as e:
                            logger.error(f"Error during forced discovery: {e}")
                    else:
                        logger.warning("No pending URLs found in either filtered list or raw file. All URLs have been processed.")
            
            # Check for any remaining pending URLs before vector generation
            remaining_pending = self._check_for_remaining_pending_urls(max_depth)
            
            if remaining_pending > 0:
                logger.warning(f"There are still {remaining_pending} pending URLs that need processing!")
                logger.warning("Processing remaining URLs before generating vectors...")
                
                # Process these remaining URLs one level at a time
                for level in range(1, max_depth + 1):
                    retry_result = self.process_urls_at_level(
                        level=level,
                        batch_size=batch_size,
                        force_fetch=True
                    )
                    logger.info(f"Additional processing of level {level}: {retry_result}")
            
            # PHASE 4: VECTOR GENERATION PHASE
            # ===============================
            logger.info("==================================================")
            logger.info("STARTING VECTOR GENERATION PHASE")
            logger.info("==================================================")
            logger.info(f"Processed {analysis_results['total_processed']} URLs in analysis phase")
            
            if not self.cancel_requested:
                from scripts.analysis.vector_store_generator import VectorStoreGenerator
                vector_generator = VectorStoreGenerator(self.data_folder)
                vector_result = await vector_generator.generate_vector_stores(force_update=True)
                result["vector_generation"] = vector_result
                
                # Also clean up temporary files
                vector_generator.cleanup_temporary_files()
                
                logger.info("Vector generation completed successfully")
            else:
                result["vector_generation"] = {"status": "cancelled"}
                logger.info("Vector generation cancelled")
            
            logger.info("==================================================")
            logger.info("KNOWLEDGE PROCESSING COMPLETED SUCCESSFULLY")
            logger.info("==================================================")
            logger.info(f"Discovery phase: {discovery_result['total_discovered']} URLs discovered")
            logger.info(f"Analysis phase: {analysis_results['total_processed']} URLs processed, {analysis_results['successful']} successful, {analysis_results['failed']} errors")
            
            self.processing_active = False
            
            return {
                "status": "success",
                "message": "Knowledge processing completed in phases",
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Error in phased knowledge processing: {e}", exc_info=True)
            self.processing_active = False
            return {
                "status": "error",
                "message": str(e),
                "data": {}
            }
    
    def _check_for_remaining_pending_urls(self, max_depth: int = 4) -> int:
        """
        Check for any remaining pending URLs across all levels.
        
        Args:
            max_depth: Maximum depth level to check
            
        Returns:
            Total count of pending URLs found
        """
        from scripts.analysis.url_storage import URLStorageManager
        processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)
        
        total_pending = 0
        pending_by_level = {}
        
        # Check each level for pending URLs - use max_urls=0 to get ALL pending URLs
        for level in range(1, max_depth + 1):
            pending_urls = url_storage.get_pending_urls(max_urls=0, depth=level)
            if pending_urls:
                count = len(pending_urls)
                pending_by_level[level] = count
                total_pending += count
        
        if total_pending > 0:
            level_info = ", ".join([f"level {l}: {c}" for l, c in sorted(pending_by_level.items())])
            logger.warning(f"Found {total_pending} pending URLs remaining: {level_info}")
        
        return total_pending

# Standalone function for easy import
async def process_knowledge_base(data_folder=None, request_app_state=None, **kwargs):
    """
    Process all knowledge base files as a standalone function.
    
    Args:
        data_folder: Path to data folder (if None, gets from config)
        request_app_state: FastAPI request.app.state
        **kwargs: All other KnowledgeOrchestrator parameters
    
    Returns:
        Dictionary with processing results
    """
    # Get data folder from config if not provided
    if data_folder is None:
        from utils.config import get_data_path
        data_folder = get_data_path()
    
    # Initialize orchestrator
    orchestrator = KnowledgeOrchestrator(data_folder)
    
    # Process knowledge
    return await orchestrator.process_knowledge(request_app_state=request_app_state, **kwargs)