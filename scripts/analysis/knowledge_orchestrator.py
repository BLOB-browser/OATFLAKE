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
        self.force_url_fetch = False  # Default to not forcing URL fetching
        
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
                               process_level: Optional[int] = None) -> Dict[str, Any]:
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
            for level in range(1, max_depth + 1):
                pending_urls = url_storage.get_pending_urls(depth=level)
                if pending_urls:
                    pending_urls_count += len(pending_urls)
            
            if pending_urls_count == 0 and force_url_fetch:
                logger.info("No pending URLs found, but force_url_fetch is enabled. Will trigger URL discovery on all resources.")
                # We'll process this later in the STEP 4 section
                pass
            
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
            
            # STEP 4: Process resources or URLs at specific level
            if not self.cancel_requested:
                # Import required components
                from scripts.analysis.single_resource_processor import SingleResourceProcessor
                from scripts.analysis.url_storage import URLStorageManager
                import pandas as pd
                
                # Initialize components
                single_processor = SingleResourceProcessor(self.data_folder)
                resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
                processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
                url_storage = URLStorageManager(processed_urls_file)
                
                # If force_url_fetch is enabled, set it in the SingleResourceProcessor
                if self.force_url_fetch:
                    logger.info(f"Setting force_fetch=True in SingleResourceProcessor")
                    single_processor.force_fetch = True
                
                # Check if we should process a specific level
                if process_level is not None and process_level > 0:
                    logger.info(f"STEP 4: PROCESSING ONLY LEVEL {process_level} URLS")
                else:
                    logger.info(f"STEP 4: PROCESSING ALL RESOURCES DIRECTLY WITH SINGLE RESOURCE PROCESSOR")
                
                # Initialize results
                processed_resources = 0
                processed_urls_count = 0
                skipped_resources = 0
                success_count = 0
                error_count = 0
                
                try:
                    # Load resources from CSV
                    if os.path.exists(resources_csv_path):
                        resources_df = pd.read_csv(resources_csv_path)
                        
                        # Only process resources that have a URL
                        resources_with_url = resources_df[resources_df['url'].notna()]
                        
                        # Different processing mode based on process_level parameter
                        if process_level is not None and process_level > 0:
                            # Process URLs at a specific level
                            logger.info(f"Retrieving pending URLs at level {process_level}")
                            pending_urls = url_storage.get_pending_urls(depth=process_level)
                            
                            if pending_urls:
                                logger.info(f"Found {len(pending_urls)} pending URLs at level {process_level}")
                                
                                # Group URLs by their origin (the URL that led to them)
                                urls_by_origin = {}
                                for pending_url_data in pending_urls:
                                    url = pending_url_data.get('url')
                                    origin = pending_url_data.get('origin', '')
                                    
                                    if origin not in urls_by_origin:
                                        urls_by_origin[origin] = []
                                    
                                    urls_by_origin[origin].append(pending_url_data)
                                
                                logger.info(f"URLs grouped by {len(urls_by_origin)} different origin URLs")
                                
                                # Process URLs for each origin
                                for origin_url, urls_for_origin in urls_by_origin.items():
                                    # Find the resource that this origin URL belongs to
                                    resource_for_origin = None
                                    for _, row in resources_with_url.iterrows():
                                        if row['url'] == origin_url:
                                            resource_for_origin = row.to_dict()
                                            break
                                            
                                    if resource_for_origin:
                                        logger.info(f"Processing {len(urls_for_origin)} URLs at level {process_level} from origin: {origin_url}")
                                        
                                        # Process each URL using the specific_url_processor in SingleResourceProcessor
                                        for pending_url_data in urls_for_origin:
                                            url_to_process = pending_url_data.get('url')
                                            
                                            # Skip if already processed and not forcing reprocess
                                            if url_storage.url_is_processed(url_to_process) and not self.force_url_fetch:
                                                logger.info(f"Skipping already processed URL: {url_to_process}")
                                                continue
                                                
                                            # Process this URL
                                            url_result = single_processor.process_specific_url(
                                                url=url_to_process,
                                                origin_url=origin_url,
                                                resource=resource_for_origin,
                                                depth=process_level
                                            )
                                            
                                            processed_urls_count += 1
                                            
                                            if url_result.get('success', False):
                                                success_count += 1
                                            else:
                                                error_count += 1
                                                logger.error(f"Error processing URL at level {process_level}: {url_to_process}")
                                            
                                            # Show progress
                                            if processed_urls_count % 10 == 0:
                                                logger.info(f"Processed {processed_urls_count}/{len(pending_urls)} URLs at level {process_level}")
                                    else:
                                        logger.warning(f"Could not find resource for origin URL: {origin_url}")
                                
                                logger.info(f"Level {process_level} processing complete: {processed_urls_count} URLs processed, {success_count} successful, {error_count} errors")
                            else:
                                logger.info(f"No pending URLs found at level {process_level}")
                                
                        else:
                            # Process all resources normally (starting with level 0)
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
                                if resource.get('analysis_completed', False) and not self.force_url_fetch:
                                    logger.info(f"[{resource_id}] Skipping already processed resource: {resource.get('title', 'Unnamed')}")
                                    skipped_resources += 1
                                    continue
                                    
                                logger.info(f"[{resource_id}] Processing resource: {resource.get('title', 'Unnamed')} - {resource_url}")
                                
                                # Process the resource
                                resource_result = single_processor.process_resource(
                                    resource=resource,
                                    resource_id=resource_id,
                                    idx=idx,
                                    csv_path=resources_csv_path,
                                    max_depth=max_depth  # Pass max_depth parameter through
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
                                    
                            logger.info(f"Resource processing complete: {processed_resources} processed, {skipped_resources} skipped, {success_count} successful, {error_count} errors")
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
            
            logger.info(f"Knowledge base processing completed successfully")
            self.processing_active = False
            
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
        
        # Initialize components
        from scripts.analysis.single_resource_processor import SingleResourceProcessor
        from scripts.analysis.url_storage import URLStorageManager
        import pandas as pd
        import os
        
        single_processor = SingleResourceProcessor(self.data_folder)
        resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
        processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)
        
        # Set force_fetch if enabled
        if force_fetch:
            logger.info(f"Setting force_fetch=True in SingleResourceProcessor for level {level}")
            single_processor.force_fetch = True
        
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
                        
                        # Store the original force_fetch value
                        original_force_fetch = single_processor.force_fetch
                        # Enable force_fetch for this operation
                        single_processor.force_fetch = True
                        
                        # Process the resource to discover URLs
                        result = single_processor.process_resource(
                            resource=resource,
                            resource_id=f"{idx+1}/{len(resources_with_url)}",
                            idx=idx,
                            csv_path=resources_csv_path,
                            max_depth=4,  # Use maximum depth for discovery
                            process_by_level=True  # Ensure we use level-based processing for proper URL discovery
                        )
                        
                        # Restore original force_fetch setting
                        single_processor.force_fetch = original_force_fetch
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
        Process knowledge base in separate phases:
        1. Discovery phase - Find all URLs from all resources
        2. Analysis phase - Process content from discovered URLs
        
        Args:
            request_app_state: The FastAPI request.app.state
            max_depth: Maximum depth for crawling
            force_url_fetch: If True, forces refetching of URLs even if already processed
            batch_size: Number of URLs to process in each batch
        
        Returns:
            Dictionary with processing results
        """
        logger.info("Starting knowledge processing in phases (discovery then analysis)")
        
        # Set processing flags
        self.processing_active = True
        self.cancel_requested = False
        self.force_url_fetch = force_url_fetch
        
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
            
            # PHASE 1.5: Check if URL discovery is actually needed
            check_result = content_fetcher.check_discovery_needed(max_depth=max_depth)
            result["discovery_check"] = check_result
            
            # PHASE 2: URL Discovery - Process all resources in discovery-only mode
            # Only run discovery if needed or if force_url_fetch is True
            if check_result["discovery_needed"] or force_url_fetch:
                logger.info(f"Starting URL discovery phase for {len(resources_with_url)} resources")
                
                # Get all URLs from resources
                all_urls = resources_with_url['url'].tolist()
                
                # Run discovery phase
                discovery_result = content_fetcher.discovery_phase(
                    urls=all_urls,
                    max_depth=max_depth,
                    force_reprocess=force_url_fetch
                )
                
                result["discovery_phase"] = discovery_result
                logger.info(f"Discovery phase completed: {discovery_result['total_discovered']} URLs discovered")
            else:
                logger.info("Skipping URL discovery phase - sufficient pending URLs already exist")
                result["discovery_phase"] = {
                    "status": "skipped",
                    "pending_urls": check_result["total_pending"],
                    "pending_by_level": check_result["pending_by_level"]
                }
            
            # Check for cancellation
            if self.cancel_requested:
                logger.info("Knowledge processing cancelled after discovery phase")
                self.processing_active = False
                return {
                    "status": "cancelled",
                    "message": "Knowledge processing cancelled after discovery phase",
                    "data": result
                }
                
            # PHASE 3: Analysis Phase - Process discovered URLs level by level
            logger.info("Starting analysis phase")
            
            # Get the URL storage manager
            from scripts.analysis.url_storage import URLStorageManager
            processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
            url_storage = URLStorageManager(processed_urls_file)
            
            # Process each level sequentially
            analysis_results = {
                "processed_by_level": {},
                "total_processed": 0,
                "successful": 0,
                "failed": 0
            }
            
            # Process each level
            for level in range(1, max_depth + 1):
                pending_urls = url_storage.get_pending_urls(depth=level)
                
                if not pending_urls:
                    logger.info(f"No pending URLs at level {level}")
                    analysis_results["processed_by_level"][level] = 0
                    continue
                    
                logger.info(f"Processing {len(pending_urls)} URLs at level {level}")
                
                # Process this level (using the existing process_urls_at_level method)
                level_result = self.process_urls_at_level(
                    level=level,
                    batch_size=batch_size,
                    force_fetch=force_url_fetch
                )
                
                # Record results
                processed = level_result.get('processed_urls', 0)
                successful = level_result.get('success_count', 0)
                failed = level_result.get('error_count', 0)
                
                analysis_results["processed_by_level"][level] = processed
                analysis_results["total_processed"] += processed
                analysis_results["successful"] += successful
                analysis_results["failed"] += failed
                
                # Check for cancellation
                if self.cancel_requested:
                    logger.info(f"Knowledge processing cancelled during level {level} analysis")
                    break
            
            result["analysis_phase"] = analysis_results
            
            # PHASE 4: Generate vector stores if not cancelled
            if not self.cancel_requested:
                from scripts.analysis.vector_store_generator import VectorStoreGenerator
                vector_generator = VectorStoreGenerator(self.data_folder)
                vector_result = await vector_generator.generate_vector_stores(force_update=True)
                result["vector_generation"] = vector_result
                
                # Also clean up temporary files
                vector_generator.cleanup_temporary_files()
            else:
                result["vector_generation"] = {"status": "cancelled"}
            
            logger.info("Knowledge processing in phases completed successfully")
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