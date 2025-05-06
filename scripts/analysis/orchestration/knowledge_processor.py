#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import pandas as pd
from typing import Dict, Any, Optional

from scripts.analysis.orchestration.base_orchestrator import BaseOrchestrator
from scripts.analysis.url_storage import URLStorageManager
from scripts.analysis.single_resource_processor import SingleResourceProcessor
from scripts.analysis.change_detector import ChangeDetector
from scripts.analysis.vector_store_generator import VectorStoreGenerator

logger = logging.getLogger(__name__)

class KnowledgeProcessor(BaseOrchestrator):
    """
    Handles the main knowledge processing workflow.
    This is the core processing component coordinating different steps.
    """
    
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
                await self._process_resources_and_urls(
                    result=result,
                    process_level=process_level,
                    max_depth=max_depth,
                    force_url_fetch=force_url_fetch
                )
            
            # STEP 5: Generate vector stores if not cancelled
            if not skip_vector_generation and not self.cancel_requested:
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
    
    async def _process_resources_and_urls(self, 
                                        result: Dict[str, Any],
                                        process_level: Optional[int],
                                        max_depth: int,
                                        force_url_fetch: bool) -> None:
        """
        Process resources and URLs based on parameters.
        
        Args:
            result: Dictionary to store results in
            process_level: If specified, only process URLs at this level
            max_depth: Maximum crawling depth
            force_url_fetch: Whether to force URL fetching
        """
        # Import required components
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
                    await self._process_specific_level(
                        process_level=process_level,
                        url_storage=url_storage,
                        single_processor=single_processor,
                        resources_with_url=resources_with_url,
                        processed_urls_count=processed_urls_count,
                        success_count=success_count,
                        error_count=error_count
                    )
                else:
                    # Process all resources normally (starting with level 0)
                    await self._process_all_resources(
                        resources_with_url=resources_with_url,
                        single_processor=single_processor,
                        resources_csv_path=resources_csv_path,
                        max_depth=max_depth,
                        processed_resources=processed_resources,
                        skipped_resources=skipped_resources,
                        success_count=success_count,
                        error_count=error_count
                    )
            else:
                logger.warning(f"Resources file not found: {resources_csv_path}")
                
        except Exception as e:
            logger.error(f"Error processing resources: {e}")
            error_count += 1
        
        # After processing resources, process pending URLs at different levels if needed
        await self._process_pending_urls(
            url_storage=url_storage,
            single_processor=single_processor,
            resources_csv_path=resources_csv_path,
            resources_with_url=resources_df[resources_df['url'].notna()] if 'resources_df' in locals() else None,
            max_depth=max_depth,
            processed_resources=processed_resources, 
            success_count=success_count,
            error_count=error_count
        )
        
        # Store processing results
        result["url_processing"] = {
            "resources_processed": processed_resources,
            "resources_skipped": skipped_resources,
            "success_count": success_count,
            "error_count": error_count,
            "force_fetch_enabled": self.force_url_fetch
        }
        
        # Reset force_url_fetch if it was temporarily enabled
        if self.force_url_fetch != force_url_fetch:
            logger.info("Resetting force_url_fetch to original value")
            self.force_url_fetch = force_url_fetch
    
    async def _process_specific_level(self, 
                                    process_level: int,
                                    url_storage: URLStorageManager,
                                    single_processor: SingleResourceProcessor,
                                    resources_with_url: pd.DataFrame,
                                    processed_urls_count: int,
                                    success_count: int,
                                    error_count: int) -> Dict[str, int]:
        """Process URLs at a specific level."""
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
                        try:
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
                        except Exception as url_error:
                            processed_urls_count += 1
                            error_count += 1
                            logger.error(f"Exception processing URL at level {process_level}: {url_to_process} - Error: {url_error}")
                            # Also remove this URL from the pending queue to avoid getting stuck
                            try:
                                url_storage.remove_pending_url(url_to_process)
                                logger.info(f"Removed problematic URL from pending queue: {url_to_process}")
                            except Exception as rm_error:
                                logger.error(f"Error removing URL from pending queue: {rm_error}")
                            # Continue to next URL even after an exception
                        
                        # Show progress
                        if processed_urls_count % 10 == 0:
                            logger.info(f"Processed {processed_urls_count}/{len(pending_urls)} URLs at level {process_level}")
                else:
                    logger.warning(f"Could not find resource for origin URL: {origin_url}")
            
            logger.info(f"Level {process_level} processing complete: {processed_urls_count} URLs processed, {success_count} successful, {error_count} errors")
        else:
            logger.info(f"No pending URLs found at level {process_level}")
            
        return {
            "processed_urls_count": processed_urls_count,
            "success_count": success_count,
            "error_count": error_count
        }
    
    async def _process_all_resources(self,
                                   resources_with_url: pd.DataFrame,
                                   single_processor: SingleResourceProcessor,
                                   resources_csv_path: str,
                                   max_depth: int,
                                   processed_resources: int,
                                   skipped_resources: int,
                                   success_count: int,
                                   error_count: int) -> Dict[str, int]:
        """Process all resources starting from level 0."""
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
        
        return {
            "processed_resources": processed_resources,
            "skipped_resources": skipped_resources,
            "success_count": success_count,
            "error_count": error_count
        }
    
    async def _process_pending_urls(self,
                                  url_storage: URLStorageManager,
                                  single_processor: SingleResourceProcessor,
                                  resources_csv_path: str,
                                  resources_with_url: pd.DataFrame,
                                  max_depth: int,
                                  processed_resources: int,
                                  success_count: int,
                                  error_count: int) -> Dict[str, Any]:
        """Process any pending URLs at different levels."""
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
            if resources_with_url is not None:
                unanalyzed_mask = ~resources_with_url['analysis_completed'].fillna(False).astype(bool)
                unanalyzed_resources_with_url = resources_with_url[unanalyzed_mask]
                unanalyzed_resources_count = len(unanalyzed_resources_with_url)
            
            logger.info(f"No pending URLs found at any level with {unanalyzed_resources_count} unanalyzed resources")
            
            if unanalyzed_resources_count > 0 or self.force_url_fetch:
                # Use URLProcessor to process level 1 for discovery
                from scripts.analysis.orchestration.url_processor import URLProcessor
                url_processor = URLProcessor(self.data_folder)
                # Share cancellation state
                url_processor.cancel_requested = self.cancel_requested
                url_processor.processing_active = self.processing_active
                url_processor.force_url_fetch = True  # Force URL fetch for discovery
                
                # Process all resources to discover URLs at level 1
                level_result = url_processor.process_urls_at_level(level=1, force_fetch=True)
                logger.info(f"URL discovery result: {level_result}")
                
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
                # Use URLProcessor to process next level
                from scripts.analysis.orchestration.url_processor import URLProcessor
                url_processor = URLProcessor(self.data_folder)
                # Share cancellation state
                url_processor.cancel_requested = self.cancel_requested
                url_processor.processing_active = self.processing_active
                url_processor.force_url_fetch = self.force_url_fetch
                
                level_result = url_processor.process_urls_at_level(level=next_level, force_fetch=self.force_url_fetch)
                
                # Update counters based on level_result
                if level_result.get("status") == "success":
                    processed_resources += level_result.get("processed_urls", 0)
                    success_count += level_result.get("success_count", 0)
                    error_count += level_result.get("error_count", 0)
                
                logger.info(f"Level {next_level} processing complete via URL processor")
            else:
                logger.info("No pending URLs found at any level - all levels have been processed")
                
        return {
            "processed_resources": processed_resources,
            "success_count": success_count,
            "error_count": error_count,
            "pending_urls_by_level": pending_by_level
        }
