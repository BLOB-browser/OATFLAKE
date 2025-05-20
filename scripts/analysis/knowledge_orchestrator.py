#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
from scripts.analysis.url_discovery_manager import URLDiscoveryManager
from scripts.analysis.vector_generator import VectorGenerator

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
        
        # Ensure process_level is valid (must be >= 1)
        if process_level is not None and process_level < 1:
            logger.warning(f"Invalid process_level {process_level} specified. Level must be >= 1. Defaulting to level 1.")
            process_level = 1
        
        logger.info(f"Starting comprehensive knowledge base processing with force_update={force_update}")
        
        try:
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
            
            # STEP 1: GET PENDING URLS AND DETERMINE IF DISCOVERY IS NEEDED
            # This is the single decision point for URL discovery
            from scripts.analysis.url_storage import URLStorageManager
            processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
            url_storage = URLStorageManager(processed_urls_file)
            
            # Get counts of pending URLs by level
            pending_urls_count = 0
            pending_by_level = {}
            for level in range(1, max_depth + 1):  # Start from level 1, not 0
                pending_urls = url_storage.get_pending_urls(depth=level)
                pending_by_level[level] = len(pending_urls) if pending_urls else 0
                pending_urls_count += pending_by_level[level]
            
            if pending_urls_count > 0:
                level_info = ", ".join([f"level {l}: {c}" for l, c in sorted(pending_by_level.items()) if c > 0])
                logger.info(f"Found {pending_urls_count} total pending URLs: {level_info}")
            else:
                logger.info("No pending URLs found. Discovery phase needed.")
            
            # STEP 2: DISCOVERY PHASE - Only if needed (no pending URLs)
            # The simple check - if no pending URLs, run discovery; otherwise skip it
            if pending_urls_count == 0:
                logger.info("==================================================")
                logger.info("STARTING DISCOVERY PHASE")
                logger.info("==================================================")
                
                # Check if resources.csv exists before trying discovery
                resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
                if not os.path.exists(resources_csv_path):
                    logger.error(f"Resources file not found: {resources_csv_path}")
                    return {
                        "status": "error",
                        "message": f"Resources file not found: {resources_csv_path}",
                        "data": result
                    }
                else:
                    # Initialize URL discovery manager
                    url_discovery = URLDiscoveryManager(self.data_folder)
                    
                    # Get all resources with URLs
                    resources_df = pd.read_csv(resources_csv_path)
                    resources_with_url = resources_df[resources_df['url'].notna()]
                    
                    # Log discovery details
                    logger.info(f"Found {len(resources_with_url)} resources with URLs for discovery")
                    
                    # Trigger discovery using the direct method for better separation of concerns
                    discovery_result = url_discovery.discover_urls_from_resources_direct(
                        level=process_level or 1,
                        max_depth=max_depth
                    )
                    
                    # Log discovery results
                    if discovery_result.get("status") == "success":
                        logger.info(f"Discovery succeeded: Found {discovery_result.get('discovered_urls', 0)} pending URLs")
                    else:
                        logger.warning(f"Discovery status: {discovery_result.get('status')}, reason: {discovery_result.get('reason')}")
                    
                    # Add discovery results to the overall result
                    result["discovery_phase"] = discovery_result
                
                # After discovery, refresh our pending URLs count
                pending_urls_count = 0
                pending_by_level = {}
                for level in range(1, max_depth + 1):
                    pending_urls = url_storage.get_pending_urls(depth=level)
                    pending_by_level[level] = len(pending_urls) if pending_urls else 0
                    pending_urls_count += pending_by_level[level]
                
                logger.info(f"After discovery: found {pending_urls_count} total pending URLs")

            else:
                logger.info("Skipping discovery phase - found existing pending URLs")
                result["discovery_phase"] = {"status": "skipped", "reason": "existing_pending_urls"}
                
            # STEP 3: PROCESS CONTENT (PDFs, markdown, etc.)
            if process_all_steps or unanalyzed_resources_exist:
                logger.info("==================================================")
                logger.info("STARTING CONTENT PROCESSING PHASE")
                logger.info("==================================================")
                
                from scripts.analysis.critical_content_processor import CriticalContentProcessor
                content_processor = CriticalContentProcessor(self.data_folder)
                
                # Process content
                content_result = await content_processor.process_content(
                    request_app_state=request_app_state,
                    skip_markdown_scraping=skip_markdown_scraping,
                    analyze_resources=analyze_resources,
                    analyze_all_resources=analyze_all_resources,
                    batch_size=batch_size,
                    resource_limit=resource_limit,
                    check_unanalyzed=check_unanalyzed
                )
                
                # Update result with content processing results
                result["content_processing"] = content_result
            
            # STEP 4: URL ANALYSIS PHASE 
            # Now process URLs at the specified level or by priority
            if not self.cancel_requested and pending_urls_count > 0:
                # Determine which level to process
                current_level = process_level

                if current_level is None:
                    # If no level specified, determine the level with the most pending URLs
                    current_level = max(pending_by_level, key=pending_by_level.get) if pending_by_level else 1
                    logger.info(f"No level specified. Using level {current_level} which has the most pending URLs")
                
                logger.info("==================================================")
                logger.info(f"STARTING URL ANALYSIS PHASE AT LEVEL {current_level}")
                logger.info("==================================================")
                
                # Process URLs at the selected level
                url_analysis_result = self.process_urls_at_level(
                    level=current_level,
                    batch_size=batch_size
                )
                
                # Update result with URL analysis results
                result["url_analysis"] = url_analysis_result
                
            elif not self.cancel_requested:
                logger.info("Skipping URL analysis phase - no pending URLs found")
                result["url_analysis"] = {"status": "skipped", "reason": "no_pending_urls"}
            
            # STEP 5: VECTOR GENERATION
            if not skip_vector_generation and not self.cancel_requested:
                logger.info("==================================================")
                logger.info("STARTING VECTOR GENERATION PHASE")
                logger.info("==================================================")
                
                # Generate vectors
                vector_generator = VectorGenerator(self.data_folder)
                vector_result = await vector_generator.generate_vectors(
                    request_app_state=request_app_state,
                    generate_questions=not skip_questions
                )
                
                # Update result with vector generation results
                result["vector_generation"] = vector_result
                
            elif self.cancel_requested:
                logger.info("Skipping vector generation phase - processing cancelled")
                result["vector_generation"] = {"status": "skipped", "reason": "processing_cancelled"}
            else:
                logger.info("Skipping vector generation phase - skip_vector_generation=True")
                result["vector_generation"] = {"status": "skipped", "reason": "skip_vector_generation"}
            
            # STEP 6: GOAL EXTRACTION
            if skip_goals or self.cancel_requested:
                logger.info("Skipping goal extraction phase - skip_goals=True or cancelled")
                result["goal_extraction"] = {"status": "skipped", "reason": "skip_goals_or_cancelled"}
            else:
                logger.info("==================================================")
                logger.info("STARTING GOAL EXTRACTION PHASE")
                logger.info("==================================================")
                
                from scripts.analysis.goal_extractor import GoalExtractor
                goal_extractor = GoalExtractor(self.data_folder)
                
                # Extract goals
                goal_extraction_result = await goal_extractor.extract_goals(request_app_state=request_app_state)
                
                # Update result with goal extraction results
                result["goal_extraction"] = goal_extraction_result
            
            # STEP 7: QUESTION GENERATION
            if skip_questions or self.cancel_requested:
                logger.info("Skipping question generation phase - skip_questions=True or cancelled")
                result["question_generation"] = {"status": "skipped", "reason": "skip_questions_or_cancelled"}
            else:
                logger.info("==================================================")
                logger.info("STARTING QUESTION GENERATION PHASE")
                logger.info("==================================================")
                
                from scripts.analysis.question_generator_step import QuestionGeneratorStep
                question_generator = QuestionGeneratorStep(self.data_folder)
                
                # Generate questions
                question_generation_result = await question_generator.generate_questions(request_app_state=request_app_state)
                
                # Update result with question generation results
                result["question_generation"] = question_generation_result
            
            # Save the file state with updated timestamps
            change_detector.save_file_state()
            
            # Include whether this was a force update
            result["force_update"] = force_update
            
            # Auto-advance level logic
            result["current_level"] = process_level
            
            # If auto_advance_level is enabled, determine the next level to process
            if auto_advance_level and process_level is not None and not self.cancel_requested:
                # Advance to the next level
                advancement_result = self.advance_to_next_level(process_level, max_depth)
                result["level_advancement"] = advancement_result
                result["next_level"] = advancement_result.get("next_level")
                logger.info(f"Advanced to next level: {advancement_result.get('next_level')}")
            
            # Reset processing flag temporarily
            self.processing_active = False
            
            # Check if we should continue processing with the next level
            if continue_until_end and auto_advance_level and process_level is not None and not self.cancel_requested:
                # Get the next level from advancement result
                next_level = result.get("next_level")
                
                if next_level is not None:
                    logger.info(f"Continuing with next level {next_level} (continue_until_end=True)")
                    
                    # Process the next level (recursive call)
                    next_result = await self.process_knowledge(
                        request_app_state=request_app_state,
                        skip_markdown_scraping=skip_markdown_scraping,
                        analyze_resources=analyze_resources,
                        analyze_all_resources=analyze_all_resources,
                        batch_size=batch_size,
                        resource_limit=resource_limit,
                        force_update=False,  # Only force update once
                        skip_vector_generation=skip_vector_generation,
                        check_unanalyzed=check_unanalyzed,
                        skip_questions=skip_questions,
                        skip_goals=skip_goals,
                        max_depth=max_depth,
                        force_url_fetch=force_url_fetch,  # Keep the force_url_fetch setting for URL processor
                        process_level=next_level,
                        auto_advance_level=auto_advance_level,
                        continue_until_end=continue_until_end
                    )
                    
                    # Add next level result to the overall result
                    result["next_level_result"] = next_result
            
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
    
    def process_urls_at_level(self, level: int, batch_size: int = 50) -> Dict[str, Any]:
        """
        Process all pending URLs at a specific level.
        
        Args:
            level: The depth level to process (1=first level, 2=second level, etc.)
            batch_size: Maximum number of URLs to process in one batch
            
        Returns:
            Dictionary with processing results
        """
        if level < 1:
            logger.warning(f"Invalid level {level} specified. Level must be >= 1. Defaulting to level 1.")
            level = 1
            
        logger.info(f"Processing all pending URLs at level {level}")
        
        # Initialize components
        from scripts.analysis.single_resource_processor import SingleResourceProcessor
        from scripts.analysis.url_storage import URLStorageManager
        from scripts.analysis.url_discovery_manager import URLDiscoveryManager
        import pandas as pd
        import os
        import csv
        
        single_processor = SingleResourceProcessor(self.data_folder)
        resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
        processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)
        
        # Get all pending URLs at this level
        pending_urls = url_storage.get_pending_urls(depth=level)
        
        # If no pending URLs at this level, trigger URL discovery
        if not pending_urls:
            logger.info(f"No pending URLs found at level {level}, checking if URL discovery is needed")
            
            # Check if resources.csv exists for discovery
            if not os.path.exists(resources_csv_path):
                logger.error(f"Resources file not found: {resources_csv_path}")
                return {
                    "status": "error", 
                    "reason": "no_resources_file",
                    "message": f"Resources file not found: {resources_csv_path}"
                }
            
            # Load resources with URLs for discovery
            resources_df = pd.read_csv(resources_csv_path)
            resources_with_url = resources_df[resources_df['url'].notna()]
            
            if len(resources_with_url) > 0:
                logger.info(f"Found {len(resources_with_url)} resources with URLs for discovery")
                
                # Initialize URL discovery manager
                url_discovery = URLDiscoveryManager(self.data_folder)
                
                # Trigger discovery starting at the current level
                logger.info(f"Starting URL discovery at level {level}")
                discovery_result = url_discovery.discover_urls_from_resources(level=level, max_depth=level+1)
                
                # Log discovery results
                if discovery_result.get("status") == "success":
                    logger.info(f"Discovery succeeded: Found {discovery_result.get('discovered_urls', 0)} pending URLs")
                else:
                    logger.warning(f"Discovery status: {discovery_result.get('status')}, reason: {discovery_result.get('reason')}")
                
                # Check again for pending URLs after discovery
                pending_urls = url_storage.get_pending_urls(depth=level)
                
                if not pending_urls:
                    logger.warning(f"Still no pending URLs after discovery at level {level}")
                    return {
                        "status": "warning",
                        "level": level,
                        "reason": "no_pending_urls_after_discovery",
                        "discovery_result": discovery_result
                    }
                
                logger.info(f"After discovery: found {len(pending_urls)} pending URLs at level {level}")
            else:
                logger.warning(f"No resources with URLs found for discovery at level {level}")
                return {
                    "status": "warning", 
                    "reason": "no_resources_with_url",
                    "message": "No resources with URLs found for discovery"
                }
        
        # Initialize counters
        total_urls = len(pending_urls)
        processed_urls_count = 0
        success_count = 0
        error_count = 0
        skipped_count = 0
        failed_urls = []  # Track URLs that failed processing
        
        try:
            # Load resources CSV if it exists
            if not os.path.exists(resources_csv_path):
                logger.warning(f"Resources file not found: {resources_csv_path}")
                logger.info("Will still process URLs without resource context")
                resources_df = pd.DataFrame(columns=['url', 'title'])
                resources_with_url = resources_df
            else:
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
                
                # If we don't have resource context, create a minimal one
                if resource_for_origin is None:
                    resource_for_origin = {
                        'url': origin_url,
                        'title': f"Origin URL: {origin_url}"
                    }
                    logger.info(f"No resource found for origin URL: {origin_url}, using minimal context")
                    
                logger.info(f"Processing {len(urls_for_origin)} URLs at level {level} from origin: {origin_url}")
                origin_resource_title = resource_for_origin.get('title', 'Unnamed')
                
                # Process each URL using the specific_url_processor
                for idx, pending_url_data in enumerate(urls_for_origin):
                    # Check for cancellation
                    if self.cancel_requested:
                        logger.info(f"URL processing at level {level} cancelled")
                        break
                    
                    url_to_process = pending_url_data.get('url')
                    url_depth = pending_url_data.get('depth', level)
                    origin = pending_url_data.get('origin', '')
                    attempt_count = pending_url_data.get('attempt_count', 0)
                    
                    # Check if max attempts reached, but DON'T mark as processed if not analyzed
                    max_attempts = 3
                    if attempt_count >= max_attempts:
                        logger.warning(f"URL {url_to_process} has reached max attempts ({max_attempts}), skipping")
                        # Just remove from pending list but don't mark as processed
                        url_storage.remove_pending_url(url_to_process)
                        # Keep track of failed URLs
                        failed_urls.append(url_to_process)
                        error_count += 1
                        processed_urls_count += 1
                        continue
                    
                    # Process this URL
                    logger.info(f"Processing URL {idx + 1}/{len(urls_for_origin)}: {url_to_process}")
                    
                    try:
                        # Increment the attempt count
                        url_storage.increment_url_attempt(url_to_process)
                        
                        # Process the URL using SingleResourceProcessor's process_specific_url method
                        # This is the ANALYSIS phase - we're not discovering URLs anymore
                        logger.info(f"ANALYSIS PHASE: Processing URL {url_to_process} at level {url_depth} using SingleResourceProcessor")
                        success_result = single_processor.process_specific_url(
                            url=url_to_process,
                            origin_url=origin,
                            resource=resource_for_origin,
                            depth=url_depth
                        )
                        
                        success = success_result.get('success', False)
                        
                        if success:
                            logger.info(f"Successfully processed URL: {url_to_process}")
                            success_count += 1
                            # URL is already marked as processed by process_specific_url if successful
                            # It also removes the URL from pending queue
                        else:
                            logger.warning(f"Failed to process URL: {url_to_process}")
                            error_count += 1
                            # Keep in pending queue for retry unless max attempts reached
                            if attempt_count + 1 >= max_attempts:
                                logger.warning(f"URL {url_to_process} reached max attempts, removing from pending")
                                # Just remove from pending but don't mark as processed
                                url_storage.remove_pending_url(url_to_process)
                                failed_urls.append(url_to_process)
                            
                        processed_urls_count += 1
                        
                    except Exception as url_error:
                        logger.error(f"Error processing URL {url_to_process}: {url_error}")
                        error_count += 1
                        processed_urls_count += 1
                        # Add to failed URLs list
                        failed_urls.append(url_to_process)
                
        except Exception as e:
            logger.error(f"Error processing URLs at level {level}: {e}", exc_info=True)
            return {
                "status": "error",
                "level": level,
                "error": str(e),
                "processed_urls": processed_urls_count,
                "success_count": success_count,
                "error_count": error_count,
                "failed_urls": failed_urls
            }
                
        logger.info(f"Level {level} processing complete: {processed_urls_count} URLs processed, {success_count} successful, {error_count} errors")
        
        # Save failed URLs to a separate file for future reference
        if failed_urls:
            try:
                failed_urls_file = os.path.join(self.data_folder, f"failed_urls_level_{level}.csv")
                with open(failed_urls_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["url", "level", "timestamp"])
                    for url in failed_urls:
                        writer.writerow([url, level, datetime.now().isoformat()])
                logger.info(f"Saved {len(failed_urls)} failed URLs to {failed_urls_file}")
            except Exception as e:
                logger.error(f"Error saving failed URLs: {e}")
        
        return {
            "status": "success" if processed_urls_count > 0 else "warning",
            "level": level,
            "total_urls": total_urls,
            "processed_urls": processed_urls_count,
            "success_count": success_count,
            "error_count": error_count,
            "failed_urls_count": len(failed_urls),
            "message": "No URLs were processed" if processed_urls_count == 0 else None
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
        """
        logger.info("==================================================")
        logger.info("STARTING KNOWLEDGE PROCESSING IN STRICT PHASES:")
        logger.info("1. DISCOVERY PHASE - Find all URLs at all levels")
        logger.info("2. ANALYSIS PHASE - Process URLs level by level")
        logger.info("==================================================")
        logger.info(f"Max depth: {max_depth}")
        
        # Set processing flags
        self.processing_active = True
        self.cancel_requested = False
        
        try:
            # Initialize result object
            result = {
                "discovery_phase": {},
                "analysis_phase": {},
                "vector_generation": {}
            }
            
            # Get URL storage manager
            from scripts.analysis.url_storage import URLStorageManager
            processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
            url_storage = URLStorageManager(processed_urls_file)
            
            # Phase loop - we may need to repeat the process if discovery adds new URLs
            max_iterations = 3  # Limit the number of iterations to prevent infinite loops
            iteration = 0
            while iteration < max_iterations:
                iteration += 1
                logger.info(f"Starting iteration {iteration}/{max_iterations}")
                
                # ----------------------------------------------------------------
                # STEP 1: Check if there are already pending URLs we need to process
                # ----------------------------------------------------------------
                pending_by_level = {}
                total_pending_urls = 0
                
                for level in range(1, max_depth + 1):
                    pending_urls = url_storage.get_pending_urls(depth=level)
                    count = len(pending_urls)
                    pending_by_level[level] = count
                    total_pending_urls += count
                    logger.info(f"Found {count} pending URLs at level {level}")
                
                # ----------------------------------------------------------------
                # STEP 2: If no pending URLs, run discovery phase
                # ----------------------------------------------------------------
                if total_pending_urls == 0:
                    logger.info("==================================================")
                    logger.info("STARTING DISCOVERY PHASE")
                    logger.info("==================================================")
                    
                    # Check if resources.csv exists before trying discovery
                    resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
                    if not os.path.exists(resources_csv_path):
                        logger.error(f"Resources file not found: {resources_csv_path}")
                        logger.error("Cannot discover URLs without resources. Please add resources first.")
                        result["discovery_phase"] = {
                            "status": "error",
                            "message": "No resources.csv file found. Please add resources first."
                        }
                    else:
                        # Run discovery phase using URL Discovery Manager with direct method
                        discovery = URLDiscoveryManager(self.data_folder)
                        
                        # Call discover_urls_from_resources_direct with the correct parameters
                        # This uses our new direct method that doesn't rely on SingleResourceProcessor
                        logger.info(f"Starting direct URL discovery from resources with max_depth={max_depth}")
                        discovery_result = discovery.discover_urls_from_resources_direct(
                            level=1,  # Start with level 1
                            max_depth=max_depth
                        )
                        
                        result["discovery_phase"] = discovery_result
                        logger.info(f"URL discovery completed: {discovery_result}")
                    
                    # Re-check pending URLs after discovery
                    pending_by_level = {}
                    total_pending_urls = 0
                    for level in range(1, max_depth + 1):
                        pending_urls = url_storage.get_pending_urls(depth=level)
                        count = len(pending_urls)
                        pending_by_level[level] = count
                        total_pending_urls += count
                        logger.info(f"After discovery: Found {count} pending URLs at level {level}")
                
                # If there are still no pending URLs after discovery, we're done
                if total_pending_urls == 0:
                    logger.warning("No pending URLs found after discovery phase. Nothing to analyze.")
                    result["analysis_phase"] = {
                        "status": "skipped",
                        "reason": "No pending URLs found"
                    }
                    return {
                        "status": "success",
                        "message": "Process completed, but no URLs to analyze were found",
                        "data": result
                    }
                    
                # ----------------------------------------------------------------
                # STEP 3: Run analysis phase to process all pending URLs
                # ----------------------------------------------------------------
                logger.info("==================================================")
                logger.info("STARTING ANALYSIS PHASE")
                logger.info("==================================================")
                logger.info("Processing discovered URLs for content analysis")
                
                # Process each level sequentially
                analysis_results = {
                    "processed_by_level": {},
                    "total_processed": 0,
                    "successful": 0,
                    "failed": 0
                }
                
                # Process URLs at each level, starting from level 1
                for level in range(1, max_depth + 1):
                    # Check if we should continue to this level
                    if level > 1 and self.cancel_requested:
                        logger.info(f"Cancellation requested, stopping at level {level-1}")
                        break
                        
                    # Skip if no URLs at this level
                    if pending_by_level.get(level, 0) == 0:
                        logger.info(f"No pending URLs at level {level}, skipping")
                        analysis_results["processed_by_level"][level] = {
                            "total_urls": 0,
                            "processed": 0,
                            "success": 0,
                            "error": 0,
                            "skipped": 0
                        }
                        continue
                    
                    # Process URLs at this level using process_urls_at_level
                    level_result = self.process_urls_at_level(
                        level=level,
                        batch_size=batch_size
                    )
                    
                    # Update analysis results
                    analysis_results["processed_by_level"][level] = {
                        "total_urls": level_result.get("total_urls", 0),
                        "processed": level_result.get("processed_urls", 0),
                        "success": level_result.get("success_count", 0),
                        "error": level_result.get("error_count", 0),
                        "skipped": 0
                    }
                    
                    analysis_results["total_processed"] += level_result.get("processed_urls", 0)
                    analysis_results["successful"] += level_result.get("success_count", 0)
                    analysis_results["failed"] += level_result.get("error_count", 0)
                
                # Update result with analysis results for this iteration
                result["analysis_phase"] = analysis_results
                
                # ----------------------------------------------------------------
                # STEP 4: Check if there are still pending URLs
                # ----------------------------------------------------------------
                remaining_pending = self._check_for_remaining_pending_urls(max_depth)
                
                if remaining_pending == 0:
                    logger.info("No pending URLs remaining - all discovered URLs were successfully processed.")
                    logger.info(f"Breaking out of iteration loop after {iteration}/{max_iterations} iterations - all done!")
                    # Break out of the iteration loop - we're done!
                    break
                else:
                    # There are still pending URLs - if this is the last iteration, log a warning
                    if iteration >= max_iterations:
                        logger.warning(f"Maximum iterations ({max_iterations}) reached with {remaining_pending} pending URLs remaining.")
                        logger.warning("Some URLs may not have been processed. Consider running the process again.")
                    else:
                        logger.warning(f"Found {remaining_pending} pending URLs remaining after analysis phase.")
                        logger.warning(f"Will run another iteration ({iteration + 1}/{max_iterations}).")
                        # Continue to the next iteration automatically
            
            # Final update of pending URL count
            result["remaining_pending_urls"] = self._check_for_remaining_pending_urls(max_depth)
            
            # PHASE 5: Advance level in config if needed
            # =========================================
            if not self.cancel_requested:
                try:
                    # Get current crawl level from config
                    import json
                    config_file = os.path.join(os.path.dirname(self.data_folder), "config.json")
                    
                    if os.path.exists(config_file):
                        with open(config_file, 'r') as f:
                            config_data = json.load(f)
                            current_level = config_data.get("current_process_level", 1)
                            
                        logger.info(f"Current process level from config: {current_level}")
                        
                        # If current level URLs are all processed, advance to next level
                        level_pending = pending_by_level.get(current_level, 0)
                        if level_pending == 0 and current_level < max_depth:
                            next_level = current_level + 1
                            logger.info(f"All URLs at level {current_level} processed, advancing to level {next_level}")
                            
                            # Update config
                            config_data["current_process_level"] = next_level
                            with open(config_file, 'w') as f:
                                json.dump(config_data, f, indent=2)
                                
                            # Update level completion status
                            if "crawl_config" not in config_data:
                                config_data["crawl_config"] = {
                                    "current_max_level": next_level,
                                    "max_depth": max_depth,
                                    "level_completion": {}
                                }
                                
                            # Add or update level completion entry
                            level_key = f"level_{current_level}"
                            if "level_completion" not in config_data["crawl_config"]:
                                config_data["crawl_config"]["level_completion"] = {}
                                
                            config_data["crawl_config"]["level_completion"][level_key] = {
                                "is_complete": True,
                                "last_processed": datetime.now().isoformat(),
                                "urls_processed": analysis_results["processed_by_level"].get(current_level, {}).get("processed", 0)
                            }
                            
                            # Update config with new level completion status
                            with open(config_file, 'w') as f:
                                json.dump(config_data, f, indent=2)
                                
                            logger.info(f"Updated config: current_process_level={next_level}, marked level {current_level} as complete")
                            result["level_advanced"] = {
                                "previous_level": current_level,
                                "next_level": next_level
                            }
                except Exception as e:
                    logger.error(f"Error updating process level in config: {e}")
            
            logger.info("==================================================")
            logger.info("KNOWLEDGE PROCESSING COMPLETED")
            logger.info("==================================================")
            logger.info(f"Discovery phase: {result['discovery_phase'].get('total_discovered', 0)} URLs discovered")
            logger.info(f"Analysis phase: {result['analysis_phase']['total_processed']} URLs processed, "
                      f"{result['analysis_phase']['successful']} successful, "
                      f"{result['analysis_phase']['failed']} failed")
            
            if result["remaining_pending_urls"] > 0:
                logger.warning(f"Remaining pending URLs: {result['remaining_pending_urls']}")
                
            return {
                "status": "success",
                "message": "Knowledge processing completed successfully",
                "data": result
            }
            
        except Exception as e:
            logger.error(f"Error in process_knowledge_in_phases: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Error processing knowledge: {str(e)}",
                "data": result if 'result' in locals() else {}
            }
        finally:
            self.processing_active = False
    
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

    def advance_to_next_level(self, current_level: int, max_depth: int) -> Dict[str, Any]:
        """
        Advance to the next crawl level, marking all URLs from the current level as processed.
        This ensures that we properly move forward in the crawling process and don't
        get stuck processing the same level over and over.
        
        Args:
            current_level: The current crawl level
            max_depth: Maximum depth to crawl
            
        Returns:
            Dictionary with advancement results
        """
        logger.info(f"Advancing from level {current_level} to level {current_level + 1}")
        
        # Mark all pending URLs from the current level as processed
        from scripts.analysis.url_storage import URLStorageManager
        processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)
        
        # Make sure to correctly mark the current level as processed
        mark_result = url_storage.mark_level_as_processed(current_level)
        logger.info(f"Marked {mark_result.get('urls_processed', 0)} URLs at level {current_level} as processed")
        
        # Calculate next level
        next_level = current_level + 1
        
        # If we've reached max depth, reset to level 1 to restart the process
        if next_level > max_depth:
            next_level = 1
            logger.info(f"Reached maximum depth ({max_depth}), resetting to level {next_level}")
        
        # Update the config file with the new level
        try:
            from scripts.services.training_scheduler import update_process_level
            if update_process_level(next_level):
                logger.info(f"Updated current_process_level in config.json to {next_level}")
            else:
                logger.error(f"Failed to update current_process_level in config.json")
        except Exception as e:
            logger.error(f"Error updating current_process_level in config.json: {e}")
        
        # Update level completion status in config.json
        try:
            # Load the current config
            config_path = Path(__file__).parents[2] / "config.json"
            if config_path.exists():
                # Fix: Using self.force_url_fetch instead of undefined force_fetch
                with open(config_path, 'r') as f:
                    config_data = json.loads(f.read())
                    
                # Update the level completion data
                if 'crawl_config' in config_data and 'level_completion' in config_data['crawl_config']:
                    level_key = f"level_{current_level}"
                    if level_key in config_data['crawl_config']['level_completion']:
                        # Mark this level as completed
                        config_data['crawl_config']['level_completion'][level_key]['is_complete'] = True
                        config_data['crawl_config']['level_completion'][level_key]['last_processed'] = datetime.now().isoformat()
                        
                    # Update the current max level
                    config_data['crawl_config']['current_max_level'] = next_level
                    
                    # Write back the updated config
                    with open(config_path, 'w') as f:
                        json.dump(config_data, f, indent=2)
                        
                    logger.info(f"Updated level completion status for level {current_level} in config.json")
            else:
                logger.warning(f"Config file not found at {config_path}")
        except Exception as e:
            logger.error(f"Error updating level completion in config.json: {e}")
        
        return {
            "previous_level": current_level,
            "next_level": next_level,
            "urls_processed": mark_result.get('urls_processed', 0),
            "max_depth": max_depth
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