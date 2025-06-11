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
    def __init__(self, data_folder: str = None):
        """
        Initialize the knowledge orchestrator.
        
        Args:
            data_folder: Path to the data directory (if None, gets from config)
        """
        # Always use the configured data path instead of passed parameter
        from utils.config import get_data_path
        self.data_folder = get_data_path()
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
                               max_depth: int = 5,
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
            
            # CRITICAL FIX: Check URLs in discovery mode to see actual pending URLs
            # The analysis mode filters out processed URLs, but we need to see all URLs that need analysis
            url_storage.set_discovery_mode(True)
            discovery_pending_count = 0
            discovery_pending_by_level = {}
            for level in range(1, max_depth + 1):
                pending_urls = url_storage.get_pending_urls(depth=level)
                discovery_pending_by_level[level] = len(pending_urls) if pending_urls else 0
                discovery_pending_count += discovery_pending_by_level[level]
            
            # Also check analysis mode counts for comparison
            url_storage.set_discovery_mode(False)
            pending_urls_count = 0
            pending_by_level = {}
            for level in range(1, max_depth + 1):  # Start from level 1, not 0
                pending_urls = url_storage.get_pending_urls(depth=level)
                pending_by_level[level] = len(pending_urls) if pending_urls else 0
                pending_urls_count += pending_by_level[level]
            
            # Use discovery mode counts for decision making, but log both
            logger.info(f"Analysis mode pending URLs: {pending_urls_count} total")
            logger.info(f"Discovery mode pending URLs: {discovery_pending_count} total")
            
            # Use discovery mode counts to make decisions
            actual_pending_count = discovery_pending_count
            actual_pending_by_level = discovery_pending_by_level            
            if actual_pending_count > 0:
                level_info = ", ".join([f"level {l}: {c}" for l, c in sorted(actual_pending_by_level.items()) if c > 0])
                logger.info(f"Found {actual_pending_count} total pending URLs (discovery mode): {level_info}")
            else:
                logger.info("No pending URLs found. Discovery phase needed.")
            
            # STEP 2: DISCOVERY PHASE - Only if needed (no pending URLs)
            # Use discovery mode counts for this decision
            if actual_pending_count == 0:
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
                else:                    # Initialize URL discovery manager
                    url_discovery = URLDiscoveryManager(self.data_folder)
                      # Get all resources with URLs
                    resources_df = pd.read_csv(resources_csv_path)
                    # Use origin_url instead of url for compatibility with universal schema
                    resources_with_url = resources_df[resources_df['origin_url'].notna() if 'origin_url' in resources_df.columns else resources_df['url'].notna()]
                    
                    # Log discovery details
                    logger.info(f"Found {len(resources_with_url)} resources with URLs for discovery")
                    
                    # Call discover_urls_from_resources with correct method name
                    logger.info(f"Starting URL discovery from resources with max_depth={max_depth}")
                    discovery_result = await url_discovery.discover_urls_from_resources(
                        max_depth=max_depth,
                        force_reprocess=force_url_fetch
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
            # Use discovery mode counts to make the decision
            if not self.cancel_requested and actual_pending_count > 0:
                # Determine which level to process
                current_level = process_level

                if current_level is None:
                    # If no level specified, determine the level with the most pending URLs (using discovery counts)
                    current_level = max(actual_pending_by_level, key=actual_pending_by_level.get) if actual_pending_by_level else 1
                    logger.info(f"No level specified. Using level {current_level} which has the most pending URLs")
                
                logger.info("==================================================")
                logger.info(f"STARTING URL ANALYSIS PHASE AT LEVEL {current_level}")
                logger.info("==================================================")
                
                # Process URLs at the selected level
                url_analysis_result = await self.process_urls_at_level(
                    level=current_level,
                    batch_size=batch_size
                )
                
                # Update result with URL analysis results
                result["url_analysis"] = url_analysis_result
                
            elif not self.cancel_requested:
                logger.info("Skipping URL analysis phase - no pending URLs found")
                result["url_analysis"] = {"status": "skipped", "reason": "no_pending_urls"}# STEP 5: VECTOR GENERATION
            # Only generate vectors if:
            # 1. We successfully processed URLs and created content, OR
            # 2. This is a rebuild/continuation with existing content, OR  
            # 3. Force update is enabled
            should_generate_vectors = False
            vector_generation_reason = ""
            
            if skip_vector_generation:
                vector_generation_reason = "skip_vector_generation=True"
            elif self.cancel_requested:
                vector_generation_reason = "processing_cancelled"
            else:
                # Check if URL analysis created new content
                url_analysis_successful = result.get("url_analysis", {}).get("status") == "success"
                url_analysis_processed_urls = result.get("url_analysis", {}).get("success_count", 0) > 0
                
                # Check if there's existing content to vectorize
                from scripts.storage.content_storage_service import ContentStorageService
                content_storage = ContentStorageService(self.data_folder)
                content_paths = list(content_storage.temp_storage_path.glob("*.jsonl"))
                has_existing_content = len(content_paths) > 0
                
                if url_analysis_successful and url_analysis_processed_urls:
                    should_generate_vectors = True
                    vector_generation_reason = "new_content_from_url_analysis"
                elif has_existing_content and force_update:
                    should_generate_vectors = True  
                    vector_generation_reason = "existing_content_with_force_update"
                elif has_existing_content and process_all_steps:
                    should_generate_vectors = True
                    vector_generation_reason = "existing_content_with_process_all_steps"
                else:
                    vector_generation_reason = "no_new_content_to_vectorize"
                    
            if should_generate_vectors:
                logger.info("==================================================")
                logger.info("STARTING VECTOR GENERATION PHASE")
                logger.info(f"Reason: {vector_generation_reason}")
                logger.info("==================================================")
                
                # STEP 5.1: Fix any broken FAISS indexes before generating new content
                logger.info("🔧 Checking and fixing FAISS vector stores...")
                try:
                    from scripts.storage.faiss_fixer import fix_all_faiss_stores
                    fix_result = fix_all_faiss_stores(self.data_folder)
                    
                    if fix_result.get("success"):
                        fixed_count = fix_result.get("fixed_stores", 0)
                        if fixed_count > 0:
                            logger.info(f"✅ Fixed {fixed_count} FAISS vector stores")
                        else:
                            logger.info("✅ All FAISS vector stores are already in good condition")
                    else:
                        logger.warning(f"⚠️ FAISS fix completed with warnings: {fix_result.get('message', 'Unknown issue')}")
                        
                except Exception as e:
                    logger.error(f"❌ Error during FAISS fix: {e}")
                    # Continue anyway - this shouldn't block vector generation
                    
                # STEP 5.2: Generate vectors
                from scripts.analysis.vector_generator import VectorGenerator
                vector_generator = VectorGenerator(self.data_folder)
                
                # Get content paths from the content storage service
                content_paths = list(content_storage.temp_storage_path.glob("*.jsonl"))
                
                vector_result = await vector_generator.generate_vector_stores(content_paths)
                
                # Update result with vector generation results
                result["vector_generation"] = vector_result
            else:
                logger.info("==================================================")
                logger.info("SKIPPING VECTOR GENERATION PHASE")
                logger.info(f"Reason: {vector_generation_reason}")
                logger.info("==================================================")
                result["vector_generation"] = {"status": "skipped", "reason": vector_generation_reason}
            
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
                
                # Extract goals with the correct parameter name
                goal_extraction_result = await goal_extractor.extract_goals(ollama_client=request_app_state)
                
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
                question_generation_result = await question_generator.generate_questions(num_questions=15)
                
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
            # Use origin_url instead of url for compatibility with universal schema
            resources_with_url = resources_df['origin_url'].notna().sum() if 'origin_url' in resources_df.columns else resources_df['url'].notna().sum()
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
    
    async def process_urls_at_level(self, level: int, batch_size: int = 50) -> Dict[str, Any]:
        """
        Process all pending URLs at a specific level.
        
        Args:
            level: The depth level to process (1=first level, 2=second level, etc.)
            batch_size: Maximum number of URLs to process in one batch
            
        Returns:
            Dictionary with processing results
        """
        if level < 1:
            logger.error(f"Invalid level {level} specified. Level must be >= 1.")
            return {
                "status": "error",
                "error": f"Invalid level {level} specified. Level must be >= 1.",
                "level": level
            }
            
        logger.info(f"Processing all pending URLs at level {level}")
        
        # Initialize components
        from scripts.analysis.single_resource_processor_universal import SingleResourceProcessorUniversal
        from scripts.analysis.url_storage import URLStorageManager
        from scripts.analysis.url_discovery_manager import URLDiscoveryManager
        import pandas as pd
        import os
        import csv
        
        single_processor = SingleResourceProcessorUniversal(self.data_folder)
        resources_csv_path = os.path.join(self.data_folder, 'resources.csv')
        processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)        # Get all pending URLs at this level
        # CRITICAL FIX: Use discovery mode to see ALL URLs at this level, not just unprocessed ones
        url_storage.set_discovery_mode(True)
        pending_urls = url_storage.get_pending_urls(depth=level)
        url_storage.set_discovery_mode(False)  # Switch back to analysis mode for processing
          # If no pending URLs at this level, check if we should trigger discovery
        if not pending_urls:
            # Check if there are pending URLs at ANY level before triggering discovery
            total_pending_count = 0
            for check_level in range(1, 6):  # Check levels 1-5
                check_pending = url_storage.get_pending_urls(depth=check_level)
                total_pending_count += len(check_pending) if check_pending else 0
            
            # Also check if URLs at this level have been processed (level complete)
            # Switch to discovery mode temporarily to see ALL URLs at this level
            url_storage.set_discovery_mode(True)
            url_storage.load_pending_urls_cache()
            urls_at_level_in_discovery = url_storage.get_pending_urls(depth=level)
            total_urls_at_level = len(urls_at_level_in_discovery) if urls_at_level_in_discovery else 0
            
            # Switch back to analysis mode
            url_storage.set_discovery_mode(False)
            url_storage.load_pending_urls_cache()
            
            # If there are URLs at this level in discovery mode but none in analysis mode,
            # it means they've all been processed
            if total_urls_at_level > 0:
                logger.info(f"Level {level} appears to be complete: {total_urls_at_level} URLs exist but all have been processed")
                return {
                    "status": "completed",
                    "level": level,
                    "reason": "level_completed_all_urls_processed",
                    "message": f"Level {level} is complete: all {total_urls_at_level} URLs have been processed",
                    "total_urls_at_level": total_urls_at_level,
                    "total_pending_other_levels": total_pending_count
                }
            
            # Only trigger discovery if there are NO pending URLs anywhere AND no URLs at this level
            if total_pending_count == 0 and total_urls_at_level == 0:
                logger.info(f"No pending URLs found at level {level}, and no pending URLs at any other level. Triggering URL discovery.")
                
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
                # Use origin_url instead of url for compatibility with universal schema
                resources_with_url = resources_df[resources_df['origin_url'].notna() if 'origin_url' in resources_df.columns else resources_df['url'].notna()]
                
                if len(resources_with_url) > 0:
                    logger.info(f"Found {len(resources_with_url)} resources with URLs for discovery")
                    
                    # Initialize URL discovery manager
                    url_discovery = URLDiscoveryManager(self.data_folder)
                      # Trigger discovery starting at the current level
                    logger.info(f"Starting URL discovery at level {level}")
                    # FIXED: Use the full max_depth from config instead of limiting to level+1
                    # Get max_depth from config
                    from utils.config import load_config
                    config = load_config()
                    config_max_depth = config.get("crawl_config", {}).get("max_depth", 2)                    # Use the config value - this method IS async, so await it
                    discovery_result = await url_discovery.discover_urls_from_resources(level=level, max_depth=config_max_depth)
                    
                    # Log discovery results
                    if discovery_result.get("status") == "success":
                        logger.info(f"Discovery succeeded: Found {discovery_result.get('discovered_urls', 0)} pending URLs")
                    else:
                        logger.warning(f"Discovery status: {discovery_result.get('status')}, reason: {discovery_result.get('reason')}")
                      # Check again for pending URLs after discovery at the requested level
                    pending_urls = url_storage.get_pending_urls(depth=level)
                    
                    # If no URLs at the requested level, check other levels before giving up
                    if not pending_urls:
                        # Check if there are pending URLs at any other level
                        total_pending_after_discovery = 0
                        for check_level in range(1, 6):  # Check levels 1-5
                            check_pending = url_storage.get_pending_urls(depth=check_level)
                            total_pending_after_discovery += len(check_pending) if check_pending else 0
                        
                        if total_pending_after_discovery == 0:
                            logger.warning(f"No pending URLs found at any level after discovery")
                            return {
                                "status": "warning",
                                "level": level,
                                "reason": "no_pending_urls_after_discovery",
                                "discovery_result": discovery_result
                            }
                        else:
                            logger.info(f"No pending URLs at level {level} after discovery, but found {total_pending_after_discovery} pending URLs at other levels")
                            return {
                                "status": "skipped",
                                "level": level,
                                "reason": "no_pending_urls_at_this_level_but_pending_at_others_after_discovery",
                                "message": f"No pending URLs at level {level} after discovery, but {total_pending_after_discovery} pending URLs exist at other levels",
                                "total_pending_other_levels": total_pending_after_discovery
                            }
                    
                    logger.info(f"After discovery: found {len(pending_urls)} pending URLs at level {level}")
                else:
                    logger.warning(f"No resources with URLs found for discovery at level {level}")
                    return {
                        "status": "warning", 
                        "reason": "no_resources_with_url",
                        "message": "No resources with URLs found for discovery"
                    }
            else:
                # There are pending URLs at other levels, so don't trigger discovery
                logger.info(f"No pending URLs at level {level}, but found {total_pending_count} pending URLs at other levels. Skipping discovery.")
                return {
                    "status": "skipped",
                    "level": level, 
                    "reason": "no_pending_urls_at_this_level_but_pending_at_others",
                    "message": f"No pending URLs at level {level}, but {total_pending_count} pending URLs exist at other levels",
                    "total_pending_other_levels": total_pending_count
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
                resources_df = pd.DataFrame(columns=['origin_url', 'title'])
                resources_with_url = resources_df
            else:
                resources_df = pd.read_csv(resources_csv_path)
                # Use origin_url instead of url for compatibility with universal schema
                resources_with_url = resources_df[resources_df['origin_url'].notna() if 'origin_url' in resources_df.columns else resources_df['url'].notna()]
            
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
                    break                # Find the resource this origin URL belongs to
                resource_for_origin = None
                for _, row in resources_with_url.iterrows():
                    # Check origin_url first, then url as fallback
                    url_field = 'origin_url' if 'origin_url' in row else 'url'
                    if row[url_field] == origin_url:
                        resource_for_origin = row.to_dict()
                        break
                
                # If we don't have resource context, try to find it in resource_urls mapping
                if resource_for_origin is None:
                    # Check if this URL is associated with a resource in resource_urls.csv
                    resource_id_from_map = url_storage.get_resource_id_for_url(origin_url)
                    if resource_id_from_map:
                        # Look up the resource by this ID
                        for _, row in resources_df.iterrows():
                            if str(row.get('title', '')) == resource_id_from_map:
                                resource_for_origin = row.to_dict()
                                logger.info(f"Found resource for URL {origin_url} via resource_urls mapping: {resource_id_from_map}")
                                break
                  # If we still don't have resource context, create a minimal one
                if resource_for_origin is None:                    # During discovery phase, just use origin URL without generating a title
                    resource_for_origin = {
                        'origin_url': origin_url,  # Use origin_url as the primary URL field
                        'url': origin_url,        # Keep url for backward compatibility
                        'title': resource_id_from_map or ""  # Use existing resource ID if available
                    }
                    logger.info(f"No resource found for origin URL: {origin_url}. Using existing resource ID: {resource_id_from_map or 'none'}")
                    
                logger.info(f"Processing {len(urls_for_origin)} URLs at level {level} from origin: {origin_url}")
                origin_resource_id = resource_for_origin.get('title', 'Unnamed')
                
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
                        
                        # Process the URL using SingleResourceProcessorUniversal's process_specific_url method
                        # This is the ANALYSIS phase - we're not discovering URLs anymore
                        logger.info(f"ANALYSIS PHASE: Processing URL {url_to_process} at level {url_depth} using SingleResourceProcessorUniversal")
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
                                    max_depth: int = 5,  # Changed from 4 to 5 to match config
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
                          # Call discover_urls_from_resources with the correct parameters
                        # This uses our URL discovery manager for discovery operations
                        logger.info(f"Starting URL discovery from resources with max_depth={max_depth}")
                        discovery_result = await discovery.discover_urls_from_resources(
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
                    level_result = await self.process_urls_at_level(
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
        Advance to the next level after completing the current level.
        
        Args:
            current_level: The level that was just completed
            max_depth: Maximum depth to process
            
        Returns:
            Dictionary with advancement results
        """
        try:
            # Check if current level is actually complete
            from scripts.analysis.url_storage import URLStorageManager
            processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
            url_storage = URLStorageManager(processed_urls_file)
            
            # Use discovery mode to see actual pending URLs
            url_storage.set_discovery_mode(True)
            pending_urls = url_storage.get_pending_urls(depth=current_level)
            url_storage.set_discovery_mode(False)
            
            pending_count = len(pending_urls) if pending_urls else 0
            
            if pending_count > 0:
                logger.warning(f"Level {current_level} is not complete yet - {pending_count} URLs still pending")
                return {
                    "status": "not_complete",
                    "current_level": current_level,
                    "next_level": current_level,
                    "pending_urls": pending_count,
                    "message": f"Level {current_level} still has {pending_count} pending URLs"
                }
            
            # Determine next level
            if current_level >= max_depth:
                logger.info(f"Reached maximum depth {max_depth}, processing complete")
                return {
                    "status": "completed",
                    "current_level": current_level,
                    "next_level": None,
                    "message": f"Processing complete at maximum depth {max_depth}"
                }
            
            next_level = current_level + 1
            
            # Check if next level has URLs to process
            url_storage.set_discovery_mode(True)
            next_level_urls = url_storage.get_pending_urls(depth=next_level)
            url_storage.set_discovery_mode(False)
            
            next_level_count = len(next_level_urls) if next_level_urls else 0
            
            if next_level_count == 0:
                logger.info(f"No URLs found at level {next_level}, skipping to find next available level")
                
                # Find the next level with URLs
                for check_level in range(next_level + 1, max_depth + 1):
                    url_storage.set_discovery_mode(True)
                    check_urls = url_storage.get_pending_urls(depth=check_level)
                    url_storage.set_discovery_mode(False)
                    
                    if len(check_urls) > 0:
                        next_level = check_level
                        next_level_count = len(check_urls)
                        logger.info(f"Found {next_level_count} URLs at level {next_level}")
                        break
                else:
                    logger.info("No more URLs found at any level, processing complete")
                    return {
                        "status": "completed",
                        "current_level": current_level,
                        "next_level": None,
                        "message": "No more URLs to process at any level"
                    }
            
            # Update config with new level
            self.update_config_level(next_level)
            
            logger.info(f"Advanced from level {current_level} to level {next_level} ({next_level_count} URLs)")
            
            return {
                "status": "advanced",
                "current_level": current_level,
                "next_level": next_level,
                "next_level_urls": next_level_count,
                "message": f"Advanced to level {next_level} with {next_level_count} URLs to process"
            }
            
        except Exception as e:
            logger.error(f"Error advancing to next level: {e}")
            return {
                "status": "error",
                "current_level": current_level,
                "next_level": current_level,
                "error": str(e)
            }
    
    def update_config_level(self, new_level: int) -> bool:
        """
        Update the current_process_level in config.json.
        
        Args:
            new_level: The new level to set in config
            
        Returns:
            Success flag
        """
        try:
            from utils.config import load_config, save_config
            
            config = load_config()
            old_level = config.get("current_process_level", 1)
            config["current_process_level"] = new_level
            save_config(config)
            
            logger.info(f"Updated config.json: current_process_level {old_level} -> {new_level}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating config level: {e}")
            return False

# Standalone function for easy import
async def process_knowledge_base(data_folder=None, request_app_state=None, **kwargs):
    """
    Process all knowledge base files as a standalone function.
    
    Args:
        data_folder: Path to data folder (deprecated - now always uses config)
        request_app_state: FastAPI request.app.state
        **kwargs: All other KnowledgeOrchestrator parameters
    
    Returns:
        Dictionary with processing results
    """
    # Initialize orchestrator (always uses config-based path now)
    orchestrator = KnowledgeOrchestrator()
    
    # Process knowledge
    return await orchestrator.process_knowledge(request_app_state=request_app_state, **kwargs)