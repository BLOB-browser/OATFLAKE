#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import pandas as pd
from typing import Dict, Any, Optional

from scripts.analysis.orchestration.base_orchestrator import BaseOrchestrator
from scripts.analysis.url_storage import URLStorageManager
from scripts.analysis.content_fetcher import ContentFetcher
from scripts.analysis.vector_store_generator import VectorStoreGenerator

logger = logging.getLogger(__name__)

class PhasedProcessor(BaseOrchestrator):
    """
    Handles phased knowledge processing:
    - Discovery phase: Find all URLs from resources
    - Analysis phase: Process content from discovered URLs
    """
    
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
            content_fetcher = ContentFetcher(timeout=120)  # Increased timeout for discovery
            
            # PHASE 1.5: Check if URL discovery is actually needed
            check_result = content_fetcher.check_discovery_needed(max_depth=max_depth)
            result["discovery_check"] = check_result
            
            # PHASE 2: URL Discovery - Process all resources in discovery-only mode
            # Only run discovery if needed or if force_url_fetch is True
            if check_result["discovery_needed"] or force_url_fetch:
                logger.info(f"Starting URL discovery phase for {len(resources_with_url)} resources")
                
                # Get all URLs from resources and create resource ID mapping
                all_urls = resources_with_url['url'].tolist()
                resource_ids = {}
                
                # Create mapping of URLs to resource titles for tracking
                for _, row in resources_with_url.iterrows():
                    url = row.get('url', '')
                    title = row.get('title', '')
                    if url and title:
                        # Use title as resource ID, or fallback to URL-based ID
                        resource_id = title.strip() or url.split('//')[-1].replace('/', '_')
                        resource_ids[url] = resource_id
                
                logger.info(f"Created resource ID mapping for {len(resource_ids)} resources")
                
                # Run discovery phase with resource ID mapping
                discovery_result = content_fetcher.discovery_phase(
                    urls=all_urls,
                    max_depth=max_depth,
                    force_reprocess=force_url_fetch,
                    resource_ids=resource_ids
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
            logger.info("==================================================")
            logger.info("STARTING ANALYSIS PHASE")
            logger.info("==================================================")
            logger.info("All URL discovery is now complete - starting content analysis")
            
            # Get the URL storage manager
            processed_urls_file = os.path.join(self.data_folder, "processed_urls.csv")
            url_storage = URLStorageManager(processed_urls_file)
            
            # Process each level
            from scripts.analysis.orchestration.url_processor import URLProcessor
            url_processor = URLProcessor(self.data_folder)
            # Share cancellation state
            url_processor.cancel_requested = self.cancel_requested
            url_processor.processing_active = self.processing_active
            url_processor.force_url_fetch = self.force_url_fetch
            
            # Get total pending URLs by level before processing
            pending_urls_by_level = {}
            total_pending_urls = 0
            
            for level in range(1, max_depth + 1):
                pending_urls = url_storage.get_pending_urls(depth=level)
                url_count = len(pending_urls) if pending_urls else 0
                pending_urls_by_level[level] = url_count
                total_pending_urls += url_count
            
            # Log detailed information about pending URLs
            level_info = ", ".join([f"level {level}: {count}" for level, count in pending_urls_by_level.items() if count > 0])
            if total_pending_urls > 0:
                logger.info(f"Found {total_pending_urls} total pending URLs to process: {level_info}")
            else:
                logger.info("No pending URLs found to process")
                
            # Rather than processing level by level, use the more efficient URLProcessor method
            logger.info(f"Processing all URLs up to level {max_depth} with batch_size={batch_size}")
            
            # Process all URLs by level using the specialized processor with our parameters
            analysis_result = url_processor.process_urls_by_levels(
                max_depth=max_depth,
                batch_size=batch_size,
                force_fetch=force_url_fetch
            )
            
            # Store analysis results
            analysis_results = {
                "processed_by_level": analysis_result.get("processed_by_level", {}),
                "total_processed": analysis_result.get("total_processed", 0),
                "successful": analysis_result.get("successful", 0),
                "failed": analysis_result.get("failed", 0)
            }
            
            # Log detailed results
            logger.info(f"Analysis phase completed: {analysis_results['total_processed']} URLs processed, "
                       f"{analysis_results['successful']} successful, {analysis_results['failed']} errors")
            
            result["analysis_phase"] = analysis_results
            
            # PHASE 4: Generate vector stores if not cancelled
            if not self.cancel_requested:
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