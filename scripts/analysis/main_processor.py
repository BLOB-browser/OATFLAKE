#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import asyncio

logger = logging.getLogger(__name__)

# Import interruptible handling
from scripts.analysis.interruptible_llm import is_interrupt_requested, clear_interrupt

class MainProcessor:
    """
    Main orchestrator that coordinates content fetching, analysis, and storage.
    Delegates to specialized components for different aspects of processing.
    """
    
    def __init__(self, data_folder: str, url_batch_size: int = 10):
        """
        Initialize the main processor.
        
        Args:
            data_folder: Path to the data directory
            url_batch_size: Batch size for URL processing before vector generation (default: 10)
        """
        self.data_folder = data_folder
        self.url_batch_size = url_batch_size  # Added configurable batch size
        
        # Track the last checked URL count to avoid excessive checks
        self.last_checked_url_count = 0
        
        # Import specialized components
        from scripts.analysis.single_resource_processor import SingleResourceProcessor
        from scripts.analysis.cleanup_manager import CleanupManager
        from scripts.analysis.vector_generator import VectorGenerator
        
        # Initialize components
        self.single_processor = SingleResourceProcessor(data_folder)
        self.cleanup_manager = CleanupManager(data_folder)
        self.vector_generator = VectorGenerator(data_folder)
        
        # For backward compatibility, still import these directly
        from scripts.storage.content_storage_service import ContentStorageService
        from scripts.storage.temporary_storage_service import TemporaryStorageService
        self.content_storage = ContentStorageService(data_folder)
        self.temp_storage = TemporaryStorageService(data_folder)
        
        # Track processing state
        self.vector_generation_needed = False
        
        logger.info(f"MainProcessor initialized with data folder: {data_folder} and URL batch size: {url_batch_size}")
    
    def _should_generate_vectors(self, url_count: int) -> bool:
        """
        Determine if we should generate vectors based on the current URL count.
        Generates vectors when URL count reaches a multiple of url_batch_size.
        
        Args:
            url_count: Current count of processed URLs
            
        Returns:
            Boolean indicating whether vector generation should occur
        """
        # Generate vectors when URL count is a multiple of batch size
        should_generate = url_count > 0 and url_count % self.url_batch_size == 0
        if should_generate:
            logger.info(f"URL count {url_count} is a multiple of batch size {self.url_batch_size}, should generate vectors")
        return should_generate
    
    @classmethod
    def reset_cancellation(cls):
        """Reset the cancellation flag for backward compatibility"""
        clear_interrupt()
    
    @classmethod
    def check_cancellation(cls):
        """Check if processing should be cancelled for backward compatibility"""
        return is_interrupt_requested()
    
    def process_resources(self, csv_path: str = None, max_resources: int = None, force_reanalysis: bool = False, 
                      skip_vector_generation: bool = False, process_by_level: bool = True, max_depth: int = 4) -> Dict[str, Any]:
        """
        Process resources from a CSV file.
        
        Args:
            csv_path: Path to the CSV file containing resources to process
            max_resources: Maximum number of resources to process
            force_reanalysis: Whether to force reanalysis of already processed resources
            skip_vector_generation: Whether to skip vector generation after processing
            process_by_level: If True, processes URLs strictly level by level (all level 1 URLs before level 2)
            max_depth: Maximum crawl depth (1=just main page links, 2=two levels, 3=three levels, etc.)
            
        Returns:
            Dictionary with processing statistics
        """
        # Reset cancellation flag when starting new processing
        self.reset_cancellation()
        
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
            "skipped_count": 0,  # Track skipped resources
            "start_time": datetime.now()
        }
        
        try:
            # Read resources CSV file
            df = pd.read_csv(csv_path)
            total_resources = len(df)
            logger.info(f"Read {total_resources} resources from {csv_path}")
            
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
                skipped_count = total_resources - unprocessed_count
                
                stats["skipped_count"] = skipped_count
                
                if skipped_count > 0:
                    logger.info(f"Skipping {skipped_count} already processed resources (analysis_completed=True)")
                    df = df[unprocessed_mask].copy()  # Use .copy() to avoid SettingWithCopyWarning
            else:
                logger.info(f"Force reanalysis enabled - processing all {total_resources} resources regardless of analysis_completed")
            
            # Limit resources if specified
            if max_resources is not None and len(df) > max_resources:
                df = df.head(max_resources)
                logger.info(f"Limited to {len(df)} resources as requested")
            
            if len(df) == 0:
                logger.info("No resources to process after filtering")
                return {
                    "status": "completed",
                    "resources_processed": 0,
                    "skipped_count": stats.get("skipped_count", 0),
                    "message": "No resources needed processing"
                }
            
            # Create collectors for processed items
            processed_resources = []
            all_definitions = []
            all_projects = []
            all_methods = []
            
            # Get URL storage to track processed URLs
            from scripts.analysis.url_storage import URLStorageManager
            from utils.config import get_data_path
            processed_urls_file = os.path.join(get_data_path(), "processed_urls.csv")
            url_storage = URLStorageManager(processed_urls_file)
            
            # Get the initial count of processed URLs
            initial_url_count = len(url_storage.get_processed_urls())
            logger.info(f"Starting with {initial_url_count} already processed URLs")
            self.last_checked_url_count = initial_url_count
            
            # Process each resource
            for idx, row in df.iterrows():
                resource = row.to_dict()
                resource_id = f"#{idx+1}/{len(df)}"
                
                # Check for cancellation
                if self.check_cancellation():
                    logger.info("Resource processing cancelled by external request")
                    break
                
                # Log which resource we're processing
                logger.info(f"Processing resource {resource_id}: {resource.get('title', 'Unnamed')}")
                
                try:
                    # Use our dedicated single resource processor for this resource
                    result = self.single_processor.process_resource(
                        resource, 
                        resource_id, 
                        idx, 
                        csv_path,
                        on_subpage_processed=lambda: self._check_and_generate_vectors(url_storage, stats, skip_vector_generation),
                        process_by_level=process_by_level,  # Pass the level-processing parameter
                        max_depth=max_depth  # Pass the maximum crawl depth
                    )
                    
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
                    
                    # Check if vector generation is needed
                    if self.single_processor.vector_generation_needed:
                        self.vector_generation_needed = True
                    
                    # Check if we need to generate vectors based on processed URLs count
                    # This check happens after each resource is fully processed
                    self._check_and_generate_vectors(url_storage, stats, skip_vector_generation)
                    
                except Exception as e:
                    logger.error(f"Error processing resource {resource_id}: {e}")
                    stats["error_count"] += 1
                
                # Check for cancellation again after processing each resource
                if self.check_cancellation():
                    logger.info("Resource processing cancelled after processing resource")
                    break
            
            # Calculate duration
            stats["duration_seconds"] = (datetime.now() - stats["start_time"]).total_seconds()
            stats["status"] = "completed"
            
            # Ensure all entity types are saved to CSV files
            # Import DataSaver to save these directly
            from scripts.services.storage import DataSaver
            data_saver = DataSaver()
            
            # Save each entity type if we have any
            if all_definitions:
                logger.info(f"Ensuring {len(all_definitions)} definitions are saved to CSV")
                data_saver.save_definitions(all_definitions)
                
            if all_projects:
                logger.info(f"Ensuring {len(all_projects)} projects are saved to CSV")
                data_saver.save_projects(all_projects)
                
            if all_methods:
                logger.info(f"Ensuring {len(all_methods)} methods are saved to CSV")
                data_saver.save_methods(all_methods)
            
            # Include processed items in the stats
            stats["processed_resources"] = processed_resources
            stats["definitions"] = all_definitions
            stats["projects"] = all_projects
            stats["methods"] = all_methods
            
            # Comprehensive vector store generation after all resources are processed
            if self.vector_generation_needed and not skip_vector_generation:
                logger.info("Starting vector generation after all resources were processed")
                
                # Get all content files from the temporary storage path
                content_paths = list(self.content_storage.temp_storage_path.glob("*.jsonl"))
                logger.info(f"Found {len(content_paths)} content files for vector generation")
                
                # Handle async vector generation
                try:
                    # Run the vector generation
                    vector_stats = asyncio.run(self.vector_generator.generate_vector_stores(content_paths))
                    stats["vector_stats"] = vector_stats
                    logger.info(f"Vector generation complete: {vector_stats}")
                    
                except Exception as ve:
                    logger.error(f"Error during vector generation: {ve}")
                    stats["vector_error"] = str(ve)
                
                # Add comprehensive cleanup after vector generation (whether successful or not)
                self.cleanup_manager.cleanup_temporary_files()
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
        Now delegates to the SingleResourceProcessor.
        
        Args:
            resource: The resource dictionary 
            resource_id: String identifier for logging
            idx: Index in the CSV file
            csv_path: Path to the CSV file for saving updates
            
        Returns:
            Dictionary with processing results
        """
        # Simply delegate to our dedicated single resource processor
        return self.single_processor.process_resource(resource, resource_id, idx, csv_path)
    
    def save_single_resource(self, resource, csv_path, idx):
        """Save a single resource to the CSV file."""
        # Delegate to the single processor
        return self.single_processor.save_single_resource(resource, csv_path, idx)

    async def generate_vector_stores(self) -> Dict[str, Any]:
        """
        Generate vector stores from content files.
        Now delegates to the VectorGenerator.
        """
        logger.info("Generating vector stores using VectorGenerator")
        
        # Get all content files from the temporary storage path
        content_paths = list(self.content_storage.temp_storage_path.glob("*.jsonl"))
        
        # Delegate to the vector generator
        return await self.vector_generator.generate_vector_stores(content_paths)
    
    def _cleanup_temporary_files(self):
        """
        Clean up all temporary files and directories after processing.
        Now delegates to the CleanupManager.
        """
        # Delegate to the cleanup manager
        self.cleanup_manager.cleanup_temporary_files()
    
    async def generate_vector_stores_for_url_list(self, processed_urls_csv: str = None) -> Dict[str, Any]:
        """
        Generate vector stores specifically from the processed URLs CSV file.
        This allows vector store creation from subpages without waiting for resource thresholds.
        
        Args:
            processed_urls_csv: Path to the processed URLs CSV file. If None, use default.
            
        Returns:
            Dictionary with vector generation statistics
        """
        from scripts.analysis.url_storage import URLStorageManager
        from utils.config import get_data_path
        import os
        
        # Set default path if not provided
        if processed_urls_csv is None:
            data_path = get_data_path()
            processed_urls_csv = os.path.join(data_path, "processed_urls.csv")
        
        logger.info(f"Generating vector stores from processed URLs: {processed_urls_csv}")
        
        try:
            # Initialize URL storage to read the processed URLs
            url_storage = URLStorageManager(processed_urls_csv)
            processed_urls = url_storage.get_processed_urls()
            
            if not processed_urls:
                logger.warning("No processed URLs found in CSV file")
                return {"status": "warning", "message": "No processed URLs found"}
            
            logger.info(f"Found {len(processed_urls)} processed URLs to use for vector store generation")
            
            # Extract content from these URLs (this needs to be implemented in a content extractor)
            from scripts.analysis.content_extractor import extract_content_for_urls
            content_paths = await extract_content_for_urls(processed_urls, self.data_folder)
            
            if not content_paths:
                logger.warning("No content files generated from processed URLs")
                return {"status": "warning", "message": "No content files generated"}
                
            logger.info(f"Generated {len(content_paths)} content files from processed URLs")
            
            # Generate vector stores from the content files
            stats = await self.vector_generator.generate_vector_stores(content_paths)
            
            # Just log the current URL count but don't save it separately
            url_storage = URLStorageManager(processed_urls_csv)
            current_url_count = len(url_storage.get_processed_urls())
            logger.info(f"Current URL count after manual vector generation: {current_url_count}")
            
            return {
                "status": "success",
                "processed_urls_count": len(processed_urls),
                "content_files_count": len(content_paths),
                "vector_stats": stats,
                "current_url_count": current_url_count,
            }
            
        except Exception as e:
            logger.error(f"Error generating vector stores from URL list: {e}")
            return {"status": "error", "error": str(e)}
    
    def _check_and_generate_vectors(self, url_storage, stats=None, skip_vector_generation=False):
        """
        Check if we need to generate vectors based on current URL count, and generate them if needed.
        
        Args:
            url_storage: The URL storage manager to get URL counts from
            stats: Optional stats dictionary to update with checkpoint information
            skip_vector_generation: Whether to skip vector generation
        
        Returns:
            Boolean indicating whether vector generation was performed
        """
        # Get current URL count
        current_url_count = len(url_storage.get_processed_urls())
        
        # Check if URL count has changed since last check
        if current_url_count == self.last_checked_url_count:
            return False
        
        logger.info(f"URL count check: Current count is {current_url_count}, last checked was {self.last_checked_url_count}")
        self.last_checked_url_count = current_url_count
        
        # Check if we should generate vectors
        if (self._should_generate_vectors(current_url_count) and 
            self.vector_generation_needed and not skip_vector_generation):
            
            logger.info(f"Starting vector store generation at URL count milestone: {current_url_count} (batch size: {self.url_batch_size})")
            
            try:
                # Get all content files from the temporary storage path
                content_paths = list(self.content_storage.temp_storage_path.glob("*.jsonl"))
                logger.info(f"Found {len(content_paths)} content files for vector generation")
                
                # Run the vector generation as a checkpoint - this doesn't delete anything
                incremental_stats = asyncio.run(self.vector_generator.generate_vector_stores(content_paths))
                
                # Store the stats of this checkpoint build if stats dictionary is provided
                if stats is not None:
                    if "url_checkpoint_builds" not in stats:
                        stats["url_checkpoint_builds"] = []
                        
                    stats["url_checkpoint_builds"].append({
                        "build_number": len(stats.get("url_checkpoint_builds", [])) + 1,
                        "total_urls": current_url_count,
                        "milestone": f"{current_url_count} (batch size: {self.url_batch_size})",
                        "content_paths": len(content_paths),
                        "stats": incremental_stats
                    })
                
                logger.info(f"Completed vector store checkpoint build at URL count {current_url_count}")
                return True
                
            except Exception as ve:
                logger.error(f"Error during checkpoint vector generation: {ve}")
                return False
                
        return False
