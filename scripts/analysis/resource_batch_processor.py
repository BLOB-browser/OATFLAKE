#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from scripts.analysis.interruptible_llm import is_interrupt_requested, clear_interrupt

logger = logging.getLogger(__name__)

class ResourceBatchProcessor:
    """
    Handles processing batches of resources from CSV files.
    Coordinates the overall processing flow and delegates individual resource processing.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the ResourceBatchProcessor.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        
        # Import required components
        from scripts.analysis.single_resource_processor import SingleResourceProcessor
        from scripts.analysis.cleanup_manager import CleanupManager
        from scripts.analysis.vector_generator import VectorGenerator
        from scripts.analysis.interruptible_llm import is_interrupt_requested, clear_interrupt
        
        # Initialize components
        self.single_processor = SingleResourceProcessor(data_folder)
        self.cleanup_manager = CleanupManager(data_folder)
        self.vector_generator = VectorGenerator(data_folder)
        
        # Track processing state
        self.vector_generation_needed = False
        self._cancel_processing = False
    
    def reset_cancellation(self):
        """Reset the cancellation flag"""
        self._cancel_processing = False
    
    def check_cancellation(self):
        """Check if processing should be cancelled"""
        return self._cancel_processing or is_interrupt_requested()
    
    def process_resources(self, csv_path: str = None, max_resources: int = None, force_reanalysis: bool = False, skip_vector_generation: bool = False) -> Dict[str, Any]:
        """
        Process resources from a CSV file.
        
        Args:
            csv_path: Path to the CSV file containing resources to process
            max_resources: Maximum number of resources to process
            force_reanalysis: Whether to force reanalysis of already processed resources
            skip_vector_generation: Whether to skip vector generation after processing
            
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
            
            # Track the count for incremental vector store generation
            resources_since_last_vector_build = 0
            incremental_build_threshold = 10  # Build vector store after every 10 resources
            
            # Process each resource
            for idx, row in df.iterrows():
                resource = row.to_dict()
                resource_id = f"#{idx+1}/{len(df)}"
                
                # Check for cancellation
                if self.check_cancellation():
                    logger.info("Resource processing cancelled by external request")
                    break
                
                logger.info(f"Processing resource {resource_id}: {resource.get('title', 'Unnamed')}")
                
                try:
                    # Process this resource using the single resource processor
                    result = self.single_processor.process_resource(resource, resource_id, idx, csv_path)
                    
                    # Update statistics
                    stats["resources_processed"] += 1
                    resources_since_last_vector_build += 1
                    
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
                    
                    # Flag if vector generation is needed
                    if self.single_processor.vector_generation_needed:
                        self.vector_generation_needed = True
                    
                    # Incremental vector store building after every N resources
                    if (resources_since_last_vector_build >= incremental_build_threshold and 
                        self.vector_generation_needed and not skip_vector_generation):
                        
                        logger.info(f"Starting incremental vector store build after {resources_since_last_vector_build} resources")
                        
                        try:
                            # Get all content files from the temporary storage path
                            from scripts.storage.content_storage_service import ContentStorageService
                            content_storage = ContentStorageService(self.data_folder)
                            content_paths = list(content_storage.temp_storage_path.glob("*.jsonl"))
                            # Also check vector_data folder if it exists
                            vector_data_path = content_storage.temp_storage_path.parent / "vector_data"
                            if vector_data_path.exists():
                                content_paths.extend(list(vector_data_path.glob("*.jsonl")))
                            
                            logger.info(f"Found {len(content_paths)} content files for incremental vector generation")
                            
                            # Run the incremental vector generation
                            incremental_stats = asyncio.run(self.vector_generator.generate_vector_stores(content_paths))
                            
                            # Store the stats of this incremental build
                            if "incremental_builds" not in stats:
                                stats["incremental_builds"] = []
                                
                            stats["incremental_builds"].append({
                                "build_number": len(stats.get("incremental_builds", [])) + 1,
                                "resources_processed": resources_since_last_vector_build,
                                "content_paths": len(content_paths),
                                "stats": incremental_stats
                            })
                            
                            # Don't delete temporary files yet - keep them for final comprehensive build
                            # Reset the counter for next batch
                            resources_since_last_vector_build = 0
                            logger.info(f"Completed incremental vector store build #{len(stats.get('incremental_builds', []))}")
                            
                        except Exception as ve:
                            logger.error(f"Error during incremental vector generation: {ve}")
                            # Continue processing resources despite vector generation error
                        
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
            
            # Include processed items in the stats
            stats["processed_resources"] = processed_resources
            stats["definitions"] = all_definitions
            stats["projects"] = all_projects
            stats["methods"] = all_methods
            
            # Generate vector stores if needed and not explicitly skipped
            if self.vector_generation_needed and not skip_vector_generation:
                logger.info("Starting vector generation after all resources were processed")
                
                # Get all content paths that need processing
                from scripts.storage.content_storage_service import ContentStorageService
                content_storage = ContentStorageService(self.data_folder)
                
                # Get all content files from the temporary storage path
                content_paths = list(content_storage.temp_storage_path.glob("*.jsonl"))
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
