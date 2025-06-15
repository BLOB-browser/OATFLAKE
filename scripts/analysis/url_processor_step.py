#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class URLProcessorStep:
    """
    Processes URLs extracted from resources and discovers deeper level URLs.
    This component handles STEP 4 of the knowledge processing workflow.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the URL processor step.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = Path(data_folder)
        self.processed_urls_file = self.data_folder / "processed_urls.csv"
        
        # Load config to get current level
        self.config_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / "config.json"
        self.current_level = self._get_current_level()
    
    def _get_current_level(self) -> int:
        """
        Get the current URL processing level from config.json
        
        Returns:
            Current URL level (defaults to 1 if not found)
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Get current level if it exists, otherwise default to 1
                    return config.get("current_url_level", 1)
        except Exception as e:
            logger.warning(f"Could not load config.json, defaulting to level 1: {e}")
        
        return 1
    
    def _save_current_level(self, level: int) -> None:
        """
        Save the current URL processing level to config.json
        
        Args:
            level: Current URL level to save
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Update current level
                config["current_url_level"] = level
                
                with open(self.config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                    
                logger.info(f"Updated current_url_level to {level} in config.json")
            else:
                logger.warning(f"Config file not found at {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving current level to config.json: {e}")
    
    async def process_pending_urls(self, target_level: int = None, max_depth: int = 3) -> Dict[str, Any]:
        """
        Process pending URLs at the current level and increment level counter in config.
        Simplified to remove substeps and use a single flow with level counter.
        
        Args:
            target_level: Target level to process (defaults to current level from config)
            max_depth: Maximum depth level to process
            
        Returns:
            Dictionary with URL processing results
        """
        logger.info(f"STEP 4: PROCESSING URLS AT LEVEL {target_level if target_level is not None else self.current_level}")
        logger.info("=================================================")
        
        try:
            # Import necessary components
            from scripts.analysis.url_storage import URLStorageManager
            from scripts.analysis.level_processor import LevelBasedProcessor
            
            # Use provided target_level or current level from config
            current_level = target_level if target_level is not None else self.current_level
            
            # Initialize URL storage and level processor
            url_storage = URLStorageManager(str(self.processed_urls_file))
            level_processor = LevelBasedProcessor(str(self.data_folder))
            
            # Seed level 1 with URLs from resources if needed
            if current_level == 1:
                resources_csv_path = self.data_folder / 'resources.csv'
                if resources_csv_path.exists():
                    logger.info("Seeding level 1 with URLs from resources.csv")
                    try:
                        resources_df = pd.read_csv(resources_csv_path)
                        if not resources_df.empty:
                            added_count = 0
                            # Use universal schema only - require origin_url field
                            if 'origin_url' not in resources_df.columns:
                                logger.error("Resources DataFrame missing 'origin_url' column - universal schema required")
                                return {
                                    "success": False,
                                    "message": "Resources file missing universal schema (origin_url field)",
                                    "urls_added": 0
                                }
                                
                            resources_with_url = resources_df[resources_df['origin_url'].notna()]
                                
                            for _, row in resources_with_url.iterrows():
                                url = row.get('origin_url')
                                # Generate a unique resource_id for this resource (use row index + 1)
                                resource_id = str(row.name + 1)  # row.name is the index, +1 for 1-based IDs
                                if url and isinstance(url, str) and url.startswith('http'):
                                    # Check if already in pending URLs
                                    pending_urls = url_storage.get_pending_urls(depth=1)
                                    if not any(p.get('origin_url') == url for p in pending_urls):
                                        # Save with the generated resource_id
                                        url_storage.save_pending_url(url, depth=1, origin=url, resource_id=resource_id)
                                        added_count += 1
                                        logger.info(f"Added URL {url} with resource_id {resource_id}")
                            logger.info(f"Added {added_count} URLs from resources.csv to level 1 queue")
                    except Exception as e:
                        logger.error(f"Error seeding URLs from resources: {e}")
            
            # Process URLs at the current level
            logger.info(f"Processing URLs at level {current_level}")
            level_result = level_processor.process_level(
                level=current_level,
                csv_path=str(self.data_folder / 'resources.csv'),
                max_urls=100,
                skip_vector_generation=True
            )
            
            # Calculate the next level to process
            next_level = min(current_level + 1, max_depth)
            
            # Get status of pending URLs at current level
            pending_urls = url_storage.get_pending_urls(depth=current_level)
            has_pending = len(pending_urls) > 0
            
            # Check if we're at max depth or have no pending URLs
            if current_level >= max_depth:
                logger.info(f"Reached maximum depth level {max_depth}")
                if has_pending:
                    logger.info(f"Still have {len(pending_urls)} pending URLs at level {current_level}, will continue on next run")
                else:
                    logger.info(f"No more pending URLs at level {current_level}, finished URL processing at max depth")
            elif not has_pending:
                # No more URLs at this level, increment level counter in config
                logger.info(f"No more pending URLs at level {current_level}, advancing to level {next_level}")
                if target_level is None:  # Only update config if using config-based level
                    self._save_current_level(next_level)
            else:
                logger.info(f"Still have {len(pending_urls)} pending URLs at level {current_level}, will continue on next run")
            
            # Get overall status
            level_status = level_processor.get_level_status()
            
            return {
                "status": "completed",
                "level_processed": current_level,
                "urls_processed": level_result.get("urls_processed", 0),
                "urls_added_to_resources": level_result.get("resources_created", 0),
                "current_level_remaining": len(pending_urls),
                "next_level": next_level if not has_pending and current_level < max_depth else current_level,
                "level_status": level_status
            }
            
        except Exception as e:
            logger.error(f"Error processing URLs: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "urls_processed": 0
            }


# Standalone function for easy import
async def process_pending_urls(
    data_folder: str,
    target_level: int = None, 
    max_depth: int = 3
) -> Dict[str, Any]:
    """
    Process pending URLs at the target level and discover URLs for deeper levels.
    
    Args:
        data_folder: Path to data folder
        target_level: Target level to process (defaults to current level from config)
        max_depth: Maximum depth level to process
        
    Returns:
        Dictionary with URL processing results
    """
    processor = URLProcessorStep(data_folder)
    return await processor.process_pending_urls(target_level=target_level, max_depth=max_depth)
