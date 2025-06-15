#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class ChangeDetector:
    """
    Detects changes in knowledge base files to determine if processing is needed.
    This component handles STEP 0 of the knowledge processing workflow.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the change detector.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = Path(data_folder)
        
    def get_last_processed_time(self) -> Optional[datetime]:
        """
        Get the timestamp of the last processing.
        
        Returns:
            Datetime object of last processing or None if not found
        """
        stats_path = self.data_folder / "stats" / "vector_stats.json"
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    previous_stats = json.load(f)
                    last_updated = previous_stats.get("last_updated")
                    if last_updated:
                        return datetime.fromisoformat(last_updated)
            except Exception as e:
                logger.warning(f"Could not read last processing time: {e}")
        
        return None
        
    def check_for_changes(self, last_processed_time: Optional[datetime] = None) -> Tuple[bool, List[str]]:
        """
        Check if there are changes in the data folder since last processing.
        
        Args:
            last_processed_time: The last time processing was run
            
        Returns:
            Tuple of (changes_detected, changed_files)
        """
        from api.routes.stats import check_for_file_changes
        
        # Define which file patterns to check
        patterns_to_check = [
            "*.csv",                  # CSV files in data directory
            "markdown/**/*.md",       # Markdown files
            "materials/**/*",         # Material files including PDFs
            "vector_stores/**/*.json" # Vector store metadata
        ]
        
        # Check for changes
        changes_detected, changed_files = check_for_file_changes(
            self.data_folder, patterns_to_check, last_processed_time
        )
        
        return changes_detected, changed_files
    
    def check_unanalyzed_resources(self) -> Tuple[bool, int]:
        """
        Check if there are resources that need analysis.
        
        Returns:
            Tuple of (unanalyzed_resources_exist, count_of_unanalyzed_resources)
        """
        import pandas as pd
        
        resources_path = self.data_folder / "resources.csv"
        if not resources_path.exists():
            logger.info("No resources.csv file found - will need content processing to create initial resources")
            return True, 0  # Return True to trigger content processing, but 0 count since file doesn't exist
            
        try:
            df = pd.read_csv(resources_path)
            # Check how many resources need analysis (analysis_completed is False)
            needs_analysis = []
            for _, row in df.iterrows():
                # Convert to boolean and check if analysis_completed is False
                if pd.isna(row.get('analysis_completed')) or row.get('analysis_completed') == False:
                    needs_analysis.append(row)
            
            unanalyzed_count = len(needs_analysis)
            if unanalyzed_count > 0:
                logger.info(f"Found {unanalyzed_count} resources with incomplete analysis")
                return True, unanalyzed_count
                
        except Exception as e:
            logger.error(f"Error checking for unanalyzed resources: {e}")
            
        return False, 0
    
    def determine_processing_needs(self, force_update: bool = False, check_unanalyzed: bool = True) -> Dict[str, Any]:
        """
        Determine what processing steps are needed based on changes and unanalyzed resources.
        
        Args:
            force_update: If True, forces a full update regardless of changes
            check_unanalyzed: If True, checks for unanalyzed resources
            
        Returns:
            Dictionary with processing decisions
        """
        logger.info("STEP 0: CHECKING FOR CHANGES SINCE LAST PROCESSING")
        logger.info("=================================================")
        
        if force_update:
            logger.info("Force update requested - proceeding with full processing regardless of changes")
            return {
                "process_all_steps": True,
                "unanalyzed_resources_exist": False,
                "reason": "force_update",
                "changes_detected": True,
                "changed_files": []
            }
        
        # Get the timestamp of the last processing
        last_processed_time = self.get_last_processed_time()
        if last_processed_time:
            logger.info(f"Last processing time: {last_processed_time}")
        
        # Check for changes
        changes_detected, changed_files = self.check_for_changes(last_processed_time)
        
        # Even if no file changes, check for unanalyzed resources if requested
        unanalyzed_resources_exist = False
        unanalyzed_count = 0
        
        if not changes_detected and check_unanalyzed:
            logger.info("No file changes detected, checking for unanalyzed resources...")
            unanalyzed_resources_exist, unanalyzed_count = self.check_unanalyzed_resources()
            
        # Determine if we need to process all steps or just a subset
        process_all_steps = changes_detected
        reason = "changes_detected" if changes_detected else "no_changes"
        
        if not changes_detected and unanalyzed_resources_exist:
            logger.info(f"Found {unanalyzed_count} resources with incomplete analysis - will perform partial processing")
            reason = "unanalyzed_resources"
        elif not changes_detected:
            logger.info("No changes detected since last processing and all resources are analyzed")
            logger.info("Only vector generation will be performed if requested")
            reason = "no_changes_or_unanalyzed"
        
        return {
            "process_all_steps": process_all_steps,
            "unanalyzed_resources_exist": unanalyzed_resources_exist,
            "unanalyzed_count": unanalyzed_count,
            "reason": reason,
            "changes_detected": changes_detected,
            "changed_files": changed_files
        }
    
    def save_file_state(self):
        """
        Save the current file state for future change detection.
        """
        try:
            from scripts.services.training_scheduler import save_file_state
            file_state = {
                "last_check": datetime.now().isoformat(),
                "file_timestamps": {},
                "file_hashes": {}
            }
            
            # Save the state to persist file tracking between runs
            save_file_state(file_state)
            logger.info("Saved file state after processing")
        except Exception as e:
            logger.error(f"Error saving file state: {e}")

# Standalone function for easy import
def check_for_processing_needs(data_folder: str, force_update: bool = False, check_unanalyzed: bool = True) -> Dict[str, Any]:
    """
    Check if knowledge processing is needed and which steps should be run.
    
    Args:
        data_folder: Path to data folder
        force_update: If True, forces a full update regardless of changes
        check_unanalyzed: If True, checks for unanalyzed resources
        
    Returns:
        Dictionary with processing decisions
    """
    detector = ChangeDetector(data_folder)
    return detector.determine_processing_needs(force_update, check_unanalyzed)