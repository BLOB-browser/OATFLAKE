#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CriticalContentProcessor:
    """
    Processes high priority content such as PDFs and methods files.
    This component handles STEP 1 of the knowledge processing workflow.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the critical content processor.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        
    async def process_critical_content(self) -> Dict[str, Any]:
        """
        Process PDFs and methods files which are considered highest priority content.
        
        Returns:
            Dictionary with processing results
        """
        logger.info("STEP 1: PROCESSING CRITICAL CONTENT (PDFs AND METHODS)")
        logger.info("=====================================================")
        
        try:
            from scripts.data.data_processor import DataProcessor
            processor = DataProcessor(self.data_folder, "default")
            
            # Process critical content (PDFs and methods)
            critical_content_result = await processor.process_critical_content()
            
            logger.info(f"Critical content processing completed: {critical_content_result}")
            return critical_content_result
            
        except Exception as e:
            logger.error(f"Error processing critical content: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

# Standalone function for easy import
async def process_critical_content(data_folder: str) -> Dict[str, Any]:
    """
    Process PDFs and methods files which are considered highest priority content.
    
    Args:
        data_folder: Path to data folder
        
    Returns:
        Dictionary with processing results
    """
    processor = CriticalContentProcessor(data_folder)
    return await processor.process_critical_content()