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
        
    async def process_content(self, 
                             request_app_state=None,
                             skip_markdown_scraping: bool = True,
                             analyze_resources: bool = True,
                             analyze_all_resources: bool = False,
                             batch_size: int = 5,
                             resource_limit: int = None,
                             check_unanalyzed: bool = True) -> Dict[str, Any]:
        """
        Process content including critical content like PDFs and methods files.
        This is the method called by knowledge_orchestrator.
        
        Args:
            request_app_state: The FastAPI request.app.state
            skip_markdown_scraping: If True, don't scrape web content from markdown links
            analyze_resources: If True, analyze resources with LLM
            analyze_all_resources: If True, analyze all resources even if already analyzed
            batch_size: Number of resources to process at once
            resource_limit: Maximum number of resources to process
            check_unanalyzed: If True, always processes resources that haven't been analyzed yet
            
        Returns:
            Dictionary with processing results
        """
        logger.info("PROCESSING CONTENT INCLUDING CRITICAL CONTENT")
        logger.info("=============================================")
        
        try:
            # First process critical content (PDFs and methods)
            critical_result = await self.process_critical_content()
            
            # Return the results
            return {
                "status": "success",
                "critical_content": critical_result,
                "message": "Content processing completed"
            }
            
        except Exception as e:
            logger.error(f"Error in content processing: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
        
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