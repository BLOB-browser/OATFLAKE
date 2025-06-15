#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from typing import Dict, Any, List
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class MarkdownProcessingStep:
    """
    Processes markdown files to extract resources and content.
    This component handles STEP 2 of the knowledge processing workflow.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the markdown processor step.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = Path(data_folder)
        self.markdown_path = self.data_folder / "markdown"
        self.markdown_path.mkdir(parents=True, exist_ok=True)
        
    def get_markdown_files(self) -> List[Path]:
        """
        Get all markdown files in the markdown directory.
        
        Returns:
            List of Path objects for markdown files
        """
        return list(self.markdown_path.glob("**/*.md")) if self.markdown_path.exists() else []

    async def process_markdown_files(self, skip_scraping: bool = True, group_id: str = "default") -> Dict[str, Any]:
        """
        Process markdown files to extract resources and content.
        
        Args:
            skip_scraping: If True, don't scrape web content from markdown links
            group_id: Optional group ID (default: uses 'default')
            
        Returns:
            Dictionary with processing results
        """
        logger.info("STEP 2: PROCESSING MARKDOWN FILES FOR URL EXTRACTION")
        logger.info("=================================================")
        
        try:
            from scripts.data.markdown_processor import MarkdownProcessor
            
            # Initialize the markdown processor with data path and group id
            markdown_processor = MarkdownProcessor(self.data_folder, group_id)
            
            # Get all markdown files
            markdown_files = self.get_markdown_files()
            
            if not markdown_files:
                logger.info("No markdown files found to process")
                return {"status": "skipped", "data_extracted": {}}
            
            # Process markdown files 
            markdown_result = await markdown_processor.process_markdown_files(
                skip_scraping=skip_scraping
            )
            
            logger.info("=================================================")
            logger.info(f"Markdown processing completed: extracted {markdown_result.get('data_extracted', {}).get('resources', 0)} URLs")
            
            return markdown_result
            
        except Exception as e:
            logger.error(f"Error processing markdown files: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

# Standalone function for easy import
async def process_markdown_files(data_folder: str, skip_scraping: bool = True, group_id: str = "default") -> Dict[str, Any]:
    """
    Process markdown files to extract resources and content.
    
    Args:
        data_folder: Path to data folder
        skip_scraping: If True, don't scrape web content from markdown links
        group_id: Optional group ID (default: uses 'default')
        
    Returns:
        Dictionary with processing results
    """
    processor = MarkdownProcessingStep(data_folder)
    return await processor.process_markdown_files(skip_scraping, group_id)
