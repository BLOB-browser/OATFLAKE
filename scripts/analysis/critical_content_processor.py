#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class CriticalContentProcessor:
    """
    Processes high priority content such as PDFs.
    This component handles STEP 1 of the knowledge processing workflow.
    
    Methods CSV is handled separately by the reference store processing pipeline.
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
        Process content including critical content like PDFs.
        This is the method called by knowledge_orchestrator.
        
        Methods CSV processing is handled separately by the reference store pipeline.
        
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
        logger.info("PROCESSING CRITICAL CONTENT (HIGHEST PRIORITY)")
        logger.info("===============================================")
        logger.info("Critical content (PDFs) is always processed first,")
        logger.info("regardless of vector store existence or other conditions.")
        
        try:
            # First process critical content (PDFs)
            critical_result = await self.process_critical_content()
            
            # Process markdown files if not skipped
            markdown_result = {"status": "skipped", "reason": "skip_markdown_scraping=True"}
            if not skip_markdown_scraping:
                logger.info("STEP 2: PROCESSING MARKDOWN FILES FOR URL EXTRACTION")
                logger.info("=====================================================")
                
                from scripts.analysis.markdown_processor_step import MarkdownProcessingStep
                markdown_processor = MarkdownProcessingStep(self.data_folder)
                
                # Process markdown files to extract URLs
                markdown_result = await markdown_processor.process_markdown_files(
                    skip_scraping=skip_markdown_scraping,
                    group_id="default"
                )
                
                logger.info(f"Markdown processing completed: {markdown_result.get('status')}")
            
            # Return the results
            return {
                "status": "success",
                "critical_content": critical_result,
                "markdown_processing": markdown_result,
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
        Check for critical content (PDFs) but defer actual processing to universal rebuild system.
        This prevents duplication where PDFs are processed both here and in the universal rebuild.
        Methods CSV is handled separately by the reference store processing.
        
        Returns:
            Dictionary with processing results
        """
        logger.info("STEP 1: CHECKING FOR CRITICAL CONTENT (PDFs)")
        logger.info("=============================================")
        logger.info("📋 Critical content will be processed via universal rebuild system to avoid duplication")
        
        try:
            # Check for PDF files in materials folder (auto-downloaded PDFs)
            materials_folder = os.path.join(self.data_folder, 'materials')
            pdf_files = []
            
            if os.path.exists(materials_folder):
                for filename in os.listdir(materials_folder):
                    if filename.lower().endswith('.pdf'):
                        pdf_files.append(filename)
            
            # Check for materials.csv (uploaded PDFs)
            materials_csv_path = os.path.join(self.data_folder, 'materials.csv')
            materials_csv_exists = os.path.exists(materials_csv_path)
            
            # Determine if we have critical content (PDFs only)
            has_pdfs = len(pdf_files) > 0 or materials_csv_exists
            
            if has_pdfs:
                if len(pdf_files) > 0:
                    logger.info(f"✅ Found {len(pdf_files)} auto-downloaded PDF(s) in materials folder: {pdf_files[:3]}{'...' if len(pdf_files) > 3 else ''}")
                if materials_csv_exists:
                    logger.info(f"✅ Found materials.csv with uploaded PDFs - will be processed by universal rebuild")
            
            if not has_pdfs:
                logger.info("ℹ️  No critical content found (no PDFs in materials folder, no materials.csv)")
                return {
                    "status": "skipped",
                    "reason": "no_critical_content_files",
                    "materials_processed": 0
                }
            
            # Return successful result indicating critical content was found
            # Actual processing will be handled by universal rebuild system
            return {
                "status": "deferred_to_universal_rebuild",
                "materials_found": has_pdfs,
                "pdf_files_count": len(pdf_files),
                "materials_csv_exists": materials_csv_exists,
                "materials_processed": 0,  # Will be processed by universal rebuild
                "message": f"Critical content found ({len(pdf_files)} auto-downloaded PDFs, materials.csv: {materials_csv_exists}) - will be processed by universal rebuild system"
            }
            
        except Exception as e:
            logger.error(f"Error checking critical content: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

# Standalone function for easy import
async def process_critical_content(data_folder: str) -> Dict[str, Any]:
    """
    Process PDFs which are considered highest priority content.
    Methods CSV is handled separately by the reference store processing.
    
    Args:
        data_folder: Path to data folder
        
    Returns:
        Dictionary with processing results
    """
    processor = CriticalContentProcessor(data_folder)
    return await processor.process_critical_content()