#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class KnowledgeBaseProcessor:
    """
    Processes remaining knowledge base documents.
    This component handles STEP 4 of the knowledge processing workflow.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the knowledge base processor.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        
    async def process_remaining_knowledge(self, force_update: bool = False) -> Dict[str, Any]:
        """
        Process remaining knowledge base documents including definitions, projects, etc.
        
        Args:
            force_update: If True, forces a full update regardless of existing data
            
        Returns:
            Dictionary with processing results
        """
        logger.info("STEP 4: PROCESSING REMAINING KNOWLEDGE BASE DOCUMENTS")
        logger.info("====================================================")
        
        try:
            from scripts.data.data_processor import DataProcessor
            processor = DataProcessor(self.data_folder, "default")
            
            # Process the remaining content (definitions, projects, etc.)
            # Skip vector generation since we'll do it at the end
            logger.info("Processing remaining knowledge base content")
            logger.info("💡 Force update mode: " + ("Enabled" if force_update else "Disabled"))
            
            # Use incremental mode as specified by the force_update parameter
            result = await processor.process_knowledge_base(
                incremental=not force_update, 
                skip_vector_generation=True
            )
            
            logger.info(f"Remaining knowledge base processing completed")
            return result
            
        except Exception as e:
            logger.error(f"Error processing remaining knowledge base documents: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

# Standalone function for easy import
async def process_remaining_knowledge(data_folder: str, force_update: bool = False) -> Dict[str, Any]:
    """
    Process remaining knowledge base documents including definitions, projects, etc.
    
    Args:
        data_folder: Path to data folder
        force_update: If True, forces a full update regardless of existing data
        
    Returns:
        Dictionary with processing results
    """
    processor = KnowledgeBaseProcessor(data_folder)
    return await processor.process_remaining_knowledge(force_update)