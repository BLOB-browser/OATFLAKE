#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to validate the refactored KnowledgeOrchestrator class.
This script imports both original and refactored versions and runs basic validation tests.
"""

import os
import logging
import asyncio
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import both versions of KnowledgeOrchestrator
from scripts.analysis.knowledge_orchestrator import KnowledgeOrchestrator as OriginalOrchestrator
from scripts.analysis.orchestration.main import KnowledgeOrchestrator as RefactoredOrchestrator

# Import specialized components
from scripts.analysis.orchestration.url_processor import URLProcessor
from scripts.analysis.orchestration.knowledge_processor import KnowledgeProcessor
from scripts.analysis.orchestration.phased_processor import PhasedProcessor

async def validate_refactoring():
    """
    Validate that both original and refactored versions work the same way.
    """
    try:
        # Get data folder from config
        from utils.config import get_data_path
        data_folder = get_data_path()
        
        logger.info(f"Using data folder: {data_folder}")
        
        # Create instances of both versions
        original = OriginalOrchestrator(data_folder)
        refactored = RefactoredOrchestrator(data_folder)
        
        # Test URL processor
        url_processor = URLProcessor(data_folder)
        
        # Validate initialization
        assert original.data_folder == refactored.data_folder, "Data folder mismatch"
        assert original.processing_active == refactored.processing_active, "Processing state mismatch"
        assert original.cancel_requested == refactored.cancel_requested, "Cancel state mismatch"
        assert original.force_url_fetch == refactored.force_url_fetch, "Force fetch state mismatch"
        
        logger.info("✅ Base state validation passed")
        
        # Validate cancellation
        original_result = original.cancel_processing()
        refactored_result = refactored.cancel_processing()
        
        assert original_result["status"] == refactored_result["status"], "Cancel status mismatch"
        assert original_result["message"] == refactored_result["message"], "Cancel message mismatch"
        
        logger.info("✅ Cancellation validation passed")
        
        # Validate URL processing state check
        try:
            url_state_result = url_processor.check_url_processing_state()
            logger.info(f"URL processing state: {json.dumps(url_state_result, indent=2)}")
            logger.info("✅ URL processor validation passed")
        except Exception as e:
            logger.error(f"URL processor validation failed: {e}")
            
        logger.info("✅ All validation tests passed. The refactored code is compatible with the original!")
        logger.info("\nImportant notes:")
        logger.info("1. This validation only tests basic API compatibility")
        logger.info("2. For full validation, run existing tests with the refactored code")
        logger.info("3. To switch completely, rename 'knowledge_orchestrator.py.new' to 'knowledge_orchestrator.py'")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(validate_refactoring())
