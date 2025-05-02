#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add the parent directory to the path so we can import from scripts
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def process_urls_at_level(level=2):
    """
    Process all pending URLs at a specific level.
    
    Args:
        level: The depth level to process (1=first level, 2=second level, etc.)
    """
    try:
        from scripts.analysis.knowledge_orchestrator import KnowledgeOrchestrator
        from utils.config import get_data_path
        
        logger.info(f"Starting to process URLs at level {level}")
        
        # Get data path
        data_path = get_data_path()
        logger.info(f"Using data folder: {data_path}")
        
        # Initialize orchestrator
        orchestrator = KnowledgeOrchestrator(data_path)
        
        # Process URLs at the specified level
        result = orchestrator.process_urls_at_level(
            level=level,
            batch_size=50,
            force_fetch=False
        )
        
        # Print results
        logger.info(f"URL processing at level {level} result: {result}")
            
        return result
    
    except Exception as e:
        logger.error(f"Error processing URLs at level {level}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    asyncio.run(process_urls_at_level(level=level))
