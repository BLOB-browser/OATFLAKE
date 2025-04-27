#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import sys
import os
import logging
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import from scripts
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def process_urls_at_level(level: int, force_fetch: bool = False, max_depth: int = 4):
    """Process URLs at a specific level."""
    try:
        from scripts.analysis.knowledge_orchestrator import KnowledgeOrchestrator
        from utils.config import get_data_path
        
        logger.info(f"Processing URLs at level {level}")
        
        # Get data path from config
        data_path = get_data_path()
        logger.info(f"Using data folder: {data_path}")
        
        # Create the knowledge orchestrator
        orchestrator = KnowledgeOrchestrator(data_path)
        
        # Process URLs at the specified level
        result = await orchestrator.process_knowledge(
            skip_markdown_scraping=True,
            analyze_resources=False,  # Skip resource analysis to focus on URL processing
            skip_vector_generation=True,  # Skip vector generation for faster processing
            skip_questions=True,  # Skip question generation
            skip_goals=True,  # Skip goal extraction
            max_depth=max_depth,
            force_url_fetch=force_fetch,
            process_level=level  # Process only URLs at this level
        )
        
        # Print results
        if result["status"] == "success":
            url_processing = result["data"].get("url_processing", {})
            logger.info(f"URLs processed: {url_processing.get('processed_urls_count', 0)}")
            logger.info(f"Success count: {url_processing.get('success_count', 0)}")
            logger.info(f"Error count: {url_processing.get('error_count', 0)}")
            
            # Check if there are still URLs pending at this level
            pending_by_level = url_processing.get("pending_urls_by_level", {})
            if level in pending_by_level and pending_by_level[level] > 0:
                logger.info(f"Still {pending_by_level[level]} URLs pending at level {level}")
            else:
                logger.info(f"All URLs at level {level} have been processed")
        else:
            logger.error(f"Processing failed: {result.get('message', 'Unknown error')}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing URLs at level {level}: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Process URLs at a specific level.')
    parser.add_argument('level', type=int, help='The level to process (e.g., 2 for level 2 URLs)')
    parser.add_argument('--force', action='store_true', help='Force reprocessing of URLs even if already processed')
    parser.add_argument('--max-depth', type=int, default=4, help='Maximum depth for discovery (default: 4)')
    
    args = parser.parse_args()
    
    # Run the URL processing
    asyncio.run(process_urls_at_level(args.level, args.force, args.max_depth))

if __name__ == "__main__":
    main()
