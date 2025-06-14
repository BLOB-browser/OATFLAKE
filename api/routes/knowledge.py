from fastapi import APIRouter, Request
from pathlib import Path
import pandas as pd
import json
import logging
import os
from datetime import datetime
from typing import Optional
from scripts.analysis.goal_extractor import GoalExtractor
from utils.config import BACKEND_CONFIG, get_data_path

# Add imports for process cancellation
import threading
import time

# FLOW ANALYSIS AND RECOMMENDATIONS:
# ===================================
# 
# CURRENT ISSUE: Goals function and topic store generation are out of flow
# 
# PROBLEMS IDENTIFIED:
# 1. Goals extraction often skipped due to time constraints (skip_goals=True when <60min left)
# 2. Topic stores generated embedded within vector rebuild, not as orchestrated step
# 3. No dedicated time allocation for these critical components
# 4. No fallback mechanisms if these steps fail
#
# RECOMMENDED IMPROVED FLOW:
# 1. Change Detection → Check for file changes
# 2. Discovery Phase → Find URLs from resources  
# 3. Analysis Phase → Process URLs level by level
# 4. Content Processing → Handle PDFs, markdown
# 5. Vector Generation → Create basic embeddings
# 6. Topic Store Generation → DEDICATED STEP with fallback
# 7. Goal Extraction → DEDICATED STEP with priority
# 8. Question Generation → DEDICATED STEP 
#
# SCHEDULER IMPROVEMENTS NEEDED:
# - Allocate minimum 30 minutes for goals/topics even in short windows
# - Make goals extraction higher priority than vector rebuilding
# - Separate topic store generation from vector rebuilding
# - Add retry mechanisms for failed goals/topic generation
# - Track these steps separately in progress reporting

# Global flag for cancellation
_processing_active = False
_processor_cancel_event = threading.Event()

# Create router with a completely different prefix to avoid conflicts
router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])
logger = logging.getLogger(__name__)

@router.post("/cancel")
async def cancel_knowledge_processing(force: bool = False):
    """
    Cancel any running knowledge processing jobs.
    
    Args:
        force: If True, attempt to forcefully terminate processing
    """
    global _processing_active, _processor_cancel_event
    
    logger.info(f"Knowledge processing cancellation requested (force={force})")
    
    try:
        if not _processing_active:
            return {"status": "success", "message": "No active processing to cancel"}
        
        # Set cancellation flag
        _processor_cancel_event.set()
        
        # Import and use the orchestrator's cancel method if available
        from scripts.analysis.knowledge_orchestrator import KnowledgeOrchestrator
        data_path = get_data_path()
        orchestrator = KnowledgeOrchestrator(data_path)
        cancel_result = orchestrator.cancel_processing()
        
        # Also cancel any running LLM processes
        from scripts.analysis.interruptible_llm import request_interrupt
        request_interrupt()
        
        return {
            "status": "success", 
            "message": "Cancellation request sent to knowledge processor",
            "details": cancel_result
        }
        
    except Exception as e:
        logger.error(f"Error canceling knowledge processing: {e}", exc_info=True)
        return {"status": "error", "message": f"Error during cancellation: {str(e)}"}

@router.post("/process")  # Now the full path will be /api/knowledge/process
async def process_knowledge_base(
    request: Request,
    group_id: str = None, 
    skip_markdown_scraping: bool = True, 
    analyze_resources: bool = True,
    analyze_all_resources: bool = False,
    batch_size: int = 1,
    resource_limit: Optional[int] = None,
    force_update: bool = False,
    skip_vector_generation: bool = False,
    check_unanalyzed: bool = True,  # New parameter to check for unanalyzed resources
    skip_questions: bool = False,   # Skip question generation (useful when time is limited)
    skip_goals: bool = False,       # Skip goal extraction (useful when time is limited)
    max_depth: int = 5,             # Maximum depth for crawling - increased to 5 for full discovery
    force_url_fetch: bool = False,  # Only discover new URLs, don't reprocess already processed URLs (set to False to skip already processed URLs)
    process_level: Optional[int] = None,  # If specified, only process URLs at this level
    auto_advance_level: bool = False,     # If True, automatically advance to next level after completion
    continue_until_end: bool = False      # If True, process all levels until completion or cancellation
):
    """
    Process all knowledge base files and generate embeddings.
    
    Revised Steps:
    1. Check for changes since last processing
    2. Process PDFs from materials.csv (highest priority)
    3. Process methods from methods.csv (high priority)
    4. Process markdown files to extract structured data
    5. Optionally analyze resources with LLM
    6. Process remaining knowledge base CSVs to generate embeddings
    7. Extract goals from vector stores
    8. Generate questions from the processed knowledge
    
    Data Format Notes:
    - Definitions CSV should have 'title' and 'description' columns (preferred) or 'term' and 'definition'
    - Projects CSV should have 'title', 'description', and 'goals' columns
    - Methods CSV should have 'name', 'description', and 'steps' columns
    
    Processing Triggers:
    - Manual trigger from frontend UI (via direct API call)
    - Force update parameter (overrides change detection)
    - File changes detected since last processing
    - Scheduled via external task scheduler
    - Post upload of new material/resources
    - After model retraining processes
    
    Args:
        group_id: Optional group ID (default: uses 'default')
        skip_markdown_scraping: If True, don't scrape web content from markdown links
        analyze_resources: If True, analyze resources with LLM
        analyze_all_resources: If True, analyze all resources even if already analyzed
        batch_size: Number of resources to process at once
        resource_limit: Maximum number of resources to process
        force_update: If True, forces a full update regardless of changes
        skip_vector_generation: If True, skip vector generation in this step (for when it will be done later)
        check_unanalyzed: If True, always processes resources that haven't been analyzed yet,
                          even when no file changes are detected
        skip_questions: If True, skip question generation step (useful when time is limited)
        force_url_fetch: If True, force reprocess URLs even if they've been processed before
        skip_goals: If True, skip goal extraction step (useful when time is limited)
        process_level: If specified, only process URLs at this level (e.g., 2 for level 2 URLs)
    """
    global _processing_active, _processor_cancel_event
    
    # Reset cancellation flag at the start of processing
    _processor_cancel_event.clear()
    _processing_active = True
    
    # Load config to get the proper max_depth and validate process_level
    config_data = None
    config_path = None
    try:
        config_path = Path(__file__).parents[2] / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
            # Get max_depth from crawl_config, default to 4 if not specified
            crawl_config = config_data.get("crawl_config", {})
            config_max_depth = crawl_config.get("max_depth", 4)
            
            # ALWAYS use the config max_depth, regardless of whether it's higher or lower than the parameter
            if config_max_depth != max_depth:
                logger.info(f"Overriding max_depth={max_depth} with config value of {config_max_depth}")
                max_depth = config_max_depth
                
            # If no process_level specified, get from config but ensure it's valid
            if process_level is None:
                config_level = config_data.get("current_process_level", 1)
                # Only use config level if it's valid (>= 1)
                if config_level >= 1:
                    process_level = config_level
                    logger.info(f"Using process_level={process_level} from config")
                else:
                    logger.warning(f"Invalid current_process_level={config_level} in config, using level 1")
                    process_level = 1
            
            # Validate process_level is not 0
            if process_level is not None and process_level < 1:
                logger.warning(f"Invalid process_level={process_level}, must be >= 1. Using level 1.")
                process_level = 1
                
    except Exception as e:
        logger.error(f"Error reading config: {e}")
        # Use safe defaults
        if max_depth < 4:
            max_depth = 4  # Default to 4 if config reading fails
        if process_level is not None and process_level < 1:
            process_level = 1
    
    logger.info(f"Starting comprehensive knowledge base processing... group_id={group_id}, force_update={force_update}")
    logger.info(f"Using max_depth={max_depth}, process_level={process_level}")
    
    try:
        # Get data path from config
        data_path = get_data_path()
        
        # Use the knowledge orchestrator to handle the processing
        from scripts.analysis.knowledge_orchestrator import KnowledgeOrchestrator
        orchestrator = KnowledgeOrchestrator(data_path)
        
        # Process based on specified level or use full iterative flow
        if process_level is not None:
            logger.info(f"Processing URLs starting from level {process_level}")
            # Check if we should process pending URLs directly
            from scripts.analysis.url_storage import URLStorageManager
            
            # Check for pending URLs at the specified level first
            processed_urls_file = os.path.join(data_path, "processed_urls.csv")
            url_storage = URLStorageManager(processed_urls_file)
            
            # ALWAYS START FROM LOWEST AVAILABLE LEVEL
            # Check all levels to find the lowest level with pending URLs
            logger.info(f"Scanning all levels to find lowest available level with pending URLs...")
            
            pending_by_level = {}
            selected_level = None
            
            for level in range(1, max_depth + 1):
                level_urls = url_storage.get_pending_urls(depth=level)
                pending_count = len(level_urls) if level_urls else 0
                pending_by_level[level] = pending_count
                
                # Save the FIRST (lowest) level that has pending URLs
                if pending_count > 0 and selected_level is None:
                    selected_level = level
            
            # If we found a level with pending URLs and it's different from process_level, use the lowest
            if selected_level and selected_level != process_level:
                logger.info(f"Found pending URLs at lower level {selected_level}, using that instead of {process_level}")
                process_level = selected_level
            elif selected_level:
                logger.info(f"Confirmed level {process_level} has pending URLs, proceeding")
            
            # If no pending URLs at any level, let the orchestrator handle advancement
            if not selected_level:
                logger.warning("No pending URLs found at any level")
                
                # Log the counts at each level for debugging
                level_info = ", ".join([f"level {l}: {c}" for l, c in sorted(pending_by_level.items()) if c > 0])
                if level_info:
                    logger.info(f"Found pending URLs: {level_info}")
                else:
                    logger.warning("No pending URLs found at any level")
                
                # Let the orchestrator handle level advancement and discovery
                logger.info("Proceeding with orchestrator to handle level advancement or URL discovery")
            
            # Use the iterative approach from the orchestrator instead of individual level processing
            result = await orchestrator.process_knowledge(
                request_app_state=request.app.state,
                skip_markdown_scraping=skip_markdown_scraping,
                analyze_resources=analyze_resources,
                analyze_all_resources=analyze_all_resources,
                batch_size=batch_size,
                resource_limit=resource_limit,
                force_update=force_update,
                skip_vector_generation=skip_vector_generation,
                check_unanalyzed=check_unanalyzed,
                skip_questions=skip_questions,
                skip_goals=skip_goals,
                max_depth=max_depth,
                force_url_fetch=force_url_fetch,
                process_level=process_level,
                auto_advance_level=auto_advance_level,
                continue_until_end=continue_until_end
            )
            
            return result
        else:
            # Process in phases with the orchestrator
            result = await orchestrator.process_knowledge(
                request_app_state=request.app.state,
                skip_markdown_scraping=skip_markdown_scraping,
                analyze_resources=analyze_resources,
                analyze_all_resources=analyze_all_resources,
                batch_size=batch_size,
                resource_limit=resource_limit,
                force_update=force_update,
                skip_vector_generation=skip_vector_generation,
                check_unanalyzed=check_unanalyzed,
                skip_questions=skip_questions,
                skip_goals=skip_goals,
                max_depth=max_depth,
                force_url_fetch=force_url_fetch
            )
            
            return result
        
    except Exception as e:
        logger.error(f"Error processing knowledge base: {e}", exc_info=True)
        _processing_active = False
        return {"status": "error", "message": str(e)}

@router.post("/extract-goals")  # Now the full path will be /api/knowledge/extract-goals
async def extract_goals(request: Request):
    """
    Extract learning goals from vector stores and save them to goals.csv.
    
    Args:
        request: FastAPI request object containing the Ollama client
    """
    logger.info("Starting standalone goal extraction")
    try:
        # Get data path from config
        data_path = get_data_path()
        
        # Create goal extractor
        extractor = GoalExtractor(data_path)
        
        # Extract goals with client from request
        if hasattr(request.app.state, "ollama_client"):
            result = await extractor.extract_goals(request.app.state.ollama_client)
        else:
            # Try without client
            result = await extractor.extract_goals()
            
        # Return results
        return result
    except Exception as e:
        logger.error(f"Error extracting goals: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# Add a stats endpoint in the knowledge router that calls the original in stats.py
@router.get("/stats")
async def get_knowledge_stats(group_id: str = None):
    """
    Get statistics about knowledge data and embeddings
    
    This endpoint redirects to the existing implementation in stats.py
    """
    # Import the original function from stats
    from .stats import get_stats
    return await get_stats(group_id)
