from fastapi import APIRouter, HTTPException, Request
from pathlib import Path
import pandas as pd
import json
import logging
import os
import glob
from datetime import datetime
from typing import Optional
from scripts.data.data_processor import DataProcessor
from scripts.data.markdown_processor import MarkdownProcessor
from scripts.services.data_analyser import DataAnalyser
from scripts.services.question_generator import generate_questions, save_questions, get_config_path
from scripts.analysis.goal_extractor import GoalExtractor
from utils.config import BACKEND_CONFIG, get_data_path

# Import check_for_file_changes from stats file to maintain compatibility
from .stats import check_for_file_changes

# Fix imports from storage module - this is already correct but the error suggests
# there might be another direct import somewhere else in the code
from scripts.services.storage import DataSaver  # Import DataSaver directly instead of individual methods

# Add imports for process cancellation
import threading
import time

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
        # Import our knowledge orchestrator
        from scripts.analysis.knowledge_orchestrator import KnowledgeOrchestrator
        from utils.config import get_data_path
        
        # Create orchestrator to send cancellation request
        orchestrator = KnowledgeOrchestrator(get_data_path())
        result = orchestrator.cancel_processing()
        
        # For backward compatibility, also set the old cancel event
        _processor_cancel_event.set()
        
        # If force is True, also try to cancel via interruptible_llm directly
        if force:
            from scripts.analysis.interruptible_llm import request_interrupt
            request_interrupt()
            
            # Also try cancellation in MainProcessor if available
            try:
                from scripts.analysis.main_processor import MainProcessor
                logger.info("Sending cancellation signal to all MainProcessor instances")
                
                # Set global cancellation flag if present
                if hasattr(MainProcessor, '_cancel_processing'):
                    MainProcessor._cancel_processing = True
                    logger.info("Set MainProcessor._cancel_processing = True")
            except Exception as e:
                logger.warning(f"Could not signal MainProcessor: {e}")
        
        # Wait a short time for processing to stop
        waited_time = 0
        max_wait_time = 5  # seconds
        check_interval = 0.5  # seconds
        
        while _processing_active and waited_time < max_wait_time:
            time.sleep(check_interval)
            waited_time += check_interval
        
        return {
            "status": "success",
            "message": "Cancellation signal sent" if _processing_active else "Processing stopped",
            "processing_stopped": not _processing_active
        }
        
    except Exception as e:
        logger.error(f"Error during cancellation: {e}")
        
        # Set cancellation flag as fallback
        _processor_cancel_event.set()
        
        return {
            "status": "partial_success",
            "message": f"Cancellation signal sent with warning: {e}"
        }

@router.post("/process")  # Now the full path will be /api/knowledge/process
async def process_knowledge_base(
    request: Request,
    group_id: str = None, 
    skip_markdown_scraping: bool = True, 
    analyze_resources: bool = True,
    analyze_all_resources: bool = False,
    batch_size: int = 5,
    resource_limit: Optional[int] = None,
    force_update: bool = False,
    skip_vector_generation: bool = False,
    check_unanalyzed: bool = True,  # New parameter to check for unanalyzed resources
    skip_questions: bool = False,   # Skip question generation (useful when time is limited)
    skip_goals: bool = False,       # Skip goal extraction (useful when time is limited)
    max_depth: int = 4,             # Maximum depth for crawling (higher = deeper crawling)
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
    
    logger.info(f"Starting comprehensive knowledge base processing... group_id={group_id}, force_update={force_update}")
    
    try:
        # Import our modular knowledge processing components
        from scripts.analysis.knowledge_orchestrator import KnowledgeOrchestrator
        
        # Get data path from config
        from utils.config import get_data_path
        data_path = get_data_path()
        
        # Create the knowledge orchestrator
        orchestrator = KnowledgeOrchestrator(data_path)
        
        # Process the knowledge base using the orchestrator
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
            process_level=process_level,  # Pass through the level to process
            auto_advance_level=auto_advance_level,  # Whether to advance to next level automatically
            continue_until_end=continue_until_end   # Whether to continue processing all levels
        )
        
        # Add group_id for frontend compatibility
        if "data" in result:
            result["data"]["group_id"] = group_id
        
        # Update processing flag
        _processing_active = False
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing knowledge base: {e}", exc_info=True)
        _processing_active = False
        raise HTTPException(status_code=500, detail=str(e))

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
        from utils.config import get_data_path
        data_path = get_data_path()
        
        # Import our goal extractor component
        from scripts.analysis.goal_extractor_step import GoalExtractorStep
        
        # Initialize goal extractor
        goal_extractor = GoalExtractorStep(data_path)
        
        # Extract goals using the client from the request state
        goals_result = await goal_extractor.extract_goals(ollama_client=request.app.state.ollama_client)
        
        if goals_result.get("status") == "success":
            return {
                "status": "success",
                "message": f"Successfully extracted {goals_result.get('goals_extracted', 0)} goals",
                "data": goals_result
            }
        else:
            return {
                "status": "warning",
                "message": goals_result.get("message", "Unknown issue during goal extraction"),
                "data": goals_result
            }
            
    except Exception as e:
        logger.error(f"Error extracting goals: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Add a stats endpoint in the knowledge router that calls the original in stats.py
@router.get("/stats")
async def get_knowledge_stats(group_id: str = None):
    """
    Get statistics about knowledge data and embeddings
    
    This endpoint redirects to the existing implementation in stats.py
    """
    # Import the original function from stats
    from .stats import get_knowledge_stats as original_stats_function
    
    # Call and return the result from the original implementation
    return await original_stats_function(group_id)
