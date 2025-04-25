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
    
    if not _processing_active:
        return {
            "status": "success", 
            "message": "No active processing to cancel"
        }
    
    # Set cancellation flag
    _processor_cancel_event.set()
    logger.info("Cancellation flag set for knowledge processing")
    
    if force:
        # Send termination signal to MainProcessor if we can find a reference
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
    max_depth: int = 4              # Maximum depth for crawling (higher = deeper crawling)
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
        skip_goals: If True, skip goal extraction step (useful when time is limited)
    """
    global _processing_active, _processor_cancel_event
    
    # Reset cancellation flag at the start of processing
    _processor_cancel_event.clear()
    _processing_active = True
    
    logger.info(f"Starting comprehensive knowledge base processing... group_id={group_id}, force_update={force_update}")
    try:
        # Ensure we're using the imported BACKEND_CONFIG
        from utils.config import BACKEND_CONFIG as config
        # Get data path from config
        data_path = Path(config['data_path'])
        # Always use "default" for group_id since we don't store in group-specific folders
        group_id_to_use = "default"
        
        # Step 0: Check for changes to determine if processing is needed
        process_all_steps = True  # Default to processing all steps
        unanalyzed_resources_exist = False
        
        if not force_update:
            logger.info("STEP 0: CHECKING FOR CHANGES SINCE LAST PROCESSING")
            logger.info("=================================================")
            
            # Get the timestamp of the last processing
            last_processed_time = None
            stats_path = data_path / "stats" / "vector_stats.json"
            if stats_path.exists():
                try:
                    with open(stats_path, 'r') as f:
                        previous_stats = json.load(f)
                        last_updated = previous_stats.get("last_updated")
                        if last_updated:
                            last_processed_time = datetime.fromisoformat(last_updated)
                            logger.info(f"Last processing time: {last_processed_time}")
                except Exception as e:
                    logger.warning(f"Could not read last processing time: {e}")
            
            # Define which file patterns to check
            patterns_to_check = [
                "*.csv",                  # CSV files in data directory
                "markdown/**/*.md",       # Markdown files
                "materials/**/*",         # Material files including PDFs
                "vector_stores/**/*.json" # Vector store metadata
            ]
            
            # Check for changes - this updates internal file states if needed
            from scripts.services.training_scheduler import load_file_state
            
            # Get initial state to track file changes
            _file_state = load_file_state()
            
            # Check for changes
            changes_detected, changed_files = check_for_file_changes(data_path, patterns_to_check, last_processed_time)
            
            # Even if no file changes, check for unanalyzed resources if requested
            if not changes_detected and check_unanalyzed and analyze_resources:
                logger.info("No file changes detected, checking for unanalyzed resources...")
                
                resources_path = data_path / "resources.csv"
                if resources_path.exists():
                    try:
                        df = pd.read_csv(resources_path)
                        # Check how many resources need analysis (analysis_completed is False)
                        needs_analysis = []
                        for _, row in df.iterrows():
                            # Convert to boolean and check if analysis_completed is False
                            if pd.isna(row.get('analysis_completed')) or row.get('analysis_completed') == False:
                                needs_analysis.append(row)
                        
                        unanalyzed_count = len(needs_analysis)
                        if unanalyzed_count > 0:
                            logger.info(f"Found {unanalyzed_count} resources with incomplete analysis")
                            unanalyzed_resources_exist = True
                            # We'll do partial processing - just for resources
                            process_all_steps = False
                        else:
                            logger.info("All resources have been analyzed")
                            
                    except Exception as e:
                        logger.error(f"Error checking for unanalyzed resources: {e}")
                
                if not unanalyzed_resources_exist:
                    logger.info("No changes detected since last processing and all resources are analyzed - skipping content processing steps")
                    logger.info("BUT still generating vector stores as requested")
                    # We'll skip content processing steps but still do vector generation
                    process_all_steps = False
                    
                # Even if no changes, update the last check time
                try:
                    from scripts.services.training_scheduler import save_file_state
                    _file_state["last_check"] = datetime.now().isoformat()
                    save_file_state(_file_state)
                except Exception as e:
                    logger.error(f"Error updating file state: {e}")
            else:
                if changes_detected:
                    logger.info(f"Changes detected in {len(changed_files)} files - proceeding with full processing")
                    if len(changed_files) <= 5:  # Only log all files if there aren't too many
                        logger.info(f"Changed files: {changed_files}")
        else:
            logger.info("Force update requested - proceeding with full processing regardless of changes")
        
        # Initialize result object
        result = {
            "vector_generation": {},
            "content_processing": {"status": "skipped" if not process_all_steps else "pending"}
        }
        
        # Check for cancellation before starting processing
        if _processor_cancel_event.is_set():
            logger.info("Knowledge processing cancelled before starting")
            _processing_active = False
            return {
                "status": "cancelled",
                "message": "Knowledge processing cancelled before starting",
                "data": result
            }
        
        # Only process content if changes detected or force_update is True OR unanalyzed resources exist
        if process_all_steps or unanalyzed_resources_exist:
            # Add check for cancellation before each major step
            
            # Step 1: Process PDFs and Methods first (highest priority content)
            if _processor_cancel_event.is_set():
                logger.info("Knowledge processing cancelled before step 1")
                _processing_active = False
                return {"status": "cancelled", "message": "Processing cancelled", "data": result}
                
            logger.info("STEP 1: PROCESSING CRITICAL CONTENT (PDFs AND METHODS)")
            logger.info("=====================================================")
            processor = DataProcessor(data_path, group_id_to_use)
            critical_content_result = await processor.process_critical_content()
            logger.info(f"Critical content processing completed: {critical_content_result}")
            
            # Step 2: Process markdown files to extract resources
            if _processor_cancel_event.is_set():
                logger.info("Knowledge processing cancelled before step 2")
                _processing_active = False
                return {"status": "cancelled", "message": "Processing cancelled", "data": result}
                
            markdown_result = {}
            markdown_path = data_path / "markdown"
            markdown_files = list(markdown_path.glob("**/*.md")) if markdown_path.exists() else []

            if markdown_files:
                logger.info("STEP 2: PROCESSING MARKDOWN FILES FOR URL EXTRACTION")
                logger.info("=================================================")
                markdown_processor = MarkdownProcessor(data_path, group_id_to_use)
                
                # Process markdown files to extract resources
                # Setting analyze_resources=False here to do it separately in Step 3
                markdown_result = await markdown_processor.process_markdown_files(
                    skip_scraping=skip_markdown_scraping,
                    analyze_resources=False  # We'll analyze in a separate step for better control
                )
                
                logger.info("=================================================")
                logger.info(f"Markdown processing completed: extracted {markdown_result.get('data_extracted', {}).get('resources', 0)} URLs")
            else:
                logger.info("No markdown files found to process")
                markdown_result = {"status": "skipped", "data_extracted": {}}
            
            # Step 3: Analyze resources
            if _processor_cancel_event.is_set():
                logger.info("Knowledge processing cancelled before step 3")
                _processing_active = False
                return {"status": "cancelled", "message": "Processing cancelled", "data": result}
                
            resource_analysis_result = {}
            resources_path = data_path / "resources.csv"
            
            # Always check if resources exist, and analyze them if requested
            if resources_path.exists() and analyze_resources:
                # Check total resources to potentially analyze
                try:
                    df = pd.read_csv(resources_path)
                    total_resources = len(df)
                    
                    logger.info(f"Found {total_resources} total resources")
                    
                    # Find resources that need analysis if not analyzing all
                    needs_analysis_count = 0
                    if not analyze_all_resources:
                        # Check how many resources need analysis (analysis_completed is False)
                        needs_analysis = []
                        for _, row in df.iterrows():
                            # Convert to boolean and check if analysis_completed is False
                            if pd.isna(row.get('analysis_completed')) or row.get('analysis_completed') == False:
                                needs_analysis.append(row)
                        needs_analysis_count = len(needs_analysis)
                        logger.info(f"Found {needs_analysis_count} resources with analysis_completed=False that need analysis")
                    
                    # Skip analysis ONLY if nothing needs analysis AND we're not forcing reanalysis
                    if needs_analysis_count == 0 and not analyze_all_resources and not force_update:
                        logger.info("All resources already have analysis results - skipping resource analysis")
                        resource_analysis_result = {
                            "status": "skipped",
                            "reason": "all resources already analyzed",
                            "resources_analyzed": 0
                        }
                    else:
                        # Analyze resources if any need it or we're forcing reanalysis
                        logger.info("STEP 3: ANALYZING RESOURCES WITH LLM")
                        logger.info("===================================")
                        analyzer = DataAnalyser()
                        
                        # If we're only processing due to unanalyzed resources, adjust the message
                        if unanalyzed_resources_exist and not process_all_steps:
                            logger.info("Processing ONLY resources with incomplete analysis (skipping other steps)")
                        
                        try:
                            logger.info(f"Starting resource analysis with LLM analyzer")
                            logger.info(f"Processing options: batch_size={batch_size}, limit={resource_limit}, analyze_all={analyze_all_resources}")
                            
                            # Use the full resource limit as provided (or None for all)
                            actual_limit = resource_limit
                            
                            # Analyze resources using LLM
                            updated_resources, projects = analyzer.analyze_resources(
                                csv_path=str(resources_path),
                                batch_size=batch_size,
                                max_resources=actual_limit,
                                force_reanalysis=analyze_all_resources or force_update
                            )
                            
                            # Get extracted definitions
                            definitions = analyzer._get_definitions_from_resources(updated_resources)
                            
                            # We can't get methods directly from processing_result since analyze_resources doesn't return it
                            # Instead, extract methods from CSV or create an empty list
                            methods_path = os.path.join(get_data_path(), 'methods.csv')
                            methods = []
                            
                            # If methods CSV exists, read it
                            if os.path.exists(methods_path):
                                try:
                                    methods_df = pd.read_csv(methods_path)
                                    # Get only recent methods (last 24 hours)
                                    recent_time = (datetime.now() - pd.Timedelta(days=1)).isoformat()
                                    if 'created_at' in methods_df.columns:
                                        recent_methods = methods_df[methods_df['created_at'] > recent_time]
                                        methods = recent_methods.to_dict('records')
                                        logger.info(f"Retrieved {len(methods)} recently added methods from {methods_path}")
                                except Exception as m_err:
                                    logger.error(f"Error retrieving methods from CSV: {m_err}")
                            
                            # Count resources that were actually processed by the LLM
                            resources_with_llm = sum(1 for r in updated_resources 
                                                if r.get('analysis_results') and 
                                                isinstance(r.get('analysis_results'), dict))
                            
                            logger.info(f"Successfully processed {resources_with_llm} resources with LLM analysis")
                            
                            # Save processed data
                            analyzer.save_updated_resources(updated_resources)
                            logger.info(f"✅ Resources saved successfully, moving to next step in pipeline")
                            
                            # Save projects separately if any were found
                            if projects:
                                analyzer.save_projects_csv(projects)
                                logger.info(f"Saved {len(projects)} identified projects to projects.csv")
                            
                            # Save methods separately if any were found
                            if methods:
                                # Use data_saver directly to ensure methods get saved
                                from scripts.analysis.data_saver import DataSaver
                                from utils.config import BACKEND_CONFIG
                                data_saver = DataSaver(BACKEND_CONFIG['data_path'])
                                data_saver.save_methods(methods)
                                logger.info(f"Saved {len(methods)} extracted methods to methods.csv")
                            
                            logger.info(f"Found {len(definitions)} definitions, {len(projects)} projects, and {len(methods)} methods")
                            logger.info(f"✅ Resource analysis phase complete - continuing to vector embedding phase")
                            
                            resource_analysis_result = {
                                "resources_processed": len(updated_resources),
                                "resources_analyzed": resources_with_llm,
                                "resources_updated": sum(1 for r in updated_resources 
                                                    if r.get('tags') and 
                                                    isinstance(r.get('tags'), list) and 
                                                    len(r.get('tags')) > 0),
                                "definitions_extracted": len(definitions),
                                "projects_identified": len(projects)
                            }
                            logger.info(f"Resource analysis completed: {resource_analysis_result}")
                        except Exception as analyze_error:
                            logger.error(f"Error during resource analysis: {analyze_error}", exc_info=True)
                            resource_analysis_result = {
                                "status": "error",
                                "error": str(analyze_error)
                            }
                except Exception as resource_error:
                    logger.error(f"Error checking resources: {resource_error}")
                    resource_analysis_result = {"status": "error", "error": str(resource_error)}
            else:
                if not resources_path.exists():
                    logger.info("No resources.csv found, skipping analysis")
                    resource_analysis_result = {"status": "skipped", "reason": "no resources.csv found"}
                else:
                    logger.info("Resource analysis disabled by parameter, skipping")
                    resource_analysis_result = {"status": "skipped", "reason": "analyze_resources=False"}
            
            # Only continue with other steps if full processing is needed
            if process_all_steps:
                # Step 4: Process remaining knowledge base documents
                logger.info("STEP 4: PROCESSING REMAINING KNOWLEDGE BASE DOCUMENTS")
                logger.info("====================================================")
                
                # We already created the processor in Step 1, so reuse it
                # Process the remaining content (definitions, projects, etc)
                # Pass skip_vector_generation flag to avoid redundant processing
                # We want to do vector generation in one place - at the end of this method
                logger.info("Processing remaining knowledge base content")
                logger.info("💡 Force update mode: " + ("Enabled" if force_update else "Disabled"))
                
                # Use incremental mode as specified by the force_update parameter
                # Skip vector generation in MainProcessor since we'll do it here
                result = await processor.process_knowledge_base(incremental=not force_update, skip_vector_generation=True)
                
                # Add markdown processing results to the overall result
                result["markdown_processing"] = markdown_result
                
                # Add resource analysis results
                result["resource_analysis"] = resource_analysis_result
                
                # After processing all content, update result
                result["content_processing"] = {"status": "success"}
        
        # Step 4.5: Process pending URLs at deeper levels (if any)
        if not _processor_cancel_event.is_set():
            logger.info("STEP 4.5: CHECKING FOR PENDING URLS AT DEEPER LEVELS")
            logger.info("===================================================")
            try:
                # Import necessary components
                from scripts.analysis.url_storage import URLStorageManager
                from scripts.analysis.level_processor import LevelBasedProcessor
                import os
                
                # Initialize URL storage manager
                processed_urls_file = os.path.join(get_data_path(), "processed_urls.csv")
                url_storage = URLStorageManager(processed_urls_file)
                
                # Get the CSV path for resources
                resources_csv_path = str(resources_path) if resources_path.exists() else None
                
                # Counter for total URLs processed across all iterations
                total_urls_processed = 0
                all_level_processing_results = []
                
                # Process pending URLs in a continuous loop until no more pending URLs are found
                # or processing is cancelled by the training scheduler
                processing_rounds = 0
                
                while not _processor_cancel_event.is_set():
                    processing_rounds += 1
                    
                    # Get the status of all levels
                    level_processor = LevelBasedProcessor(str(data_path))
                    level_status = level_processor.get_level_status()
                    
                    pending_levels = []
                    total_pending_urls = 0
                    
                    # Find levels with pending URLs
                    for level, status in level_status.items():
                        if not status["is_complete"]:
                            pending_count = status["pending"]
                            if pending_count > 0:
                                pending_levels.append(level)
                                total_pending_urls += pending_count
                    
                    # If no more pending URLs, check if we should advance to the next level
                    if not pending_levels:
                        logger.info(f"No more pending URLs found after {processing_rounds} processing rounds")
                        
                        # Check if we should discover URLs at the next level
                        current_max_level = max([int(level) for level in level_status.keys()]) if level_status else 1
                        next_level = current_max_level + 1
                        
                        # Only proceed if we're still within max_depth
                        if next_level <= max_depth:
                            logger.info(f"Advancing to next depth level: {next_level} (max_depth: {max_depth})")
                            
                            # Initialize ContentFetcher to discover URLs at the next level
                            from scripts.analysis.content_fetcher import ContentFetcher
                            content_fetcher = ContentFetcher()
                            
                            # Get some processed URLs from the previous level to use as starting points
                            processed_urls = url_storage.get_processed_urls_by_level(current_max_level)
                            
                            if processed_urls:
                                logger.info(f"Found {len(processed_urls)} processed URLs from level {current_max_level} to use as starting points")
                                
                                # Take a sample of URLs to process (to avoid excessive processing)
                                sample_size = min(len(processed_urls), 20)
                                urls_to_process = processed_urls[:sample_size]
                                
                                # Process each URL to discover links at the next level
                                for url_data in urls_to_process:
                                    url = url_data.get('url')
                                    if url:
                                        logger.info(f"Discovering level {next_level} URLs from {url}")
                                        try:
                                            # Use get_main_page_with_links with discover_only_level set to next_level
                                            content_fetcher.get_main_page_with_links(
                                                url=url,
                                                max_depth=max_depth,
                                                discover_only_level=next_level
                                            )
                                        except Exception as e:
                                            logger.error(f"Error discovering URLs for level {next_level} from {url}: {e}")
                                
                                logger.info(f"Completed discovery for level {next_level}")
                                
                                # Check if we discovered any new URLs for the next level
                                level_status = level_processor.get_level_status()
                                if str(next_level) in level_status and level_status[str(next_level)]["pending"] > 0:
                                    logger.info(f"Successfully discovered {level_status[str(next_level)]['pending']} URLs for level {next_level}")
                                    # Continue the loop to process these newly discovered URLs
                                    continue
                            else:
                                logger.info(f"No processed URLs found from level {current_max_level} to use as starting points")
                        else:
                            logger.info(f"Reached maximum depth ({max_depth}), not discovering deeper levels")
                        
                        # If we didn't continue the loop above, break out now
                        break
                        
                    logger.info(f"Round {processing_rounds}: Found {total_pending_urls} pending URLs across {len(pending_levels)} levels: {pending_levels}")
                    
                    # Process all pending levels, starting from the lowest level
                    pending_levels.sort()  # Sort levels in ascending order
                    round_level_results = []
                    
                    for next_level in pending_levels:
                        logger.info(f"Round {processing_rounds}: Processing level {next_level} which has pending URLs")
                        
                        # Process this level with a larger batch size to process more URLs at once
                        level_result = level_processor.process_level(
                            level=next_level,
                            csv_path=resources_csv_path,
                            max_urls=100,  # Process more URLs at once
                            skip_vector_generation=True  # Skip vector generation here, we'll do it in the next step
                        )
                        
                        urls_processed = level_result.get("urls_processed", 0)
                        total_urls_processed += urls_processed
                        
                        round_level_results.append({
                            "level": next_level,
                            "result": level_result
                        })
                        
                        logger.info(f"Round {processing_rounds}: Level {next_level} processing completed with result: processed {urls_processed} URLs")
                        
                        # Check if processing was cancelled mid-way
                        if _processor_cancel_event.is_set():
                            logger.info("Processing cancelled during level processing")
                            break
                    
                    # Add the results from this round to the overall results
                    all_level_processing_results.append({
                        "round": processing_rounds,
                        "levels_processed": round_level_results
                    })
                    
                    # Check again if we should continue processing
                    if level_processor.check_cancellation():
                        logger.info("Processing cancelled after level processing round")
                        break
                
                # Add the final results to the overall result
                if total_urls_processed > 0:
                    result["level_processing"] = {
                        "status": "success",
                        "total_urls_processed": total_urls_processed,
                        "processing_rounds": processing_rounds,
                        "details": all_level_processing_results
                    }
                else:
                    logger.info("No pending URLs were processed at any level")
                    result["level_processing"] = {"status": "skipped", "reason": "no pending URLs processed"}
                    
            except Exception as level_error:
                logger.error(f"Error processing pending URLs at deeper levels: {level_error}", exc_info=True)
                result["level_processing"] = {"status": "error", "error": str(level_error)}
        
        # Generate vector stores if not cancelled
        if not skip_vector_generation and not _processor_cancel_event.is_set():
            logger.info("STEP 5: GENERATING VECTOR STORES FROM PROCESSED CONTENT")
            logger.info("======================================================")
            try:
                # Get all content files for vector generation
                from scripts.analysis.vector_generator import VectorGenerator
                from scripts.storage.vector_store_manager import VectorStoreManager
                import asyncio
                
                # Get temp storage path from ContentStorageService
                from scripts.storage.content_storage_service import ContentStorageService
                content_storage = ContentStorageService(str(data_path))
                
                # Look for JSONL files in multiple locations
                content_paths = list(content_storage.temp_storage_path.glob("*.jsonl"))
                temp_dir = data_path / "temp"
                if temp_dir.exists():
                    content_paths.extend(list(temp_dir.glob("*.jsonl")))
                    # Also check vector_data folder if it exists
                    vector_data_path = temp_dir / "vector_data"
                    if vector_data_path.exists():
                        content_paths.extend(list(vector_data_path.glob("*.jsonl")))
                    
                # Check if we should proceed even if no content files found
                if not content_paths and force_update:
                    logger.info("No new content files found. Looking for existing content to vectorize.")
                    # Look for existing content files
                    existing_content_dir = data_path / "content"
                    if existing_content_dir.exists():
                        content_paths.extend(list(existing_content_dir.glob("*.jsonl")))
                        logger.info(f"Found {len(content_paths)} existing content files to vectorize from content directory")
                
                if content_paths:
                    logger.info(f"Found {len(content_paths)} content files for vector generation")
                    
                    # Instead of using VectorGenerator, directly execute the rebuild_faiss_indexes.py script
                    logger.info("Running rebuild_faiss_indexes.py script to generate vector stores")
                    
                    import subprocess
                    import sys
                    
                    # Get the path to the rebuild script
                    script_path = Path(__file__).parents[2] / "scripts" / "tools" / "rebuild_faiss_indexes.py"
                    
                    # Run the script as a separate process to ensure it has full stdout/stderr
                    logger.info(f"Executing rebuild script: {script_path}")
                    rebuild_start_time = time.time()
                    
                    try:
                        # Run with the same Python interpreter
                        python_path = sys.executable
                        cmd = [python_path, str(script_path), "--data-path", str(data_path)]
                        
                        # Execute with real-time output
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True,
                            bufsize=1
                        )
                        
                        # Stream the output in real-time
                        logger.info("===== BEGIN REBUILD SCRIPT OUTPUT =====")
                        rebuild_output = []
                        for line in process.stdout:
                            line = line.strip()
                            rebuild_output.append(line)
                            if line:
                                logger.info(f"REBUILD: {line}")
                        
                        # Wait for process to complete
                        return_code = process.wait()
                        logger.info("===== END REBUILD SCRIPT OUTPUT =====")
                        rebuild_duration = time.time() - rebuild_start_time
                        
                        # Parse output to extract stats
                        rebuilt_stores = []
                        total_documents = 0
                        
                        for line in rebuild_output:
                            if "Successfully rebuilt" in line and "store" in line:
                                store_name = line.split("Successfully rebuilt")[-1].strip()
                                rebuilt_stores.append(store_name)
                            elif "documents" in line and ":" in line:
                                try:
                                    store_info = line.strip().split(":")
                                    if len(store_info) >= 2 and "documents" in store_info[1]:
                                        doc_count = int(store_info[1].split("documents")[0].strip())
                                        total_documents += doc_count
                                except:
                                    pass
                        
                        if return_code == 0:
                            vector_stats = {
                                "status": "success",
                                "documents_processed": total_documents,
                                "embeddings_generated": total_documents,
                                "stores_created": rebuilt_stores,
                                "topic_stores_created": [store for store in rebuilt_stores if store.startswith("topic_")],
                                "duration_seconds": rebuild_duration
                            }
                            logger.info(f"✅ Successfully rebuilt all FAISS indexes")
                        else:
                            vector_stats = {
                                "status": "error",
                                "message": f"Rebuild script failed with return code {return_code}",
                                "duration_seconds": rebuild_duration
                            }
                            logger.error(f"❌ Failed to rebuild indexes, return code: {return_code}")
                    
                    except Exception as e:
                        logger.error(f"Error executing rebuild script: {e}")
                        vector_stats = {
                            "status": "error",
                            "message": str(e),
                            "duration_seconds": time.time() - rebuild_start_time
                        }
                    
                    # Add vector stats to the result
                    result["vector_generation"] = vector_stats
                    logger.info(f"Vector generation complete: {vector_stats}")
                    
                    # Note: Topic store generation is now handled directly in the rebuild_faiss_indexes.py script
                    # We don't need to regenerate them here, as that would cause duplication
                    logger.info("Topic store generation is now handled by the rebuild_faiss_indexes.py script")
                else:
                    # Even if no content files found, try to rebuild existing stores
                    logger.info("No content files found, attempting to rebuild existing vector stores")
                    
                    # Use VectorStoreManager's capabilities directly
                    try:
                        from scripts.storage.vector_store_manager import VectorStoreManager
                        
                        # Initialize VectorStoreManager
                        vector_store_manager = VectorStoreManager(base_path=data_path)
                        
                        # List existing stores
                        stores = vector_store_manager.list_stores()
                        
                        # Record which stores were successfully rebuilt
                        rebuilt_stores = []
                        topic_stores = []
                        
                        # Force rebuild of existing stores
                        for store in stores:
                            store_name = store.get("name")
                            
                            # Skip if not a real store (metadata only)
                            if not store_name:
                                continue
                                
                            # For each store, try to rebuild topic stores
                            if store_name == "content_store":
                                logger.info("Attempting to rebuild topic stores from content_store")
                                
                                # Get representative chunks
                                rep_docs = await vector_store_manager.get_representative_chunks(
                                    store_name=store_name, 
                                    num_chunks=100
                                )
                                
                                if rep_docs:
                                    # Create topic stores
                                    topic_results = await vector_store_manager.create_topic_stores(rep_docs)
                                    
                                    # Track rebuilt topic stores
                                    for topic, success in topic_results.items():
                                        if success:
                                            topic_stores.append(f"topic_{topic}")
                            
                            # Track this store as rebuilt
                            rebuilt_stores.append(store_name)
                        
                        # Record rebuild results
                        result["vector_generation"] = {
                            "status": "success",
                            "message": "Rebuilt existing vector stores",
                            "stores_created": rebuilt_stores,
                            "topic_stores_created": topic_stores,
                            "duration_seconds": 0
                        }
                        logger.info(f"Vector store rebuild complete: rebuilt {len(rebuilt_stores)} stores and {len(topic_stores)} topic stores")
                        
                    except Exception as rebuild_error:
                        logger.error(f"Error rebuilding vector stores: {rebuild_error}")
                        result["vector_generation"] = {
                            "status": "error",
                            "message": "Error rebuilding vector stores",
                            "error": str(rebuild_error)
                        }
            except Exception as ve:
                logger.error(f"Error during vector generation: {ve}")
                result["vector_generation"] = {"status": "error", "error": str(ve)}
                
            # Clean up all temporary files after vector generation
            try:
                processor = DataProcessor(data_path, group_id_to_use)
                processor._cleanup_temporary_files()
            except Exception as ce:
                logger.error(f"Error during cleanup: {ce}")
        elif _processor_cancel_event.is_set():
            logger.info("Vector generation skipped due to cancellation")
            result["vector_generation"] = {"status": "cancelled"}
        else:
            logger.info("Vector generation explicitly skipped")
            result["vector_generation"] = {"status": "skipped", "reason": "explicitly skipped"}

        # Step 6: Extract goals from vector stores (unless skipped)
        if skip_goals:
            logger.info("STEP 6: SKIPPING GOAL EXTRACTION (skip_goals=True)")
            result["goals"] = {
                "status": "skipped",
                "message": "Goal extraction explicitly skipped"
            }
        else:
            logger.info("STEP 6: EXTRACTING GOALS FROM VECTOR STORES")
            logger.info("=========================================")
            
            try:
                # Initialize goal extractor
                goal_extractor = GoalExtractor(str(data_path))
                
                # Extract goals using the current ollama client
                goals_result = await goal_extractor.extract_goals(ollama_client=request.app.state.ollama_client)
                
                if goals_result.get("status") == "success":
                    logger.info(f"Successfully extracted {goals_result.get('stats', {}).get('goals_extracted', 0)} goals")
                    result["goals"] = {
                        "goals_extracted": goals_result.get("stats", {}).get("goals_extracted", 0),
                        "stores_analyzed": goals_result.get("stats", {}).get("stores_analyzed", []),
                        "duration_seconds": goals_result.get("stats", {}).get("duration_seconds", 0)
                    }
                else:
                    logger.warning(f"Goal extraction issue: {goals_result.get('message')}")
                    result["goals"] = {
                        "status": "warning",
                        "message": goals_result.get("message", "Unknown issue during goal extraction")
                    }
            except Exception as goal_error:
                logger.error(f"Error during goal extraction: {goal_error}", exc_info=True)
                result["goals"] = {
                    "status": "error",
                    "error": str(goal_error)
                }
        
        # Step 7: Generate questions after processing the knowledge base (unless skipped)
        if skip_questions:
            logger.info("STEP 7: SKIPPING QUESTION GENERATION (skip_questions=True)")
            questions_result = {
                "status": "skipped",
                "message": "Question generation explicitly skipped"
            }
        else:
            logger.info("STEP 7: GENERATING QUESTIONS FROM PROCESSED KNOWLEDGE")
            logger.info("===================================================")
            try:
                # Generate more questions to ensure variety
                questions = await generate_questions(num_questions=15)
                questions_result = {}
                
                if questions:
                    questions_saved = await save_questions(questions)
                    questions_result = {
                        "questions_generated": len(questions),
                        "questions_saved": questions_saved
                    }
                    logger.info(f"Successfully generated {len(questions)} new questions from knowledge base")
                else:
                    logger.warning("No questions could be generated")
                    questions_result = {
                        "questions_generated": 0,
                        "questions_saved": False,
                        "error": "No questions generated - possible timeout or empty context"
                    }
            except Exception as question_error:
                logger.error(f"Error generating questions: {question_error}", exc_info=True)
                questions_result = {
                    "questions_generated": 0,
                "questions_saved": False,
                "error": str(question_error)
            }
        
        # Include questions in the result
        result["questions"] = questions_result
        # Include group_id for frontend compatibility
        result["group_id"] = group_id
        
        # Include critical content result from step 1
        result["critical_content"] = critical_content_result
        
        # Include whether this was a force update
        result["force_update"] = force_update
        
        # Save the file state with updated timestamps after successful processing
        try:
            from scripts.services.training_scheduler import save_file_state
            file_state = {
                "last_check": datetime.now().isoformat(),
                "file_timestamps": {},
                "file_hashes": {}
            }
            
            # If we have updated file timestamps from the change check, use them
            if "_file_state" in locals():
                file_state.update(locals().get("_file_state", {}))
                
            # Also grab the file stats from the vector processing result
            vector_stats = result.get("stats", {})
            
            # Save the state to persist file tracking between runs
            save_file_state(file_state)
            logger.info("Saved file state after successful processing")
        except Exception as e:
            logger.error(f"Error saving file state: {e}")
        
        logger.info(f"Knowledge base processing completed: {result}")
        _processing_active = False
        return {
            "status": "success",
            "message": "Knowledge base processed successfully",
            "data": result
        }

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
        data_path = Path(BACKEND_CONFIG['data_path'])
        
        # Initialize goal extractor
        goal_extractor = GoalExtractor(str(data_path))
        
        # Extract goals using the client from the request state
        goals_result = await goal_extractor.extract_goals(ollama_client=request.app.state.ollama_client)
        
        if goals_result.get("status") == "success":
            return {
                "status": "success",
                "message": f"Successfully extracted {goals_result.get('stats', {}).get('goals_extracted', 0)} goals",
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
