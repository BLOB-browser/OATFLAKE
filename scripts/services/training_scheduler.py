import logging
import threading
import time
import asyncio
from pathlib import Path
import json
import os
import glob
from datetime import datetime, time as dt_time, timedelta

logger = logging.getLogger(__name__)

# Initial state
_scheduler_thread = None
_stop_event = threading.Event()
_running = False

# Default training schedule times
_training_start_hour = 0   # 12:00 AM
_training_start_minute = 0 
_training_stop_hour = 6    # 6:00 AM
_training_stop_minute = 0

# Last processing tracking
_last_data_check = None
_last_data_timestamp = None

def set_training_time(start_hour, start_minute, stop_hour, stop_minute):
    """Set the training schedule time window"""
    global _training_start_hour, _training_start_minute, _training_stop_hour, _training_stop_minute
    
    # Validate input
    if not (0 <= start_hour < 24 and 0 <= start_minute < 60 and 
            0 <= stop_hour < 24 and 0 <= stop_minute < 60):
        raise ValueError("Invalid time values")
    
    _training_start_hour = start_hour
    _training_start_minute = start_minute
    _training_stop_hour = stop_hour
    _training_stop_minute = stop_minute
    
    logger.info(f"Training schedule set to {start_hour:02d}:{start_minute:02d} - {stop_hour:02d}:{stop_minute:02d}")
    
    # Save configuration to file
    try:
        config_path = get_config_path()
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        config['training_schedule'] = {
            'start_hour': start_hour,
            'start_minute': start_minute,
            'stop_hour': stop_hour,
            'stop_minute': stop_minute
        }
        
        # Make parent directory if needed
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
    except Exception as e:
        logger.error(f"Failed to save training schedule: {e}")

def get_status():
    """Get the current status of the scheduler"""
    return {
        "active": _running,
        "last_update": None,  # We don't track this currently
        "schedule": {
            "start": f"{_training_start_hour:02d}:{_training_start_minute:02d}",
            "stop": f"{_training_stop_hour:02d}:{_training_stop_minute:02d}"
        },
        "tasks_pending": 0,   # We don't track pending tasks
        "tasks_completed": 0  # We don't track completed tasks
    }

def get_config_path():
    """Get the path to the config file in the project directory"""
    # First try to use config.json in the project root
    local_config = Path("config.json")
    if local_config.exists():
        return local_config
    
    # If it doesn't exist, try the user's home directory as fallback
    home_config = Path.home() / '.blob' / 'config.json'
    if home_config.exists():
        return home_config
    
    # If neither exists, return the local path as default
    return local_config
    
def get_data_path():
    """Get the configured data path from config.json"""
    try:
        config_path = get_config_path()
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                return Path(config.get('data_path', '.'))
        return Path('.')
    except Exception as e:
        logger.error(f"Error getting data path: {e}")
        return Path('.')
        
def get_file_state_path():
    """Get the path to the file state tracking JSON file"""
    data_path = get_data_path()
    state_dir = data_path / "stats"
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "file_state.json"

def load_file_state():
    """Load the file state tracking data"""
    try:
        state_path = get_file_state_path()
        if state_path.exists():
            with open(state_path, 'r') as f:
                return json.load(f)
        return {
            "last_check": None,
            "file_timestamps": {},
            "file_hashes": {}
        }
    except Exception as e:
        logger.error(f"Error loading file state: {e}")
        return {
            "last_check": None,
            "file_timestamps": {},
            "file_hashes": {}
        }

def save_file_state(state):
    """Save the file state tracking data"""
    try:
        state_path = get_file_state_path()
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving file state: {e}")
        return False

def get_file_hash(file_path):
    """Get a hash of the file contents"""
    try:
        import hashlib
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            # Read in 64k chunks to handle large files
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing file {file_path}: {e}")
        return None

def check_for_new_data():
    """
    Check if there are new data files or updates to existing files
    that need processing.
    
    Returns:
        bool: True if new data is detected, False otherwise
    """
    try:
        # Load the file state
        file_state = load_file_state()
        last_check_str = file_state.get("last_check")
        last_check = datetime.fromisoformat(last_check_str) if last_check_str else None
        file_timestamps = file_state.get("file_timestamps", {})
        file_hashes = file_state.get("file_hashes", {})
        
        # If we've never checked before, assume there's new data
        if last_check is None:
            logger.info("First data check, assuming new data exists")
            
            # Initialize file state with current timestamp
            file_state["last_check"] = datetime.now().isoformat()
            save_file_state(file_state)
            
            return True
        
        # Get the data directory
        data_path = get_data_path()
        if not data_path.exists():
            logger.warning(f"Data path {data_path} does not exist")
            return False
        
        # Define file patterns to check for modifications
        patterns_to_check = [
            "*.csv",                  # Check CSV files in the data directory
            "markdown/**/*.md",       # Check markdown files
            "materials/**/*",         # Material files including PDFs
            "vector_stores/**/*.json" # Check vector store metadata
        ]
        
        # Find all matching files
        all_files = []
        for pattern in patterns_to_check:
            path_pattern = data_path / pattern
            matching_files = glob.glob(str(path_pattern), recursive=True)
            all_files.extend(matching_files)
            
        # Check for new or modified files
        has_new_files = False
        changed_files = []
        new_file_timestamps = {}
        new_file_hashes = {}
        
        for file_path in all_files:
            try:
                file_path_str = str(file_path)
                
                # Skip directories
                if os.path.isdir(file_path):
                    continue
                    
                # Get file modification time and size
                file_mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                file_size = os.path.getsize(file_path)
                
                # First check timestamp - if it hasn't changed, skip hash calculation
                previous_mod_time = file_timestamps.get(file_path_str)
                
                # Store the new timestamp regardless
                new_file_timestamps[file_path_str] = file_mod_time.isoformat()
                
                if previous_mod_time:
                    # Convert string timestamp to datetime
                    prev_time = datetime.fromisoformat(previous_mod_time)
                    
                    # If modification time unchanged, it's probably not modified
                    if file_mod_time == prev_time:
                        # Preserve previous hash if we have one
                        if file_path_str in file_hashes:
                            new_file_hashes[file_path_str] = file_hashes[file_path_str]
                        continue
                
                # For CSV files and small files, check hash to detect actual content changes
                if file_path_str.endswith('.csv') or file_size < 1000000:  # 1MB
                    file_hash = get_file_hash(file_path)
                    new_file_hashes[file_path_str] = file_hash
                    
                    # If hash matches previous hash, content hasn't changed
                    if file_hash and file_hash == file_hashes.get(file_path_str):
                        continue
                
                # File is new or modified
                has_new_files = True
                changed_files.append(file_path_str)
                logger.info(f"Modified file detected: {file_path_str}")
                
            except Exception as e:
                logger.error(f"Error checking file {file_path}: {e}")
                
        # Check for deleted files
        previous_files = set(file_timestamps.keys())
        current_files = set(new_file_timestamps.keys())
        deleted_files = previous_files - current_files
        
        if deleted_files:
            logger.info(f"Deleted files detected: {len(deleted_files)} files")
            has_new_files = True
            
        # Update and save file state
        file_state["last_check"] = datetime.now().isoformat()
        file_state["file_timestamps"] = new_file_timestamps
        file_state["file_hashes"] = new_file_hashes
        save_file_state(file_state)
        
        # Log detail about what changed
        if has_new_files:
            logger.info(f"Detected {len(changed_files)} changed files")
            if len(changed_files) <= 5:  # Only log details if not too many
                for file in changed_files:
                    logger.info(f"  - {file}")
        else:
            logger.info("No file changes detected")
            
        return has_new_files
        
    except Exception as e:
        logger.error(f"Error checking for new data: {e}")
        # If there's an error, assume no new data to be safe
        return False

def _training_loop():
    """The main training loop that runs in a separate thread"""
    global _running
    
    _running = True
    logger.info("Training scheduler started")
    last_check_time = None
    
    while not _stop_event.is_set():
        try:
            # Wait for 5 seconds between checks
            if _stop_event.wait(5):
                break
                
            # Get current time
            now = datetime.now()
            current_time = now.time()
            
            # Define training start and stop times
            training_start = dt_time(hour=_training_start_hour, minute=_training_start_minute)
            training_stop = dt_time(hour=_training_stop_hour, minute=_training_stop_minute)
            
            # Check if we're in the training window
            is_training_time = False
            
            # Handle case where training window crosses midnight
            if (training_start <= training_stop):
                # Normal case (e.g., 01:00 - 06:00)
                is_training_time = training_start <= current_time <= training_stop
            else:
                # Crosses midnight (e.g., 22:00 - 04:00)
                is_training_time = current_time >= training_start or current_time <= training_stop
            
            # Execute training tasks if it's time and we haven't run in the last hour
            if is_training_time and (last_check_time is None or (now - last_check_time).seconds >= 3600):
                logger.info(f"Training time window active ({training_start} - {training_stop}), checking for new data")
                
                # First check if there's any new data to process
                has_new_data = check_for_new_data()
                
                if has_new_data:
                    logger.info("New data detected, starting knowledge processing")
                    
                    # Use the API endpoint to trigger knowledge processing
                    import requests
                    try:
                        # Call the updated API endpoint
                        url = "http://localhost:8999/api/knowledge/process"  # Updated path
                        
                        # Set parameters for scheduled processing
                        params = {
                            "skip_markdown_scraping": "false",  # Process markdown files
                            "analyze_resources": "true",        # Analyze resources with LLM
                            "analyze_all_resources": "false",   # Only analyze resources that need it
                            "batch_size": "1",                  # Process 1 resource at a time for stability
                            "resource_limit": "",               # No limit on resources
                            "force_update": "false",            # Let the API decide if processing is needed
                            "skip_vector_generation": "false"   # Do vector generation in the process endpoint
                        }
                        
                        logger.info(f"Calling knowledge processing API endpoint: {url}")
                        response = requests.post(url, params=params)
                        
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("status") == "success":
                                logger.info("Knowledge processing completed successfully via API")
                                
                                # Check if vector generation was already performed
                                vector_generation = result.get("data", {}).get("vector_generation", {})
                                if vector_generation.get("status") == "success":
                                    logger.info("Vector generation was already performed during knowledge processing")
                                    # No need to rebuild FAISS indexes again
                                else:
                                    # Only rebuild if vector generation wasn't already done
                                    logger.info("Vector generation wasn't performed during knowledge processing, rebuilding now...")
                                    rebuild_url = "http://localhost:8999/api/rebuild-faiss-indexes"
                                    
                                    try:
                                        rebuild_response = requests.post(rebuild_url)
                                        if rebuild_response.status_code == 200:
                                            rebuild_result = rebuild_response.json()
                                            if rebuild_result.get("status") == "success":
                                                logger.info("FAISS index rebuild completed successfully")
                                                logger.info(f"Rebuilt {len(rebuild_result.get('stores_rebuilt', []))} stores with {rebuild_result.get('total_documents', 0)} documents")
                                            else:
                                                logger.warning(f"FAISS rebuild issue: {rebuild_result.get('error', 'unknown error')}")
                                        else:
                                            logger.error(f"FAISS rebuild API error: {rebuild_response.status_code} - {rebuild_response.text}")
                                    except Exception as rebuild_error:
                                        logger.error(f"Error rebuilding FAISS indexes: {rebuild_error}")
                                # API endpoint already generates questions, no need for separate call
                                questions_result = result.get("data", {}).get("questions", {})
                                questions_generated = questions_result.get("questions_generated", 0)
                                logger.info(f"Generated {questions_generated} questions during processing")
                            elif result.get("status") == "skipped":
                                logger.info(f"Knowledge processing skipped: {result.get('message')}")
                            else:
                                logger.error(f"Knowledge processing failed: {result.get('message')}")
                        else:
                            logger.error(f"API request failed: {response.status_code} - {response.text}")
                    
                    except Exception as e:
                        logger.error(f"Error calling knowledge processing API: {e}")
                else:
                    logger.info("No new data detected, skipping knowledge processing")
                
                # Update last check time regardless of whether we ran processing
                last_check_time = now
                
            
        except Exception as e:
            logger.error(f"Error in training loop: {e}")
            
    _running = False
    logger.info("Training scheduler stopped")

def start():
    """Start the scheduler thread if not already running"""
    global _scheduler_thread, _stop_event
    
    if (_running):
        logger.info("Scheduler already running")
        return
        
    _stop_event.clear()
    _scheduler_thread = threading.Thread(target=_training_loop)
    _scheduler_thread.daemon = True  # Thread will exit when main program exits
    _scheduler_thread.start()
    
    logger.info("Started training scheduler")

def stop():
    """Stop the scheduler thread if running"""
    global _stop_event, _scheduler_thread
    
    if not _running:
        logger.info("Scheduler not running")
        return
        
    logger.info("Stopping training scheduler...")
    _stop_event.set()
    
    if (_scheduler_thread):
        _scheduler_thread.join(timeout=5.0)
        if _scheduler_thread.is_alive():
            logger.warning("Scheduler thread did not stop gracefully")
        else:
            logger.info("Scheduler thread stopped successfully")
    
    _scheduler_thread = None
    
    # When training is stopped, rebuild FAISS indexes to ensure consistency
    logger.info("Training stopped - rebuilding FAISS indexes to ensure consistency...")
    try:
        import requests
        rebuild_url = "http://localhost:8999/api/rebuild-faiss-indexes"
        rebuild_response = requests.post(rebuild_url)
        
        if rebuild_response.status_code == 200:
            rebuild_result = rebuild_response.json()
            if rebuild_result.get("status") == "success":
                logger.info("FAISS index rebuild completed successfully")
                logger.info(f"Rebuilt {len(rebuild_result.get('stores_rebuilt', []))} stores with {rebuild_result.get('total_documents', 0)} documents")
            else:
                logger.warning(f"FAISS rebuild issue: {rebuild_result.get('error', 'unknown error')}")
        else:
            logger.error(f"FAISS rebuild API error: {rebuild_response.status_code} - {rebuild_response.text}")
    except Exception as rebuild_error:
        logger.error(f"Error rebuilding FAISS indexes: {rebuild_error}")
