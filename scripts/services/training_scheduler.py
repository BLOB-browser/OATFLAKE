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

def load_training_schedule():
    """Load training schedule from config.json if available"""
    global _training_start_hour, _training_start_minute, _training_stop_hour, _training_stop_minute
    
    try:
        config_path = get_config_path()
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Check if training_schedule exists in config
            if 'training_schedule' in config:
                schedule = config['training_schedule']
                # Set times if they exist in config
                if all(k in schedule for k in ['start_hour', 'start_minute', 'stop_hour', 'stop_minute']):
                    _training_start_hour = schedule['start_hour']
                    _training_start_minute = schedule['start_minute']
                    _training_stop_hour = schedule['stop_hour']
                    _training_stop_minute = schedule['stop_minute']
                    logger.info(f"Loaded training schedule from config: "
                               f"{_training_start_hour:02d}:{_training_start_minute:02d} - "
                               f"{_training_stop_hour:02d}:{_training_stop_minute:02d}")
                    return True
    except Exception as e:
        logger.error(f"Error loading training schedule from config: {e}")
    
    # Use defaults if couldn't load from config
    logger.info(f"Using default training schedule: "
               f"{_training_start_hour:02d}:{_training_start_minute:02d} - "
               f"{_training_stop_hour:02d}:{_training_stop_minute:02d}")
    return False

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
    
    # Set up interruption handling flag
    processing_active = False
    
    while not _stop_event.is_set():
        try:
            # Wait for 2 seconds between checks - this is where we check for stop events
            if _stop_event.wait(2):
                logger.info("Stop event detected in training loop")
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
                
            # Check if we're approaching the end of the training window
            # Calculate time until end of window
            time_to_end_minutes = 0
            if is_training_time:
                # Calculate minutes until end of training window
                if training_start <= training_stop:
                    # Normal case (e.g., 01:00 - 06:00)
                    stop_datetime = now.replace(hour=_training_stop_hour, minute=_training_stop_minute)
                else:
                    # Crosses midnight (e.g., 22:00 - 04:00)
                    stop_datetime = now.replace(hour=_training_stop_hour, minute=_training_stop_minute)
                    if now.time() >= dt_time(hour=0, minute=0) and now.time() <= training_stop:
                        # We're after midnight but before stop time
                        pass  # stop_datetime is already correct
                    else:
                        # We're after start time but before midnight
                        stop_datetime = stop_datetime + timedelta(days=1)
                
                time_to_end_minutes = (stop_datetime - now).total_seconds() / 60
                logger.debug(f"Minutes until end of training window: {time_to_end_minutes:.1f}")
                
                # If we're close to the end (< 60 minutes), don't start new resource analysis
                # but still allow embedding generation and other quick tasks
                if time_to_end_minutes < 60:
                    logger.info(f"Approaching end of training window ({time_to_end_minutes:.1f} min left), "
                               "will skip resource analysis but allow embeddings")
            
            # Execute training tasks if:
            # - It's within the training window AND
            # - We haven't run in the last hour (to avoid excessive processing)
            if is_training_time and (last_check_time is None or (now - last_check_time).seconds >= 3600):
                # Check again for stop event before heavy processing
                if _stop_event.is_set():
                    logger.info("Stop event detected before data check")
                    break
                    
                logger.info(f"Training time window active ({training_start} - {training_stop}), checking for new data")
                
                # First check if there's any new data to process
                has_new_data = check_for_new_data()
                
                # Check again for stop event after data check
                if _stop_event.is_set():
                    logger.info("Stop event detected after data check")
                    break
                
                # Always check for unanalyzed resources, even if no file changes detected
                # This ensures we process any resources that were interrupted or not fully analyzed
                should_process = has_new_data
                
                # If no new data detected, check for unanalyzed resources anyway
                if not has_new_data:
                    try:
                        import requests
                        import json
                        
                        # Make a quick check for unanalyzed resources without calling the full API
                        # This is more efficient than making an API call every time
                        data_path = get_data_path()
                        resources_path = data_path / "resources.csv"
                        
                        if resources_path.exists():
                            try:
                                import pandas as pd
                                df = pd.read_csv(resources_path)
                                
                                # Count resources that need analysis
                                needs_analysis_count = 0
                                for _, row in df.iterrows():
                                    # Check if analysis_completed is False or missing
                                    if pd.isna(row.get('analysis_completed')) or row.get('analysis_completed') == False:
                                        needs_analysis_count += 1
                                        if needs_analysis_count > 0:
                                            # We found at least one that needs analysis, no need to check more
                                            break
                                
                                if needs_analysis_count > 0:
                                    logger.info(f"Found {needs_analysis_count} resources with incomplete analysis")
                                    should_process = True
                                else:
                                    logger.info("All resources have been analyzed")
                            except Exception as e:
                                logger.error(f"Error checking for unanalyzed resources: {e}")
                    except Exception as e:
                        logger.warning(f"Could not check for unanalyzed resources: {e}")
                
                # Now decide whether to process based on has_new_data OR unanalyzed resources
                if should_process:
                    if has_new_data:
                        logger.info("New data detected, starting knowledge processing")
                    else:
                        logger.info("No file changes, but found unanalyzed resources - starting processing")
                    
                    # Use the API endpoint to trigger knowledge processing
                    import requests
                    try:
                        # Mark that we're in processing mode
                        processing_active = True
                        
                        url = "http://localhost:8999/api/knowledge/process"  # Updated path
                        
                        # Set parameters for scheduled processing
                        params = {
                            "skip_markdown_scraping": "false",   # Process markdown files
                            "analyze_resources": str(time_to_end_minutes >= 60).lower(),  # Only analyze resources if >60 min left
                            "analyze_all_resources": "false",    # Only analyze resources that need it
                            "batch_size": "1",                   # Process 1 resource at a time for stability
                            "force_update": "false",             # Let the API decide if processing is needed
                            "skip_vector_generation": "false",   # Do vector generation in the process endpoint
                            "check_unanalyzed": "true"           # Always check for unanalyzed resources
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
                        processing_active = False  # Ensure flag is reset even on error
                else:
                    logger.info("No new data detected and all resources analyzed, skipping knowledge processing")
                
                # Update last check time regardless of whether we ran processing
                last_check_time = now
                
            # Add specific handling for keyboard interrupts
            # If we're in the middle of processing, we want to catch Ctrl+C
            # and handle it gracefully
            
        except KeyboardInterrupt:
            # If we catch a keyboard interrupt inside our thread, 
            # we need to trigger a proper shutdown
            logger.warning("Keyboard interrupt caught in training thread")
            _stop_event.set()  # This will cause the thread to exit gracefully
            break
        except Exception as e:
            logger.error(f"Error in training loop: {e}")
            processing_active = False  # Reset processing flag on error
            
    # Make sure we properly mark as not running when exiting loop
    _running = False
    logger.info("Training scheduler loop exited")

def start():
    """Start the scheduler thread if not already running"""
    global _scheduler_thread, _stop_event, _running
    
    if (_running):
        logger.info("Scheduler already running")
        return
    
    # Load training schedule from config before starting
    load_training_schedule()
        
    # Reset stop event and thread state
    _stop_event.clear()
    
    # Check if thread exists and is still alive
    if _scheduler_thread and _scheduler_thread.is_alive():
        logger.warning("Thread still alive, attempting to stop it first")
        stop()
    
    # Create and start a new thread
    _scheduler_thread = threading.Thread(target=_training_loop, name="TrainingScheduler")
    _scheduler_thread.daemon = True  # Thread will exit when main program exits
    _scheduler_thread.start()
    
    # Wait a moment to ensure the thread has properly started
    import time
    time.sleep(0.5)
    
    if _scheduler_thread.is_alive():
        logger.info(f"Started training scheduler thread (id: {_scheduler_thread.ident})")
    else:
        logger.error("Failed to start training scheduler thread")

def stop(timeout=15.0):
    """Stop the scheduler thread if running"""
    global _stop_event, _scheduler_thread, _running
    
    if not _running and (_scheduler_thread is None or not _scheduler_thread.is_alive()):
        logger.info("Scheduler not running")
        return
        
    logger.info("Stopping training scheduler...")
    
    # First try to send cancellation signal to any running knowledge process
    try:
        import requests
        try:
            # Send cancellation signal to the knowledge processing endpoint
            logger.info("Sending cancellation signal to knowledge process...")
            cancel_response = requests.post(
                "http://localhost:8999/api/knowledge/cancel", 
                json={"force": True},
                timeout=3
            )
            if cancel_response.status_code == 200:
                logger.info("Cancellation signal sent successfully")
            else:
                logger.warning(f"Error sending cancellation signal: {cancel_response.status_code}")
        except Exception as cancel_e:
            logger.error(f"Error during cancellation request: {cancel_e}")
    except ImportError:
        pass
    
    # Set the stop event and wait for thread to finish
    _stop_event.set()
    
    if _scheduler_thread:
        # Check if any embedding jobs are in progress
        embedding_in_progress = False
        try:
            import requests
            try:
                # Check if there's an active job
                status_response = requests.get("http://localhost:8999/api/knowledge/stats", timeout=1)
                if status_response.status_code == 200:
                    data = status_response.json()
                    status = data.get('status', '').lower()
                    if status == 'processing' or 'embedding' in status:
                        embedding_in_progress = True
                        logger.info("Embedding is in progress, waiting longer for completion before stopping")
            except Exception as e:
                logger.error(f"Error checking embedding status: {e}")
        except ImportError:
            pass
        
        # Join with a timeout adjusted based on embedding status
        join_timeout = min(timeout, 20.0 if embedding_in_progress else 10.0)
        logger.info(f"Waiting up to {join_timeout} seconds for scheduler to stop...")
        _scheduler_thread.join(timeout=join_timeout)
        
        # Check if thread is still alive
        if _scheduler_thread.is_alive():
            logger.warning("Scheduler thread did not stop gracefully within timeout")
            # Try more aggressive approach for testing environments
            try:
                import ctypes
                if hasattr(ctypes, 'pythonapi') and hasattr(ctypes.pythonapi, 'PyThreadState_SetAsyncExc'):
                    thread_id = _scheduler_thread.ident
                    result = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(thread_id), 
                        ctypes.py_object(SystemExit)
                    )
                    if result == 0:
                        logger.error(f"Invalid thread ID: {thread_id}")
                    elif result > 1:
                        # If more than one thread affected, reset state
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
                        logger.error("Failed to terminate thread")
                    else:
                        logger.info("Thread forcefully terminated")
                        time.sleep(0.5)  # Give it a moment to terminate
            except Exception as e:
                logger.error(f"Error attempting to forcefully terminate thread: {e}")
        else:
            logger.info("Scheduler thread stopped successfully")
    
    # Reset thread and state regardless
    _scheduler_thread = None
    _running = False
    
    # When training is stopped, rebuild FAISS indexes to ensure consistency
    # Only do this if the server is likely running (skip during app shutdown)
    try:
        # Check if server is running before attempting API call
        import requests
        try:
            # Try a quick status check first
            status_response = requests.get("http://localhost:8999/api/status", timeout=1)
            server_running = status_response.status_code == 200
        except:
            server_running = False
            
        if server_running:
            logger.info("Training stopped - rebuilding FAISS indexes to ensure consistency...")
            rebuild_url = "http://localhost:8999/api/rebuild-faiss-indexes"
            rebuild_response = requests.post(rebuild_url, timeout=5)
            
            if rebuild_response.status_code == 200:
                rebuild_result = rebuild_response.json()
                if rebuild_result.get("status") == "success":
                    logger.info("FAISS index rebuild completed successfully")
                    logger.info(f"Rebuilt {len(rebuild_result.get('stores_rebuilt', []))} stores with {rebuild_result.get('total_documents', 0)} documents")
                else:
                    logger.warning(f"FAISS rebuild issue: {rebuild_result.get('error', 'unknown error')}")
            else:
                logger.error(f"FAISS rebuild API error: {rebuild_response.status_code} - {rebuild_response.text}")
        else:
            logger.info("Server appears to be shutting down, skipping FAISS rebuild")
            
    except Exception as rebuild_error:
        logger.error(f"Error rebuilding FAISS indexes: {rebuild_error}")

# Call load_training_schedule when module is imported to set correct times immediately
load_training_schedule()
