from fastapi import APIRouter, HTTPException, Request
from pathlib import Path
import pandas as pd
import json
import logging
import os
import glob
from datetime import datetime
from scripts.data.data_processor import DataProcessor
from scripts.data.markdown_processor import MarkdownProcessor
from scripts.services.question_generator import generate_questions, save_questions, get_config_path
from utils.config import BACKEND_CONFIG
from scripts.analysis.goal_extractor import GoalExtractor

router = APIRouter(prefix="/api/data/stats", tags=["stats"])
logger = logging.getLogger(__name__)

def get_config_path():
    """Get the path to the config file in the project directory"""
    # Use project root config.json
    project_root = Path(__file__).parent.parent.parent
    local_config = project_root / 'config.json'
    if local_config.exists():
        return local_config
    
    # If it doesn't exist, try the user's home directory as fallback
    home_config = Path.home() / '.blob' / 'config.json'
    if home_config.exists():
        return home_config
    
    # If neither exists, return the local path as default
    return local_config

def count_csv_rows(folder_path: Path, file_pattern: str) -> int:
    """Count rows in a CSV file if it exists"""
    try:
        csv_path = folder_path / file_pattern
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            logger.info(f"Found {len(df)} rows in {csv_path}")
            return len(df)
    except Exception as e:
        logger.error(f"Error counting rows in {file_pattern}: {e}")
    return 0

@router.get("/knowledge")
async def get_knowledge_stats(group_id: str = None):
    """Get statistics about knowledge data and embeddings"""
    logger.info(f"Starting knowledge stats collection... group_id={group_id}")
    try:
        data_path = Path(BACKEND_CONFIG['data_path'])
        base_path = data_path
        # Always use default folder for storage, regardless of group_id
        vector_path = data_path / "vector_stores" / "default"
        
        logger.info(f"Reading from data path: {base_path}")

        # Count CSV rows
        source_counts = {
            "definitions": count_csv_rows(base_path, "definitions.csv"),
            "materials": count_csv_rows(base_path, "materials.csv"),
            "methods": count_csv_rows(base_path, "methods.csv"),
            "projects": count_csv_rows(base_path, "projects.csv"),
            "resources": count_csv_rows(base_path, "resources.csv")
        }

        # Calculate store totals
        reference_total = source_counts["definitions"] + source_counts["methods"] + source_counts["projects"]
        content_total = source_counts["materials"] + source_counts["resources"]
        total_items = sum(source_counts.values())

        logger.info(f"Source counts: {source_counts}")
        logger.info(f"Total items found: {total_items}")

        # Get embedding stats for each store, including topic stores
        stores_stats = {
            "reference_store": {
                "embedding_count": 0,  # Default values to prevent KeyError
                "chunk_count": 0
            },
            "content_store": {
                "embedding_count": 0,  # Default values to prevent KeyError
                "chunk_count": 0
            },
            "topic_stores": {}
        }
        
        # Check for topic stores
        topic_stores = []
        for store_dir in vector_path.glob("topic_*"):
            if store_dir.is_dir():
                topic_stores.append(store_dir.name)
        
        logger.info(f"Found {len(topic_stores)} topic stores")
        
        # First read main stores and their stats
        total_embeddings = 0
        total_chunks = 0
        for store in ["reference_store", "content_store"]:
            stats_path = vector_path / store / "embedding_stats.json"
            if stats_path.exists():
                try:
                    with open(stats_path) as f:
                        store_stats = json.load(f)
                        stores_stats[store] = store_stats
                        # Use get with default value to avoid KeyError
                        # Get embedding and chunk counts with detailed logging
                        store_embeddings = store_stats.get("embedding_count", 0)
                        store_chunks = store_stats.get("chunk_count", 0)
                        
                        # Log store stats for debugging
                        logger.info(f"{store} stats: {store_embeddings} embeddings, {store_chunks} chunks")
                        
                        # Add to totals
                        total_embeddings += store_embeddings
                        total_chunks += store_chunks
                except Exception as store_error:
                    logger.error(f"Error reading stats for {store}: {store_error}")
                    # Keep the default values in stores_stats
                    stores_stats[store]["updated_at"] = datetime.now().isoformat()
        
        # Now add stats for all topic stores
        topic_embeddings = 0
        topic_chunks = 0
        for store in topic_stores:
            stats_path = vector_path / store / "embedding_stats.json"
            if stats_path.exists():
                try:
                    with open(stats_path) as f:
                        store_stats = json.load(f)
                        topic_name = store.replace("topic_", "")
                        stores_stats["topic_stores"][topic_name] = store_stats
                        
                        # Get counts with detailed logging
                        topic_count = store_stats.get("embedding_count", 0)
                        chunk_count = store_stats.get("chunk_count", 0)
                        
                        # Add to totals
                        topic_embeddings += topic_count
                        topic_chunks += chunk_count
                        
                        # Log details for each topic store
                        logger.info(f"Topic store '{topic_name}': {topic_count} embeddings, {chunk_count} chunks, ratio: {topic_count/chunk_count if chunk_count > 0 else 'N/A'}")
                except Exception as store_error:
                    logger.error(f"Error reading stats for topic store {store}: {store_error}")
                    # Continue with other topic stores rather than stopping

        # Add topic counts to the totals
        total_embeddings += topic_embeddings
        total_chunks += topic_chunks
        
        # Calculate and log chunk-to-embedding ratio - prevent division by zero
        ratio = total_chunks / total_embeddings if total_embeddings > 0 else 0
        logger.info(f"Found total of {total_embeddings} embeddings and {total_chunks} chunks across all stores")
        logger.info(f"Chunk-to-embedding ratio: {ratio:.2f} (ideally should be close to 1.0 for 1:1 mapping)")

        # Get update timestamp from stats (could be created_at or updated_at)
        def get_timestamp(store_stats):
            if not store_stats:
                return None
            # Try different field names for timestamp
            for field in ['updated_at', 'created_at', 'last_updated']:
                if field in store_stats:
                    return store_stats[field]
            return None
        
        # Prepare topic stores stats - Make sure this works even with no topic stores
        topic_store_stats = {}
        if stores_stats.get("topic_stores", {}):
            for topic_name, topic_stats in stores_stats["topic_stores"].items():
                topic_store_stats[topic_name] = {
                    "total_documents": topic_stats.get("document_count", 0),
                    "embeddings": topic_stats.get("embedding_count", 0),
                    "chunks": topic_stats.get("chunk_count", 0),  # Add chunk count
                    "last_updated": get_timestamp(topic_stats)
                }
        
        # Ensure reference_store and content_store always have required fields
        for store_key in ["reference_store", "content_store"]:
            if store_key not in stores_stats or not isinstance(stores_stats[store_key], dict):
                stores_stats[store_key] = {"embedding_count": 0, "chunk_count": 0}
        
        # Combine all stats
        knowledge_stats = {
            "sources": {
                "definitions": source_counts["definitions"],
                "methods": source_counts["methods"],
                "materials": source_counts["materials"],
                "projects": source_counts["projects"],
                "resources": source_counts["resources"]
            },
            "stores": {
                "reference_store": {
                    "total_documents": reference_total,
                    "embeddings": stores_stats["reference_store"].get("embedding_count", 0),
                    "chunks": stores_stats["reference_store"].get("chunk_count", 0),
                    "last_updated": get_timestamp(stores_stats["reference_store"])
                },
                "content_store": {
                    "total_documents": content_total,
                    "embeddings": stores_stats["content_store"].get("embedding_count", 0),
                    "chunks": stores_stats["content_store"].get("chunk_count", 0),
                    "last_updated": get_timestamp(stores_stats["content_store"])
                },
                "topic_stores": topic_store_stats
            },
            "totals": {
                "documents": total_items,
                "embeddings": total_embeddings,
                "chunks": total_chunks,  # Number of actual text chunks processed for embeddings
                "topic_stores": len(topic_store_stats)
            },
            "last_updated": datetime.now().isoformat(),
            # Include group_id in response for frontend compatibility
            "group_id": group_id
        }

        logger.info(f"Returning combined stats: {knowledge_stats}")
        return {
            "status": "success",
            "data": knowledge_stats
        }

    except Exception as e:
        logger.error(f"Error getting knowledge stats: {e}", exc_info=True)
        # Return a valid fallback response instead of raising an exception
        return {
            "status": "error",
            "data": {
                "sources": {
                    "definitions": 0,
                    "methods": 0,
                    "materials": 0,
                    "projects": 0,
                    "resources": 0
                },
                "stores": {
                    "reference_store": {
                        "total_documents": 0,
                        "embeddings": 0,
                        "chunks": 0,
                        "last_updated": None
                    },
                    "content_store": {
                        "total_documents": 0,
                        "embeddings": 0,
                        "chunks": 0,
                        "last_updated": None
                    },
                    "topic_stores": {}
                },
                "totals": {
                    "documents": 0,
                    "embeddings": 0,
                    "chunks": 0,
                    "topic_stores": 0
                },
                "last_updated": datetime.now().isoformat(),
                "group_id": group_id,
                "error": str(e)
            }
        }


def check_for_file_changes(data_path: Path, file_patterns, last_check_time=None):
    """
    Check if there are changes to any files matching the patterns since last check time
    
    Args:
        data_path: Base data directory path
        file_patterns: List of file patterns to check
        last_check_time: Optional datetime of last check
        
    Returns:
        bool: True if changes detected, False otherwise
        list: List of changed file paths
    """
    try:
        # If no last check time provided, assume everything needs updating
        if last_check_time is None:
            return True, ["Initial check - no previous timestamp"]
        
        # Use the more sophisticated tracking from training_scheduler
        from scripts.services.training_scheduler import load_file_state, save_file_state, get_file_hash
        
        # Load the file state for comparison
        file_state = load_file_state()
        file_timestamps = file_state.get("file_timestamps", {})
        file_hashes = file_state.get("file_hashes", {})
        
        # Find all matching files
        all_files = []
        for pattern in file_patterns:
            matching_files = glob.glob(str(data_path / pattern), recursive=True)
            all_files.extend(matching_files)
            
        # Check for new or modified files
        has_new_files = False
        changed_files = []
        new_file_timestamps = file_timestamps.copy()
        new_file_hashes = file_hashes.copy()
        
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
                    prev_time = datetime.fromisoformat(previous_mod_time) if isinstance(previous_mod_time, str) else previous_mod_time
                    
                    # If modification time unchanged, it's probably not modified
                    if file_mod_time == prev_time:
                        continue
                    
                    # If modification time is older than last_check_time, it hasn't changed
                    if file_mod_time < last_check_time:
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
                
            except Exception as e:
                logger.error(f"Error checking file {file_path}: {e}")
                
        # Check for deleted files
        previous_files = set(file_timestamps.keys())
        current_files = set(new_file_timestamps.keys())
        deleted_files = previous_files - current_files
        
        if deleted_files:
            logger.info(f"Deleted files detected: {len(deleted_files)} files")
            has_new_files = True
            for file in deleted_files:
                if file in new_file_timestamps:
                    del new_file_timestamps[file]
                if file in new_file_hashes:
                    del new_file_hashes[file]
            changed_files.extend(deleted_files)
            
        # If necessary, update and save the file state (only needed if processing)
        if has_new_files:
            # Save updated state to the tracking file
            file_state["file_timestamps"] = new_file_timestamps
            file_state["file_hashes"] = new_file_hashes
            
            # We don't save the file state here - that will be done after processing
            # This avoids saving if the user decides not to proceed with processing
            
            # Log detailed changes
            logger.info(f"Detected {len(changed_files)} changed files")
            if len(changed_files) <= 5:  # Only log details if not too many
                for file in changed_files:
                    logger.info(f"  - {file}")
        else:
            logger.info("No file changes detected")
            
        return has_new_files, changed_files
        
    except Exception as e:
        logger.error(f"Error checking for file changes: {e}")
        # If error occurs, assume changes to be safe
        return True, ["Error checking for changes"]

@router.get("/embeddings")
async def get_embeddings_info(request: Request):
    """Get information about the embedding model and vector stores"""
    try:
        # Get the OllamaClient instance
        ollama_client = request.app.state.ollama_client
        
        if not ollama_client:
            return {
                "status": "error",
                "message": "OllamaClient is not initialized"
            }
            
        # Get config info
        # Get embedding model info
        model_info = {
            "model_name": ollama_client.embeddings.model_name,
            "base_url": ollama_client.embeddings.base_url,
            "dimension": ollama_client.embeddings.dimension
        }
        
        # Get vector store info
        stores_info = {
            "reference_store": {
                "loaded": ollama_client.reference_store is not None,
                "document_count": len(ollama_client.reference_store.docstore._dict) if ollama_client.reference_store else 0,
                "path": str(Path(BACKEND_CONFIG['data_path']) / "vector_stores" / "default" / "reference_store")
            },
            "content_store": {
                "loaded": ollama_client.content_store is not None,
                "document_count": len(ollama_client.content_store.docstore._dict) if ollama_client.content_store else 0,
                "path": str(Path(BACKEND_CONFIG['data_path']) / "vector_stores" / "default" / "content_store")
            }
        }
        
        # Get the project config path
        project_config_path = Path(__file__).parent.parent.parent / 'config.json'
        
        return {
            "status": "success",
            "data": {
                "embeddings_model": model_info,
                "vector_stores": stores_info,
                "config_path": str(project_config_path)
            }
        }
            
    except Exception as e:
        logger.error(f"Error getting embeddings info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
