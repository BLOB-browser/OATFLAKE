from fastapi import APIRouter, HTTPException, Request
from scripts.models.settings import ModelSettings, TrainingSchedule, LLMProvider
from scripts.services import training_scheduler
from scripts.services.settings_manager import SettingsManager
from utils.config import BACKEND_CONFIG
from scripts.llm.ollama_client import OllamaClient
from scripts.llm.open_router_client import OpenRouterClient
import httpx
import os
import logging
import json
import asyncio
from pathlib import Path
from datetime import datetime, time
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# Create router with proper configuration
router = APIRouter(tags=["system"])

logger = logging.getLogger(__name__)

@router.get("/api/status")  # Full path
async def status(request: Request):  # Add request parameter
    """Get system status"""
    try:
        # Read both config files
        user_config_path = Path.home() / '.blob' / 'config.json'
        project_config_path = Path(__file__).parent.parent.parent / 'config.json'
        
        # Get data path directly from project config.json
        data_path = BACKEND_CONFIG['data_path']  # Default
        if project_config_path.exists():
            try:
                with open(project_config_path) as f:
                    project_config = json.load(f)
                    if 'data_path' in project_config:
                        data_path = project_config['data_path']
            except Exception as e:
                logger.error(f"Error reading project config: {e}")
        
        # Get user credentials from user config
        user_config = {}
        if user_config_path.exists():
            try:
                with open(user_config_path) as f:
                    user_config = json.load(f)
            except Exception as e:
                logger.error(f"Error reading user config: {e}")
                
        # Check OpenRouter connection
        openrouter_status = "disconnected"
        if hasattr(request.app.state, 'openrouter_client') and request.app.state.openrouter_client.api_key:
            # Check if OpenRouter client has a valid connection
            try:
                is_connected, _ = await request.app.state.openrouter_client.check_connection()
                openrouter_status = "connected" if is_connected else "disconnected"
            except:
                pass

        return {
            "server": "running",
            "ollama": await check_ollama_status(),
            "openrouter": openrouter_status,
            "tunnel": "connected" if hasattr(request.app.state, 'ngrok_url') else "disconnected",
            "ngrok_url": getattr(request.app.state, 'ngrok_url', ''),
            "data_path": str(data_path),  # Use freshly loaded path 
            "group_id": user_config.get('group_id'),
            "group_image": user_config.get('group_image'),
            "group_name": user_config.get('group_name'),
            "last_connected": user_config.get('last_connected')
        }
    except Exception as e:
        logger.error(f"Status error: {e}")
        return {
            "server": "running",
            "ollama": "disconnected",
            "openrouter": "disconnected",
            "tunnel": "disconnected",
            "ngrok_url": "",
            "data_path": str(BACKEND_CONFIG['data_path'])
        }

@router.get("/health")  # Keep as is
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "server": "local",
        "version": "0.1.0",
        "port": BACKEND_CONFIG['PORT']
    }

@router.get("/api/check-update")  # Full path
async def check_update():
    """Check for system updates"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.github.com/repos/yourusername/blob/releases/latest"
            )
            if response.status_code == 200:
                latest = response.json()
                current_version = "0.1.0"
                latest_version = latest['tag_name'].strip('v')
                return {
                    "update_available": latest_version > current_version,
                    "current_version": current_version,
                    "latest_version": latest_version
                }
            return {"update_available": False, "current_version": "0.1.0"}
    except Exception as e:
        logger.error(f"Update check error: {e}")
        raise HTTPException(status_code=500, detail="Failed to check for updates")

@router.get("/api/system-settings")  # Full path
async def get_system_settings():
    """Get system settings"""
    try:
        settings_manager = SettingsManager()
        settings = settings_manager.load_settings()
        
        # Add current training schedule
        try:
            # Import directly
            from scripts.services.training_scheduler import get_status
            scheduler_status = get_status()
            
            # Set training schedule in settings if not already set
            if not settings.training and scheduler_status.get("schedule"):
                from scripts.models.settings import TrainingSchedule
                settings.training = TrainingSchedule(
                    start=scheduler_status["schedule"]["start"],
                    stop=scheduler_status["schedule"]["stop"]
                )
            
            # Add scheduler status to the response
            return {
                **settings.model_dump(),
                "scheduler_status": scheduler_status
            }
        except Exception as e:
            logger.error(f"Error getting scheduler status: {e}")
            return {
                **settings.model_dump(),
                "scheduler_status": {
                    "active": False,
                    "error": str(e)
                }
            }
    except Exception as e:
        logger.error(f"Settings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/system-settings")  # Full path
async def update_system_settings(settings: ModelSettings):
    """Update system settings"""
    try:
        settings_manager = SettingsManager()
        
        # Update training schedule if provided
        if settings.training:
            try:
                # Log the received values
                logger.info(f"Received training schedule: start={settings.training.start}, stop={settings.training.stop}")
                
                # Ensure we have valid strings
                if not isinstance(settings.training.start, str) or not isinstance(settings.training.stop, str):
                    raise ValueError(f"Invalid types: start={type(settings.training.start)}, stop={type(settings.training.stop)}")
                
                # Parse start and stop times (HH:MM format)
                try:
                    start_parts = settings.training.start.strip().split(':')
                    stop_parts = settings.training.stop.strip().split(':')
                    
                    if len(start_parts) != 2 or len(stop_parts) != 2:
                        raise ValueError(f"Invalid format: start={settings.training.start}, stop={settings.training.stop}")
                    
                    start_hour = int(start_parts[0])
                    start_minute = int(start_parts[1])
                    stop_hour = int(stop_parts[0])
                    stop_minute = int(stop_parts[1])
                    
                except ValueError as e:
                    logger.error(f"Failed to parse time components: {e}")
                    raise ValueError(f"Invalid time format: {str(e)}")
                
                # Validate times
                if not (0 <= start_hour < 24 and 0 <= start_minute < 60 and
                       0 <= stop_hour < 24 and 0 <= stop_minute < 60):
                    raise ValueError(f"Time values out of range: start={start_hour}:{start_minute}, stop={stop_hour}:{stop_minute}")

                # Import directly and use the function
                from scripts.services.training_scheduler import set_training_time
                set_training_time(
                    start_hour=start_hour,
                    start_minute=start_minute,
                    stop_hour=stop_hour,
                    stop_minute=stop_minute
                )
                logger.info(f"Updated training schedule: {start_hour:02d}:{start_minute:02d} - {stop_hour:02d}:{stop_minute:02d}")
            except Exception as e:
                logger.error(f"Error setting training schedule: {e}")
                raise HTTPException(status_code=400, detail=f"Invalid training schedule format: {str(e)}")

        # Save all settings
        if settings_manager.save_settings(settings):
            # Import directly
            from scripts.services.training_scheduler import get_status
            return {
                "status": "success",
                "message": "Settings updated",
                "training_schedule": get_status() if settings.training else None
            }
        raise HTTPException(status_code=500, detail="Failed to save settings")
    except Exception as e:
        logger.error(f"Settings update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class WebConfig(BaseModel):
    prompt: str
    context_k: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    
class OpenRouterKeyRequest(BaseModel):
    api_key: str
    
class OpenRouterModelRequest(BaseModel):
    model_id: str
    
class OpenRouterResponse(BaseModel):
    status: str
    message: str
    
class ModelListResponse(BaseModel):
    models: List[Dict[str, Any]]

@router.post("/api/web")  # Full path
async def process_web_request(request: Request):
    """Handle LLM web requests using settings from model_settings.json"""
    try:
        # Log the raw request body for debugging
        body = await request.body()
        raw_content = body.decode('utf-8')
        logger.info(f"Raw web request content: {raw_content}")
        
        # Parse the request
        try:
            data = await request.json()
            # Handle different formats - support both 'prompt' and 'query'
            prompt = data.get('prompt', '') or data.get('query', '')
            
            # Check if this is a "phase 2" request - when the frontend is asking for the completed response
            is_phase_two = data.get('phase', 0) == 2
            
            # If this is phase 2, get the response from the state
            if is_phase_two:
                request_id = data.get('request_id')
                if not request_id:
                    return {"error": "Missing request_id for phase 2 request", "status": "error"}
                
                # Check if we have a pending response
                if hasattr(request.app.state, 'pending_responses') and request_id in request.app.state.pending_responses:
                    response_data = request.app.state.pending_responses.get(request_id, {})
                    if response_data.get('complete'):
                        logger.info(f"Returning completed response for request_id {request_id}")
                        
                        # Ensure model_info is populated with default values if missing
                        model_info = response_data.get('model_info', {})
                        if not model_info or 'provider' not in model_info:
                            # If model info is missing, try to create it from available data
                            model_info = {
                                'provider': 'ollama',  # Default to ollama
                                'model_name': getattr(request.app.state, 'ollama_client', {}).model if hasattr(request.app.state, 'ollama_client') else 'unknown'
                            }
                        
                        # Format response with model info and timing
                        return {
                            "response": response_data.get('response', ''),
                            "status": "success",
                            "complete": True,
                            "phase": 2,
                            "word_count": response_data.get('word_count', len(response_data.get('response', '').split())),
                            "model_info": model_info,
                            "timestamp": datetime.now().isoformat(),  # Add server timestamp
                            "query_timestamp": response_data.get('start_time', ''),  # When the query started
                            "completion_timestamp": response_data.get('end_time', ''),  # When processing completed
                            "timing": {
                                "total_seconds": response_data.get('total_duration_seconds', 0),
                                "retrieval_seconds": response_data.get('stages', {}).get('retrieval', {}).get('duration_ms', 0) / 1000 
                                                    if 'stages' in response_data else 0,
                                "generation_seconds": response_data.get('stages', {}).get('generation', {}).get('duration_ms', 0) / 1000
                                                     if 'stages' in response_data else 0,
                            }
                        }
                    else:
                        # Response not ready yet, but being processed
                        return {
                            "status": "processing",
                            "phase": 1,
                            "request_id": request_id,
                            "message": "Response still generating..."
                        }
                else:
                    return {"status": "error", "error": f"No pending response found for request_id: {request_id}"}
            
            # Log the extracted prompt/query clearly
            logger.info(f"Extracted query for processing: '{prompt}'")
            
            if not prompt:
                logger.error("No prompt/query found in request")
                return {"status": "error", "error": "No prompt or query provided in request"}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return {"status": "error", "error": f"Invalid JSON format: {str(e)}"}
        
        # Generate unique request ID for this conversation
        import uuid
        request_id = str(uuid.uuid4())
        
        # Load current settings to determine which provider to use
        settings_manager = SettingsManager()
        settings = settings_manager.load_settings()
        
        # Get Ollama client and ensure consistent chunking
        client = getattr(request.app.state, 'ollama_client', None)
        if not client:
            client = OllamaClient()
            request.app.state.ollama_client = client
        
        # Process the prompt using the Ollama client's chunking parameters
        processed_prompt = client.process_text(prompt)
        
        # Initialize the pending_responses dict if it doesn't exist
        if not hasattr(request.app.state, 'pending_responses'):
            request.app.state.pending_responses = {}
        
        # Create an initial response record
        request.app.state.pending_responses[request_id] = {
            'prompt': prompt,
            'processing': True,
            'complete': False,
            'response': '',
            'timestamp': datetime.now().isoformat()
        }
        
        # Start a background task to generate the response
        # Choose client based on provider setting
        if settings.provider == LLMProvider.OPENROUTER:
            # Use OpenRouter if configured
            client = getattr(request.app.state, 'openrouter_client', None)
            if client and client.api_key:
                logger.info("Using OpenRouter for web request")
                model = settings.openrouter_model
                
                # Start background task with model information
                task = asyncio.create_task(
                    generate_response_background(
                        request.app.state,
                        client, 
                        prompt,
                        request_id,
                        model=model,  # Pass the model name from settings
                        is_openrouter=True
                    )
                )
            else:
                # Fall back to Ollama if OpenRouter not configured
                logger.warning("OpenRouter selected but not configured, falling back to Ollama")
                client = getattr(request.app.state, 'ollama_client', None)
                if not client:
                    client = OllamaClient()
                    request.app.state.ollama_client = client
                
                # Start background task
                task = asyncio.create_task(
                    generate_response_background(
                        request.app.state,
                        client, 
                        processed_prompt,
                        request_id
                    )
                )
        else:
            # Use Ollama
            client = getattr(request.app.state, 'ollama_client', None)
            if not client:
                client = OllamaClient()
                request.app.state.ollama_client = client
            logger.info("Using Ollama for web request")
            
            # Start background task
            task = asyncio.create_task(
                generate_response_background(
                    request.app.state,
                    client, 
                    processed_prompt,
                    request_id
                )
            )
        
        # Ensure the task is tracked but we don't wait for it
        # This prevents an error in task creation from causing this handler to fail
        task.add_done_callback(lambda t: logger.info(f"Background processing for {request_id} completed"))
        
        # Return the initial response immediately with the request ID
        # This tells the frontend to maintain the loading state and poll for completion
        return {
            "status": "processing",
            "phase": 1,
            "request_id": request_id,
            "message": "Processing query and generating response..."
        }
        
    except Exception as e:
        logger.error(f"Web request error: {e}", exc_info=True)
        # Return a JSON response instead of raising an HTTP exception
        # This makes it easier for the frontend to handle
        return {
            "status": "error",
            "error": str(e),
            "message": "An error occurred processing your request."
        }

# Helper function to generate response in the background
async def generate_response_background(app_state, client, prompt, request_id, model=None, is_openrouter=False):
    """Generate a response in the background and store it in app state"""
    try:
        logger.info(f"Background response generation started for request_id: {request_id}")
        
        # Use the new RAG service for processing with tracking
        from scripts.services.rag_service import process_request_with_tracking
        
        # Process using the service
        result = await process_request_with_tracking(
            app_state=app_state,
            query=prompt,
            request_id=request_id,
            use_openrouter=is_openrouter,
            model=model
        )
        
        # The result is already stored in app_state.pending_responses by the service
        logger.info(f"Response generation complete for request_id: {request_id}, {result.get('word_count', 0)} words")
        
    except Exception as e:
        # Store the error, but only if the request_id still exists
        logger.error(f"Error generating response in background: {e}", exc_info=True)
        
        if hasattr(app_state, 'pending_responses') and request_id in app_state.pending_responses:
            # Update the response with an error message
            error_msg = f"Sorry, an error occurred while generating the response: {str(e)}"
            app_state.pending_responses[request_id] = {
                'prompt': prompt,
                'response': error_msg,
                'error': str(e),
                'processing': False,
                'complete': True,
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'provider': 'openrouter' if is_openrouter else 'ollama',
                    'model_name': model if model else (
                        client.model if hasattr(client, 'model') else 'unknown'
                    ),
                }
            }
        else:
            logger.warning(f"Request_id {request_id} no longer in app_state, cannot store error")

@router.post("/api/references")  # Full path
async def get_references(request: Request):
    """
    Get relevant references and content for a query from the vector stores
    Returns structured data without running through LLM processing
    """
    try:
        body = await request.json()
        query = body.get('prompt') or body.get('query')
        k_value = body.get('context_k', 5)  # Allow configurable number of results
        
        if not query:
            raise HTTPException(status_code=422, detail="No query provided")

        # Use local Ollama client for vector search regardless of configured LLM provider
        # because vector search happens locally in all cases
        client = getattr(request.app.state, 'ollama_client', None)
        if not client:
            client = OllamaClient()
            request.app.state.ollama_client = client
        
        # Process the query using the Ollama client's chunking parameters
        processed_query = client.process_text(query)
        
        logger.info(f"Fetching comprehensive references for query: '{query}' with k={k_value}")
        
        # Get references from reference_store (definitions, methods, projects)
        references = await client.get_relevant_references(processed_query, k=k_value)
        logger.info(f"Found {len(references)} references from all stores")
        
        # Add additional metadata for categorization
        for ref in references:
            if "topic_store" in ref:
                ref["source_category"] = "topic"
            else:
                ref["source_category"] = "reference"
        
        # Get content from content_store (resources, materials)
        content_results = []
        if client.content_store is not None:
            try:
                # Get query embedding
                query_embedding = await client.embeddings.aembeddings([query])
                
                # Check for specific terms to help with relevance evaluation
                query_terms = set(term.lower() for term in query.split() if len(term) > 3)
                
                # Use a safer search approach with error handling
                content_docs = []
                try:
                    # Define a safer search function
                    def safe_content_search():
                        try:
                            # Try the standard search
                            return client.content_store.similarity_search_by_vector(
                                query_embedding[0],
                                k=k_value
                            )
                        except KeyError as key_err:
                            # Handle the specific KeyError issue with index_to_docstore_id
                            logger.error(f"KeyError in content search: {key_err} - index may be inconsistent with docstore")
                            # Try a smaller k value as a fallback
                            try:
                                smaller_k = max(1, k_value // 2)
                                logger.info(f"Attempting with smaller k={smaller_k}")
                                return client.content_store.similarity_search_by_vector(
                                    query_embedding[0],
                                    k=smaller_k
                                )
                            except:
                                # If that still fails, return empty list
                                logger.error("Even reduced k search failed, returning empty results")
                                return []
                        except Exception as e:
                            # Handle any other errors
                            logger.error(f"Unexpected error in content search: {e}")
                            return []
                    
                    # Execute the safe search
                    content_docs = await asyncio.get_event_loop().run_in_executor(
                        None,
                        safe_content_search
                    )
                except Exception as e:
                    logger.error(f"Error executing content store search: {e}")
                
                logger.info(f"Found {len(content_docs)} content items in content_store")
                
                # Get start time for performance tracking
                content_search_start = datetime.now()
                
                # Process the standard search results and calculate term overlap metrics
                # This approach works with the documents we have from the safer search
                processed_count = 0
                try:
                    for i, doc in enumerate(content_docs):
                        processed_count += 1
                        meta = doc.metadata
                        
                        # Calculate term overlap for relevance evaluation
                        doc_terms = set(term.lower() for term in doc.page_content.split() if len(term) > 3)
                        term_overlap = query_terms.intersection(doc_terms)
                        term_overlap_count = len(term_overlap)
                        term_overlap_list = list(term_overlap)[:10]  # Limit to 10 terms
                        
                        # Calculate score based on position in results
                        position_score = 1.0 - (i / (len(content_docs) or 1))
                        
                        # Calculate hybrid score (50% position, 50% term overlap)
                        term_overlap_score = term_overlap_count / max(len(query_terms), 1)
                        hybrid_score = (0.5 * position_score) + (0.5 * term_overlap_score)
                        
                        # Log detailed info about this result with improved scoring
                        logger.info(f"Content #{i+1}: '{meta.get('resource_title', meta.get('title', 'Untitled'))}' - Hybrid score: {hybrid_score:.4f} (position: {position_score:.4f}, terms: {term_overlap_count})")
                        
                        # Create structured content item with rich metadata
                        content_item = {
                            "title": meta.get("resource_title", meta.get("material_title", meta.get("title", "Untitled"))),
                            "content": doc.page_content,
                            "type": meta.get("source_type", "content"),
                            "resource_url": meta.get("resource_url", ""),
                            "url": meta.get("url", ""),
                            "tags": meta.get("tags", []),
                            "source": meta.get("source", ""),
                            "source_category": "content",  # Added for categorization
                            "term_overlap": term_overlap_list,
                            "hybrid_score": float(hybrid_score),  # Calculated hybrid score
                            "position_score": float(position_score),  # Position-based score
                            "term_overlap_count": term_overlap_count,  # Number of matching terms
                            "created_at": meta.get("created_at", meta.get("processed_at", "")),
                            # Include additional metadata that might be useful
                            "metadata": {
                                "chunk_index": meta.get("chunk_index"),
                                "total_chunks": meta.get("total_chunks"),
                                "content_type": meta.get("content_type", ""),
                                "document_count": meta.get("document_count"),
                                "page_name": meta.get("page_name", ""),
                                "resource_id": meta.get("resource_id", "")
                            }
                        }
                        content_results.append(content_item)
                    
                    # Sort by hybrid score for better relevance
                    content_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
                    
                    # Log performance metrics
                    content_search_duration = (datetime.now() - content_search_start).total_seconds()
                    logger.info(f"Content search processing completed in {content_search_duration:.2f}s")
                    
                except Exception as e:
                    # Log if we hit any errors during processing
                    logger.error(f"Error processing content results: {e} (processed {processed_count}/{len(content_docs)} items)")
                    
                    # No fallback processing needed - we've already processed results above
                    logger.warning("The error was handled, but we've already processed available results")
                    
            except Exception as e:
                logger.error(f"Error searching content store: {e}", exc_info=True)
        
        # Get raw context string (combined from both stores)
        # This is useful for frontend processing or simple display
        try:
            context = await client.get_relevant_context(query, k=k_value)
            context_length = len(context.split()) if context else 0
            logger.info(f"Retrieved combined context with {context_length} words")
        except Exception as e:
            logger.error(f"Error getting combined context: {e}")
            context = ""
        
        # Get topic store stats for metadata
        topic_store_info = {}
        if hasattr(client, 'topic_stores'):
            for name, store in client.topic_stores.items():
                if store is not None and hasattr(store, 'docstore'):
                    doc_count = len(store.docstore._dict) if hasattr(store.docstore, '_dict') else 0
                    if doc_count > 0:
                        # Format the topic name for display
                        display_name = name.replace('topic_', '').replace('_', ' ').title()
                        topic_store_info[name] = {
                            "name": display_name,
                            "document_count": doc_count
                        }
        
        # Return all data in a comprehensive, well-structured format
        return {
            "status": "success",
            "query": query,
            "references": references,  # From reference_store and topic stores
            "content": content_results,  # From content_store (resources, materials)
            "context": context,  # Combined context string
            "metadata": {
                "reference_count": len(references),
                "content_count": len(content_results),
                "k_value": k_value,
                "context_words": len(context.split()) if context else 0,
                "timestamp": datetime.now().isoformat(),
                "stores_info": {
                    "reference_store_loaded": client.reference_store is not None,
                    "content_store_loaded": client.content_store is not None,
                    "topic_stores_loaded": len(topic_store_info),
                    "reference_count": len(client.reference_store.docstore._dict) if client.reference_store and hasattr(client.reference_store, 'docstore') else 0,
                    "content_count": len(client.content_store.docstore._dict) if client.content_store and hasattr(client.content_store, 'docstore') else 0,
                    "topic_stores": topic_store_info
                }
            }
        }
    except Exception as e:
        logger.error(f"References request error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def check_ollama_status():
    """Check Ollama availability"""
    try:
        ollama_url = f"http://{os.getenv('OLLAMA_HOST', 'localhost')}:{os.getenv('OLLAMA_PORT', '11434')}/api/version"
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(ollama_url)
            return "connected" if resp.status_code == 200 else "disconnected"
    except:
        return "disconnected"

@router.get("/api/vector-store-status")
async def vector_store_status(request: Request):
    """Get detailed status of vector stores for debugging"""
    try:
        # Use local Ollama client to check vector stores
        client = getattr(request.app.state, 'ollama_client', None)
        if not client:
            client = OllamaClient()
            request.app.state.ollama_client = client
            
        # Get config info
        config_path = client.get_config_path() if hasattr(client, 'get_config_path') else None
        config_data = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                
        # Get vector store paths
        data_path = Path(config_data.get('data_path', ''))
        vector_path = data_path / "vector_stores" / "default"
        reference_path = vector_path / "reference_store"
        content_path = vector_path / "content_store"
        
        # Check for FAISS index files
        reference_index_exists = (reference_path / "index.faiss").exists() if reference_path.exists() else False
        content_index_exists = (content_path / "index.faiss").exists() if content_path.exists() else False
        
        # Get document counts
        reference_count = len(client.reference_store.docstore._dict) if client.reference_store and hasattr(client.reference_store, 'docstore') else 0
        content_count = len(client.content_store.docstore._dict) if client.content_store and hasattr(client.content_store, 'docstore') else 0
        
        # Count topic stores
        topic_stores = []
        if vector_path.exists():
            topic_store_paths = list(vector_path.glob("topic_*"))
            for topic_path in topic_store_paths:
                if (topic_path / "index.faiss").exists():
                    topic_stores.append(topic_path.name)
        
        # Get FAISS index size if possible
        reference_index_size = 0
        content_index_size = 0
        try:
            if reference_index_exists:
                reference_index_size = (reference_path / "index.faiss").stat().st_size
            if content_index_exists:
                content_index_size = (content_path / "index.faiss").stat().st_size
        except Exception as e:
            logger.error(f"Error getting index sizes: {e}")
            
        return {
            "status": "success",
            "vector_stores": {
                "data_path": str(data_path),
                "vector_path": str(vector_path),
                "reference_store": {
                    "path": str(reference_path),
                    "index_exists": reference_index_exists,
                    "index_size_bytes": reference_index_size,
                    "document_count": reference_count,
                    "loaded": client.reference_store is not None
                },
                "content_store": {
                    "path": str(content_path),
                    "index_exists": content_index_exists, 
                    "index_size_bytes": content_index_size,
                    "document_count": content_count,
                    "loaded": client.content_store is not None
                },
                "topic_stores": topic_stores,
                "topic_store_count": len(topic_stores)
            },
            "action_tips": [
                "To rebuild all indexes, run python scripts/tools/rebuild_faiss_indexes.py",
                "For full reprocessing, run python run_complete_processing.py to regenerate all indexes",
                "For incremental processing, use python scripts/tools/run_incremental_processing.py"
            ]
        }
    except Exception as e:
        logger.error(f"Vector store status error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "suggestion": "Try running python run_complete_processing.py to regenerate all vector stores"
        }

@router.post("/api/rebuild-faiss-indexes")
async def rebuild_faiss_indexes(request: Request):
    """
    Rebuild all FAISS indexes from existing document stores.
    This ensures consistency between document content and vector indexes.
    """
    try:
        # Use local Ollama client to access vector stores
        client = getattr(request.app.state, 'ollama_client', None)
        if not client:
            client = OllamaClient()
            request.app.state.ollama_client = client
            
        # Get config info to identify data_path
        config_path = client.get_config_path() if hasattr(client, 'get_config_path') else None
        config_data = {}
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        
        data_path = Path(config_data.get('data_path', ''))
        
        # Start time for performance tracking
        start_time = datetime.now()
        logger.info("Starting FAISS index rebuild for all stores")
        
        # Use the new modular code for rebuilding if it exists, otherwise fall back to old method
        try:
            from scripts.data.processing_manager import ProcessingManager
            processing_manager = ProcessingManager(data_path)
            result = await processing_manager.rebuild_all_indexes()
            logger.info("Rebuilding indexes using new modular code")
        except ImportError:
            # Fall back to old implementation
            from scripts.data.data_processor import DataProcessor
            data_processor = DataProcessor(data_path)
            result = await data_processor.rebuild_all_vector_stores()
            logger.info("Rebuilding indexes using legacy code")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Reload the vector stores in the client to use the new indexes
        client.load_vector_stores()
        request.app.state.ollama_client = client
        
        return {
            "status": "success",
            "message": "All FAISS indexes have been rebuilt successfully",
            "processing_time_seconds": processing_time,
            "stores_rebuilt": result.get("stores_rebuilt", []),
            "document_counts": result.get("document_counts", {}),
            "total_documents": result.get("total_documents", 0)
        }
    except Exception as e:
        logger.error(f"Error rebuilding FAISS indexes: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "suggestion": "Check server logs for details"
        }
