from fastapi import APIRouter, HTTPException, Request
from scripts.services.settings_manager import SettingsManager
from scripts.services.conversation_logger import log_search_conversation
from scripts.services.simple_search_cache import get_search_cache
from scripts.models.settings import LLMProvider
from scripts.llm.ollama_client import OllamaClient
from scripts.llm.open_router_client import OpenRouterClient
import httpx
import os
import logging
import json
import asyncio
import uuid
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# Create router with proper configuration
router = APIRouter(tags=["search"])

logger = logging.getLogger(__name__)

class WebConfig(BaseModel):
    prompt: str
    context_k: Optional[int] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

@router.post("/api/web")
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
            original_query = data.get('query', '')
            skip_search = data.get('skip_search', False)
            references_provided = data.get('references_provided', False)
            
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
                        formatted_response = {
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
                        
                        # Log the completed conversation
                        try:
                            conversation_data = {
                                'request_id': request_id,
                                'query': response_data.get('query', original_query),
                                'prompt': response_data.get('prompt', ''),
                                'response': response_data.get('response', ''),
                                'word_count': formatted_response['word_count'],
                                'model_info': model_info,
                                'skip_search': response_data.get('skip_search', False),
                                'timing': formatted_response['timing'],
                                'timestamp': formatted_response['timestamp'],
                                'references': response_data.get('references', []),
                                'content': response_data.get('content', [])
                            }
                            log_search_conversation(conversation_data)
                        except Exception as e:
                            logger.warning(f"Failed to log conversation {request_id}: {e}")
                        
                        return formatted_response
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
                        request_id,
                        skip_search=skip_search,
                        original_query=original_query
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
                    request_id,
                    skip_search=skip_search,
                    original_query=original_query
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
async def generate_response_background(app_state, client, prompt, request_id, model=None, is_openrouter=False, skip_search=False, original_query=None):
    """Generate a response in the background and store it in app state"""
    try:
        logger.info(f"Background response generation started for request_id: {request_id}")
        logger.info(f"Skip search flag: {skip_search}")
        
        if skip_search and original_query:
            logger.info(f"Skipping search phase - using pre-formatted prompt for query: '{original_query}'")
            # When skip_search is True, we bypass the RAG search and use the prompt directly
            # The prompt already contains the formatted references from the frontend
            
            # Use a direct LLM call instead of going through the RAG service
            if is_openrouter:
                # Use OpenRouter directly
                openrouter_client = getattr(app_state, 'openrouter_client', None)
                if openrouter_client:
                    from scripts.llm.open_router_client import OpenRouterClient
                    
                    # Create a simplified processing approach for pre-formatted prompts
                    logger.info("Using OpenRouter for direct LLM processing (no search)")
                    
                    # Call OpenRouter with the pre-formatted prompt
                    response = await openrouter_client.generate_response_direct(
                        prompt=prompt,
                        model=model,
                        max_tokens=2000,
                        temperature=0.7
                    )
                    
                    # Store the result
                    if hasattr(app_state, 'pending_responses'):
                        app_state.pending_responses[request_id] = {
                            'request_id': request_id,
                            'query': original_query,
                            'prompt': prompt,
                            'response': response.get('response', ''),
                            'processing': False,
                            'complete': True,
                            'word_count': len(response.get('response', '').split()),
                            'model_info': {
                                'provider': 'openrouter',
                                'model_name': model
                            },
                            'skip_search': True,
                            'timestamp': datetime.now().isoformat()
                        }
                    return
                else:
                    logger.warning("OpenRouter client not available, falling back to RAG service")
            else:
                # Use Ollama directly  
                logger.info("Using Ollama for direct LLM processing (no search)")
                if hasattr(client, 'generate_response_direct'):
                    response = await client.generate_response_direct(
                        prompt=prompt,
                        max_tokens=2000,
                        temperature=0.7
                    )
                    
                    # Store the result
                    if hasattr(app_state, 'pending_responses'):
                        app_state.pending_responses[request_id] = {
                            'request_id': request_id,
                            'query': original_query,
                            'prompt': prompt,
                            'response': response.get('response', ''),
                            'processing': False,
                            'complete': True,
                            'word_count': len(response.get('response', '').split()),
                            'model_info': {
                                'provider': 'ollama',
                                'model_name': getattr(client, 'model', 'unknown')
                            },
                            'skip_search': True,
                            'timestamp': datetime.now().isoformat()
                        }
                    return
                else:
                    logger.warning("Ollama client does not support direct generation, falling back to RAG service")
        
        # Fall back to normal RAG processing if skip_search is False or direct methods fail
        
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

@router.post("/api/references")
async def get_references(request: Request):
    """
    Get relevant references and content for a query from the vector stores
    Returns structured data without running through LLM processing
    """
    try:
        body = await request.json()
        query = body.get('prompt') or body.get('query')
        k_value = body.get('context_k', 10)  # Increased from 5 to 10 for better results
        
        if not query:
            raise HTTPException(status_code=422, detail="No query provided")

        # Check cache first to avoid duplicate searches
        search_cache = get_search_cache()
        cached_results = search_cache.get(query, k_value)
        if cached_results:
            logger.info(f"Returning cached search results for query: '{query[:50]}...'")
            return cached_results

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
                # Check for specific terms to help with relevance evaluation
                query_terms = set(term.lower() for term in query.split() if len(term) > 3)
                
                # Use optimized FAISS text-based search (no redundant embedding generation)
                content_docs = []
                try:
                    # Define an optimized search function using FAISS's built-in text search
                    def optimized_content_search():
                        try:
                            # Use FAISS's similarity_search which handles embedding internally
                            return client.content_store.similarity_search(
                                query,
                                k=k_value
                            )
                        except KeyError as key_err:
                            # Handle the specific KeyError issue with index_to_docstore_id
                            logger.error(f"KeyError in content search: {key_err} - index may be inconsistent with docstore")
                            # Try a smaller k value as a fallback
                            try:
                                smaller_k = max(1, k_value // 2)
                                logger.info(f"Attempting with smaller k={smaller_k}")
                                return client.content_store.similarity_search(
                                    query,
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
                    
                    # Execute the optimized search
                    content_docs = await asyncio.get_event_loop().run_in_executor(
                        None,
                        optimized_content_search
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
                        logger.info(f"Content #{i+1}: '{meta.get('resource_id', meta.get('title', 'Untitled'))}' - Hybrid score: {hybrid_score:.4f} (position: {position_score:.4f}, terms: {term_overlap_count})")
                        # Debug the actual content structure - this helps identify schema issues
                        if i == 0:  # Only for the first item to avoid log spam
                            logger.debug(f"First content item metadata keys: {list(meta.keys())}")
                        
                        # Create structured content item with rich metadata
                        content_item = {
                            "title": meta.get("resource_id", meta.get("material_title", meta.get("title", "Untitled"))),
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
        response_data = {
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
        
        # Cache the results for potential reuse by /api/web endpoint
        search_cache = get_search_cache()
        search_cache.put(query, response_data, k_value)
        
        return response_data
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

class UnifiedSearchRequest(BaseModel):
    query: str
    k_reference: Optional[int] = 3  # Increased: references for attribution/sourcing
    k_content: Optional[int] = 7    # Adjusted: content for detailed information (total 10)
    
@router.post("/api/search/unified")
async def unified_search(request: UnifiedSearchRequest, req: Request):
    """
    Perform a search across both reference and content stores with a single embedding.
    
    This optimized endpoint uses a single query embedding for both searches to reduce latency.
    """
    try:
        query = request.query
        if not query or not query.strip():
            return {"error": "Query cannot be empty", "status": "error"}
        
        # Get client from app state or create new one
        client = getattr(req.app.state, "openrouter_client", None)
        if not client:
            # Import here to avoid circular imports
            from scripts.llm.open_router_client import OpenRouterClient
            client = OpenRouterClient()
            req.app.state.openrouter_client = client
            
        # Perform unified search
        results = await client.unified_search(
            query=query,
            k_reference=request.k_reference,
            k_content=request.k_content
        )
        
        # Add extra metadata
        results["query"] = query
        results["timestamp"] = datetime.now().isoformat()
        
        # Return cached status to client
        results["metadata"]["cache_enabled"] = True
        
        return results
    
    except Exception as e:
        logger.error(f"Error in unified search: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}

class PinItemRequest(BaseModel):
    item: Dict[str, Any]
    item_type: str = "search_result"
    
class UnpinItemRequest(BaseModel):
    item_id: str
    
@router.post("/api/search/pin")
async def pin_search_item(request: PinItemRequest, req: Request):
    """Pin a search result or chunk for later use."""
    try:
        # Get or create pinned items manager
        if not hasattr(req.app.state, "pinned_items_manager"):
            from scripts.services.pinned_items_manager import PinnedItemsManager
            req.app.state.pinned_items_manager = PinnedItemsManager()
        
        manager = req.app.state.pinned_items_manager
        
        # Pin the item
        item_id = manager.pin_item(request.item, request.item_type)
        
        return {
            "status": "success",
            "message": f"Item pinned successfully",
            "item_id": item_id
        }
        
    except Exception as e:
        logger.error(f"Error pinning item: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

@router.post("/api/search/unpin")
async def unpin_search_item(request: UnpinItemRequest, req: Request):
    """Unpin a previously pinned item."""
    try:
        # Get pinned items manager
        if not hasattr(req.app.state, "pinned_items_manager"):
            from scripts.services.pinned_items_manager import PinnedItemsManager
            req.app.state.pinned_items_manager = PinnedItemsManager()
        
        manager = req.app.state.pinned_items_manager
        
        # Unpin the item
        success = manager.unpin_item(request.item_id)
        
        if success:
            return {
                "status": "success",
                "message": f"Item unpinned successfully"
            }
        else:
            return {
                "status": "error",
                "error": f"Item with ID {request.item_id} not found"
            }
        
    except Exception as e:
        logger.error(f"Error unpinning item: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
        
@router.get("/api/search/pinned")
async def get_pinned_items(req: Request, item_type: Optional[str] = None):
    """Get all pinned items, optionally filtered by type."""
    try:
        # Get pinned items manager
        if not hasattr(req.app.state, "pinned_items_manager"):
            from scripts.services.pinned_items_manager import PinnedItemsManager
            req.app.state.pinned_items_manager = PinnedItemsManager()
        
        manager = req.app.state.pinned_items_manager
        
        # Get items
        items = manager.get_all_pinned_items(item_type)
        
        return {
            "status": "success",
            "count": len(items),
            "items": items
        }
        
    except Exception as e:
        logger.error(f"Error getting pinned items: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

class DeeperSearchRequest(BaseModel):
    query: str
    pinned_item_ids: List[str] = []
    k: Optional[int] = 5

@router.post("/api/search/deeper")
async def search_deeper(request: DeeperSearchRequest, req: Request):
    """
    Perform a deeper search using pinned items as additional context.
    
    This endpoint allows users to search more deeply by leveraging pinned items
    as additional context. It uses the embedding cache to optimize performance.
    """
    try:
        query = request.query
        if not query or not query.strip():
            return {"error": "Query cannot be empty", "status": "error"}
            
        # Get pinned items manager
        if not hasattr(req.app.state, "pinned_items_manager"):
            from scripts.services.pinned_items_manager import PinnedItemsManager
            req.app.state.pinned_items_manager = PinnedItemsManager()
            
        pinned_manager = req.app.state.pinned_items_manager
        
        # Get client from app state or create new one
        client = getattr(req.app.state, "openrouter_client", None)
        if not client:
            # Import here to avoid circular imports
            from scripts.llm.open_router_client import OpenRouterClient
            client = OpenRouterClient()
            req.app.state.openrouter_client = client
        
        # Collect all requested pinned items
        pinned_items = []
        if request.pinned_item_ids:
            for item_id in request.pinned_item_ids:
                item = pinned_manager.get_pinned_item(item_id)
                if item:
                    pinned_items.append(item)
            
            logger.info(f"Retrieved {len(pinned_items)}/{len(request.pinned_item_ids)} pinned items for deeper search")
            
        # If no specific pinned items were requested, use the most recent 3
        if not pinned_items:
            # Get the 3 most recent search result items
            all_search_results = pinned_manager.get_all_pinned_items("search_result")
            # Sort by pinned_at date (newest first)
            all_search_results.sort(key=lambda x: x.get("pinned_at", ""), reverse=True)
            pinned_items = all_search_results[:3]
            logger.info(f"Using {len(pinned_items)} most recent pinned search results for deeper search")
            
        # Perform deeper search
        results = await client.search_deeper(
            query=query,
            pinned_items=pinned_items,
            k=request.k
        )
        
        # Add extra metadata
        results["query"] = query
        results["timestamp"] = datetime.now().isoformat()
        results["pinned_items_used"] = [
            {"id": item.get("id"), "title": item.get("title", "Untitled")} 
            for item in pinned_items
        ]
        
        return results
    
    except Exception as e:
        logger.error(f"Error in deeper search: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}

@router.get("/api/conversations/stats")
async def get_conversation_stats():
    """Get statistics about logged conversations."""
    try:
        from scripts.services.conversation_logger import get_conversation_logger
        
        logger_instance = get_conversation_logger()
        stats = logger_instance.get_stats()
        
        return {
            "status": "success",
            "data": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation stats: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}
