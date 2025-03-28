import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from scripts.llm.rag_handler import RAGHandler
from scripts.llm.ollama_client import OllamaClient
from scripts.llm.open_router_client import OpenRouterClient

logger = logging.getLogger(__name__)

class RAGService:
    """
    Service to coordinate RAG operations between RAGHandler and LLM clients,
    with tracking of processing state and timing.
    """
    
    def __init__(self, data_path: str = None, group_id: str = "default"):
        self.rag_handler = RAGHandler(docs_path=data_path, group_id=group_id)
        # We'll initialize LLM clients as needed
        
    async def process_query(self, 
                     query: str, 
                     request_id: str,
                     context_k: int = 5, 
                     ollama_client: Optional[OllamaClient] = None,
                     openrouter_client: Optional[OpenRouterClient] = None,
                     model: str = None,
                     use_openrouter: bool = False) -> Dict[str, Any]:
        """
        Process a query using RAG with detailed tracking of processing steps
        """
        start_time = datetime.now()
        processing_info = {
            'request_id': request_id,
            'query': query,
            'processing': True,
            'complete': False,
            'start_time': start_time.isoformat(),
            'stages': {
                'retrieval': {
                    'status': 'pending',
                    'start_time': None,
                    'end_time': None,
                    'duration_ms': None
                },
                'generation': {
                    'status': 'pending',
                    'start_time': None,
                    'end_time': None,
                    'duration_ms': None
                },
            },
            'model_info': {
                'provider': 'openrouter' if use_openrouter else 'ollama',
                'model_name': model if model else (
                    openrouter_client.model if use_openrouter and openrouter_client else 
                    (ollama_client.model if ollama_client else 'unknown')
                ),
            },
            'response': '',
            'error': None
        }
        
        try:
            # 1. RETRIEVAL PHASE
            retrieval_start = datetime.now()
            processing_info['stages']['retrieval']['status'] = 'in_progress'
            processing_info['stages']['retrieval']['start_time'] = retrieval_start.isoformat()
            
            # Run query through RAG handler to get context
            try:
                # Use run_query for detailed metrics
                query_results = await self.rag_handler.run_query(
                    query=query, 
                    k=context_k, 
                    include_metadata=True
                )
                
                # Record timing info from retrieval
                retrieval_end = datetime.now()
                retrieval_duration = (retrieval_end - retrieval_start).total_seconds() * 1000
                processing_info['stages']['retrieval']['status'] = 'complete'
                processing_info['stages']['retrieval']['end_time'] = retrieval_end.isoformat()
                processing_info['stages']['retrieval']['duration_ms'] = retrieval_duration
                
                # Add retrieval metrics
                processing_info['retrieval_info'] = {
                    'reference_count': len(query_results['reference_results']),
                    'content_count': len(query_results['content_results']),
                    'timing': query_results['timing']
                }
                
                # Get context from query results
                context = self._format_context_from_results(query_results)
                
                # Format context into the expected structure
                context_info = {
                    'context': context,
                    'context_word_count': len(context.split()) if context else 0
                }
                processing_info['context_info'] = context_info
                
            except Exception as e:
                logger.error(f"Error in retrieval phase: {e}")
                processing_info['stages']['retrieval']['status'] = 'error'
                processing_info['stages']['retrieval']['error'] = str(e)
                # Use empty context if retrieval fails
                context = "No specific information found about this topic."
            
            # 2. GENERATION PHASE
            generation_start = datetime.now()
            processing_info['stages']['generation']['status'] = 'in_progress'
            processing_info['stages']['generation']['start_time'] = generation_start.isoformat()
            
            # Choose the appropriate client
            if use_openrouter and openrouter_client:
                logger.info(f"Using OpenRouter model '{model if model else openrouter_client.model}' for request_id: {request_id}")
                # Track model details
                processing_info['model_info']['provider'] = 'openrouter'
                processing_info['model_info']['model_name'] = model if model else openrouter_client.model
                # Generate using OpenRouter
                response = await openrouter_client.generate_response(
                    prompt=query,
                    model=model
                )
            elif ollama_client:
                logger.info(f"Using Ollama model '{ollama_client.model}' for request_id: {request_id}")
                # Track model details
                processing_info['model_info']['provider'] = 'ollama'
                processing_info['model_info']['model_name'] = ollama_client.model
                # Generate using Ollama with the retrieved context
                response = await ollama_client.generate_response(
                    prompt=query,
                    context_k=context_k
                )
            else:
                raise ValueError("No valid LLM client provided")
            
            # Record generation completion
            generation_end = datetime.now()
            generation_duration = (generation_end - generation_start).total_seconds() * 1000
            processing_info['stages']['generation']['status'] = 'complete'
            processing_info['stages']['generation']['end_time'] = generation_end.isoformat()
            processing_info['stages']['generation']['duration_ms'] = generation_duration
            
            # Update final response and timing
            processing_info['response'] = response
            processing_info['word_count'] = len(response.split())
            processing_info['processing'] = False
            processing_info['complete'] = True
            processing_info['end_time'] = datetime.now().isoformat()
            processing_info['total_duration_ms'] = (datetime.now() - start_time).total_seconds() * 1000
            processing_info['total_duration_seconds'] = round((datetime.now() - start_time).total_seconds(), 2)
            
            # Log completion with performance metrics
            model_name = processing_info['model_info']['model_name']
            provider = processing_info['model_info']['provider']
            logger.info(f"Request {request_id} completed in {processing_info['total_duration_seconds']}s "
                       f"using {provider}/{model_name} - "
                       f"Retrieval: {processing_info['stages']['retrieval']['duration_ms']/1000:.2f}s, "
                       f"Generation: {processing_info['stages']['generation']['duration_ms']/1000:.2f}s, "
                       f"Response: {processing_info['word_count']} words")
            
            return processing_info
            
        except Exception as e:
            # Handle any errors in the overall process
            logger.error(f"Error processing query: {e}")
            end_time = datetime.now()
            processing_info['processing'] = False
            processing_info['complete'] = True
            processing_info['error'] = str(e)
            processing_info['end_time'] = end_time.isoformat()
            processing_info['total_duration_ms'] = (end_time - start_time).total_seconds() * 1000
            processing_info['total_duration_seconds'] = round((end_time - start_time).total_seconds(), 2)
            processing_info['response'] = f"Error processing query: {str(e)}"
            
            return processing_info
    
    def _format_context_from_results(self, query_results: Dict[str, Any]) -> str:
        """Format the query results into a context string for the LLM"""
        context_parts = []
        
        # Add reference results
        if query_results.get('reference_results'):
            context_parts.append("\nReference materials:")
            for item in query_results['reference_results']:
                if 'metadata' in item and item['metadata'].get('title'):
                    title = item['metadata'].get('title', 'Untitled')
                    source_type = item['metadata'].get('source_type', 'reference')
                    context_parts.append(f"[{source_type}] {title}: {item['text']}")
                else:
                    context_parts.append(item['text'])
        
        # Add content results
        if query_results.get('content_results'):
            context_parts.append("\nDetailed content:")
            for item in query_results['content_results']:
                if 'metadata' in item:
                    title = item['metadata'].get('title', 'Untitled Content')
                    source_type = item['metadata'].get('source_type', 'content')
                    context_parts.append(f"From {source_type} '{title}': {item['text']}")
                else:
                    context_parts.append(item['text'])
        
        # Combine all parts
        return "\n".join(context_parts)

# Example of how to use the service:
async def process_request_with_tracking(app_state, query, request_id, use_openrouter=False, model=None):
    """Process a request with tracking using the RAG service"""
    try:
        # Get the appropriate clients
        ollama_client = getattr(app_state, 'ollama_client', None)
        openrouter_client = getattr(app_state, 'openrouter_client', None) if use_openrouter else None
        
        # Get model info if not provided
        if model is None and use_openrouter and openrouter_client:
            from scripts.services.settings_manager import SettingsManager
            settings_manager = SettingsManager()
            settings = settings_manager.load_settings()
            model = settings.openrouter_model
        
        # Initialize the service
        rag_service = RAGService()
        
        # Process the query
        result = await rag_service.process_query(
            query=query,
            request_id=request_id,
            ollama_client=ollama_client,
            openrouter_client=openrouter_client,
            model=model,
            use_openrouter=use_openrouter
        )
        
        # Store the result in app state for later retrieval
        if hasattr(app_state, 'pending_responses'):
            app_state.pending_responses[request_id] = result
            
        return result
    except Exception as e:
        logger.error(f"Error in process_request_with_tracking: {e}")
        return {
            'request_id': request_id,
            'query': query,
            'processing': False,
            'complete': True,
            'error': str(e),
            'response': f"Error processing request: {str(e)}",
            'model_info': {
                'provider': 'openrouter' if use_openrouter else 'ollama',
                'model_name': model if model else 'unknown',
            }
        }
