from pathlib import Path
import httpx
import logging
from typing import Dict, Any, Tuple, List, Optional
import json
import asyncio
from scripts.services.settings_manager import SettingsManager
from scripts.models.settings import ModelSettings
from .ollama_embeddings import OllamaEmbeddings
import os

logger = logging.getLogger(__name__)

class OpenRouterClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://openrouter.ai/api/v1"):
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("No OpenRouter API key provided. Set OPENROUTER_API_KEY environment variable.")
        
        self.settings_manager = SettingsManager()
        self.settings = self.settings_manager.load_settings()
        self.default_model = "mistralai/mistral-nemo:free"  # Default model
        
        # Keep using local embeddings via Ollama
        self.embeddings = OllamaEmbeddings()
        
        # Initialize vector stores
        self.reference_store = None
        self.content_store = None
        self.load_vector_stores()
        
        logger.info("Initialized OpenRouter client")
    
    def get_config_path(self):
        """Get the path to the config file in the project directory"""
        # First try to use config.json in the project root
        local_config = Path("config.json")
        if local_config.exists():
            return local_config
        
        # If it doesn't exist, try the user's home directory as fallback
        home_config = Path('config.json')
        if home_config.exists():
            # Copy the home config to the local config for future use
            os.makedirs(Path.home() / '.blob', exist_ok=True)
            with open(home_config, 'r') as src:
                config_data = json.load(src)
            
            with open(local_config, 'w') as dst:
                json.dump(config_data, dst, indent=2)
            
            logger.info(f"Copied config from {home_config} to {local_config}")
            return local_config
        
        # If neither exists, create a new local config
        os.makedirs(Path(local_config).parent, exist_ok=True)
        with open(local_config, 'w') as f:
            default_config = {"data_path": str(Path.cwd() / "data")}
            json.dump(default_config, f, indent=2)
        
        logger.info(f"Created new config file at {local_config}")
        return local_config

    def load_vector_stores(self):
        """Load both vector stores if they exist"""
        try:
            # Reset stores to None before reloading
            self.reference_store = None
            self.content_store = None
            
            config_path = self.get_config_path()
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            data_path = Path(config.get('data_path', ''))
            vector_path = data_path / "vector_stores" / "default"
            logger.info(f"Looking for vector stores in: {vector_path}")

            # Import here to avoid circular imports
            from langchain_community.vectorstores import FAISS

            # Load reference store
            reference_path = vector_path / "reference_store"
            if reference_path.exists() and (reference_path / "index.faiss").exists():
                try:
                    self.reference_store = FAISS.load_local(
                        str(reference_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    ref_count = len(self.reference_store.docstore._dict) if hasattr(self.reference_store, 'docstore') else 0
                    logger.info(f"Successfully loaded reference store with {ref_count} documents")
                except Exception as e:
                    logger.error(f"Error loading reference store: {e}")
            else:
                logger.warning(f"Reference store not found at {reference_path} or index.faiss missing")

            # Load content store
            content_path = vector_path / "content_store"
            if content_path.exists() and (content_path / "index.faiss").exists():
                try:
                    self.content_store = FAISS.load_local(
                        str(content_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    content_count = len(self.content_store.docstore._dict) if hasattr(self.content_store, 'docstore') else 0
                    logger.info(f"Successfully loaded content store with {content_count} documents")
                except Exception as e:
                    logger.error(f"Error loading content store: {e}")
            else:
                logger.warning(f"Content store not found at {content_path} or index.faiss missing")

        except Exception as e:
            logger.error(f"Error loading vector stores: {e}")
            
    async def _hybrid_search(self, store, query: str, query_embedding: list, k: int = 5, alpha: float = 0.5) -> list:
        """Perform hybrid search combining vector similarity with term overlap"""
        if not store:
            return []
            
        try:
            # Extract key terms for relevance checking
            query_terms = set(term.lower() for term in query.split() if len(term) > 3)
            
            # Get documents using standard similarity search
            vector_docs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: store.similarity_search_by_vector(
                    query_embedding,
                    k=k * 2  # Get more candidates for reranking
                )
            )
            
            # Calculate scores using term overlap for additional relevance
            results = []
            for doc in vector_docs:
                # Calculate term overlap score
                doc_terms = set(term.lower() for term in doc.page_content.split() if len(term) > 3)
                term_overlap = len(query_terms.intersection(doc_terms))
                term_overlap_score = term_overlap / max(len(query_terms), 1)
                
                # Since we don't have actual vector scores, estimate based on position
                # (earlier results are more relevant in vector search)
                position_score = 1.0 - (vector_docs.index(doc) / (len(vector_docs) or 1))
                
                # Calculate hybrid score (combine position + term overlap)
                hybrid_score = (alpha * position_score) + ((1 - alpha) * term_overlap_score)
                
                # Add metadata to track relevance factors
                doc.metadata["position_score"] = float(position_score)
                doc.metadata["term_overlap"] = term_overlap
                doc.metadata["hybrid_score"] = float(hybrid_score)
                
                # Add to results for reranking
                results.append((doc, hybrid_score))
            
            # Sort by hybrid score and return top k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    async def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Get relevant context from both stores using hybrid search"""
        context_parts = []
        
        try:
            from datetime import datetime
            search_start = datetime.now()
            
            # Get query embedding
            query_embedding = await self.embeddings.aembeddings([query])
            logger.info(f"Generated query embedding with {len(query_embedding[0])} dimensions")
            
            # Check for specific terms to help with relevance evaluation
            query_terms = set(term.lower() for term in query.split() if len(term) > 3)
            logger.info(f"Query terms: {', '.join(query_terms) if query_terms else 'none'}")
            
            # Reference store search with hybrid approach
            if self.reference_store is not None:
                try:
                    # Use hybrid search for better relevance
                    ref_results = await self._hybrid_search(
                        self.reference_store,
                        query,
                        query_embedding[0],
                        k=k,
                        alpha=0.7  # Weight vector search more heavily for references
                    )
                    
                    if ref_results:
                        context_parts.append("\nReference materials:")
                        logger.info(f"Found {len(ref_results)} relevant reference items")
                        
                        # Log and add retrieved documents
                        for i, (doc, score) in enumerate(ref_results):
                            meta = doc.metadata
                            source = f"[{meta.get('source_type', 'unknown')}]"
                            
                            # Log the metadata and relevance
                            vector_score = meta.get('vector_score', 0)
                            term_overlap = meta.get('term_overlap', 0)
                            logger.info(f"Reference #{i+1}: {source} - Score: {score:.4f} (vector: {vector_score:.4f}, terms: {term_overlap})")
                            
                            # Add to context with rich formatting
                            title = meta.get('title', 'Untitled Reference')
                            context_parts.append(f"{source} {title}: {doc.page_content}")
                except Exception as e:
                    logger.error(f"Error searching reference store: {e}")
            else:
                logger.warning("Reference store is not available")

            # Content store search with hybrid approach
            if self.content_store is not None:
                try:
                    # Use hybrid search with more weight on term overlap for content
                    content_results = await self._hybrid_search(
                        self.content_store,
                        query,
                        query_embedding[0],
                        k=k,
                        alpha=0.5  # Equal weight to vector and term overlap
                    )
                    
                    if content_results:
                        context_parts.append("\nDetailed content:")
                        logger.info(f"Found {len(content_results)} relevant content items")
                        
                        # Log retrieved documents
                        for i, (doc, score) in enumerate(content_results):
                            meta = doc.metadata
                            title = meta.get('material_title', meta.get('title', 'Untitled Content'))
                            source_type = meta.get('source_type', 'content')
                            
                            # Log the metadata
                            vector_score = meta.get('vector_score', 0)
                            term_overlap = meta.get('term_overlap', 0)
                            logger.info(f"Content #{i+1}: '{title}' [{source_type}] - Score: {score:.4f} (vector: {vector_score:.4f}, terms: {term_overlap})")
                            
                            # Add to context with source information
                            context_parts.append(f"From {source_type} '{title}': {doc.page_content}")
                except Exception as e:
                    logger.error(f"Error searching content store: {e}")
            else:
                logger.warning("Content store is not available")

            # Combine all context parts
            context = "\n".join(context_parts)
            
            # Log search performance
            search_duration = (datetime.now() - search_start).total_seconds()
            if context:
                word_count = len(context.split())
                logger.info(f"Retrieved context: {word_count} words in {search_duration:.2f}s")
            else:
                logger.warning(f"No context retrieved in {search_duration:.2f}s")
                
            return context

        except Exception as e:
            logger.error(f"Error getting context: {e}", exc_info=True)
            return ""

    async def get_relevant_references(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get relevant references with metadata using hybrid search"""
        try:
            references = []
            
            # Get query embedding
            query_embedding = await self.embeddings.aembeddings([query])
            
            if self.reference_store is not None:
                try:
                    # Use hybrid search for better relevance
                    ref_results = await self._hybrid_search(
                        self.reference_store,
                        query,
                        query_embedding[0],
                        k=k,
                        alpha=0.7  # Weight vector search more heavily for references
                    )
                    
                    # Process each reference with scores
                    for doc, score in ref_results:
                        meta = doc.metadata
                        vector_score = meta.get('vector_score', 0)
                        term_overlap = meta.get('term_overlap', 0)
                        
                        # Create rich reference object with scores
                        reference = {
                            "title": meta.get("title", "Untitled"),
                            "content": doc.page_content,
                            "type": meta.get("source_type", "unknown"),
                            "tags": meta.get("tags", []),
                            "relevance_score": float(score),  # Overall hybrid score
                            "vector_score": float(vector_score),  # Vector component
                            "term_overlap": term_overlap,  # Term match count
                            "source": meta.get("source", ""),
                            "created_at": meta.get("created_at", ""),
                        }
                        references.append(reference)
                        
                    logger.info(f"Found {len(references)} relevant references with hybrid search")
                    
                except Exception as e:
                    logger.error(f"Error searching references: {e}")
            
            return references

        except Exception as e:
            logger.error(f"Error getting references: {e}", exc_info=True)
            return []
            
    async def check_connection(self) -> Tuple[bool, str]:
        """Check if OpenRouter API is accessible and verify model availability"""
        try:
            if not self.api_key:
                return False, "No API key provided"
                
            async with httpx.AsyncClient(timeout=10.0) as client:
                logger.info("Checking OpenRouter connection...")
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://blob.iaac.net",  # Replace with your site
                    "X-Title": "Blob RAG Assistant"
                }
                response = await client.get(
                    f"{self.base_url}/models", 
                    headers=headers
                )
                
                if response.status_code != 200:
                    return False, f"API error: {response.status_code}"
                    
                # Check available models
                models_data = response.json()
                
                if not models_data.get("data"):
                    return False, "No models available"
                
                # Check if our configured model is available
                self.available_models = [model.get("id") for model in models_data.get("data", [])]
                logger.info(f"Connected to OpenRouter, found {len(self.available_models)} models")
                
                # Check current model setting
                settings = self.settings_manager.load_settings()
                configured_model = settings.openrouter_model
                
                # Validate that the configured model is in the available models
                if configured_model not in self.available_models:
                    logger.warning(f"Configured model '{configured_model}' not found in available models")
                    # We'll still return success but warn about the model
                    return True, f"Connected to OpenRouter, but configured model '{configured_model}' may not be available"
                    
                return True, "Connected to OpenRouter API"

        except Exception as e:
            logger.error(f"Connection check failed: {str(e)}")
            return False, str(e)

    async def generate_response(self, prompt: str, model: str = None, context_k: int = None, max_tokens: int = None) -> str:
        """Generate a response using RAG with OpenRouter"""
        try:
            # Verify OpenRouter is accessible
            ready, status = await self.check_connection()
            if not ready:
                logger.error(f"OpenRouter not accessible: {status}")
                return "OpenRouter API not available. Please check your API key and connection."

            # Use settings from model_settings.json or defaults
            settings = self.settings_manager.load_settings()
            
            # Override with provided values if specified
            k_value = context_k if context_k is not None else 5
            max_tokens_value = max_tokens if max_tokens is not None else int(settings.max_tokens)
            temperature_value = float(settings.temperature) if hasattr(settings, 'temperature') else 0.3
            
            # If model is provided, use it; otherwise use default from settings
            model_to_use = model or settings.openrouter_model
            
            # Validate model exists in available models
            if hasattr(self, 'available_models') and model_to_use not in self.available_models:
                # Get a list of available free models to suggest
                free_models = [m for m in self.available_models if ":free" in m]
                suggestion = f"Try one of these free models: {', '.join(free_models[:3])}" if free_models else "Try a different model"
                logger.warning(f"Selected model '{model_to_use}' may not be available. {suggestion}")
            
            # Reload vector stores to ensure we have latest data
            try:
                logger.info("Reloading vector stores to ensure latest data access")
                self.load_vector_stores()  # Not async in this class
            except Exception as e:
                logger.warning(f"Could not reload vector stores: {e}")

            # Get relevant context with configurable k
            context = await self.get_relevant_context(prompt, k=k_value)
            logger.info(f"Context found: {len(context.split()) if context else 0} words")
            
            # Check if we have meaningful context
            if not context or len(context.strip()) < 50:
                logger.warning("Initial search found no relevant context, trying with k=10")
                # Try a more aggressive search with higher k
                context = await self.get_relevant_context(prompt, k=10)
                logger.info(f"Second attempt context found: {len(context.split()) if context else 0} words")
                
            # If still no context
            if not context or len(context.strip()) < 50:
                logger.warning("No relevant context found in knowledge base")
                
                # Check more carefully if we have any data in either store
                has_reference_data = (self.reference_store is not None and 
                                    hasattr(self.reference_store, 'docstore') and 
                                    len(self.reference_store.docstore._dict) > 0)
                                    
                has_content_data = (self.content_store is not None and 
                                   hasattr(self.content_store, 'docstore') and 
                                   len(self.content_store.docstore._dict) > 0)
                
                logger.info(f"Vector stores status - Reference store: {has_reference_data} ({len(self.reference_store.docstore._dict) if has_reference_data else 0} docs), Content store: {has_content_data} ({len(self.content_store.docstore._dict) if has_content_data else 0} docs)")
                
                # If neither store has data, return a simple message (no API call needed)
                if not has_reference_data and not has_content_data:
                    logger.info("Knowledge base is empty, returning direct message")
                    return "I don't have any data in my knowledge base yet. Please upload some content first."
                
                # If stores exist but no relevant context found, try to broaden the search
                logger.info("Trying a more general search with a simpler query...")
                simplified_query = " ".join(prompt.split()[:3])  # Use first few words
                broader_context = await self.get_relevant_context(simplified_query, k=k_value*2)
                
                if broader_context and len(broader_context.strip()) > 50:
                    logger.info(f"Found broader context with simplified query: {len(broader_context.split())} words")
                    context = broader_context
                else:
                    context = "No specific information found about this topic in my knowledge base."

            # Use system prompt from settings
            system_prompt = settings.system_prompt if hasattr(settings, 'system_prompt') else ""
            
            # Create a more detailed prompt that encourages the model to use the context
            messages = [
                {
                    "role": "system", 
                    "content": f"""
{system_prompt}

CONTEXT INFORMATION:
{context}

INSTRUCTIONS:
1. Use ONLY the information provided in the CONTEXT INFORMATION section above to answer the question.
2. If the context doesn't contain information specific to the question, say "I don't have specific information about that in my knowledge base."
3. Provide a detailed, comprehensive answer using the context provided.
4. Include specific facts, figures, and details from the context when relevant.
5. Format your response in well-structured paragraphs.
6. The response should be at least 100-150 words if the context contains relevant information.
"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            try:
                async with httpx.AsyncClient() as client:
                    logger.info(f"Generating response with OpenRouter model={model_to_use}, k={k_value}, max_tokens={max_tokens_value}, temp={temperature_value}")
                    
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://blob.iaac.net",  # Replace with your site
                        "X-Title": "Blob RAG Assistant"
                    }
                    
                    # Try with increased timeout for more robust handling
                    try:
                        response = await client.post(
                            f"{self.base_url}/chat/completions",
                            json={
                                "model": model_to_use,
                                "messages": messages,
                                "temperature": temperature_value,
                                "max_tokens": max_tokens_value,
                                "top_p": 0.9,
                                "frequency_penalty": 0.0,
                                "presence_penalty": 0.0
                            },
                            headers=headers,
                            timeout=90.0  # Increase timeout for better reliability
                        )
                    except httpx.ReadTimeout:
                        # Handle timeout specifically
                        logger.error(f"Timeout while waiting for OpenRouter response with model {model_to_use}")
                        return "The request timed out. This could be because the selected model is experiencing high load. Please try again later or choose a different model."

                    # Improved error handling with specific feedback
                    if response.status_code != 200:
                        error_message = f"Error from OpenRouter API: {response.status_code}"
                        error_suggestion = ""
                        
                        # Provide specific error messages for common status codes
                        if response.status_code == 404:
                            error_suggestion = f"The model '{model_to_use}' was not found. It may have been removed or renamed."
                        elif response.status_code == 402:
                            error_suggestion = "This model requires payment. Please choose a free model or add credit to your account."
                        elif response.status_code == 429:
                            error_suggestion = "Rate limit exceeded. Please wait a moment and try again."
                        elif response.status_code == 500:
                            error_suggestion = "OpenRouter is experiencing internal server errors. Please try again later."
                        
                        # Try to parse the error response for additional details
                        try:
                            error_details = response.json()
                            if 'error' in error_details:
                                error_message += f", Details: {error_details['error'].get('message', '')}"
                        except:
                            # If we can't parse the response, use the text
                            if response.text:
                                error_message += f", Response: {response.text[:200]}"
                        
                        # Get some free model suggestions
                        free_models_suggestion = ""
                        if hasattr(self, 'available_models'):
                            free_models = [m for m in self.available_models if ":free" in m][:3]
                            if free_models:
                                free_models_suggestion = f" Suggested free models: {', '.join(free_models)}"
                        
                        logger.error(error_message)
                        user_message = f"Sorry, I encountered an error connecting to the language model (Status: {response.status_code}). "
                        if error_suggestion:
                            user_message += f"{error_suggestion} "
                        user_message += f"Please try a different model or try again later.{free_models_suggestion}"
                        
                        return user_message

                    result = response.json()
                    
                    # Check for error in the response (even with 200 status code)
                    if 'error' in result:
                        error_msg = result.get('error', {}).get('message', 'Unknown error')
                        error_code = result.get('error', {}).get('code', 'unknown')
                        logger.error(f"OpenRouter API error: {error_code} - {error_msg}")
                        logger.error(f"Full error response: {result}")
                        
                        # Provide more helpful message about the specific error
                        if error_code == 502:
                            # Get some free model suggestions
                            free_models_suggestion = ""
                            if hasattr(self, 'available_models'):
                                free_models = [m for m in self.available_models if ":free" in m][:3]
                                if free_models:
                                    free_models_suggestion = f"Please try a different model like {', '.join(free_models)}"
                                else:
                                    free_models_suggestion = "Please try a different model"
                            return f"The selected model is currently unavailable. {free_models_suggestion}"
                        return f"Sorry, the model returned an error: {error_msg}. Please try a different model."
                        
                    if not result or 'choices' not in result or not result['choices']:
                        logger.error(f"Invalid response format: {result}")
                        return "Sorry, I received an invalid response format from the model."

                    # Extract text from OpenAI-style response
                    generated_text = result['choices'][0]['message']['content']
                    
                    # Add model info if available
                    if 'model' in result:
                        logger.info(f"Response generated by model: {result['model']}")
                    
                    return generated_text

            except httpx.RequestError as e:
                logger.error(f"Request error: {e}")
                return "Sorry, I couldn't reach the language model."
            except asyncio.TimeoutError:
                logger.error("Request timeout")
                return "Sorry, the request took too long. Please try a simpler question or a different model."
            except Exception as e:
                logger.error(f"API error: {e}")
                return f"Sorry, an error occurred while processing your request: {str(e)}"

        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return f"Sorry, something went wrong with your request: {str(e)}"
            
    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models from OpenRouter with improved error handling and filtering"""
        try:
            if not self.api_key:
                logger.error("Cannot list models: No API key provided")
                return []
                
            async with httpx.AsyncClient() as client:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://blob.iaac.net",  # Replace with your site
                    "X-Title": "Blob RAG Assistant"
                }
                
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=headers
                )
                
                if response.status_code != 200:
                    logger.error(f"Error getting models: {response.status_code}")
                    return []
                    
                data = response.json()
                models = []
                
                # Track whether current configured model is in the list
                settings = self.settings_manager.load_settings()
                configured_model = settings.openrouter_model
                configured_model_found = False
                
                for model in data.get("data", []):
                    model_id = model.get("id")
                    
                    # Check if this is the configured model
                    if model_id == configured_model:
                        configured_model_found = True
                    
                    # Create a structured model object with additional fields
                    model_obj = {
                        "id": model_id,
                        "name": model.get("name"),
                        "description": model.get("description", ""),
                        "context_length": model.get("context_length", 0),
                        "is_current": model_id == configured_model,  # Indicate if this is our current model
                        "pricing": model.get("pricing", {}),
                        "is_free": ":free" in model_id,
                        # Extract provider and model from ID (e.g., "anthropic/claude-3-haiku" → "anthropic", "claude-3-haiku")
                        "provider": model_id.split("/")[0] if "/" in model_id else "",
                        "model_type": model_id.split("/")[1] if "/" in model_id else model_id
                    }
                    models.append(model_obj)
                    
                # Sort models to put free models first and current model at the top
                models.sort(key=lambda m: (not m["is_current"], not m["is_free"]))
                
                # Set available models for later use
                self.available_models = [model["id"] for model in models]
                
                # Log warning if configured model not found
                if not configured_model_found:
                    logger.warning(f"Currently configured model '{configured_model}' not found in available models list!")
                    
                return models
                
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []