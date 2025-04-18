from pathlib import Path
import httpx
import logging
from typing import Dict, Any, Tuple, List
import json
import asyncio
from scripts.services.settings_manager import SettingsManager
from scripts.models.settings import ModelSettings
from langchain_community.vectorstores import FAISS  # Updated import
from .ollama_embeddings import OllamaEmbeddings
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.settings_manager = SettingsManager()
        self.settings = self.settings_manager.load_settings()
        
        # Use the model name from settings instead of hardcoding it
        self.model = self.settings.model_name
        logger.info(f"Initializing Ollama client with {self.model} from settings")
        
        # Resource processing model is separate in resource_llm.py (set to mistral:7b-instruct-v0.2-q4_0)
        self.embeddings = OllamaEmbeddings()
        self.reference_store = None
        self.content_store = None
        self.load_vector_stores()
        
        # Initialize text splitter with consistent parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            separators=[
                "\n\n",          # First try to split on double newlines
                "\n",            # Then single newlines
                ". ",            # Then sentences
                ", ",            # Then clauses
                " ",             # Then words
                ""               # Finally characters
            ],
            length_function=len,
        )

    def get_config_path(self):
        """Get the path to the config file in the project directory"""
        # First try to use config.json in the project root
        local_config = Path("config.json")
        if (local_config.exists()):
            return local_config
        
        # If it doesn't exist, try the user's home directory as fallback
        home_config = Path.home() / '.blob' / 'config.json'
        if (home_config.exists()):
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
            config_path = self.get_config_path()
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            data_path = Path(config.get('data_path', ''))
            vector_path = data_path / "vector_stores" / "default"
            logger.info(f"Looking for vector stores in: {vector_path}")

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
                
            # Initialize topic stores dictionary
            self.topic_stores = {}
            
            # Only search for topic stores if the vector path exists
            if vector_path.exists():
                # Get a list of all potential topic stores
                topic_dirs = list(vector_path.glob("topic_*"))
                
                if topic_dirs:
                    logger.info(f"Found {len(topic_dirs)} potential topic stores, checking each...")
                    
                    # Track loading statistics
                    loaded_count = 0
                    failed_count = 0
                    
                    # Try to load each topic store that has an index.faiss file
                    for topic_dir in topic_dirs:
                        if topic_dir.is_dir() and (topic_dir / "index.faiss").exists():
                            try:
                                topic_name = topic_dir.name
                                # Load the store
                                self.topic_stores[topic_name] = FAISS.load_local(
                                    str(topic_dir),
                                    self.embeddings,
                                    allow_dangerous_deserialization=True
                                )
                                
                                # Verify it loaded correctly and has documents
                                if hasattr(self.topic_stores[topic_name], 'docstore'):
                                    doc_count = len(self.topic_stores[topic_name].docstore._dict) 
                                    if doc_count > 0:
                                        logger.info(f"Successfully loaded topic store '{topic_name}' with {doc_count} documents")
                                        loaded_count += 1
                                    else:
                                        logger.warning(f"Topic store '{topic_name}' loaded but contains 0 documents")
                                        # Keep it loaded anyway in case it's valid but empty
                                        loaded_count += 1
                                else:
                                    logger.warning(f"Topic store '{topic_name}' loaded but has no docstore attribute")
                                    # Remove it since it's not a valid store
                                    del self.topic_stores[topic_name]
                                    failed_count += 1
                            except Exception as e:
                                logger.error(f"Error loading topic store '{topic_dir.name}': {e}")
                                # Make sure it's not in the dictionary if loading failed
                                if topic_dir.name in self.topic_stores:
                                    del self.topic_stores[topic_dir.name]
                                failed_count += 1
                    
                    # Log summary
                    logger.info(f"Topic store loading complete: {loaded_count} loaded successfully, {failed_count} failed")
                else:
                    logger.info("No topic_* directories found")
            else:
                logger.warning(f"Vector store path not found: {vector_path}")
                
            # Log info about successfully loaded stores
            logger.info(f"Vector stores loaded: reference_store: {'Yes' if self.reference_store else 'No'}, " 
                        f"content_store: {'Yes' if self.content_store else 'No'}, "
                        f"topic_stores: {len(self.topic_stores)}")

        except Exception as e:
            logger.error(f"Error loading vector stores: {e}")
            # Initialize empty topic_stores if something went wrong
            self.topic_stores = {}

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
            
            # Manually calculate scores
            # First, get the vector from the query
            faiss_index = store.index
            
            # Get the raw docstore to extract IDs directly
            docstore = store.docstore._dict if hasattr(store, 'docstore') else {}
            
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
                
            # Topic stores search
            if self.topic_stores and len(self.topic_stores) > 0:
                logger.info(f"Searching {len(self.topic_stores)} topic stores")
                
                # Search each topic store
                topic_results = []
                for store_name, store in self.topic_stores.items():
                    if store is not None:
                        try:
                            # Search this topic store
                            store_results = await self._hybrid_search(
                                store,
                                query,
                                query_embedding[0],
                                k=max(1, k//2),  # Use fewer results per topic store
                                alpha=0.6  # Balanced weight for topic stores
                            )
                            
                            # Add to consolidated results
                            if store_results:
                                logger.info(f"Found {len(store_results)} relevant items in topic store '{store_name}'")
                                for doc, score in store_results:
                                    # Add store name to metadata for organization
                                    doc.metadata['topic_store'] = store_name
                                    # Store this as a tuple with score for sorting later
                                    topic_results.append((doc, score, store_name))
                        except Exception as e:
                            logger.error(f"Error searching topic store '{store_name}': {e}")
                
                # Sort all topic results by score and pick top k
                if topic_results:
                    topic_results.sort(key=lambda x: x[1], reverse=True)
                    top_topic_results = topic_results[:k]
                    
                    # Add to context
                    context_parts.append("\nTopic-specific information:")
                    
                    for i, (doc, score, store_name) in enumerate(top_topic_results):
                        meta = doc.metadata
                        topic_name = store_name.replace('topic_', '').replace('_', ' ').title()
                        title = meta.get('title', meta.get('topic', topic_name))
                        
                        # Log what we found
                        logger.info(f"Topic #{i+1}: '{title}' from {store_name} - Score: {score:.4f}")
                        
                        # Add to context with topic information
                        context_parts.append(f"From topic '{title}': {doc.page_content}")
                        
                    logger.info(f"Added {len(top_topic_results)} items from topic stores to context")
            else:
                logger.info("No topic stores available for search")

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

    async def get_relevant_context_from_store(self, query: str, store_name: str, k: int = 3) -> str:
        """Get relevant context from a specific store using hybrid search"""
        context_parts = []
        
        try:
            # Validate store exists before attempting to use it
            if store_name == "reference_store":
                if self.reference_store is None:
                    logger.warning("Reference store is not available")
                    return ""
                store = self.reference_store
            elif store_name == "content_store":
                if self.content_store is None:
                    logger.warning("Content store is not available")
                    return ""
                store = self.content_store
            elif store_name in self.topic_stores:
                store = self.topic_stores[store_name]
                if store is None:
                    logger.warning(f"Topic store '{store_name}' exists but is not properly loaded")
                    return ""
            else:
                logger.warning(f"Store '{store_name}' not found (available topic stores: {', '.join(self.topic_stores.keys()) if self.topic_stores else 'none'})")
                return ""
            
            # Get query embedding
            query_embedding = await self.embeddings.aembeddings([query])
            
            # Use hybrid search for this store
            results = await self._hybrid_search(
                store,
                query,
                query_embedding[0],
                k=k,
                alpha=0.6  # Balanced weight between vector and term matching
            )
            
            if results:
                logger.info(f"Found {len(results)} relevant items in {store_name}")
                
                # Format results based on store type
                is_topic_store = store_name.startswith("topic_")
                header = f"Information from {store_name}:" if is_topic_store else ""
                if header:
                    context_parts.append(header)
                
                # Add documents to context
                for i, (doc, score) in enumerate(results):
                    meta = doc.metadata
                    title = meta.get('title', meta.get('material_title', 'Untitled'))
                    
                    # For topic stores, format with more detail
                    if is_topic_store:
                        context_parts.append(f"Topic: {title} (Relevance: {score:.2f})")
                        context_parts.append(f"{doc.page_content}")
                        context_parts.append("---")
                    else:
                        # Standard formatting for regular stores
                        source_type = meta.get('source_type', 'document')
                        context_parts.append(f"From {source_type} '{title}': {doc.page_content}")
            else:
                logger.info(f"No relevant results found in {store_name}")
            
            # Combine context parts
            context = "\n".join(context_parts)
            return context
            
        except Exception as e:
            logger.error(f"Error getting context from {store_name}: {e}")
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
                        
                    logger.info(f"Found {len(references)} relevant references from reference_store")
                    
                except Exception as e:
                    logger.error(f"Error searching references: {e}")
            
            # Add results from topic stores
            if self.topic_stores and len(self.topic_stores) > 0:
                logger.info(f"Searching {len(self.topic_stores)} topic stores for references")
                
                # Search each topic store
                for store_name, store in self.topic_stores.items():
                    if store is not None:
                        try:
                            # Search this topic store (fewer results per store)
                            store_results = await self._hybrid_search(
                                store,
                                query,
                                query_embedding[0],
                                k=max(1, k//2),  # Use fewer results per topic store
                                alpha=0.6  # Balanced weight for topic stores
                            )
                            
                            # Process results from this store
                            if store_results:
                                topic_name = store_name.replace('topic_', '').replace('_', ' ').title()
                                logger.info(f"Found {len(store_results)} relevant items in topic store '{topic_name}'")
                                
                                for doc, score in store_results:
                                    meta = doc.metadata
                                    vector_score = meta.get('vector_score', 0)
                                    term_overlap = meta.get('term_overlap', 0)
                                    
                                    # Create reference object with topic store information
                                    reference = {
                                        "title": meta.get("title", topic_name),
                                        "content": doc.page_content,
                                        "type": "topic_reference",
                                        "topic": topic_name,
                                        "topic_store": store_name,
                                        "tags": meta.get("tags", []),
                                        "relevance_score": float(score),
                                        "vector_score": float(vector_score),
                                        "term_overlap": term_overlap,
                                        "source": meta.get("source", "topic store"),
                                        "created_at": meta.get("created_at", ""),
                                    }
                                    references.append(reference)
                        except Exception as e:
                            logger.error(f"Error searching topic store '{store_name}' for references: {e}")
                
                # Sort all references by relevance score and take top k overall
                if references:
                    references.sort(key=lambda x: x["relevance_score"], reverse=True)
                    references = references[:k*2]  # Allow more results when including topic stores
                    logger.info(f"Combined references from all stores: {len(references)} items")
            
            return references

        except Exception as e:
            logger.error(f"Error getting references: {e}", exc_info=True)
            return []

    async def check_connection(self) -> Tuple[bool, str]:
        """Check if Ollama is running and model is loaded"""
        try:
            # Refresh settings to ensure we're checking for the current model
            self.settings = self.settings_manager.load_settings()
            self.model = self.settings.model_name
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                logger.info(f"Checking Ollama connection for model {self.model}...")
                response = await client.get(f"{self.base_url}/api/tags")
                models_data = response.json()
                available_models = [m.get("name") for m in models_data.get("models", [])]
                
                # Check for our chat model
                if self.model not in available_models:
                    return False, f"{self.model} model not available"
                
                # Check for resource processing model as well
                if "mistral:7b-instruct-v0.2-q4_0" not in available_models:
                    logger.warning("Resource processing model (mistral:7b-instruct-v0.2-q4_0) not available! Resource processing may fail.")
                
                logger.info(f"Using {self.model} for chat interactions and mistral:7b-instruct-v0.2-q4_0 for resource processing")
                return True, f"Connected to Ollama. Using {self.model} for chat"

        except Exception as e:
            logger.error(f"Connection check failed: {str(e)}")
            return False, str(e)

    async def generate_response(self, prompt: str, context_k: int = None, max_tokens: int = None) -> str:
        """Generate a response using RAG"""
        try:
            # Refresh settings to ensure we're using the latest model configuration
            self.settings = self.settings_manager.load_settings()
            self.model = self.settings.model_name  # Update model to latest setting in case it changed since client initialization
            
            # Verify Ollama is ready
            ready, status = await self.check_connection()
            if not ready:
                logger.error(f"Ollama not ready: {status}")
                return f"Model not available. Please check Ollama installation. Details: {status}"

            # Use settings from model_settings.json or defaults
            settings = self.settings_manager.load_settings()
            # Values from settings take precedence over hardcoded defaults
            k_value = context_k if context_k is not None else 5
            max_tokens_value = max_tokens if max_tokens is not None else int(settings.max_tokens)
            temperature_value = float(settings.temperature) if hasattr(settings, 'temperature') else 0.3
            
            # Get relevant context with configurable k
            logger.info(f"Getting context for prompt (k={k_value})...")
            try:
                context = await self.get_relevant_context(prompt, k=k_value)
                logger.info(f"Context found: {len(context.split()) if context else 0} words")
            except Exception as context_err:
                logger.error(f"Error getting context: {context_err}")
                context = "Error retrieving context. Proceeding with general knowledge."
            
            # Check if we have meaningful context
            if not context or len(context.strip()) < 50:
                logger.warning("Initial search found no relevant context, trying with k=10")
                # Try a more aggressive search with higher k
                try:
                    context = await self.get_relevant_context(prompt, k=10)
                    logger.info(f"Second attempt context found: {len(context.split()) if context else 0} words")
                except Exception as retry_err:
                    logger.error(f"Error in second context attempt: {retry_err}")
                
            # If still no context
            if not context or len(context.strip()) < 50:
                logger.warning("No relevant context found in knowledge base")
                
                # Default context if everything fails
                context = "No specific information found about this topic in my knowledge base."

            # Use system prompt from settings
            system_prompt = settings.system_prompt if hasattr(settings, 'system_prompt') else ""
            
            # Create a more detailed prompt that encourages the model to use the context
            system_context = f"""{system_prompt}

CONTEXT INFORMATION:
{context}

INSTRUCTIONS:
1. Use ONLY the information provided in the CONTEXT INFORMATION section above to answer the question.
2. If the context doesn't contain information specific to the question, say "I don't have specific information about that in my knowledge base."
3. Provide a detailed, comprehensive answer using the context provided.
4. Include specific facts, figures, and details from the context when relevant.
5. Format your response in well-structured paragraphs.
6. The response should be at least 100-150 words if the context contains relevant information.
USER QUESTION: {prompt}

DETAILED ANSWER:
"""

            try:
                logger.info("Context preparation complete, now generating response from model...")
                
                # For troubleshooting: log if Ollama is reachable
                if not await self.check_health():
                    logger.error("Ollama health check failed before making request!")
                    return "Error: Ollama API is not responding. Please check if Ollama is running."
                
                async with httpx.AsyncClient() as client:
                    logger.info(f"Generating response with: k={k_value}, max_tokens={max_tokens_value}, temp={temperature_value}")
                    # Use a slightly higher temperature for better elaboration
                    response_temperature = min(0.5, temperature_value + 0.1)
                    
                    # Send request with increased timeout to ensure we get the full response
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": system_context.strip(),
                            "stream": False,
                            "temperature": response_temperature,
                            "max_tokens": max_tokens_value,
                            "top_p": 0.9,
                            "repeat_penalty": 1.1
                        },
                        timeout=90.0  # Increased timeout for longer, more detailed responses
                    )
                    
                    if response.status_code != 200:
                        error_message = f"Error from Ollama API: {response.status_code}"
                        if response.text:
                            try:
                                error_details = response.json()
                                error_message += f", Details: {error_details}"
                            except:
                                error_message += f", Response: {response.text[:200]}"
                        
                        logger.error(error_message)
                        return f"Sorry, I encountered an error connecting to the language model. {error_message}"
                    
                    result = response.json()
                    if not result or 'response' not in result:
                        logger.error("Invalid response format")
                        return "Sorry, I received an invalid response format from the language model."

                    # Return the generated response
                    generated_text = result['response']
                    logger.info(f"Response generation complete: {len(generated_text.split())} words")
                    return generated_text

            except httpx.RequestError as e:
                error_msg = f"Request error: {e}"
                logger.error(error_msg)
                return f"Sorry, I couldn't reach the language model. {error_msg}"
            except asyncio.TimeoutError:
                error_msg = "Request timeout after 90 seconds"
                logger.error(error_msg)
                return f"Sorry, the request took too long. Please try a simpler question. {error_msg}"
            except Exception as e:
                error_msg = f"API error: {e}"
                logger.error(error_msg)
                return f"Sorry, an error occurred while processing your request. {error_msg}"

        except Exception as e:
            error_msg = f"Error in generate_response: {e}"
            logger.error(error_msg)
            return f"Sorry, something went wrong with your request. {error_msg}"

    async def check_health(self) -> bool:
        """Check if Ollama is responding"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def check_system_info(self) -> Dict[str, Any]:
        """Get system resource usage from Ollama"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/show")
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {}
            
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Send a chat message to Ollama API and return the response
        
        Args:
            messages: List of message objects with 'role' and 'content'
                     e.g. [{"role": "user", "content": "Hello"}]
        
        Returns:
            The text response from the model
        """
        try:
            # Refresh settings to ensure we're using the latest model
            self.settings = self.settings_manager.load_settings()
            self.model = self.settings.model_name
            
            logger.info(f"Sending chat request to Ollama with {len(messages)} messages using model {self.model}")
            
            # For synchronous operation, create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def _send_chat():
                try:
                    async with httpx.AsyncClient() as client:
                        # Format for Ollama API
                        prompt = ""
                        for msg in messages:
                            if msg["role"] == "user":
                                prompt += f"{msg['content']}\n"
                            
                        response = await client.post(
                            f"{self.base_url}/api/generate",
                            json={
                                "model": self.model,
                                "prompt": prompt.strip(),
                                "stream": False,
                                "temperature": 0.7,
                            },
                            timeout=60.0
                        )
                        
                        if response.status_code != 200:
                            logger.error(f"Error from Ollama API: {response.status_code}")
                            return "Error: Failed to get response from Ollama"
                        result = response.json()
                        return result.get('response', '')
                except Exception as e:
                    logger.error(f"Error in _send_chat: {e}")
                    return f"Error: {str(e)}"
            
            # Run the async function in the new event loop
            response = loop.run_until_complete(_send_chat())
            loop.close()
            
            return response        
            
        except Exception as e:
            logger.error(f"Error in chat method: {e}")
            return f"Error: {str(e)}"
    
    def process_text(self, text: str) -> str:
        """
        Process input text into chunks using consistent chunking parameters
        """
        logger.info(f"Processing text: {text}")
        
        # Split the text into chunks
        text_chunks = self.text_splitter.split_text(text)
        logger.info(f"Text processed into {len(text_chunks)} chunks")
        processed_text = " ".join(text_chunks)
        return processed_text
