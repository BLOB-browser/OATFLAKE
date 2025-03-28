from typing import List, Dict, Optional, Any, Tuple
import logging
from pathlib import Path
import asyncio
import json
import numpy as np
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from .ollama_embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)

class RAGHandler:
    def __init__(self, docs_path: str = None, group_id: str = "default"):
        """Initialize the RAG Handler with configurable path and group"""
        # Set up paths
        if docs_path is None:
            mdef_path = Path(__file__).parent.parent.parent  # Go up to project root
            self.docs_path = mdef_path / "data" / "docs"
        else:
            self.docs_path = Path(docs_path)
            
        self.group_id = group_id
        self.embeddings = OllamaEmbeddings(model_name="nomic-embed-text")
        self.reference_store = None
        self.content_store = None
        self.vector_store_path = None
        
        # Load config
        self._load_config()
        
        # Load vector stores
        self._load_vector_stores()
        
    def _load_config(self):
        """Load configuration from config.json"""
        try:
            config_path = Path("config.json")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                data_path = Path(config.get('data_path', ''))
                self.vector_store_path = data_path / "vector_stores" / self.group_id
                logger.info(f"Using vector store path: {self.vector_store_path}")
            else:
                logger.warning("config.json not found, using default paths")
                self.vector_store_path = self.docs_path.parent / "vector_stores" / self.group_id
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.vector_store_path = self.docs_path.parent / "vector_stores" / self.group_id
            
    def _load_vector_stores(self):
        """Load both vector stores if they exist"""
        try:
            # Load reference store
            reference_path = self.vector_store_path / "reference_store"
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
            content_path = self.vector_store_path / "content_store"
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
            
    async def _hybrid_search(self, store, query: str, k: int = 5, alpha: float = 0.5) -> List[Tuple[Any, float]]:
        """Perform hybrid search combining sparse + dense retrieval"""
        if not store:
            return []
            
        try:
            # Get query embedding
            query_embedding = await self.embeddings.aembeddings([query])
            embedding_vector = query_embedding[0]
            
            # Extract key terms for relevance checking
            query_terms = set(term.lower() for term in query.split() if len(term) > 3)
            
            # Perform dense vector search
            vector_results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: store.similarity_search_by_vector_with_relevance_scores(
                    embedding_vector,
                    k=k * 2  # Get more candidates for reranking
                )
            )
            
            # Get results and prepare for reranking
            results = []
            for doc, score in vector_results:
                # Calculate term overlap for BM25-like scoring
                doc_terms = set(term.lower() for term in doc.page_content.split() if len(term) > 3)
                term_overlap = len(query_terms.intersection(doc_terms))
                term_overlap_score = term_overlap / max(len(query_terms), 1)
                
                # Calculate hybrid score (combine dense + sparse signals)
                hybrid_score = (alpha * score) + ((1 - alpha) * term_overlap_score)
                
                # Add metadata to track relevance factors
                doc.metadata["vector_score"] = score
                doc.metadata["term_overlap"] = term_overlap
                doc.metadata["hybrid_score"] = hybrid_score
                
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
            search_start = datetime.now()
            
            # Reference store search with hybrid approach
            if self.reference_store is not None:
                try:
                    # Use hybrid search for better relevance
                    ref_results = await self._hybrid_search(
                        self.reference_store,
                        query,
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
                            logger.info(f"Reference #{i+1}: {source} - Score: {score:.4f}")
                            
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
                            logger.info(f"Content #{i+1}: '{title}' [{source_type}] - Score: {score:.4f} (vector: {vector_score:.4f}, term overlap: {term_overlap})")
                            
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
            logger.error(f"Error getting context: {e}")
            return ""
            
    async def get_relevant_references(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get relevant references with metadata using hybrid search"""
        try:
            references = []
            
            if self.reference_store is not None:
                try:
                    # Use hybrid search for better relevance
                    ref_results = await self._hybrid_search(
                        self.reference_store,
                        query,
                        k=k,
                        alpha=0.7  # Weight vector search more heavily for references
                    )
                    
                    # Process each reference with scores
                    for doc, score in ref_results:
                        meta = doc.metadata
                        reference = {
                            "title": meta.get("title", "Untitled"),
                            "content": doc.page_content,
                            "type": meta.get("source_type", "unknown"),
                            "tags": meta.get("tags", []),
                            "relevance_score": float(score),  # Convert to float for JSON serialization
                            "source": meta.get("source", ""),
                            "created_at": meta.get("created_at", ""),
                        }
                        references.append(reference)
                        
                    logger.info(f"Found {len(references)} relevant references")
                    
                except Exception as e:
                    logger.error(f"Error searching references: {e}")
            
            return references

        except Exception as e:
            logger.error(f"Error getting references: {e}")
            return []
            
    async def run_query(self, query: str, k: int = 5, include_metadata: bool = True) -> Dict:
        """Run a query and return structured results with full metadata"""
        try:
            search_start = datetime.now()
            results = {
                "reference_results": [],
                "content_results": [],
                "timing": {},
                "query_metadata": {
                    "query": query,
                    "k": k,
                    "query_time": datetime.now().isoformat()
                }
            }
            
            # Get query embedding for both searches
            embedding_start = datetime.now()
            query_embedding = await self.embeddings.aembeddings([query])
            embedding_vector = query_embedding[0]
            embedding_time = (datetime.now() - embedding_start).total_seconds()
            results["timing"]["embedding_generation"] = embedding_time
            
            # Reference store search
            if self.reference_store is not None:
                ref_start = datetime.now()
                try:
                    ref_results = await self._hybrid_search(
                        self.reference_store,
                        query,
                        k=k,
                        alpha=0.7
                    )
                    
                    for doc, score in ref_results:
                        ref_item = {
                            "text": doc.page_content,
                            "score": float(score)
                        }
                        if include_metadata:
                            ref_item["metadata"] = doc.metadata
                            
                        results["reference_results"].append(ref_item)
                except Exception as e:
                    logger.error(f"Error in reference search: {e}")
                    
                ref_time = (datetime.now() - ref_start).total_seconds()
                results["timing"]["reference_search"] = ref_time
                
            # Content store search
            if self.content_store is not None:
                content_start = datetime.now()
                try:
                    content_results = await self._hybrid_search(
                        self.content_store,
                        query,
                        k=k,
                        alpha=0.5
                    )
                    
                    for doc, score in content_results:
                        content_item = {
                            "text": doc.page_content,
                            "score": float(score)
                        }
                        if include_metadata:
                            content_item["metadata"] = doc.metadata
                            
                        results["content_results"].append(content_item)
                except Exception as e:
                    logger.error(f"Error in content search: {e}")
                    
                content_time = (datetime.now() - content_start).total_seconds()
                results["timing"]["content_search"] = content_time
            
            # Total search time
            total_search_time = (datetime.now() - search_start).total_seconds()
            results["timing"]["total_search_time"] = total_search_time
            
            logger.info(f"Completed search in {total_search_time:.2f}s: {len(results['reference_results'])} references, {len(results['content_results'])} content items")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in run_query: {e}")
            return {
                "reference_results": [],
                "content_results": [],
                "error": str(e),
                "query": query
            }