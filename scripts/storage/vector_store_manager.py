from pathlib import Path
import logging
from typing import Dict, List, Optional, Union  # Add Union
import json
from datetime import datetime
import os
import numpy as np
import faiss  # Add this import
from collections import Counter, defaultdict

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.indexes import VectorstoreIndexCreator
from ..llm.ollama_embeddings import OllamaEmbeddings

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        # Use the correct folder name "vector_stores" instead of "faiss"
        self.vectors_path = self.base_path / "vector_stores"
        self.index_path = self.vectors_path / "index.json"
        
        logger.info(f"Vector store initialized at: {self.vectors_path}")
        
        # Use Ollama embeddings with increased timeout for Raspberry Pi
        self.embeddings = OllamaEmbeddings(
            model_name="nomic-embed-text",
            batch_size=5,  # Reduced batch size for Raspberry Pi
            timeout=120.0  # Increased timeout for Raspberry Pi
        )
        
        # Ensure vector store directory exists
        self.vectors_path.mkdir(parents=True, exist_ok=True)
        
        # Load or create index
        self.index = self._load_index()
        
        logger.info(f"Vector store initialized at: {self.vectors_path}")

    def _get_store_path(self, store_name: str, group_id: str = None) -> Path:
        """Get path for a vector store"""
        # Use the provided group_id or use the folder named 'default'
        group_to_use = group_id if group_id else "default"
        logger.info(f"Using vector store path: {self.vectors_path}/{group_to_use}/{store_name}")
        return self.vectors_path / group_to_use / store_name

    def _load_index(self) -> Dict:
        """Load or create navigation index"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {
            "stores": {},
            "metadata": {
                "last_updated": None,
                "total_documents": 0
            }
        }

    def _save_index(self) -> None:
        """Save navigation index"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    async def create_or_update_store(self, 
        store_name: str, 
        documents: List[Document],
        metadata: Dict = None,
        group_id: str = "default",  # Keep parameter for backward compatibility
        update_stats: bool = True
    ) -> bool:
        """
        Create or update a vector store
        
        Note: The group_id parameter is maintained for backward compatibility
        but is no longer used as the backend only works with one group
        """
        try:
            # Get the store path
            store_path = self._get_store_path(store_name)
            store_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize metadata if None
            if metadata is None:
                metadata = {}
            
            # Get embeddings for all documents
            if not documents:
                logger.warning(f"No documents provided to create store {store_name}")
                return False
            
            # Check if documents are already chunked (from materials processing)
            already_chunked = any("chunk_index" in doc.metadata for doc in documents)
            
            if already_chunked:
                # Documents are already chunked, use them directly                logger.info("Documents appear to be pre-chunked, using existing chunks")
                chunks = documents
                chunk_count = len(chunks)
            else:
                # Split documents into chunks first with optimized settings for performance
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,    # Optimized chunk size for better performance
                    chunk_overlap=200,  # Optimized overlap for better context preservation
                    separators=["\n\n", "\n", ". ", ", ", " ", ""]
                )
                chunks = text_splitter.split_documents(documents)
                chunk_count = len(chunks)
            
            # Extract text from chunks for embedding
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            
            # Log detailed document info
            text_lengths = [len(t) for t in texts]
            min_len = min(text_lengths) if texts else 0
            max_len = max(text_lengths) if texts else 0
            avg_len = sum(text_lengths) / len(text_lengths) if texts else 0
            
            logger.info(f"Creating new vector store '{store_name}':")
            logger.info(f"  - Original documents: {len(documents)}")
            logger.info(f"  - Split into chunks: {chunk_count}")
            logger.info(f"  - Document length stats: min={min_len}, max={max_len}, avg={avg_len:.1f} chars")
            logger.info(f"  - Total text volume: {sum(text_lengths):,} characters")
            
            # Process with detailed timing
            start_time = datetime.now()
            logger.info(f"Embedding process started at {start_time.isoformat()}")
            
            # Generate embeddings
            embeddings = await self.embeddings.aembed_documents(texts)
            embedding_count = len(embeddings)
            
            # Log completion time and performance
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Embedding completed in {duration:.1f} seconds")
            logger.info(f"Performance: {duration/len(texts):.2f} seconds per chunk, {sum(text_lengths)/duration:.1f} chars/second")
            
            # Check if we got valid embeddings
            if not embeddings or len(embeddings) == 0:
                logger.error(f"Failed to generate embeddings for documents in store {store_name}")
                return False
                
            # Log embedding stats
            dimension = len(embeddings[0])
            logger.info(f"Generated {embedding_count} embeddings from {chunk_count} chunks")
            
            # Create FAISS index
            import faiss
            index = faiss.IndexFlatL2(dimension)
            
            # Add vectors to index
            index.add(np.array(embeddings, dtype=np.float32))
            
            # Save FAISS index
            faiss.write_index(index, str(store_path / "index.faiss"))
            
            # We decided not to use pkl files or symlinks, using only index.faiss
            
            # Save documents metadata - now based on chunks not original documents
            with open(store_path / "documents.json", "w") as f:
                json.dump([{
                    "content": chunks[i].page_content,
                    "metadata": chunks[i].metadata,
                    "embedding_id": i  # Add embedding ID for reference
                } for i in range(len(chunks)) if i < len(embeddings)], f, indent=2)
            
            # Update embedding stats if requested
            if update_stats:
                # Create embedding stats file with separate counts for chunks and embeddings
                embedding_stats = {
                    "embedding_count": embedding_count,
                    "document_count": len(documents),
                    "chunk_count": chunk_count,  # Count actual chunks separately from embeddings
                    "dimension": dimension,
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "model": self.embeddings.model_name
                }
                
                # Save embedding stats
                with open(store_path / "embedding_stats.json", "w") as f:
                    json.dump(embedding_stats, f, indent=2)
                    
                logger.info(f"Updated embedding stats for {store_name}: {embedding_count} embeddings from {chunk_count} chunks")
            
            # Update index
            if "default" not in self.index["stores"]:
                self.index["stores"]["default"] = {}
                
            self.index["stores"]["default"][store_name] = {
                "path": str(store_path),
                "document_count": len(documents),
                "last_updated": datetime.now().isoformat(),
                "metadata": metadata or {},
                "dimension": dimension
            }
            
            self._save_index()
            logger.info(f"Vector store updated: {store_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating/updating vector store {store_name}: {e}")
            return False
            
    async def add_documents_to_store(self,
        store_name: str,
        documents: List[Document],
        metadata: Dict = None,
        update_stats: bool = True
    ) -> bool:
        """WARNING: This method might silently fail if there are no documents to add or embeddings fail!"""
        """
        Add documents to an existing vector store (if it exists) or create a new one
        This method preserves existing documents and only adds new ones
        
        Args:
            store_name: Name of the store to add documents to
            documents: List of documents to add
            metadata: Metadata to update in the store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the store path
            store_path = self._get_store_path(store_name)
            
            # Initialize metadata if None
            if metadata is None:
                metadata = {}
                
            # Check if store exists
            if (store_path / "index.faiss").exists() and (store_path / "documents.json").exists():
                logger.info(f"Adding {len(documents)} documents to existing store: {store_name}")
                
                # Load existing documents and index
                import faiss
                existing_index = faiss.read_index(str(store_path / "index.faiss"))
                
                with open(store_path / "documents.json", "r") as f:
                    existing_documents = json.load(f)
                
                # Get existing count for embedding IDs
                existing_count = len(existing_documents)
                
                # Get embeddings for new documents
                if not documents:
                    logger.warning(f"No documents provided to add to store {store_name}")
                    return False
                    
                texts = [doc.page_content for doc in documents]
                
                # Log detailed document info
                text_lengths = [len(t) for t in texts]
                min_len = min(text_lengths)
                max_len = max(text_lengths)
                avg_len = sum(text_lengths) / len(text_lengths)
                
                logger.info(f"Starting embeddings generation for {len(texts)} documents:")
                logger.info(f"  - Store: {store_name}")
                logger.info(f"  - Document length stats: min={min_len}, max={max_len}, avg={avg_len:.1f} chars")
                logger.info(f"  - Total text volume: {sum(text_lengths):,} characters")
                
                # Process with detailed timing
                start_time = datetime.now()
                logger.info(f"Embedding process started at {start_time.isoformat()}")
                
                # Generate embeddings
                new_embeddings = await self.embeddings.aembed_documents(texts)
                
                # Log completion time and performance
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                logger.info(f"Embedding completed in {duration:.1f} seconds")
                logger.info(f"Performance: {duration/len(texts):.2f} seconds per document, {sum(text_lengths)/duration:.1f} chars/second")
                
                # Check if documents are already chunked (from materials processing)
                already_chunked = any("chunk_index" in doc.metadata for doc in documents)
                
                if already_chunked:
                    # Documents are already chunked, use them directly
                    logger.info("Documents appear to be pre-chunked, using existing chunks")
                    chunks = documents
                    chunk_count = len(chunks)                
                else:
                    # Split documents into chunks first with optimized settings for performance
                    from langchain.text_splitter import RecursiveCharacterTextSplitter
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,    # Optimized chunk size for better performance
                        chunk_overlap=200,  # Optimized overlap for better context preservation
                        separators=["\n\n", "\n", ". ", ", ", " ", ""]
                    )
                    chunks = text_splitter.split_documents(documents)
                    chunk_count = len(chunks)
                
                # Extract text from chunks for embedding
                texts = [chunk.page_content for chunk in chunks]
                metadatas = [chunk.metadata for chunk in chunks]
                
                logger.info(f"Adding to store '{store_name}':")
                logger.info(f"  - Original documents: {len(documents)}")
                logger.info(f"  - Split into chunks: {chunk_count}")
                
                # Generate embeddings
                new_embeddings = await self.embeddings.aembed_documents(texts)
                embedding_count = len(new_embeddings)
                
                # Check if we got valid embeddings
                if not new_embeddings or len(new_embeddings) == 0:
                    logger.error(f"Failed to generate embeddings for documents in store {store_name}")
                    return False
                    
                # Log embedding stats
                logger.info(f"Generated {embedding_count} embeddings from {chunk_count} chunks")
                
                # Check if dimensions match
                if len(new_embeddings[0]) != existing_index.d:
                    logger.error(f"Dimension mismatch: existing={existing_index.d}, new={len(new_embeddings[0])}")
                    return False
                
                # Add new vectors to index
                existing_index.add(np.array(new_embeddings, dtype=np.float32))
                
                # Save updated index
                faiss.write_index(existing_index, str(store_path / "index.faiss"))
                
                # We decided not to use pkl files or symlinks, using only index.faiss
                
                # Combine document metadata and save - use chunks instead of original documents
                combined_documents = existing_documents + [{
                    "content": chunks[i].page_content,
                    "metadata": chunks[i].metadata,
                    "embedding_id": i + existing_count  # Continue numbering from existing
                } for i in range(len(chunks)) if i < len(new_embeddings)]
                
                with open(store_path / "documents.json", "w") as f:
                    json.dump(combined_documents, f, indent=2)
                
                # Update embedding stats if requested
                if update_stats:
                    # Check for existing stats
                    embedding_stats = {}
                    if (store_path / "embedding_stats.json").exists():
                        try:
                            with open(store_path / "embedding_stats.json", "r") as f:
                                embedding_stats = json.load(f)
                        except Exception as e:
                            logger.warning(f"Could not read existing embedding stats: {e}")
                    
                    # Update the stats with new counts
                    current_count = embedding_stats.get("embedding_count", 0)
                    current_docs = embedding_stats.get("document_count", 0)
                    current_chunks = embedding_stats.get("chunk_count", 0)
                    
                    # Track chunks and embeddings separately
                    new_chunks = chunk_count
                    new_embedding_count = embedding_count
                    
                    embedding_stats.update({
                        "embedding_count": current_count + new_embedding_count,
                        "document_count": current_docs + len(documents),
                        "chunk_count": current_chunks + new_chunks,  # Count actual chunks separately from embeddings
                        "updated_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat(),
                        "model": self.embeddings.model_name
                    })
                    
                    # Save updated stats
                    with open(store_path / "embedding_stats.json", "w") as f:
                        json.dump(embedding_stats, f, indent=2)
                        
                    logger.info(f"Updated embedding stats for {store_name}: added {len(new_embeddings)} new embeddings, total now {embedding_stats['embedding_count']}")
                
                # Update index information
                if "default" not in self.index["stores"]:
                    self.index["stores"]["default"] = {}
                
                # Update or add store info
                if store_name in self.index["stores"]["default"]:
                    # Update existing store info
                    store_info = self.index["stores"]["default"][store_name]
                    store_info["document_count"] = len(combined_documents)
                    store_info["last_updated"] = datetime.now().isoformat()
                    
                    # Update metadata
                    if "metadata" not in store_info:
                        store_info["metadata"] = {}
                    
                    # Merge metadata (don't overwrite existing)
                    for key, value in metadata.items():
                        # For lists, append new values
                        if key in store_info["metadata"] and isinstance(store_info["metadata"][key], list) and isinstance(value, list):
                            # Add only unique values
                            store_info["metadata"][key] = list(set(store_info["metadata"][key] + value))
                        # For non-list values, only add if not present
                        elif key not in store_info["metadata"]:
                            store_info["metadata"][key] = value
                            
                    # Add resource count
                    if "resources" not in store_info["metadata"]:
                        store_info["metadata"]["resources"] = []
                        
                    # Add resource ID if present and not already in list
                    if metadata.get("resource_id") and metadata["resource_id"] not in store_info["metadata"].get("resources", []):
                        store_info["metadata"]["resources"].append(metadata["resource_id"])
                else:
                    # It's a new store in the index
                    self.index["stores"]["default"][store_name] = {
                        "path": str(store_path),
                        "document_count": len(combined_documents),
                        "last_updated": datetime.now().isoformat(),
                        "metadata": metadata or {},
                        "dimension": existing_index.d
                    }
                
                self._save_index()
                logger.info(f"Added {len(documents)} documents to store '{store_name}', now contains {len(combined_documents)} documents")
                return True
            else:
                # Store doesn't exist, create it
                logger.info(f"Store '{store_name}' doesn't exist, creating new store with {len(documents)} documents")
                return await self.create_or_update_store(
                    store_name=store_name,
                    documents=documents,
                    metadata=metadata
                )
                
        except Exception as e:
            logger.error(f"Error adding documents to store {store_name}: {e}", exc_info=True)
            return False
            
    async def add_documents_to_topics(self,
        documents: List[Document],
        update_stats: bool = True
    ) -> Dict[str, bool]:
        """
        Add documents to topic-specific stores based on document metadata
        
        Args:
            documents: List of documents to add, each with 'topics' in metadata
            
        Returns:
            Dictionary of topic names and success status
        """
        try:
            # Group documents by topic
            topic_docs: Dict[str, List[Document]] = {}
            
            for doc in documents:
                topics = doc.metadata.get('topics', [])
                
                if isinstance(topics, str):
                    topics = [topics]
                
                # Skip if no topics
                if not topics:
                    continue
                    
                for topic in topics:
                    # Clean up topic name
                    topic_name = topic.lower().strip()
                    if not topic_name:
                        continue
                        
                    # Add to topic group
                    if topic_name not in topic_docs:
                        topic_docs[topic_name] = []
                    topic_docs[topic_name].append(doc)
            
            # Log topic distribution
            logger.info(f"Adding to {len(topic_docs)} topic stores:")
            for topic, docs in topic_docs.items():
                logger.info(f"  - Topic '{topic}': {len(docs)} documents")
            
            # Process each topic
            results = {}
            for topic, docs in topic_docs.items():
                # Create store name for this topic
                store_name = f"topic_{topic}"
                
                # Add to this topic store
                success = await self.add_documents_to_store(
                    store_name=store_name,
                    documents=docs,
                    metadata={"topic": topic},
                    update_stats=update_stats
                )
                
                results[topic] = success
                
            return results
            
        except Exception as e:
            logger.error(f"Error adding documents to topic stores: {e}", exc_info=True)
            return {}

    async def create_topic_stores(self, documents: List[Document], group_id: str = "default", use_clustering: bool = False, min_docs_per_topic: int = 5) -> Dict:
        """Create separate vector stores by topic (with optional clustering)"""
        if use_clustering:
            logger.info("Using intelligent topic clustering to create meaningful topic stores")
            return await self.create_topic_stores_with_clustering(documents, min_docs_per_topic=min_docs_per_topic)
        else:
            logger.info("Using legacy individual topic store creation")
            # Original implementation for backwards compatibility
            return await self._create_individual_topic_stores(documents, group_id)
    
    async def _create_individual_topic_stores(self, documents: List[Document], group_id: str = "default") -> Dict:
        """Pure tag-based topic store creation method (no LLM or NLP interpretation)"""
        try:
            # First, extract topics from document metadata - TAGS ONLY
            topic_docs: Dict[str, List[Document]] = {}
            
            # Look for explicit tags/topics in metadata - no fallback to NLP
            for doc in documents:
                # Check if document already has topics
                topics = doc.metadata.get('topics', [])
                if isinstance(topics, str):
                    topics = [t.strip() for t in topics.split(',') if t.strip()]
                
                # If no topics found, try using tags
                if not topics:
                    tags = doc.metadata.get('tags', [])
                    if isinstance(tags, str):
                        tags = [t.strip() for t in tags.split(',') if t.strip()]
                    topics = tags
                    
                # If still no topics, try fields (some tables use fields instead of tags)
                if not topics:
                    fields = doc.metadata.get('fields', [])
                    if isinstance(fields, str):
                        fields = [t.strip() for t in fields.split(',') if t.strip()]
                    topics = fields
                
                # REMOVED: No LLM/NLP fallback - skip documents without explicit tags
                if not topics:
                    logger.debug(f"Skipping document without explicit tags: {doc.metadata.get('title', 'Unknown')}")
                    continue
                
                # Add document to each topic collection (only if explicit tags found)
                for topic in topics:
                    # Sanitize topic name for Windows file systems
                    # Remove invalid characters: [ ] " ' , and other problematic characters
                    import re
                    sanitized_topic = re.sub(r'[^\w\s-]', '', str(topic))  # Remove non-alphanumeric chars except spaces and hyphens
                    normalized_topic = sanitized_topic.lower().strip().replace(' ', '-')
                    # Remove multiple consecutive hyphens
                    normalized_topic = re.sub(r'-+', '-', normalized_topic)
                    # Remove leading/trailing hyphens
                    normalized_topic = normalized_topic.strip('-')
                    
                    # Skip empty or invalid topics
                    if not normalized_topic or len(normalized_topic) < 2:
                        continue
                        
                    if normalized_topic not in topic_docs:
                        topic_docs[normalized_topic] = []
                    topic_docs[normalized_topic].append(doc)
            
            # Calculate processing statistics
            total_docs_with_topics = sum(len(docs) for docs in topic_docs.values())
            skipped_docs = len(documents) - total_docs_with_topics
            
            # Log topic distribution and processing summary
            logger.info(f"📊 Tag-based topic extraction results:")
            logger.info(f"  - Total documents: {len(documents)}")
            logger.info(f"  - Documents with explicit tags: {total_docs_with_topics}")
            logger.info(f"  - Documents skipped (no tags): {skipped_docs}")
            logger.info(f"  - Unique topics found: {len(topic_docs)}")
            
            if topic_docs:
                logger.info("📋 Topic distribution:")
                for topic, docs in topic_docs.items():
                    logger.info(f"  - Topic '{topic}': {len(docs)} documents")
            else:
                logger.warning("⚠️  No documents with explicit tags found - no topic stores will be created")
            
            # Create vector store for each topic
            results = {}
            for topic, docs in topic_docs.items():
                store_name = f"topic_{topic}"
                
                # Skip empty topics
                if not docs:
                    continue
                    
                success = await self.create_or_update_store(
                    store_name=store_name,
                    documents=docs,
                    metadata={"topic": topic},
                    update_stats=True  # Ensure stats are updated
                )
                results[topic] = success
                
            return results
            
        except Exception as e:
            logger.error(f"Error creating topic stores: {e}", exc_info=True)
            return {}

    async def create_topic_stores_with_clustering(self, documents: List[Document], min_docs_per_topic: int = 5) -> Dict:
        """
        Create topic stores using intelligent clustering based on tags, titles, and definitions.
        Groups related topics together to avoid over-fragmentation.
        
        Args:
            documents: List of documents to cluster into topics
            min_docs_per_topic: Minimum documents required to create a topic store
            
        Returns:
            Dictionary of topic names and success status
        """
        try:
            import re
            
            logger.info(f"🧠 Creating topic stores using intelligent clustering (min {min_docs_per_topic} docs per topic)")
            
            # Step 1: Extract and analyze all available topics
            all_topics = []
            doc_topics_map = {}  # Maps doc index to list of topics
            
            for i, doc in enumerate(documents):
                doc_topics = []
                
                # Extract topics from multiple sources
                # Priority 1: Explicit topics field
                topics = doc.metadata.get('topics', [])
                if isinstance(topics, str):
                    topics = [t.strip() for t in topics.split(',')]
                doc_topics.extend(topics)
                
                # Priority 2: Tags field
                tags = doc.metadata.get('tags', [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(',')]
                doc_topics.extend(tags)
                
                # Priority 3: Use title as a topic if no other topics found
                if not doc_topics:
                    title = doc.metadata.get('title', '') or doc.metadata.get('resource_id', '')
                    if title:
                        # Simply use cleaned title as a topic
                        clean_title = self._normalize_topic(title)
                        if clean_title and len(clean_title) > 2:
                            doc_topics.append(clean_title)
                
                # Clean and normalize topics
                cleaned_topics = [self._normalize_topic(topic) for topic in doc_topics if topic.strip()]
                cleaned_topics = [t for t in cleaned_topics if t and len(t) > 2]  # Remove empty and very short topics
                
                if not cleaned_topics:
                    cleaned_topics = ['general']  # Fallback
                
                doc_topics_map[i] = cleaned_topics
                all_topics.extend(cleaned_topics)
            
            # Step 2: Analyze topic frequency and relationships
            topic_counts = Counter(all_topics)
            logger.info(f"📊 Found {len(topic_counts)} unique topics from {len(documents)} documents")
            
            # Step 3: Create semantic clusters
            topic_clusters = await self._create_semantic_clusters(topic_counts, min_docs_per_topic)
            
            # Step 4: Group documents by clusters
            cluster_docs = defaultdict(list)
            
            for doc_idx, topics in doc_topics_map.items():
                doc = documents[doc_idx]
                
                # Find which cluster(s) this document belongs to
                assigned_clusters = set()
                for topic in topics:
                    for cluster_name, cluster_topics in topic_clusters.items():
                        if topic in cluster_topics:
                            assigned_clusters.add(cluster_name)
                
                # If no cluster assigned, put in general
                if not assigned_clusters:
                    assigned_clusters.add('general')
                
                # Add document to all relevant clusters
                for cluster in assigned_clusters:
                    cluster_docs[cluster].append(doc)
            
            # Step 5: Log cluster analysis
            logger.info(f"🎯 Created {len(cluster_docs)} topic clusters:")
            for cluster, docs in cluster_docs.items():
                topics_in_cluster = topic_clusters.get(cluster, [cluster])
                logger.info(f"  - Cluster '{cluster}': {len(docs)} documents (topics: {', '.join(topics_in_cluster[:5])}{'...' if len(topics_in_cluster) > 5 else ''})")
            
            # Step 6: Create vector stores for each cluster
            results = {}
            for cluster_name, docs in cluster_docs.items():
                if len(docs) >= min_docs_per_topic:  # Only create if meets minimum threshold
                    store_name = f"topic_{cluster_name}"
                    
                    # Add cluster metadata
                    cluster_metadata = {
                        "topic": cluster_name,
                        "cluster_topics": topic_clusters.get(cluster_name, [cluster_name]),
                        "document_count": len(docs)
                    }
                    
                    success = await self.create_or_update_store(
                        store_name=store_name,
                        documents=docs,
                        metadata=cluster_metadata,
                        update_stats=True
                    )
                    results[cluster_name] = success
                    
                    if success:
                        logger.info(f"✅ Created topic store: {store_name} with {len(docs)} documents")
                    else:
                        logger.warning(f"❌ Failed to create topic store: {store_name}")
                else:
                    logger.info(f"⏭️  Skipped cluster '{cluster_name}' - only {len(docs)} documents (min: {min_docs_per_topic})")
            
            logger.info(f"🎉 Topic clustering complete: Created {len(results)} meaningful topic stores")
            return results
            
        except Exception as e:
            logger.error(f"Error in topic clustering: {e}", exc_info=True)
            return {}
    
    def _extract_semantic_topics_from_title(self, title: str) -> List[str]:
        """Extract meaningful topics from document titles using semantic analysis"""
        import re
        
        # Common academic/technical domain keywords
        domain_keywords = {
            'programming': ['python', 'javascript', 'java', 'code', 'programming', 'development', 'software'],
            'data-science': ['data', 'analysis', 'statistics', 'machine-learning', 'ai', 'analytics', 'visualization'],
            'mathematics': ['math', 'calculus', 'algebra', 'geometry', 'statistics', 'probability'],
            'web-development': ['html', 'css', 'web', 'frontend', 'backend', 'api', 'http'],
            'databases': ['sql', 'database', 'mysql', 'postgresql', 'mongodb', 'data-storage'],
            'algorithms': ['algorithm', 'sorting', 'search', 'complexity', 'optimization'],
            'networks': ['network', 'tcp', 'http', 'internet', 'protocol', 'security']
        }
        
        title_lower = title.lower()
        topics = []
        
        # Extract domain-specific topics
        for domain, keywords in domain_keywords.items():
            if any(keyword in title_lower for keyword in keywords):
                topics.append(domain)
        
        # Extract meaningful noun phrases (2-3 words)
        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', title_lower)
        filtered_words = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Add significant individual terms
        topics.extend(filtered_words[:3])  # Take up to 3 most significant terms
        
        return topics
    
    def _extract_topics_from_definition(self, definition: str) -> List[str]:
        """Extract topics from definition content"""
        import re
        
        if not definition or len(definition) < 10:
            return []
        
        # Look for key phrases that indicate topics
        topic_patterns = [
            r'is a (\w+)',  # "X is a concept"
            r'refers to (\w+)',  # "X refers to method"
            r'type of (\w+)',  # "X is a type of algorithm"
            r'method for (\w+)',  # "X is a method for analysis"
            r'used in (\w+)',  # "X is used in programming"
        ]
        
        topics = []
        definition_lower = definition.lower()
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, definition_lower)
            topics.extend(matches)
        
        # Extract domain-specific terms
        domain_terms = ['algorithm', 'method', 'technique', 'concept', 'theory', 'principle', 'approach']
        for term in domain_terms:
            if term in definition_lower:
                topics.append(term)
        
        return topics[:2]  # Limit to avoid noise
    
    def _normalize_topic(self, topic: str) -> str:
        """Normalize topic names for consistency"""
        import re
        
        if not topic:
            return ""
        
        # Convert to lowercase and clean
        normalized = topic.lower().strip()
        
        # Remove special characters except hyphens and spaces
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        
        # Replace spaces with hyphens
        normalized = normalized.replace(' ', '-')
        
        # Remove multiple consecutive hyphens
        normalized = re.sub(r'-+', '-', normalized)
        
        # Remove leading/trailing hyphens
        normalized = normalized.strip('-')
        
        return normalized
    
    async def _create_semantic_clusters(self, topic_counts: Counter, min_docs_per_topic: int) -> Dict[str, List[str]]:
        """Create semantic clusters from topics using LLM analysis"""
        try:
            # Get topics that have enough frequency to be considered
            significant_topics = [topic for topic, count in topic_counts.items() 
                                if count >= max(1, min_docs_per_topic // 3)]  # Lower threshold for clustering input
            
            if len(significant_topics) <= 3:
                # Too few topics to cluster meaningfully, return individual clusters
                clusters = {}
                for topic, count in topic_counts.most_common():
                    if count >= min_docs_per_topic:
                        clusters[topic] = [topic]
                logger.info(f"  📁 Too few topics for LLM clustering, created {len(clusters)} individual clusters")
                return clusters
            
            logger.info(f"🤖 Using LLM to cluster {len(significant_topics)} topics")
            
            # Load LLM settings for topic clustering
            settings_path = Path(__file__).parent / ".." / "settings" / "topic-clustering-settings.json"
            if not settings_path.exists():
                logger.warning("Topic clustering settings not found, falling back to frequency-based clustering")
                return await self._create_frequency_based_clusters(topic_counts, min_docs_per_topic)
            
            with open(settings_path, 'r') as f:
                llm_settings = json.load(f)
            
            # Initialize LLM client based on provider
            provider = llm_settings.get('provider', 'openrouter').lower()
            
            if provider == 'ollama':
                from scripts.llm.ollama_client import OllamaClient
                llm_client = OllamaClient()
            else:  # Default to openrouter
                from scripts.llm.open_router_client import OpenRouterClient
                llm_client = OpenRouterClient()
            
            # Prepare topic list with frequencies for context
            topic_info = []
            for topic in significant_topics:
                count = topic_counts[topic]
                topic_info.append(f"{topic} ({count} docs)")
            
            # Create prompt for LLM
            topics_text = ", ".join(topic_info)
            prompt = f"""Analyze these topics and group them into meaningful clusters:

{topics_text}

Group semantically related topics together. Consider domain relationships and conceptual similarity. Create 3-8 clusters maximum to avoid fragmentation."""
            
            # Get LLM response
            try:
                logger.info(f"🤖 Sending prompt to LLM: {prompt[:200]}...")
                if provider == 'ollama':
                    response = await llm_client.generate_response(prompt)
                else:  # OpenRouter
                    model = llm_settings.get('openrouter_model', llm_settings.get('model_name', 'meta-llama/llama-3.3-8b-instruct:free'))
                    response = await llm_client.generate_response(prompt, model=model)
                
                logger.info(f"🤖 LLM response: {response[:500]}...")
                
                if response and response.strip():
                    # Parse JSON response
                    import re
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        clusters_json = json.loads(json_match.group())
                        
                        # Validate and clean the clusters
                        validated_clusters = {}
                        for cluster_name, topics in clusters_json.items():
                            if isinstance(topics, list) and topics:
                                # Clean cluster name
                                clean_cluster_name = self._normalize_topic(cluster_name)
                                if clean_cluster_name:
                                    # Only include topics that exist in our original set
                                    valid_topics = [t for t in topics if t in significant_topics]
                                    if valid_topics:
                                        validated_clusters[clean_cluster_name] = valid_topics
                        
                        if validated_clusters:
                            logger.info(f"✅ LLM created {len(validated_clusters)} semantic clusters")
                            for cluster, topics in validated_clusters.items():
                                total_docs = sum(topic_counts[topic] for topic in topics)
                                logger.info(f"  📁 Cluster '{cluster}': {len(topics)} topics, ~{total_docs} docs")
                            return validated_clusters
                        
            except Exception as llm_error:
                logger.error(f"LLM clustering failed: {llm_error}")
                return {}  # Return empty clusters if LLM fails
            
            # No fallback - LLM-only clustering
            logger.error("LLM clustering produced no valid clusters")
            return {}
            
        except Exception as e:
            logger.error(f"Error in semantic clustering: {e}")
            return {}  # Return empty clusters if any error occurs
    
    async def _create_frequency_based_clusters(self, topic_counts: Counter, min_docs_per_topic: int) -> Dict[str, List[str]]:
        """Fallback clustering method based on topic frequency"""
        clusters = {}
        
        # Group high-frequency topics individually
        high_freq_topics = [(topic, count) for topic, count in topic_counts.most_common() 
                          if count >= min_docs_per_topic]
        
        for topic, count in high_freq_topics:
            clusters[topic] = [topic]
            logger.info(f"  📁 Individual cluster '{topic}': {count} docs")
        
        # Group remaining low-frequency topics into 'general'
        low_freq_topics = [topic for topic, count in topic_counts.items() 
                         if count < min_docs_per_topic]
        
        if low_freq_topics:
            total_remaining = sum(topic_counts[topic] for topic in low_freq_topics)
            if total_remaining >= min_docs_per_topic:
                clusters['general'] = low_freq_topics
                logger.info(f"  📁 General cluster: {len(low_freq_topics)} topics, ~{total_remaining} docs")
        
        return clusters

    async def search(self, 
        query: str, 
        store_names: List[str] = None,
        top_k: int = 5
    ) -> Dict:
        """Search across specified vector stores"""
        try:
            results = {}
            
            # Make sure we're looking in the default group path
            if "default" not in self.index["stores"]:
                return {}
                
            # Handle wildcard pattern matching for store names
            if store_names:
                import fnmatch
                available_stores = self.index["stores"]["default"].keys()
                
                # Expand wildcards to matching store names
                expanded_stores = []
                for pattern in store_names:
                    if "*" in pattern:
                        expanded_stores.extend([
                            store for store in available_stores 
                            if fnmatch.fnmatch(store, pattern)
                        ])
                    else:
                        expanded_stores.append(pattern)
                        
                stores_to_search = expanded_stores
            else:
                # Use all available stores
                stores_to_search = list(self.index["stores"]["default"].keys())
            
            # Search each matching store
            for store_name in stores_to_search:
                if store_name not in self.index["stores"]["default"]:
                    continue
                    
                store_info = self.index["stores"]["default"][store_name]
                store_path = Path(store_info["path"])
                if not (store_path / "index.faiss").exists():
                    continue
                      # Load vector store
                vector_store = FAISS.load_local(
                    str(store_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Search
                docs = vector_store.similarity_search(query, k=top_k)
                results[store_name] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, 'score', None)
                    }
                    for doc in docs
                ]
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector stores: {e}")
            return {}

    def list_stores(self, group_id: str = "default") -> List[Dict]:
        """
        List all available stores
        
        Note: group_id parameter is kept for backward compatibility
        but is no longer used since all stores are in one group
        """
        try:
            # Check if we have any stores at all
            if "default" not in self.index["stores"]:
                return []
                
            # Return all stores
            return [
                {
                    "name": name,
                    **info
                }
                for name, info in self.index["stores"]["default"].items()
            ]
            
        except Exception as e:
            logger.error(f"Error listing stores: {e}")
            return []

    async def get_representative_chunks(
        self,
        store_name: str,
        group_id: str = "default",  # Kept for backward compatibility
        num_chunks: int = 5
    ) -> List[Document]:
        """Get representative chunks from a store using clustering"""
        try:
            # Get the store path
            store_path = self._get_store_path(store_name)
            if not (store_path / "index.faiss").exists():
                logger.warning(f"No index.faiss file found at {store_path}")
                return []
                
            # Load vectors and documents
            import faiss
            index = faiss.read_index(str(store_path / "index.faiss"))
            
            if not (store_path / "documents.json").exists():
                logger.warning(f"No documents.json file found at {store_path}")
                return []
                
            with open(store_path / "documents.json", "r") as f:
                documents = json.load(f)
            
            if not documents:
                logger.warning(f"No documents found in {store_path / 'documents.json'}")
                return []
                
            # Log details for debugging
            logger.info(f"Found {len(documents)} documents in {store_name}")
            logger.info(f"Index has dimension {index.d} and {index.ntotal} vectors")
            
            # Perform clustering
            n_clusters = min(num_chunks, len(documents))
            if n_clusters < 1:
                logger.warning(f"Not enough documents for clustering in {store_name}")
                return []
                
            kmeans = faiss.Kmeans(index.d, n_clusters)
            try:
                vectors = faiss.vector_to_array(index).reshape(-1, index.d)
                # Only proceed if we have vectors
                if vectors.shape[0] == 0:
                    logger.warning(f"No vectors found in {store_name} index")
                    return []
                    
                logger.info(f"Training K-means with {vectors.shape[0]} vectors")
                kmeans.train(vectors)
                
                # Get nearest documents to centroids
                _, I = index.search(kmeans.centroids, 1)
                
                result_docs = []
                for idx in I.ravel():
                    # Make sure the index is valid
                    if idx < len(documents):
                        result_docs.append(Document(
                            page_content=documents[idx]["content"],
                            metadata=documents[idx]["metadata"]
                        ))
                    else:
                        logger.warning(f"Invalid document index: {idx} (max: {len(documents)-1})")
                
                logger.info(f"Returning {len(result_docs)} representative documents")
                return result_docs
            except Exception as cluster_err:
                logger.error(f"Error during clustering: {cluster_err}")
                
                # Fallback: if clustering fails, just return a random sample
                import random
                sample_size = min(num_chunks, len(documents))
                sample_indices = random.sample(range(len(documents)), sample_size)
                
                return [
                    Document(
                        page_content=documents[i]["content"],
                        metadata=documents[i]["metadata"]
                    )
                    for i in sample_indices
                ]
            
        except Exception as e:
            logger.error(f"Error getting representative chunks: {e}", exc_info=True)
            return []