#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from langchain.schema import Document

logger = logging.getLogger(__name__)

class ContentStorageService:
    """
    Handles storing document content in temporary files for later processing into vector stores.
    Provides standardized methods for storing different types of content.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the storage service.
        
        Args:
            data_folder: Path to the data folder for storing content
        """
        self.data_folder = data_folder
        
        # Setup temporary storage for documents
        self.temp_storage_path = Path(self.data_folder) / "temp" / "vector_data"
        self.temp_storage_path.mkdir(parents=True, exist_ok=True)
        self.content_store_path = self.temp_storage_path / "content_store.jsonl"
        self.reference_store_path = self.temp_storage_path / "reference_store.jsonl"
        
        # Track document counts for reporting
        self.content_doc_count = 0
        self.reference_doc_count = 0
        self.vector_generation_needed = False
        
        logger.info(f"ContentStorageService initialized at {self.temp_storage_path}")
        
    def store_original_content(self, title: str, url: str, content: str, resource_id: str) -> bool:
        """
        Store original scraped content for later vector indexing.
        Content is saved to a JSONL file instead of immediately creating vectors.
        
        Args:
            title: Resource title
            url: Resource URL
            content: Scraped content text
            resource_id: Identifier for logging
            
        Returns:
            True if storing succeeded, False otherwise
        """
        if not content:
            logger.warning(f"No content to store for {resource_id}")
            return False
            
        from scripts.data.document_processor import DocumentProcessor
        from langchain.schema import Document
        import json
        
        try:
            # Initialize required components
            processor = DocumentProcessor()
            
            # Create documents from content chunks
            documents = []
            
            # Process content in reasonable chunks to avoid hitting context limits
            max_chunk_size = 4000  # ~1000 words per chunk
            
            # If content is very large, process it in segments
            if len(content) > max_chunk_size:
                # Create chunks with text_splitter
                content_chunks = processor.text_splitter.split_text(content)
                
                # Process each chunk
                for i, chunk in enumerate(content_chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "source_type": "resource_content",
                            "resource_id": title,
                            "url": url,
                            "chunk_index": i,
                            "total_chunks": len(content_chunks),
                            "processed_at": datetime.now().isoformat(),
                            "content_type": "original_scraped",
                            "resource_id": title.replace(" ", "_").lower()
                        }
                    ))
                logger.info(f"Split content for {resource_id} into {len(content_chunks)} chunks")
            else:
                # For smaller content, just create a single document
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source_type": "resource_content",
                        "resource_id": title,
                        "url": url,
                        "processed_at": datetime.now().isoformat(),
                        "content_type": "original_scraped",
                        "resource_id": title.replace(" ", "_").lower()
                    }
                ))
            
            # Store documents in temporary JSONL file for later processing
            doc_count = 0
            with open(self.content_store_path, 'a') as f:
                for doc in documents:
                    # Convert Document to serializable dict
                    doc_dict = {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    # Write as JSON line
                    f.write(json.dumps(doc_dict) + '\n')
                    doc_count += 1
            
            # Update document count for reporting
            self.content_doc_count += doc_count
            # Flag that vector generation will be needed
            self.vector_generation_needed = True
            
            logger.info(f"Stored {doc_count} original content documents for {resource_id} in temporary file")
            return True
            
        except Exception as e:
            logger.error(f"Error storing original content for {resource_id}: {e}", exc_info=True)
            return False
            
    def store_content_batch(self, title: str, url: str, content_batch: str, 
                             definitions: List[Dict], projects: List[Dict], 
                             methods: List[Dict], batch_metadata: Dict = None) -> bool:
        """
        Store a batch of content with analysis results for later vector indexing.
        Content is saved to a JSONL file instead of immediately creating vectors.
        
        Args:
            title: Resource title for this batch
            url: Resource URL
            content_batch: A portion of the scraped content text
            definitions: Extracted definitions
            projects: Identified projects
            methods: Extracted methods
            batch_metadata: Additional metadata for this batch
            
        Returns:
            True if storing succeeded, False otherwise
        """
        if not content_batch:
            logger.warning(f"Empty content batch for {title}, skipping")
            return False
            
        from langchain.schema import Document
        import json
        import time
        
        try:
            # Start timing the batch processing
            batch_start = time.time()
            
            # Create enriched context for this batch
            max_content_size = min(5000, len(content_batch))  # Use all content up to 5000 chars
            
            # Create enriched content with batch information
            enriched_content = f"TITLE: {title}\nURL: {url}\n\nCONTENT BATCH:\n{content_batch[:max_content_size]}"
            
            # Add batch metadata if provided
            if batch_metadata and batch_metadata.get("batch_number"):
                enriched_content += f"\n\n[Batch {batch_metadata.get('batch_number')}, "
                enriched_content += f"Progress: {batch_metadata.get('progress_percent', 0):.1f}%]"
            
            # Add indication if content was truncated
            if len(content_batch) > max_content_size:
                enriched_content += f"...\n\n[Content truncated - batch size: {len(content_batch)/1024:.1f} KB]"
            
            enriched_content += "\n\n"
            
            # Add definitions if available - keeping this minimal for batches
            if definitions:
                enriched_content += "DEFINITIONS EXTRACTED:\n"
                for i, definition in enumerate(definitions[:3]):  # Limit more for batches
                    enriched_content += f"- {definition.get('term', '')}: {definition.get('definition', '')}\n"
            
            # Add methods if available - keeping this minimal for batches
            if methods:
                enriched_content += "\nMETHODS EXTRACTED:\n"
                for i, method in enumerate(methods[:2]):  # Limit more for batches
                    enriched_content += f"- {method.get('title', '')}\n"
            
            # Create metadata with all needed information
            metadata = {
                "source_type": "enriched_resource",
                "resource_id": title,
                "url": url,
                "resource_id": title.replace(" ", "_").lower(),
                "resource_url": url,
                "definitions_count": len(definitions),
                "projects_count": len(projects),
                "methods_count": len(methods),
                "processed_at": datetime.now().isoformat(),
                "content_type": "enriched_batch",
            }
            
            # Get resource tags if they exist
            if batch_metadata and "tags" in batch_metadata:
                tags = batch_metadata["tags"]
                # Copy tags to topics field for topic store creation
                if tags:
                    metadata["tags"] = tags
                    # Also add topics field for topic store creation
                    if isinstance(tags, str):
                        metadata["topics"] = tags.split(",")
                    else:
                        metadata["topics"] = tags
            
            # Add any additional batch metadata
            if batch_metadata:
                metadata.update(batch_metadata)
            
            # Create document for storage
            document = Document(
                page_content=enriched_content,
                metadata=metadata
            )
            
            # Store document in temporary JSONL file for later processing
            with open(self.content_store_path, 'a') as f:
                # Convert Document to serializable dict
                doc_dict = {
                    "page_content": document.page_content,
                    "metadata": document.metadata
                }
                # Write as JSON line
                f.write(json.dumps(doc_dict) + '\n')
            
            # Update document count for reporting
            self.content_doc_count += 1
            # Flag that vector generation will be needed
            self.vector_generation_needed = True
            
            # Calculate and log the processing time
            batch_duration = time.time() - batch_start
            logger.info(f"Stored batch content for {title} in temporary file (took {batch_duration:.2f}s)")
            return True
                
        except Exception as e:
            # Catch any exceptions from the outer try block
            logger.error(f"Error processing batch content for {title}: {e}")
            return False

    def store_content_with_analysis(self, title: str, url: str, content: str, 
                                    definitions: List[Dict], projects: List[Dict], 
                                    methods: List[Dict]) -> bool:
        """
        Store enriched content combining original and analysis results for later vector indexing.
        Content is saved to a JSONL file instead of immediately creating vectors.
        
        Args:
            title: Resource title
            url: Resource URL
            content: Scraped content text
            definitions: Extracted definitions
            projects: Identified projects
            methods: Extracted methods
            
        Returns:
            True if storing succeeded, False otherwise
        """
        if not content:
            logger.warning(f"No content to store enriched analysis for {title}")
            return False
            
        from langchain.schema import Document
        import json
        
        try:
            # Create enriched context
            max_content_size = 5000  # ~1250 words of content
            
            # Using a larger excerpt for analysis
            enriched_content = f"TITLE: {title}\nURL: {url}\n\nORIGINAL CONTENT EXCERPT:\n{content[:max_content_size]}"
            
            # Add indication if the content was truncated
            if len(content) > max_content_size:
                enriched_content += f"...\n\n[Content truncated - full size: {len(content)/1024:.1f} KB]"
            
            enriched_content += "\n\n"
            
            # Add definitions if available
            if definitions:
                enriched_content += "DEFINITIONS EXTRACTED:\n"
                for i, definition in enumerate(definitions[:5]):  # Limit to top 5
                    enriched_content += f"- {definition.get('term', '')}: {definition.get('definition', '')}\n"
            
            # Add projects if available
            if projects:
                enriched_content += "\nPROJECTS IDENTIFIED:\n"
                for i, project in enumerate(projects[:3]):  # Limit to top 3
                    # Handle both dictionary and string formats
                    if isinstance(project, dict):
                        enriched_content += f"- {project.get('title', '')}: {project.get('description', '')}\n"
                    elif isinstance(project, str):
                        enriched_content += f"- {project}\n"
                    else:
                        logger.warning(f"Unknown project type: {type(project)}")
                        enriched_content += f"- {str(project)}\n"
            
            # Add methods if available
            if methods:
                enriched_content += "\nMETHODS EXTRACTED:\n"
                for i, method in enumerate(methods[:3]):  # Limit to top 3
                    enriched_content += f"- {method.get('title', '')}: {method.get('description', '')}\n"
            
            # Create complete metadata 
            metadata = {
                "source_type": "enriched_resource",
                "resource_id": title,
                "url": url,
                "resource_id": title.replace(" ", "_").lower(),
                "resource_url": url,
                "definitions_count": len(definitions),
                "projects_count": len(projects),
                "methods_count": len(methods),
                "processed_at": datetime.now().isoformat(),
                "content_type": "enriched"
            }
            
            # Try to find resource tags from the definitions or projects metadata
            tags = None
            for definition in definitions[:1]:  # Check first definition
                if isinstance(definition, dict) and "resource_tags" in definition:
                    tags = definition.get("resource_tags")
                    break
                    
            # If we still don't have tags, look at projects
            if not tags and projects:
                for project in projects[:1]:  # Check first project
                    if isinstance(project, dict) and "resource_tags" in project:
                        tags = project.get("resource_tags")
                        break
            
            # Add tags to metadata for topic store creation
            if tags:
                metadata["tags"] = tags
                # Also add topics field for topic store creation  
                if isinstance(tags, str):
                    metadata["topics"] = tags.split(",")
                else:
                    metadata["topics"] = tags
            
            # Create document
            document = Document(
                page_content=enriched_content,
                metadata=metadata
            )
            
            # Store document in temporary JSONL file for later processing
            with open(self.content_store_path, 'a') as f:
                # Convert Document to serializable dict
                doc_dict = {
                    "page_content": document.page_content,
                    "metadata": document.metadata
                }
                # Write as JSON line
                f.write(json.dumps(doc_dict) + '\n')
            
            # Update document count for reporting
            self.content_doc_count += 1
            # Flag that vector generation will be needed
            self.vector_generation_needed = True
            
            logger.info(f"Stored enriched content for {title} in temporary file")
            return True
            
        except Exception as e:
            # Catch any exceptions from the outer try block
            logger.error(f"Error creating enriched content for {title}: {e}", exc_info=True)
            return False
    
    def get_doc_count(self) -> Dict[str, int]:
        """Get current document counts for reporting"""
        return {
            "content_docs": self.content_doc_count,
            "reference_docs": self.reference_doc_count,
            "total_docs": self.content_doc_count + self.reference_doc_count,
            "vector_generation_needed": self.vector_generation_needed
        }

    def cleanup(self) -> bool:
        """Clean up temporary storage files after processing is complete"""
        try:
            # Check if files exist before trying to delete
            if self.content_store_path.exists():
                self.content_store_path.unlink()
                logger.info(f"Removed temporary content store file")
                
            if self.reference_store_path.exists():
                self.reference_store_path.unlink()
                logger.info(f"Removed temporary reference store file")
                
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up temporary storage: {e}")
            return False