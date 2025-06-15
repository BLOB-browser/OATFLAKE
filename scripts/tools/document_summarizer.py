#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Document Summarizer

This script provides functionality to generate summaries of documents based on 
their chunk relationships. It's designed to work with the document relationship 
labels created during PDF processing in rebuild_faiss_indexes.py.

Usage:
    python document_summarizer.py [--data-path PATH] [--document-id ID] [--list-documents]

Options:
    --data-path PATH       Path to data directory (defaults to config.json setting)
    --document-id ID       Generate summary for specific document ID
    --list-documents       List all available documents that can be summarized
"""

import asyncio
import logging
import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DocumentSummarizer:
    """Generate summaries of documents from their chunked content in vector stores."""
    
    def __init__(self, data_path: Path):
        """Initialize the document summarizer.
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = Path(data_path)
        self.vector_stores_path = self.data_path / "vector_stores" / "default"
        
    def list_available_documents(self) -> Dict[str, Any]:
        """List all documents that can be summarized based on their chunk relationships.
        
        Returns:
            Dictionary with document information
        """
        try:
            documents_info = {}
            
            # Check content_store for PDFs/materials
            content_store_path = self.vector_stores_path / "content_store" / "documents.json"
            if content_store_path.exists():
                with open(content_store_path, 'r', encoding='utf-8') as f:
                    content_docs = json.load(f)
                
                # Group documents by document_id
                for doc in content_docs:
                    metadata = doc.get("metadata", {})
                    document_id = metadata.get("document_id")
                    if document_id:
                        if document_id not in documents_info:
                            documents_info[document_id] = {
                                "document_id": document_id,
                                "document_title": metadata.get("document_title", "Unknown"),
                                "document_name": metadata.get("document_name", "Unknown"),
                                "content_type": metadata.get("content_type", "unknown"),
                                "total_chunks": metadata.get("total_chunks", 0),
                                "chunks_found": 0,
                                "store": "content_store",
                                "file_path": metadata.get("file_path", ""),
                                "material_id": metadata.get("material_id", "")
                            }
                        documents_info[document_id]["chunks_found"] += 1
            
            # Also check reference_store for other document types
            reference_store_path = self.vector_stores_path / "reference_store" / "documents.json"
            if reference_store_path.exists():
                with open(reference_store_path, 'r', encoding='utf-8') as f:
                    reference_docs = json.load(f)
                
                # Look for documents with relationship metadata
                for doc in reference_docs:
                    metadata = doc.get("metadata", {})
                    document_id = metadata.get("document_id")
                    if document_id and document_id not in documents_info:
                        documents_info[document_id] = {
                            "document_id": document_id,
                            "document_title": metadata.get("document_title", metadata.get("title", "Unknown")),
                            "document_name": metadata.get("document_name", "Unknown"),
                            "content_type": metadata.get("content_type", "unknown"),
                            "total_chunks": metadata.get("total_chunks", 1),
                            "chunks_found": 1,
                            "store": "reference_store",
                            "source_type": metadata.get("source_type", "unknown")
                        }
            
            logger.info(f"Found {len(documents_info)} documents that can be summarized")
            return {
                "status": "success",
                "documents": documents_info,
                "total_documents": len(documents_info)
            }
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return {
                "status": "error",
                "error": str(e),
                "documents": {}
            }
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document ID.
        
        Args:
            document_id: The document ID to get chunks for
            
        Returns:
            List of document chunks with metadata
        """
        chunks = []
        
        try:
            # Check content_store
            content_store_path = self.vector_stores_path / "content_store" / "documents.json"
            if content_store_path.exists():
                with open(content_store_path, 'r', encoding='utf-8') as f:
                    content_docs = json.load(f)
                
                for doc in content_docs:
                    metadata = doc.get("metadata", {})
                    if metadata.get("document_id") == document_id:
                        chunks.append({
                            "content": doc.get("content", ""),
                            "metadata": metadata,
                            "chunk_index": metadata.get("chunk_index", 0),
                            "store": "content_store"
                        })
            
            # Check reference_store
            reference_store_path = self.vector_stores_path / "reference_store" / "documents.json"
            if reference_store_path.exists():
                with open(reference_store_path, 'r', encoding='utf-8') as f:
                    reference_docs = json.load(f)
                
                for doc in reference_docs:
                    metadata = doc.get("metadata", {})
                    if metadata.get("document_id") == document_id:
                        chunks.append({
                            "content": doc.get("content", ""),
                            "metadata": metadata,
                            "chunk_index": metadata.get("chunk_index", 0),
                            "store": "reference_store"
                        })
            
            # Sort chunks by chunk_index for proper order
            chunks.sort(key=lambda x: x.get("chunk_index", 0))
            
            logger.info(f"Found {len(chunks)} chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {e}")
            return []
    
    async def generate_document_summary(self, document_id: str, use_llm: bool = True) -> Dict[str, Any]:
        """Generate a summary for a specific document.
        
        Args:
            document_id: The document ID to summarize
            use_llm: Whether to use LLM for intelligent summarization
            
        Returns:
            Dictionary with summary results
        """
        try:
            # Get all chunks for this document
            chunks = self.get_document_chunks(document_id)
            
            if not chunks:
                return {
                    "status": "error",
                    "error": f"No chunks found for document {document_id}",
                    "document_id": document_id
                }
            
            # Get document metadata from first chunk
            first_chunk = chunks[0]
            document_metadata = first_chunk["metadata"]
            
            # Combine all chunk content
            full_content = "\n\n".join([chunk["content"] for chunk in chunks])
            
            summary_result = {
                "status": "success",
                "document_id": document_id,
                "document_title": document_metadata.get("document_title", "Unknown"),
                "document_name": document_metadata.get("document_name", "Unknown"),
                "content_type": document_metadata.get("content_type", "unknown"),
                "total_chunks": len(chunks),
                "content_length": len(full_content),
                "summary_type": "llm" if use_llm else "extractive",
                "generated_at": datetime.now().isoformat()
            }
            
            if use_llm:
                # Use LLM for intelligent summarization
                try:
                    from scripts.llm.ollama_client import OllamaClient
                    
                    ollama_client = OllamaClient()
                    
                    # Create summarization prompt
                    prompt = f"""Please provide a comprehensive summary of the following document:

Document Title: {document_metadata.get('document_title', 'Unknown')}
Document Type: {document_metadata.get('content_type', 'unknown')}
Content Length: {len(full_content)} characters

Content to summarize:
{full_content[:8000]}  # Limit content to avoid token limits

Please provide:
1. A brief overview (2-3 sentences)
2. Key topics covered
3. Main insights or findings
4. Practical applications or relevance

Format your response as a well-structured summary."""

                    # Generate summary using LLM
                    response = await ollama_client.generate_response(
                        prompt=prompt,
                        temperature=0.3,
                        max_tokens=800
                    )
                    
                    if response and response.get("success"):
                        summary_result["summary"] = response["response"]
                        summary_result["llm_model"] = response.get("model", "unknown")
                    else:
                        # Fallback to extractive summary
                        summary_result["summary"] = self._create_extractive_summary(full_content)
                        summary_result["summary_type"] = "extractive_fallback"
                        summary_result["llm_error"] = response.get("error", "LLM failed") if response else "No response from LLM"
                        
                except Exception as llm_error:
                    logger.warning(f"LLM summarization failed: {llm_error}")
                    # Fallback to extractive summary
                    summary_result["summary"] = self._create_extractive_summary(full_content)
                    summary_result["summary_type"] = "extractive_fallback"
                    summary_result["llm_error"] = str(llm_error)
            else:
                # Create extractive summary
                summary_result["summary"] = self._create_extractive_summary(full_content)
            
            # Add chunk details
            summary_result["chunks"] = [
                {
                    "chunk_index": chunk["chunk_index"],
                    "store": chunk["store"],
                    "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"]
                }
                for chunk in chunks
            ]
            
            return summary_result
            
        except Exception as e:
            logger.error(f"Error generating summary for document {document_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "document_id": document_id
            }
    
    def _create_extractive_summary(self, content: str) -> str:
        """Create an extractive summary from content.
        
        Args:
            content: The full content to summarize
            
        Returns:
            Extractive summary string
        """
        # Simple extractive summary - take first few sentences and key points
        sentences = content.split('. ')
        
        # Take first 3 sentences
        summary_parts = []
        if len(sentences) >= 3:
            summary_parts.extend(sentences[:3])
        else:
            summary_parts.extend(sentences)
        
        # Look for key indicators (headers, lists, etc.)
        lines = content.split('\n')
        key_lines = []
        
        for line in lines:
            line = line.strip()
            if line and (
                line.startswith('##') or  # Headers
                line.startswith('**') or  # Bold text
                line.startswith('TITLE:') or
                line.startswith('DESCRIPTION:') or
                'important' in line.lower() or
                'key' in line.lower()
            ):
                key_lines.append(line)
                if len(key_lines) >= 5:  # Limit key lines
                    break
        
        # Combine summary parts
        summary = '. '.join(summary_parts)
        if key_lines:
            summary += '\n\nKey Points:\n' + '\n'.join(key_lines)
        
        # Limit summary length
        if len(summary) > 1000:
            summary = summary[:1000] + "..."
        
        return summary

def get_data_path_from_config():
    """Get data path from config file."""
    # Get the project root directory
    project_root = Path(__file__).resolve().parents[2]
    
    # Prioritize config.json in the project root directory
    config_paths = [
        project_root / "config.json"  # Project root config
    ]
    
    for config_path in config_paths:
        logger.info(f"Checking for config at: {config_path}")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if 'data_path' in config:
                        data_path = Path(config['data_path'])
                        logger.info(f"Found data_path in config: {data_path}")
                        return data_path
            except Exception as e:
                logger.warning(f"Error reading config file {config_path}: {e}")
    
    # Default fallback - use the project root
    logger.warning("No data_path found in config files, using project root as fallback")
    return project_root

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate document summaries from vector store chunks")
    parser.add_argument("--data-path", type=str, help="Path to data directory (defaults to config.json setting)")
    parser.add_argument("--document-id", type=str, help="Generate summary for specific document ID")
    parser.add_argument("--list-documents", action="store_true", help="List all available documents that can be summarized")
    parser.add_argument("--no-llm", action="store_true", help="Use extractive summarization instead of LLM")
    
    args = parser.parse_args()
    
    # Get data path from arguments or config
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        data_path = get_data_path_from_config()
    
    logger.info(f"Using data path: {data_path}")
    
    # Initialize document summarizer
    summarizer = DocumentSummarizer(data_path)
    
    if args.list_documents:
        # List all available documents
        result = summarizer.list_available_documents()
        
        if result["status"] == "success":
            print(f"\n📋 Found {result['total_documents']} documents that can be summarized:\n")
            for doc_id, info in result["documents"].items():
                print(f"Document ID: {doc_id}")
                print(f"  Title: {info['document_title']}")
                print(f"  Type: {info['content_type']}")
                print(f"  Chunks: {info['chunks_found']}")
                print(f"  Store: {info['store']}")
                if info.get('file_path'):
                    print(f"  File: {info['file_path']}")
                print()
        else:
            print(f"❌ Error listing documents: {result['error']}")
            return 1
            
    elif args.document_id:
        # Generate summary for specific document
        print(f"🔄 Generating summary for document: {args.document_id}")
        
        result = await summarizer.generate_document_summary(
            args.document_id, 
            use_llm=not args.no_llm
        )
        
        if result["status"] == "success":
            print(f"\n📄 Document Summary")
            print(f"=" * 50)
            print(f"Document ID: {result['document_id']}")
            print(f"Title: {result['document_title']}")
            print(f"Type: {result['content_type']}")
            print(f"Chunks: {result['total_chunks']}")
            print(f"Summary Type: {result['summary_type']}")
            print(f"Generated: {result['generated_at']}")
            print(f"\nSummary:")
            print(f"-" * 30)
            print(result['summary'])
            
            if result.get('llm_error'):
                print(f"\n⚠️ LLM Error: {result['llm_error']}")
            
        else:
            print(f"❌ Error generating summary: {result['error']}")
            return 1
    else:
        print("Please specify either --list-documents or --document-id")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))
