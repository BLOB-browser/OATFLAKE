from pathlib import Path
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader, TextLoader, UnstructuredFileLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from ..llm.ollama_embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS  # Updated import

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_path: Path, group_id: str = "default"):
        self.base_path = Path(data_path)
        self.group_id = group_id
          # Optimized chunking settings for better performance and analysis quality
        logger.info("Using optimized chunking settings for better performance and analysis quality")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,      # Optimized chunk size for better performance
            chunk_overlap=200,    # Optimized overlap for better context preservation
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
        
        # Consistent embeddings configuration for all devices
        logger.info("Using consistent embedding settings for all devices")
        self.embeddings = OllamaEmbeddings(batch_size=20, timeout=60.0)
        
        self.embedding_stats = {
            "total_embeddings": 0,
            "reference_embeddings": 0,
            "content_embeddings": 0,
            "chunks_generated": 0
        }

    def parse_row_data(self, row: pd.Series, data_type: str) -> dict:
        """Safely parse row data from pandas Series"""
        try:
            data = {}
            for key, value in row.items():
                if pd.isna(value):
                    data[key] = ''
                elif isinstance(value, (int, float)):
                    data[key] = str(value)  # Convert numbers to strings
                else:
                    data[key] = value
            return data
        except Exception as e:
            logger.error(f"Error parsing row data: {e}")
            return {}

    async def process_pdf_materials(self, materials_path: Path, row: dict) -> List[Document]:
        """Process PDF files referenced in materials"""
        try:
            file_path = materials_path / row.get('file_path', '')
            if not file_path.exists() or not file_path.suffix == '.pdf':
                logger.warning(f"PDF not found or invalid: {file_path}")
                return []

            logger.info(f"Processing PDF: {file_path}")
            loader = PyPDFLoader(str(file_path))
            pdf_docs = loader.load()

            # Add metadata from CSV
            for doc in pdf_docs:
                doc.metadata.update({
                    'source_type': 'material_pdf',
                    'title': row.get('title', ''),
                    'description': row.get('description', ''),
                    'fields': row.get('fields', '').split(','),
                    'file_path': str(file_path),
                    'created_at': row.get('created_at', '')
                })

            return pdf_docs
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return []
            
    async def process_critical_content(self):
        """
        Process only PDFs from materials.csv and methods from methods.csv
        These are considered critical learning content that should be processed first
        """
        try:
            logger.info("Processing critical learning content (PDFs and methods)...")
            methods_docs = []
            pdf_docs = []
            
            # Process methods.csv
            methods_path = self.base_path / "methods.csv"
            if methods_path.exists():
                try:
                    df = pd.read_csv(methods_path)
                    logger.info(f"Processing {len(df)} methods from methods.csv")
                    
                    for _, row in df.iterrows():
                        try:
                            row_data = self.parse_row_data(row, "methods")
                            content = f"Method: {row_data.get('title', '')}\nDescription: {row_data.get('description', '')}\nSteps: {row_data.get('steps', '')}"
                            
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source_type": "methods",
                                    "csv_file": methods_path.name,
                                    "processed_at": datetime.now().isoformat(),
                                    **row_data
                                }
                            )
                            methods_docs.append(doc)
                        except Exception as e:
                            logger.error(f"Error processing method: {e}")
                except Exception as e:
                    logger.error(f"Error reading methods.csv: {e}")
            
            # Process materials.csv for PDFs
            materials_path = self.base_path / "materials.csv"
            if materials_path.exists():
                try:
                    df = pd.read_csv(materials_path)
                    logger.info(f"Processing PDFs from {len(df)} materials")
                    
                    # CRITICAL CONTENT PRIORITY: Process all PDFs in materials.csv first
                    # This ensures critical learning content is always processed regardless of vector store status
                    already_processed_pdfs = set()
                    
                    # For incremental optimization only (not to skip critical content):
                    # Check existing vector store to identify what's already been processed for efficiency
                    content_store_path = self.base_path / "vector_stores" / "default" / "content_store"
                    check_for_duplicates = False
                    
                    if content_store_path.exists() and (content_store_path / "embedding_stats.json").exists():
                        try:
                            # Check metadata file for processed PDF info (for optimization only)
                            with open(content_store_path / "embedding_stats.json", 'r') as f:
                                stats_data = json.load(f)
                                if "processed_files" in stats_data:
                                    already_processed_pdfs = set(stats_data["processed_files"])
                                    logger.info(f"Found {len(already_processed_pdfs)} already processed PDFs - will check for duplicates but still process critical content")
                                    check_for_duplicates = True
                        except Exception as stats_err:
                            logger.warning(f"Could not read embedding stats, will process all PDFs: {stats_err}")
                    
                    skipped_pdfs = 0
                    processed_pdfs = 0
                    
                    for _, row in df.iterrows():
                        try:
                            row_data = self.parse_row_data(row, "materials")
                            if 'file_path' in row_data and row_data['file_path']:
                                pdf_path = self.base_path / row_data['file_path']
                                pdf_path_str = str(pdf_path)
                                
                                # CRITICAL CONTENT PRIORITY: Always process PDFs from materials.csv
                                # Only skip if we can confirm the exact same file was already processed
                                should_process = True
                                
                                if check_for_duplicates and pdf_path_str in already_processed_pdfs:
                                    # For efficiency, we can skip this PDF if it was already processed
                                    # But log this as an optimization skip, not a critical content skip
                                    logger.info(f"PDF already in vector store (optimization skip): {pdf_path}")
                                    skipped_pdfs += 1
                                    should_process = False
                                
                                if should_process and pdf_path.exists() and pdf_path.suffix.lower() == '.pdf':
                                    logger.info(f"Processing PDF: {pdf_path}")
                                    try:
                                        loader = PyPDFLoader(pdf_path_str)
                                        docs = loader.load()
                                        for doc in docs:
                                            doc.metadata.update({
                                                "source_type": "pdf",
                                                "material_id": row_data.get('id', ''),
                                                "material_title": row_data.get('title', ''),
                                                "file_path": pdf_path_str
                                            })
                                            pdf_docs.append(doc)
                                        processed_pdfs += 1
                                        
                                        # Add to processed files list for tracking
                                        already_processed_pdfs.add(pdf_path_str)
                                    except Exception as pe:
                                        logger.error(f"Error processing PDF {pdf_path}: {pe}")
                        except Exception as e:
                            logger.error(f"Error processing material: {e}")
                    
                    logger.info(f"PDF processing summary: {processed_pdfs} processed, {skipped_pdfs} skipped (optimization: already vectorized)")
                except Exception as e:
                    logger.error(f"Error reading materials.csv: {e}")
            
            # Create vector stores for critical content
            if methods_docs:
                logger.info(f"Creating vector store for {len(methods_docs)} method documents")
                await self.create_vector_store(methods_docs, "reference_store", incremental=True)
            
            if pdf_docs:
                logger.info(f"Creating vector store for {len(pdf_docs)} PDF documents")
                await self.create_vector_store(pdf_docs, "content_store", incremental=True)
                
            return {
                "methods_processed": len(methods_docs),
                "pdfs_processed": len(pdf_docs)
            }
            
        except Exception as e:
            logger.error(f"Error processing critical content: {e}")
            raise

    async def process_knowledge_base(self, incremental=False, skip_vector_generation=False) -> dict:
        """
        Process knowledge base documents and create or update FAISS index
        
        Args:
            incremental: If True, will try to update existing vector stores instead of recreating them
                         and only process files that have changed since last run
            skip_vector_generation: If True, skip vector generation (will be handled at a higher level)
        """
        try:
            logger.info(f"Starting knowledge base processing from {self.base_path} (incremental={incremental})")
            
            # Initialize result variable
            result = {}

            reference_docs = []
            content_docs = []
            stats = {"total": 0, "processed": 0, "skipped": 0, "failed": 0, "by_type": {}}

            # Initialize processor outside the try block
            processor = None

            # Get last processed timestamp if incremental mode
            last_processed_time = None
            if incremental:
                stats_path = self.base_path / "stats" / "vector_stats.json"
                if stats_path.exists():
                    try:
                        with open(stats_path, 'r') as f:
                            previous_stats = json.load(f)
                            last_updated = previous_stats.get("last_updated")
                            if last_updated:
                                last_processed_time = datetime.fromisoformat(last_updated)
                                logger.info(f"Last processing time: {last_processed_time}")
                    except Exception as e:
                        logger.warning(f"Could not read last processing time: {e}")

            # Process each data type with specific formatting
            data_types = {
                "definitions": (self.base_path / "definitions.csv", 
                              lambda row: f"Term: {row.get('term', '')}\nDefinition: {row.get('definition', '')}"),
                "methods": (self.base_path / "methods.csv",
                          lambda row: f"Method: {row.get('title', '')}\nDescription: {row.get('description', '')}\nSteps: {row.get('steps', '')}"),
                "materials": (self.base_path / "materials.csv",
                          lambda row: f"Material: {row.get('title', '')}\nDescription: {row.get('description', '')}"),
                "projects": (self.base_path / "projects.csv",
                          lambda row: f"Project: {row.get('title', '')}\nDescription: {row.get('description', '')}\nGoals: {row.get('goals', '')}\nAchievement: {row.get('achievement', '')}"),
                "resources": (self.base_path / "resources.csv",
                          lambda row: f"Resource: {row.get('title', '')}\nURL: {row.get('origin_url', '')}\nDescription: {row.get('description', '')}\nTags: {row.get('tags', '')}")
            }

            # Keep track of resources that were successfully processed
            # Store file_path and resource info to update the flags after processing
            processed_resources = []

            for data_type, (file_path, formatter) in data_types.items():
                logger.info(f"Looking for {data_type} at {file_path}")
                if not file_path.exists():
                    continue

                try:
                    # Check if file has been modified since last processing
                    file_modified = True
                    if incremental and last_processed_time:
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_mtime < last_processed_time:
                            logger.info(f"Skipping {data_type} - file not modified since last processing")
                            file_modified = False
                    
                    df = pd.read_csv(file_path)
                    stats["by_type"][data_type] = len(df)
                    stats["total"] += len(df)
                    logger.info(f"Loaded {len(df)} {data_type} entries")

                    # Skip processing if file hasn't been modified and we're in incremental mode
                    if not file_modified and incremental:
                        logger.info(f"Skipping processing of {len(df)} unchanged {data_type} entries")
                        stats["skipped"] += len(df)
                        continue

                    for _, row in df.iterrows():
                        try:
                            # Parse row data safely
                            row_data = self.parse_row_data(row, data_type)
                            
                            # Check for processing flags in different data types
                            skip_row = False
                            
                            # For resources, check analysis_completed flag - only process new/incomplete
                            if data_type == "resources" and incremental:
                                if "analysis_completed" in row and pd.notna(row["analysis_completed"]):
                                    if row["analysis_completed"] == True:
                                        logger.debug(f"Skipping already analyzed resource: {row.get('title', 'Unnamed')}")
                                        stats["skipped"] += 1
                                        skip_row = True
                            
                            # For methods, check processed flag if it exists
                            elif data_type == "methods" and incremental:
                                if "processed" in row and pd.notna(row["processed"]):
                                    if row["processed"] == True:
                                        logger.debug(f"Skipping already processed method: {row.get('title', 'Unnamed')}")
                                        stats["skipped"] += 1
                                        skip_row = True
                                        
                            # For projects, check processed flag if it exists
                            elif data_type == "projects" and incremental:
                                if "processed" in row and pd.notna(row["processed"]):
                                    if row["processed"] == True:
                                        logger.debug(f"Skipping already processed project: {row.get('title', 'Unnamed')}")
                                        stats["skipped"] += 1
                                        skip_row = True
                                        
                            # For definitions, check processed flag if it exists
                            elif data_type == "definitions" and incremental:
                                if "processed" in row and pd.notna(row["processed"]):
                                    if row["processed"] == True:
                                        logger.debug(f"Skipping already processed definition: {row.get('term', 'Unnamed')}")
                                        stats["skipped"] += 1
                                        skip_row = True
                                        
                            # Skip this row if flagged
                            if skip_row:
                                continue
                                
                            # Format content for vector store
                            content = formatter(row_data)
                            
                            doc = Document(
                                page_content=content,
                                metadata={
                                    "source_type": data_type,
                                    "csv_file": file_path.name,
                                    "processed_at": datetime.now().isoformat(),
                                    **row_data  # Include all row data as metadata
                                }
                            )
                            
                            # Categorize documents by type - resources go to content_docs, most others to reference_docs
                            if data_type == "resources":
                                content_docs.append(doc)
                                # Store information for later instead of setting flag now
                                processed_resources.append({
                                    "file_path": file_path,
                                    "row_data": row_data
                                })
                            else:
                                reference_docs.append(doc)
                                
                            stats["processed"] += 1

                        except Exception as row_error:
                            logger.error(f"Error processing row in {data_type}: {row_error}")
                            stats["failed"] += 1

                except Exception as e:
                    logger.error(f"Error processing {data_type}: {e}")
                    stats["failed"] += 1

            # Create or update vector stores
            if reference_docs:
                logger.info(f"Creating/updating reference store with {len(reference_docs)} documents")
                await self.create_vector_store(reference_docs, "reference_store", incremental=incremental, skip_vector_generation=skip_vector_generation)
            else:
                logger.info("No reference documents to process")

            if content_docs:
                logger.info(f"Creating/updating content store with {len(content_docs)} documents")
                await self.create_vector_store(content_docs, "content_store", incremental=incremental, skip_vector_generation=skip_vector_generation)
            else:
                logger.info("No content documents to process")

            # Update stats
            vector_stats = {
                "total": stats["total"],
                "vectorized": stats["processed"],
                "skipped": stats.get("skipped", 0),
                "pending": stats["failed"],
                "reference_docs": len(reference_docs),
                "content_docs": len(content_docs),
                "embedding_stats": self.embedding_stats,
                "last_updated": datetime.now().isoformat()
            }

            stats_path = self.base_path / "stats"
            stats_path.mkdir(exist_ok=True)
            with open(stats_path / "vector_stats.json", 'w') as f:
                json.dump(vector_stats, f, indent=2)

            # Generate vector stores at the end of the full process if not explicitly skipped
            if not skip_vector_generation:
                logger.info("Generating vector stores after complete knowledge processing")
                try:
                    # Get all content files for vector generation
                    from scripts.analysis.vector_generator import VectorGenerator
                    import asyncio
                    
                    # Get temp storage path from ContentStorageService
                    from scripts.storage.content_storage_service import ContentStorageService
                    content_storage = ContentStorageService(str(self.base_path))
                    
                    # Look for JSONL files in multiple locations
                    content_paths = list(content_storage.temp_storage_path.glob("*.jsonl"))
                    temp_dir = self.base_path / "temp"
                    if (temp_dir.exists()):
                        content_paths.extend(list(temp_dir.glob("*.jsonl")))
                        
                    if content_paths:
                        logger.info(f"Found {len(content_paths)} content files for vector generation")
                        
                        # Pass chunking parameters to VectorGenerator
                        vector_generator = VectorGenerator(
                            str(self.base_path),
                            chunk_size=self.text_splitter.chunk_size,
                            chunk_overlap=self.text_splitter.chunk_overlap
                        )
                        vector_stats = await vector_generator.generate_vector_stores(content_paths)
                        
                        # Add vector stats to the result
                        result["vector_generation"] = vector_stats
                        logger.info(f"Vector generation complete: {vector_stats}")
                    else:
                        logger.info("No content files found for vector generation")
                        result["vector_generation"] = {"status": "skipped", "reason": "no content files found"}
                except Exception as ve:
                    logger.error(f"Error during vector generation: {ve}")
                    result["vector_generation"] = {"status": "error", "error": str(ve)}

                # Clean up all temporary files after vector generation
                try:
                    # Use self's cleanup method instead of processor
                    self._cleanup_temporary_files()
                except Exception as ce:
                    logger.error(f"Error during cleanup: {ce}")
                
                # Now mark resources as processed AFTER vector generation and cleanup
                # This ensures the flag is only set if the entire process completes
                logger.info(f"Marking {len(processed_resources)} processed resources as completed")
                for resource in processed_resources:
                    file_path = resource["file_path"]
                    row_data = resource["row_data"]
                    
                    try:
                        # Get the original DataFrame to update it
                        resources_df = pd.read_csv(file_path)
                        
                        # Find the row by ID or URL
                        match_found = False
                        if "id" in row_data and pd.notna(row_data["id"]):
                            res_idx = resources_df.index[resources_df["id"] == row_data["id"]].tolist()
                            if res_idx:
                                match_found = True
                                # Add processed column if it doesn't exist
                                if "processed" not in resources_df.columns:
                                    resources_df["processed"] = False
                                resources_df.at[res_idx[0], "processed"] = True
                                
                                # Set analysis_completed to True as well
                                if "analysis_completed" not in resources_df.columns:
                                    resources_df["analysis_completed"] = False
                                resources_df.at[res_idx[0], "analysis_completed"] = True
                                logger.info(f"Marked resource with ID {row_data['id']} as analysis_completed=True")
                        
                        elif "url" in row_data and pd.notna(row_data["url"]):
                            res_idx = resources_df.index[resources_df["url"] == row_data["url"]].tolist()
                            if res_idx:
                                match_found = True
                                # Add processed column if it doesn't exist
                                if "processed" not in resources_df.columns:
                                    resources_df["processed"] = False
                                resources_df.at[res_idx[0], "processed"] = True
                                
                                # Set analysis_completed to True as well
                                if "analysis_completed" not in resources_df.columns:
                                    resources_df["analysis_completed"] = False
                                resources_df.at[res_idx[0], "analysis_completed"] = True
                                logger.info(f"Marked resource with URL {row_data['origin_url']} as analysis_completed=True")
                        
                        # Save the updated DataFrame if changes were made
                        if match_found:
                            resources_df.to_csv(file_path, index=False)
                        else:
                            logger.warning(f"Could not find resource to update analysis_completed flag: {row_data.get('title', 'Unnamed')}")
                            
                    except Exception as save_err:
                        logger.error(f"Error saving resource completion state: {save_err}")
            else:
                logger.info("Vector generation explicitly skipped")
                result["vector_generation"] = {"status": "skipped", "reason": "explicitly skipped"}

            return {
                "status": "success",
                "stats": stats,
                "vector_stats": vector_stats
            }

        except Exception as e:
            logger.error(f"Error processing knowledge base: {e}")
            
            # Ensure result is defined even if an exception occurs
            if not result:
                result = {}
            
            return {
                "status": "error",
                "error": str(e),
                "vector_stats": result.get("vector_generation", {})
            }

    async def create_vector_store(self, documents: List[Document], store_name: str, incremental=False, skip_vector_generation=False) -> FAISS:
        """
        Create or update FAISS vector store
        
        Args:
            documents: List of documents to add to vector store
            store_name: Name of the vector store ('reference_store' or 'content_store')
            incremental: If True, try to add to existing index instead of recreating
            skip_vector_generation: If True, skip immediate vector generation (will be handled later)
        """
        try:
            # Save documents to JSONL file for consistent vector processing
            temp_json_path = self.base_path / "temp" / f"{store_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            temp_json_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save documents to JSONL
            logger.info(f"Saving {len(documents)} documents to temporary file for vector generation")
            with open(temp_json_path, 'w', encoding='utf-8') as f:
                for doc in documents:
                    doc_dict = {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    f.write(json.dumps(doc_dict) + '\n')
            
            # Store document metadata for reference
            documents_file = self.base_path / "vector_stores" / "default" / store_name / "documents.json"
            
            # Create directory if it doesn't exist
            documents_file.parent.mkdir(parents=True, exist_ok=True)
            
            doc_data = []
            for doc in documents:
                doc_data.append({
                    "content": doc.page_content, 
                    "metadata": doc.metadata
                })
            
            with open(documents_file, 'w') as f:
                json.dump(doc_data, f)
            
            logger.info(f"Saved {len(documents)} documents to {documents_file} for reference")
            
            # Defer actual vector generation to VectorGenerator
            logger.info(f"Deferring vector generation for {len(documents)} documents to VectorGenerator")
            
            # Create or update metadata
            stats_data = {}
            if (self.base_path / "vector_stores" / "default" / store_name / "embedding_stats.json").exists():
                try:
                    with open(self.base_path / "vector_stores" / "default" / store_name / "embedding_stats.json", 'r') as f:
                        stats_data = json.load(f)
                except Exception:
                    pass
            
            # Track processed files, especially PDFs
            processed_files = stats_data.get("processed_files", [])
            
            # Add any new PDF files to the processed list
            pdf_files = set()
            for doc in documents:
                if doc.metadata.get("source_type") == "pdf" and "file_path" in doc.metadata:
                    pdf_files.add(doc.metadata["file_path"])
            
            # Update processed files list with new PDFs
            if pdf_files:
                processed_files = list(set(processed_files).union(pdf_files))
                logger.info(f"Added {len(pdf_files)} new PDF files to processed files tracking")
            
            # Update stats
            stats_data.update({
                "document_count": stats_data.get("document_count", 0) + len(documents),
                "pending_documents": stats_data.get("pending_documents", 0) + len(documents),
                "updated_at": datetime.now().isoformat(),
                "created_at": stats_data.get("created_at", datetime.now().isoformat()),
                "last_updated": datetime.now().isoformat(),
                "processed_files": processed_files,
                "store_name": store_name
            })
            
            # Save embedding stats alongside the vector store
            with open(self.base_path / "vector_stores" / "default" / store_name / "embedding_stats.json", 'w') as f:
                json.dump(stats_data, f, indent=2)
            
            logger.info(f"Vector generation for {store_name} will be handled by VectorGenerator")
            
            # Import and run vector generator to immediately process the file if requested
            if not skip_vector_generation and 'IMMEDIATE_VECTOR_GENERATION' in os.environ and os.environ['IMMEDIATE_VECTOR_GENERATION'] == 'true':
                from scripts.analysis.vector_generator import VectorGenerator
                import asyncio
                
                logger.info(f"IMMEDIATE_VECTOR_GENERATION enabled - generating vectors now")
                vector_generator = VectorGenerator(str(self.base_path))
                vector_stats = await vector_generator.generate_vector_stores([temp_json_path])
                logger.info(f"Immediate vector generation complete: {vector_stats}")
                
                # Return a mock FAISS object with just enough attributes to satisfy the interface
                class MockFAISS:
                    def save_local(self, *args, **kwargs):
                        pass
                        
                return MockFAISS()
            else:
                # Skip vector generation message if explicitly skipped
                if skip_vector_generation:
                    logger.info("Vector generation skipped - will be handled at a higher level")
                
                # Return a mock FAISS object with just enough attributes to satisfy the interface
                class MockFAISS:
                    def save_local(self, *args, **kwargs):
                        pass
                        
                return MockFAISS()
                
        except Exception as e:
            logger.error(f"Error preparing for vector generation: {e}", exc_info=True)
            raise

    async def rebuild_all_vector_stores(self) -> Dict[str, Any]:
        """
        Rebuild all FAISS indexes from existing document.json files.
        This ensures consistency between document content and vector indexes.
        """
        try:
            # Get all vector store paths
            vector_path = self.base_path / "vector_stores" / "default"
            if not vector_path.exists():
                logger.warning(f"Vector store path not found: {vector_path}")
                return {"status": "error", "message": "Vector store path not found"}
                
            # Identify all stores (reference_store, content_store, and topic_*)
            store_paths = [p for p in vector_path.iterdir() if p.is_dir()]
            logger.info(f"Found {len(store_paths)} vector stores to rebuild")
            
            # Process each store
            for store_path in store_paths:
                store_name = store_path.name
                documents_file = store_path / "documents.json"
                
                # Skip if documents.json doesn't exist
                if not documents_file.exists():
                    logger.warning(f"Store {store_name} missing documents.json, skipping")
                    continue
                    
                try:
                    # Load documents from JSON
                    with open(documents_file, 'r') as f:
                        doc_data = json.load(f)
                        
                    # Convert JSON documents back to langchain Document objects
                    documents = []
                    for doc_entry in doc_data:
                        doc = Document(
                            page_content=doc_entry["content"],
                            metadata=doc_entry["metadata"]
                        )
                        documents.append(doc)
                        
                    # Generate embeddings for all documents
                    texts = [doc.page_content for doc in documents]
                    metadatas = [doc.metadata for doc in documents]
                    
                    # Generate embeddings in batches
                    embeddings_list = await self.embeddings.aembeddings(texts)
                    
                    # Create new FAISS index
                    vector_store = FAISS.from_embeddings(
                        text_embeddings=list(zip(texts, embeddings_list)),
                        embedding=self.embeddings,
                        metadatas=metadatas
                    )
                    
                    # Save the rebuilt vector store
                    vector_store.save_local(str(store_path))
                    
                except Exception as e:
                    logger.error(f"Error rebuilding store {store_name}: {e}")
        
        except Exception as e:
            logger.error(f"Error in rebuild_all_vector_stores: {e}")

    def get_vector_store_path(self) -> Path:
        """Get path for vector store"""
        return self.base_path / "vector_stores" / "default" / "faiss_index"

    def save_catalog(self) -> bool:
        """Save processed data catalog to JSON"""
        try:
            catalog_path = self.base_path / "catalog.json"
            with open(catalog_path, 'w') as f:
                json.dump(self.processed_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving catalog: {e}")
            return False

    def _cleanup_temporary_files(self):
        """Clean up temporary files after processing"""
        try:
            # Clean up temp directory
            temp_dir = self.base_path / "temp"
            if (temp_dir.exists()):
                for temp_file in temp_dir.glob("*.jsonl"):
                    try:
                        temp_file.unlink()
                        logger.debug(f"Deleted temporary file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Could not delete {temp_file}: {e}")
                        
            # Also clean ContentStorageService temp directory
            from scripts.storage.content_storage_service import ContentStorageService
            content_storage = ContentStorageService(str(self.base_path))
            if content_storage.temp_storage_path.exists():
                for temp_file in content_storage.temp_storage_path.glob("*.jsonl"):
                    try:
                        temp_file.unlink()
                        logger.debug(f"Deleted temporary content file: {temp_file}")
                    except Exception as e:
                        logger.warning(f"Could not delete content temp file {temp_file}: {e}")
                        
            logger.info("Cleaned up all temporary files")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")
