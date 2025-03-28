from pathlib import Path
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader, TextLoader, UnstructuredFileLoader, PyPDFLoader
from langchain.schema import Document

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles loading documents from various sources."""
    
    def __init__(self, base_path: Path):
        """
        Initialize document loader.
        
        Args:
            base_path: Base path where data files are located
        """
        self.base_path = Path(base_path)
    
    def parse_row_data(self, row: pd.Series, data_type: str) -> dict:
        """
        Safely parse row data from pandas Series.
        
        Args:
            row: Row data from pandas DataFrame
            data_type: Type of data being processed ('definitions', 'methods', etc.)
            
        Returns:
            Dictionary with parsed row data
        """
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
        """
        Process PDF files referenced in materials.
        
        Args:
            materials_path: Path to materials directory
            row: Row data containing PDF file reference
            
        Returns:
            List of Document objects from the PDF
        """
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
    
    def load_definitions(self, incremental=False, last_processed_time=None) -> List[Document]:
        """
        Load definitions from CSV file.
        
        Args:
            incremental: Whether to only process new or updated files
            last_processed_time: Timestamp of last processing
            
        Returns:
            List of Document objects with definitions
        """
        documents = []
        file_path = self.base_path / "definitions.csv"
        
        if not file_path.exists():
            logger.info(f"Definitions file not found: {file_path}")
            return documents
        
        try:
            # Check if file has been modified since last processing
            if incremental and last_processed_time:
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < last_processed_time:
                    logger.info(f"Skipping definitions - file not modified since last processing")
                    return documents
            
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} definitions")
            
            for _, row in df.iterrows():
                try:
                    row_data = self.parse_row_data(row, "definitions")
                    
                    # Skip already processed definitions in incremental mode
                    if incremental and "processed" in row and pd.notna(row["processed"]) and row["processed"] == True:
                        logger.debug(f"Skipping already processed definition: {row_data.get('term', 'Unnamed')}")
                        continue
                    
                    content = f"Term: {row_data.get('term', '')}\nDefinition: {row_data.get('definition', '')}"
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source_type": "definitions",
                            "csv_file": file_path.name,
                            "processed_at": datetime.now().isoformat(),
                            **row_data
                        }
                    )
                    documents.append(doc)
                    
                except Exception as row_error:
                    logger.error(f"Error processing definition row: {row_error}")
            
            return documents
        except Exception as e:
            logger.error(f"Error loading definitions: {e}")
            return documents
    
    # Similar methods for other data types (methods, projects, resources)
    # would be implemented here following the same pattern