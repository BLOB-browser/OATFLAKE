#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
import os
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class DataSaver:
    """
    Responsible for saving processed data (resources, definitions, projects)
    to CSV files and handling data persistence.
    """
    
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
    
    def save_resources(self, resources: List[Dict], csv_path: str = None) -> None:
        """
        Saves resources to CSV.
        
        Args:
            resources: List of resource dictionaries
            csv_path: Path to save CSV file (defaults to original location)
        """
        if not resources:
            logger.warning("No resources to save")
            return
        
        if csv_path is None:
            csv_path = os.path.join(self.data_folder, 'resources.csv')
        
        try:
            # Convert list objects to JSON strings for CSV storage
            processed_resources = []
            for resource in resources:
                resource_copy = resource.copy()
                for key, value in resource_copy.items():
                    # Handle lists by converting to JSON strings
                    if isinstance(value, list):
                        resource_copy[key] = json.dumps(value)
                    # Special handling for analysis_completed flag
                    elif key == 'analysis_completed':
                        if value in [True, 'True', 'true', 'TRUE', 'T', 't', '1', 1]:
                            resource_copy[key] = True
                        else:
                            resource_copy[key] = False
                processed_resources.append(resource_copy)
            
            # Do NOT propagate or override existing analysis_completed values
            # When new resources are saved, they should retain their current state
            # This allows failed/incomplete resources to keep analysis_completed=False
            
            logger.info("Saving resources with their current analysis_completed status...")
            
            # For logging purposes only - count completed resources
            completed_count = sum(1 for r in processed_resources if r.get('analysis_completed') == True)
            logger.info(f"Saving {len(processed_resources)} resources ({completed_count} have analysis_completed=True)")
            
            # Create DataFrame directly from processed resources - keeping their current status
            df = pd.DataFrame(processed_resources)
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {len(resources)} resources to {csv_path}")
        except Exception as e:
            logger.error(f"Error saving resources CSV: {e}")
    
    def save_single_resource(self, resource: Dict, csv_path: str, idx: int) -> None:
        """
        Update a single resource in the CSV file without loading the entire file.
        
        Args:
            resource: Resource dictionary to save
            csv_path: Path to the CSV file
            idx: Index of the resource in the CSV
        """
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Update the specific row
            if idx < len(df):
                # Handle tags specially to ensure proper format
                resource_copy = resource.copy()
                for key, value in resource_copy.items():
                    if isinstance(value, list):
                        resource_copy[key] = json.dumps(value)
                
                # Update the row with values from resource_copy
                for key, value in resource_copy.items():
                    if key in df.columns:
                        # Special handling for analysis_completed to ensure it's properly saved as boolean
                        if key == 'analysis_completed':
                            if value in [True, 'True', 'true', 'TRUE', 'T', 't', '1', 1]:
                                df.at[idx, key] = True
                            else:
                                df.at[idx, key] = False
                        else:
                            df.at[idx, key] = value
                
                # Save the entire dataframe back
                df.to_csv(csv_path, index=False)
                logger.info(f"Updated resource #{idx+1} in {csv_path}")
            else:
                logger.error(f"Index {idx} out of bounds for DataFrame with {len(df)} rows")
        except Exception as e:
            logger.error(f"Error saving individual resource: {e}")
    
    def save_definitions(self, definitions: List[Dict], csv_path: str = None) -> None:
        """
        Saves definitions to CSV with duplicate handling.
        
        Args:
            definitions: List of definition dictionaries
            csv_path: Path to save CSV file
        """
        if not definitions:
            logger.info("No definitions to save")
            return
        
        if csv_path is None:
            csv_path = os.path.join(self.data_folder, 'definitions.csv')
        
        try:
            # Process definitions, ensuring tags are properly formatted
            processed_definitions = []
            for d in definitions:
                d_copy = d.copy()
                if 'tags' in d_copy and isinstance(d_copy['tags'], list):
                    d_copy['tags'] = json.dumps(d_copy['tags'])
                if 'created_at' not in d_copy:
                    d_copy['created_at'] = datetime.now().isoformat()
                processed_definitions.append(d_copy)
            
            df = pd.DataFrame(processed_definitions)
            
            # Handle combining with existing data if file exists
            file_exists = os.path.isfile(csv_path)
            if file_exists:
                # Read existing to avoid duplicates
                existing_df = pd.read_csv(csv_path)
                combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['term'])
                combined_df.to_csv(csv_path, index=False)
                logger.info(f"Updated definitions in {csv_path} (total: {len(combined_df)})")
            else:
                df.to_csv(csv_path, index=False)
                logger.info(f"Created {csv_path} with {len(df)} definitions")
                
        except Exception as e:
            logger.error(f"Error saving definitions CSV: {e}")
    
    def save_projects(self, projects: List[Dict], csv_path: str = None) -> None:
        """
        Saves projects to CSV with duplicate handling.
        
        Args:
            projects: List of project dictionaries
            csv_path: Path to save CSV file
        """
        if not projects:
            logger.info("No projects to save")
            return
        
        if csv_path is None:
            csv_path = os.path.join(self.data_folder, 'projects.csv')
        
        try:
            # Process projects, ensuring fields are properly formatted
            processed_projects = []
            for p in projects:
                p_copy = p.copy()
                if 'fields' in p_copy and isinstance(p_copy['fields'], list):
                    p_copy['fields'] = json.dumps(p_copy['fields'])
                if 'created_at' not in p_copy:
                    p_copy['created_at'] = datetime.now().isoformat()
                processed_projects.append(p_copy)
            
            df = pd.DataFrame(processed_projects)
            
            # Handle combining with existing data if file exists
            file_exists = os.path.isfile(csv_path)
            if file_exists:
                # Read existing to avoid duplicates
                existing_df = pd.read_csv(csv_path)
                # Use title + source_url as a composite key to avoid duplicates
                combined_df = pd.concat([existing_df, df])
                if 'title' in combined_df.columns and 'source_url' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['title', 'source_url'])
                else:
                    combined_df = combined_df.drop_duplicates()
                combined_df.to_csv(csv_path, index=False)
                logger.info(f"Updated projects in {csv_path} (total: {len(combined_df)})")
            else:
                df.to_csv(csv_path, index=False)
                logger.info(f"Created {csv_path} with {len(df)} projects")
                
        except Exception as e:
            logger.error(f"Error saving projects CSV: {e}")
            
    def save_methods(self, methods: List[Dict], csv_path: str = None) -> None:
        """
        Saves methods to CSV with duplicate handling.
        
        Args:
            methods: List of method dictionaries
            csv_path: Path to save CSV file
        """
        if not methods:
            logger.info("No methods to save")
            return
        
        if csv_path is None:
            csv_path = os.path.join(self.data_folder, 'methods.csv')
        
        try:
            # Process methods, ensuring steps and tags are properly formatted
            processed_methods = []
            for m in methods:
                m_copy = m.copy()
                
                # Format steps as JSON if they're a list
                if 'steps' in m_copy and isinstance(m_copy['steps'], list):
                    m_copy['steps'] = json.dumps(m_copy['steps'])
                
                # Format tags as JSON if they're a list
                if 'tags' in m_copy and isinstance(m_copy['tags'], list):
                    m_copy['tags'] = json.dumps(m_copy['tags'])
                
                # Set default values required for the Method model
                if 'group_id' not in m_copy:
                    m_copy['group_id'] = 'default'
                    
                if 'creator_id' not in m_copy:
                    m_copy['creator_id'] = 'system'
                    
                if 'created_at' not in m_copy:
                    m_copy['created_at'] = datetime.now().isoformat()
                    
                # Set status flag for method
                if 'status' not in m_copy:
                    m_copy['status'] = 'active'
                    
                processed_methods.append(m_copy)
            
            df = pd.DataFrame(processed_methods)
            
            # Handle combining with existing data if file exists
            file_exists = os.path.isfile(csv_path)
            if file_exists:
                # Read existing to avoid duplicates
                existing_df = pd.read_csv(csv_path)
                # Use title + source as a composite key to avoid duplicates
                combined_df = pd.concat([existing_df, df])
                if 'title' in combined_df.columns and 'source' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['title', 'source'])
                else:
                    combined_df = combined_df.drop_duplicates()
                combined_df.to_csv(csv_path, index=False)
                logger.info(f"Updated methods in {csv_path} (total: {len(combined_df)})")
            else:
                df.to_csv(csv_path, index=False)
                logger.info(f"Created {csv_path} with {len(df)} methods")
                
        except Exception as e:
            logger.error(f"Error saving methods CSV: {e}")
            
    def save_to_vector_store(self, data_type: str, items: List[Dict]) -> None:
        """
        Immediately save extracted items to vector store.
        
        Args:
            data_type: Type of data ("definitions", "projects", "methods")
            items: List of items to save
        """
        if not items:
            logger.info(f"No {data_type} to save to vector store")
            return
            
        try:
            from scripts.storage.vector_store_manager import VectorStoreManager
            from langchain.schema import Document
            from pathlib import Path
            
            # Initialize vector store manager
            vector_store_manager = VectorStoreManager(base_path=Path(self.data_folder))
            
            # Convert items to documents with proper metadata
            documents = []
            for item in items:
                # Format content based on data type
                if data_type == "definitions":
                    content = f"TERM: {item.get('term', '')}\nDEFINITION: {item.get('definition', '')}"
                elif data_type == "projects":
                    content = f"PROJECT: {item.get('title', '')}\nDESCRIPTION: {item.get('description', '')}\nGOALS: {item.get('goals', '')}"
                elif data_type == "methods":
                    steps = item.get('steps', [])
                    if isinstance(steps, str):
                        try:
                            steps = json.loads(steps)
                        except:
                            steps = [steps]
                    steps_text = "\n".join([f"STEP {i+1}: {step}" for i, step in enumerate(steps)]) if isinstance(steps, list) else steps
                    content = f"METHOD: {item.get('title', '')}\nDESCRIPTION: {item.get('description', '')}\nSTEPS:\n{steps_text}"
                else:
                    content = json.dumps(item)
                
                # Define metadata
                metadata = {
                    "source_type": data_type,
                    "processed_at": datetime.now().isoformat(),
                    **{k: v for k, v in item.items() if isinstance(v, (str, int, float, bool)) or (isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v))}
                }
                
                # Create document
                documents.append(Document(page_content=content, metadata=metadata))
            
            # Add documents to the appropriate store
            store_type = "reference_store" if data_type in ["definitions", "methods", "projects"] else "content_store"
            
            # Add to store (use async method but run it synchronously)
            import asyncio
            
            try:
                # Check if we're already in an event loop
                loop = asyncio.get_running_loop()
                # We're in an event loop, use a different approach
                logger.warning(f"Running vector store update inside existing event loop - creating new one")
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                
                # Create metadata for better tracking
                metadata = {
                    "data_type": data_type,
                    "item_count": len(items),
                    "source": "data_saver",
                    "saved_at": datetime.now().isoformat()
                }
                
                loop.run_until_complete(vector_store_manager.add_documents_to_store(
                    store_type, 
                    documents,
                    metadata=metadata,
                    update_stats=True
                ))
            except RuntimeError:
                # No running event loop, use the traditional approach
                # Create metadata for better tracking
                metadata = {
                    "data_type": data_type,
                    "item_count": len(items),
                    "source": "data_saver",
                    "saved_at": datetime.now().isoformat()
                }
                
                asyncio.run(vector_store_manager.add_documents_to_store(
                    store_type, 
                    documents,
                    metadata=metadata,
                    update_stats=True
                ))
            except Exception as e:
                # Handle any other exceptions
                logger.error(f"Error saving {data_type} to vector store ({store_type}): {e}", exc_info=True)
                
            logger.info(f"Successfully saved {len(documents)} {data_type} to vector store ({store_type})")
            
        except Exception as e:
            logger.error(f"Error saving {data_type} to vector store: {e}", exc_info=True)