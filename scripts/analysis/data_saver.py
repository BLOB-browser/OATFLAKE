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
        Saves resources to CSV using the universal table structure.
        Sets sensible defaults for required fields when not provided.
        
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
            # Set default timestamp for use in multiple fields
            current_time = datetime.now().isoformat()
            
            # Convert list objects to JSON strings for CSV storage and add missing fields
            processed_resources = []
            for resource in resources:
                resource_copy = resource.copy()
                
                # Add required universal fields with sensible defaults if missing
                # 1. ID field - generate if missing
                if 'id' not in resource_copy:
                    import uuid
                    resource_copy['id'] = str(uuid.uuid4())
                
                # 2. Content type
                if 'content_type' not in resource_copy:
                    resource_copy['content_type'] = 'resource'
                
                # 3. Origin URL (from 'url' for backwards compatibility)
                if 'origin_url' not in resource_copy and 'url' in resource_copy:
                    resource_copy['origin_url'] = resource_copy['url']
                
                # 4. Status, visibility, timestamps
                if 'status' not in resource_copy:
                    resource_copy['status'] = 'active'
                if 'visibility' not in resource_copy:
                    resource_copy['visibility'] = 'public'
                if 'created_at' not in resource_copy:
                    resource_copy['created_at'] = current_time
                if 'last_updated_at' not in resource_copy:
                    resource_copy['last_updated_at'] = current_time
                
                # 5. Convert any list objects to JSON strings for CSV storage
                for key, value in resource_copy.items():
                    if isinstance(value, list):
                        resource_copy[key] = json.dumps(value)
                    # Special handling for analysis_completed flag
                    elif key == 'analysis_completed':
                        if value in [True, 'True', 'true', 'TRUE', 'T', 't', '1', 1]:
                            resource_copy[key] = True
                        else:
                            resource_copy[key] = False
                
                # 6. Ensure analysis_completed is present
                if 'analysis_completed' not in resource_copy:
                    resource_copy['analysis_completed'] = False
                
                processed_resources.append(resource_copy)
            
            # For logging purposes only - count completed resources
            completed_count = sum(1 for r in processed_resources if r.get('analysis_completed') == True)
            logger.info(f"Saving {len(processed_resources)} resources ({completed_count} have analysis_completed=True)")
            
            # Handle existing file - merge if exists
            if os.path.exists(csv_path):
                try:
                    # Read existing file
                    existing_df = pd.read_csv(csv_path)
                    
                    # Create dataframe from new resources
                    new_df = pd.DataFrame(processed_resources)
                    
                    # If we have IDs in both, use that for merging
                    if 'id' in existing_df.columns and 'id' in new_df.columns:
                        # Get existing IDs
                        existing_ids = set(existing_df['id'].astype(str))
                        
                        # Split new resources into updates and inserts
                        updates = new_df[new_df['id'].astype(str).isin(existing_ids)]
                        inserts = new_df[~new_df['id'].astype(str).isin(existing_ids)]
                        
                        # For updates, replace existing rows with new data
                        if not updates.empty:
                            # Remove the rows we'll be updating
                            existing_df = existing_df[~existing_df['id'].astype(str).isin(updates['id'].astype(str))]
                            
                        # Combine everything
                        combined_df = pd.concat([existing_df, updates, inserts], ignore_index=True)
                        
                        logger.info(f"Updated {len(updates)} existing resources and added {len(inserts)} new resources")
                    else:
                        # If no ID column, try to merge using URL as unique identifier
                        url_field = 'origin_url' if 'origin_url' in existing_df.columns else 'url'
                        if url_field in existing_df.columns and url_field in new_df.columns:
                            # Remove duplicates based on URL
                            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                            combined_df = combined_df.drop_duplicates(subset=[url_field], keep='last')
                            logger.info(f"Merged resources using {url_field} as unique identifier")
                        else:
                            # Otherwise, just append everything (might create duplicates)
                            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                            logger.warning("No unique identifier found for merging - may create duplicates")
                    
                    # Save the combined dataframe
                    combined_df.to_csv(csv_path, index=False)
                    logger.info(f"Saved {len(combined_df)} total resources to {csv_path}")
                    
                except Exception as merge_err:
                    logger.error(f"Error merging with existing CSV, overwriting: {merge_err}")
                    # If merge fails, fall back to overwriting
                    df = pd.DataFrame(processed_resources)
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Saved {len(processed_resources)} resources to {csv_path} (overwrite)")
            else:
                # No existing file, just save directly
                df = pd.DataFrame(processed_resources)
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {len(processed_resources)} resources to {csv_path} (new file)")
                
        except Exception as e:
            logger.error(f"Error saving resources CSV: {e}", exc_info=True)
    
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
            logger.info(f"DEBUG DEFINITIONS: File {csv_path} exists? {file_exists}")
            if file_exists:
                try:
                    # Read existing to avoid duplicates
                    existing_df = pd.read_csv(csv_path)
                    logger.info(f"DEBUG DEFINITIONS: Loaded existing file with {len(existing_df)} rows")
                    combined_df = pd.concat([existing_df, df]).drop_duplicates(subset=['term'])
                    combined_df.to_csv(csv_path, index=False)
                    logger.info(f"Updated definitions in {csv_path} (total: {len(combined_df)})")
                    
                    # Verify the file was written correctly
                    if os.path.exists(csv_path):
                        file_size = os.path.getsize(csv_path)
                        logger.info(f"DEBUG DEFINITIONS: After write: file exists with size {file_size} bytes")
                    else:
                        logger.warning(f"DEBUG DEFINITIONS: After write: file does not exist!")
                        
                except Exception as e:
                    logger.error(f"DEBUG DEFINITIONS: Error merging definitions: {e}")
                    df.to_csv(csv_path, index=False)
                    logger.info(f"DEBUG DEFINITIONS: Created new file with {len(df)} definitions after error")
            else:
                df.to_csv(csv_path, index=False)
                logger.info(f"DEBUG DEFINITIONS: Created new file {csv_path} with {len(df)} definitions")
                
                # Verify the file was written correctly
                if os.path.exists(csv_path):
                    file_size = os.path.getsize(csv_path)
                    logger.info(f"DEBUG DEFINITIONS: After write: file exists with size {file_size} bytes")
                else:
                    logger.warning(f"DEBUG DEFINITIONS: After write: file does not exist!")
                
        except Exception as e:
            logger.error(f"Error saving definitions CSV: {e}")
    
    def save_projects(self, projects: List[Dict], csv_path: str = None) -> None:
        """
        Saves projects to CSV using the universal table structure.
        Sets sensible defaults for required fields when not provided.
        
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
            # Set default timestamp for use in multiple fields
            current_time = datetime.now().isoformat()
            
            # Process projects, ensuring fields are properly formatted with universal structure
            processed_projects = []
            for p in projects:
                try:
                    p_copy = p.copy()
                    
                    # Debug information for project
                    logger.debug(f"Processing project: {p_copy.get('title', 'Unnamed')}, tags: {p_copy.get('tags', [])}")
                    
                    # Add required universal fields with sensible defaults if missing
                    # 1. ID field - generate if missing
                    if 'id' not in p_copy:
                        import uuid
                        p_copy['id'] = str(uuid.uuid4())
                    
                    # 2. Content type
                    if 'content_type' not in p_copy:
                        p_copy['content_type'] = 'project'
                    
                    # 3. Handle tags/fields conversion (for backward compatibility)
                    if 'tags' not in p_copy and 'fields' in p_copy:
                        # Convert fields to tags for the new universal structure
                        if isinstance(p_copy['fields'], list):
                            p_copy['tags'] = p_copy['fields']
                        elif isinstance(p_copy['fields'], str):
                            # Try to parse JSON string if it's a string
                            try:
                                p_copy['tags'] = json.loads(p_copy['fields'])
                            except Exception as tag_err:
                                logger.warning(f"Error parsing fields JSON: {tag_err}, treating as comma-separated")
                                # If not valid JSON, treat as comma-separated string
                                p_copy['tags'] = [tag.strip() for tag in p_copy['fields'].split(',')]
                    
                    # Ensure tags exists with at least one value
                    if 'tags' not in p_copy or not p_copy['tags']:
                        p_copy['tags'] = ["project"]
                        logger.debug(f"Added default 'project' tag to project '{p_copy.get('title', 'Unnamed')}'")
                    
                    # Make sure title and description are not empty
                    if 'title' not in p_copy or not p_copy['title']:
                        logger.warning(f"Project missing title, skipping: {p_copy}")
                        continue
                        
                    if 'description' not in p_copy or not p_copy['description']:
                        logger.warning(f"Project missing description, providing default for: {p_copy.get('title')}")
                        p_copy['description'] = f"Project related to {p_copy.get('title')}"
                    
                    # 4. Universal fields with defaults
                    if 'status' not in p_copy:
                        p_copy['status'] = 'active'
                    if 'visibility' not in p_copy and 'privacy' in p_copy:
                        # Convert privacy to visibility for backward compatibility
                        p_copy['visibility'] = p_copy['privacy']
                    elif 'visibility' not in p_copy:
                        p_copy['visibility'] = 'public'
                        
                    if 'created_at' not in p_copy:
                        p_copy['created_at'] = current_time
                    if 'last_updated_at' not in p_copy and 'modified_at' in p_copy:
                        p_copy['last_updated_at'] = p_copy['modified_at']
                    elif 'last_updated_at' not in p_copy:
                        p_copy['last_updated_at'] = current_time
                    
                    # 5. Analysis flag
                    if 'analysis_completed' not in p_copy:
                        p_copy['analysis_completed'] = True  # Projects are typically created after analysis
                    
                    # 6. Creator defaults
                    if 'creator_id' not in p_copy:
                        p_copy['creator_id'] = 'system'
                    
                    # 7. Convert lists to JSON strings for CSV storage
                    for key, value in p_copy.items():
                        if isinstance(value, list):
                            p_copy[key] = json.dumps(value)
                    
                    processed_projects.append(p_copy)
                    logger.debug(f"Successfully processed project: {p_copy.get('title')}")
                except Exception as e:
                    logger.error(f"Error processing project: {e}")
                    # Try to log some details about the problematic project
                    try:
                        logger.error(f"Project that caused error: title={p.get('title', 'unknown')}, has_tags={('tags' in p)}")
                    except:
                        pass
            
            df = pd.DataFrame(processed_projects)
            
            # Handle combining with existing data if file exists
            file_exists = os.path.isfile(csv_path)
            logger.info(f"DEBUG PROJECTS: File {csv_path} exists? {file_exists}")
            
            if file_exists:
                try:
                    # Read existing to avoid duplicates
                    existing_df = pd.read_csv(csv_path)
                    logger.info(f"DEBUG PROJECTS: Loaded existing file with {len(existing_df)} rows")
                    
                    # If we have IDs in both, use that for merging
                    if 'id' in existing_df.columns and 'id' in df.columns:
                        # Get existing IDs
                        existing_ids = set(existing_df['id'].astype(str))
                        
                        # Split new projects into updates and inserts
                        updates = df[df['id'].astype(str).isin(existing_ids)]
                        inserts = df[~df['id'].astype(str).isin(existing_ids)]
                        
                        # For updates, replace existing rows with new data
                        if not updates.empty:
                            # Remove the rows we'll be updating
                            existing_df = existing_df[~existing_df['id'].astype(str).isin(updates['id'].astype(str))]
                            
                        # Combine everything
                        combined_df = pd.concat([existing_df, updates, inserts], ignore_index=True)
                        
                        logger.info(f"DEBUG PROJECTS: Updated {len(updates)} existing projects and added {len(inserts)} new projects")
                    else:
                        # Use title as identifier if no ID
                        combined_df = pd.concat([existing_df, df], ignore_index=True)
                        if 'title' in combined_df.columns and 'origin_url' in combined_df.columns:
                            combined_df = combined_df.drop_duplicates(subset=['title', 'origin_url'], keep='last')
                        elif 'title' in combined_df.columns:
                            combined_df = combined_df.drop_duplicates(subset=['title'], keep='last')
                        else:
                            # Just keep everything if no good identifiers
                            logger.warning("DEBUG PROJECTS: No good identifiers for deduplication - may create duplicates")
                    
                    combined_df.to_csv(csv_path, index=False)
                    logger.info(f"DEBUG PROJECTS: Updated projects in {csv_path} (total: {len(combined_df)})")
                    
                    # Verify the file was written correctly
                    if os.path.exists(csv_path):
                        file_size = os.path.getsize(csv_path)
                        logger.info(f"DEBUG PROJECTS: After write: file exists with size {file_size} bytes")
                    else:
                        logger.warning(f"DEBUG PROJECTS: After write: file does not exist!")
                        
                except Exception as merge_err:
                    logger.error(f"DEBUG PROJECTS: Error merging with existing projects CSV: {merge_err}")
                    # If merge fails, fall back to overwriting
                    df.to_csv(csv_path, index=False)
                    logger.info(f"DEBUG PROJECTS: Created {csv_path} with {len(df)} projects (overwrite)")
                    
                    # Check if the file was created successfully
                    if os.path.exists(csv_path):
                        file_size = os.path.getsize(csv_path)
                        logger.info(f"DEBUG PROJECTS: After error recovery: file exists with size {file_size} bytes")
                    else:
                        logger.warning(f"DEBUG PROJECTS: After error recovery: file does not exist!")
            else:
                df.to_csv(csv_path, index=False)
                logger.info(f"DEBUG PROJECTS: Created {csv_path} with {len(df)} projects (new file)")
                
                # Verify the file was written correctly
                if os.path.exists(csv_path):
                    file_size = os.path.getsize(csv_path)
                    logger.info(f"DEBUG PROJECTS: After write: file exists with size {file_size} bytes")
                else:
                    logger.warning(f"DEBUG PROJECTS: After write: file does not exist!")
                
        except Exception as e:
            logger.error(f"Error saving projects CSV: {e}", exc_info=True)
            
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
        
        logger.info(f"Saving {len(methods)} methods to {csv_path}")
        
        try:
            # Process methods, ensuring steps and tags are properly formatted
            processed_methods = []
            for idx, m in enumerate(methods):
                try:
                    m_copy = m.copy()
                    
                    # Log method details for debugging
                    logger.debug(f"Processing method #{idx+1}: title={m_copy.get('title', 'Unnamed')}, steps_count={len(m_copy.get('steps', []))} steps")
                    
                    # Make sure title exists and is not empty
                    if 'title' not in m_copy or not m_copy['title']:
                        logger.warning(f"Method missing title, skipping: {m_copy}")
                        continue
                        
                    # Handle description field mapping to usecase for storage
                    if 'description' in m_copy and 'usecase' not in m_copy:
                        m_copy['usecase'] = m_copy['description']
                    
                    # Format steps as JSON if they're a list
                    if 'steps' in m_copy and isinstance(m_copy['steps'], list):
                        # Ensure we have at least one step
                        if not m_copy['steps']:
                            logger.warning(f"Method '{m_copy.get('title')}' has no steps, adding placeholder")
                            m_copy['steps'] = ["Step 1: Perform the method"]
                        m_copy['steps'] = json.dumps(m_copy['steps'])
                    elif 'steps' not in m_copy or not m_copy['steps']:
                        logger.warning(f"Method '{m_copy.get('title')}' missing steps, adding placeholder")
                        m_copy['steps'] = json.dumps(["Step 1: Perform the method"])
                    
                    # Format tags as JSON if they're a list - make sure tags is always present
                    if 'tags' not in m_copy:
                        m_copy['tags'] = ["method"]
                        logger.debug(f"Added default 'method' tag to method '{m_copy.get('title')}'")
                    
                    if isinstance(m_copy['tags'], list):
                        # Ensure we have at least one tag
                        if not m_copy['tags']:
                            m_copy['tags'] = ["method"]
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
                    logger.debug(f"Successfully processed method: {m_copy.get('title')}")
                except Exception as e:
                    logger.error(f"Error processing method #{idx+1}: {e}")
                    try:
                        logger.error(f"Method that caused error: title={m.get('title', 'unknown')}, has_steps={('steps' in m)}")
                    except:
                        pass
            
            df = pd.DataFrame(processed_methods)
            
            # Handle combining with existing data if file exists
            file_exists = os.path.isfile(csv_path)
            logger.info(f"DEBUG METHODS: File {csv_path} exists? {file_exists}")
            
            if file_exists:
                try:
                    # Read existing to avoid duplicates
                    existing_df = pd.read_csv(csv_path)
                    logger.info(f"DEBUG METHODS: Loaded existing file with {len(existing_df)} rows")
                    
                    # Use title + source as a composite key to avoid duplicates
                    combined_df = pd.concat([existing_df, df])
                    if 'title' in combined_df.columns and 'source' in combined_df.columns:
                        combined_df = combined_df.drop_duplicates(subset=['title', 'source'])
                        logger.info(f"DEBUG METHODS: Deduplicating by title and source")
                    else:
                        combined_df = combined_df.drop_duplicates()
                        logger.info(f"DEBUG METHODS: Deduplicating by all columns (no title/source columns found)")
                        
                    combined_df.to_csv(csv_path, index=False)
                    logger.info(f"DEBUG METHODS: Updated methods in {csv_path} (total: {len(combined_df)})")
                    
                    # Verify the file was written correctly
                    if os.path.exists(csv_path):
                        file_size = os.path.getsize(csv_path)
                        logger.info(f"DEBUG METHODS: After write: file exists with size {file_size} bytes")
                    else:
                        logger.warning(f"DEBUG METHODS: After write: file does not exist!")
                        
                except Exception as merge_err:
                    logger.error(f"DEBUG METHODS: Error merging with existing methods CSV: {merge_err}")
                    # If merge fails, fall back to overwriting
                    df.to_csv(csv_path, index=False)
                    logger.info(f"DEBUG METHODS: Created {csv_path} with {len(df)} methods (overwrite)")
                    
                    # Check if the file was created successfully
                    if os.path.exists(csv_path):
                        file_size = os.path.getsize(csv_path)
                        logger.info(f"DEBUG METHODS: After error recovery: file exists with size {file_size} bytes")
                    else:
                        logger.warning(f"DEBUG METHODS: After error recovery: file does not exist!")
            else:
                df.to_csv(csv_path, index=False)
                logger.info(f"DEBUG METHODS: Created {csv_path} with {len(df)} methods (new file)")
                
                # Verify the file was written correctly
                if os.path.exists(csv_path):
                    file_size = os.path.getsize(csv_path)
                    logger.info(f"DEBUG METHODS: After write: file exists with size {file_size} bytes")
                else:
                    logger.warning(f"DEBUG METHODS: After write: file does not exist!")
                
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