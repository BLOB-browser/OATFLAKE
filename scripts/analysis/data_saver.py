#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
import os
import uuid
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
    
    def save_universal_content(self, items: List[Dict], content_type: str, csv_path: str = None) -> None:
        """
        Universal CSV saving function that handles all content types with the same schema.
        This ensures all content follows the UniversalTable structure regardless of type.
        
        Args:
            items: List of content dictionaries
            content_type: Type of content (method, definition, project, reference, etc.)
            csv_path: Path to save CSV file (auto-generated if not provided)
        """
        if not items:
            logger.info(f"No {content_type} items to save")
            return
        
        if csv_path is None:
            csv_path = os.path.join(self.data_folder, f'{content_type}s.csv')
        
        try:
            # Set default timestamp for use in multiple fields
            current_time = datetime.now().isoformat()
            
            # Process items using universal schema structure
            processed_items = []
            for idx, item in enumerate(items):
                try:
                    item_copy = item.copy()
                    
                    # Debug information
                    logger.debug(f"Processing {content_type} #{idx+1}: title={item_copy.get('title', 'Unnamed')}")
                    
                    # 1. Universal Required Fields - ID
                    if 'id' not in item_copy:
                        import uuid
                        item_copy['id'] = str(uuid.uuid4())
                    
                    # 2. Content type - always set to the specified type
                    item_copy['content_type'] = content_type
                    
                    # 3. Title and Description - required for all types
                    if 'title' not in item_copy or not item_copy['title']:
                        # Handle different fallbacks based on content type
                        if content_type == 'definition' and 'term' in item_copy:
                            item_copy['title'] = item_copy['term']
                        else:
                            item_copy['title'] = f"Untitled {content_type} {idx + 1}"
                            logger.warning(f"{content_type.title()} missing title, using default: {item_copy['title']}")
                    
                    if 'description' not in item_copy or not item_copy['description']:
                        # Handle different fallbacks based on content type
                        if content_type == 'definition' and 'definition' in item_copy:
                            item_copy['description'] = item_copy['definition']
                        elif content_type == 'method' and 'usecase' in item_copy:
                            item_copy['description'] = item_copy['usecase']
                        else:
                            item_copy['description'] = f"{content_type.title()} extracted from content analysis"
                    
                    # 4. Universal Schema Fields - apply to ALL content types
                    if 'purpose' not in item_copy:
                        purpose_map = {
                            'definition': 'To define and clarify terminology',
                            'method': f"To provide methodology for {item_copy.get('title', 'task completion')}",
                            'project': f"To document project: {item_copy.get('title', 'implementation')}",
                            'resource': f"To provide resource information about {item_copy.get('title', 'content')}"
                        }
                        item_copy['purpose'] = purpose_map.get(content_type, f"To document {content_type} information")
                    
                    if 'location' not in item_copy:
                        item_copy['location'] = ""
                    
                    if 'status' not in item_copy:
                        item_copy['status'] = 'active'
                    
                    if 'visibility' not in item_copy:
                        # Handle backward compatibility
                        if 'privacy' in item_copy:
                            item_copy['visibility'] = item_copy['privacy']
                        else:
                            item_copy['visibility'] = 'public'
                    if 'creator_id' not in item_copy:
                        item_copy['creator_id'] = 'system'
                    
                    if 'group_id' not in item_copy:
                        item_copy['group_id'] = 'default'
                    
                    if 'collaborators' not in item_copy:
                        item_copy['collaborators'] = '[]'  # Default to empty JSON array
                    
                    if 'analysis_completed' not in item_copy:
                        item_copy['analysis_completed'] = True  # Items are created after analysis
                      # 5. Timestamps - ensure proper datetime format
                    if 'created_at' not in item_copy or not item_copy['created_at']:
                        item_copy['created_at'] = current_time
                    
                    if 'last_updated_at' not in item_copy or not item_copy['last_updated_at']:
                        # Use modified_at if available for backward compatibility
                        item_copy['last_updated_at'] = item_copy.get('modified_at', current_time)
                    
                    # 6. Tags - ensure they exist for all content types
                    if 'tags' not in item_copy or not item_copy['tags']:
                        # Handle backward compatibility with 'fields'
                        if 'fields' in item_copy and item_copy['fields']:
                            if isinstance(item_copy['fields'], list):
                                item_copy['tags'] = item_copy['fields']
                            elif isinstance(item_copy['fields'], str):
                                try:
                                    item_copy['tags'] = json.loads(item_copy['fields'])
                                except:
                                    item_copy['tags'] = [tag.strip() for tag in item_copy['fields'].split(',')]
                        else:
                            item_copy['tags'] = [content_type]
                      # 7. Content-Type Specific Fields (universal schema only)
                    # Only preserve content-type specific fields that exist in UniversalTable schema
                    
                    # For methods - ensure steps are preserved
                    if content_type == 'method':
                        if 'steps' not in item_copy or not item_copy['steps']:
                            logger.warning(f"Method '{item_copy.get('title')}' missing steps, adding placeholder")
                            item_copy['steps'] = ["Step 1: Perform the method"]
                      
                    # For projects - ensure goals are preserved  
                    elif content_type == 'project':
                        if 'goals' not in item_copy:
                            item_copy['goals'] = item_copy.get('goal', '')
                    
                    # For materials - ensure file_path is preserved
                    elif content_type == 'material':
                        if 'file_path' not in item_copy:
                            item_copy['file_path'] = None
                    
                    # 8. Handle origin_url/url field compatibility                if 'origin_url' not in item_copy and 'url' in item_copy:
                        item_copy['origin_url'] = item_copy['url']
                    elif 'url' not in item_copy and 'origin_url' in item_copy:
                        item_copy['url'] = item_copy['origin_url']
                    elif 'origin_url' not in item_copy:
                        # Ensure origin_url always exists for universal schema compliance
                        item_copy['origin_url'] = item_copy.get('url', '')
                    
                    # 9. Convert all list fields to JSON strings for CSV storage
                    # Also ensure proper formatting and non-empty values
                    for key, value in item_copy.items():
                        if isinstance(value, list):
                            # Ensure required lists have at least one value
                            if key == 'steps' and not value:
                                value = ["Step 1: Perform the method"]
                            elif key == 'tags' and not value:
                                value = [content_type]
                            elif key == 'collaborators' and not value:
                                value = []  # Keep empty for collaborators
                            item_copy[key] = json.dumps(value)
                        elif value is None:
                            # Convert None values to empty strings for CSV compatibility
                            item_copy[key] = ''
                    
                    # 10. Ensure all required UniversalTable fields are present
                    required_fields = {
                        'id': str(uuid.uuid4()) if 'id' not in item_copy else item_copy['id'],
                        'content_type': content_type,
                        'title': item_copy.get('title', f'Untitled {content_type}'),
                        'description': item_copy.get('description', f'{content_type.title()} extracted from content analysis'),
                        'tags': item_copy.get('tags', json.dumps([content_type])),
                        'purpose': item_copy.get('purpose', f'To document {content_type} information'),
                        'location': item_copy.get('location', ''),
                        'origin_url': item_copy.get('origin_url', ''),
                        'status': item_copy.get('status', 'active'),
                        'creator_id': item_copy.get('creator_id', 'system'),
                        'group_id': item_copy.get('group_id', 'default'),
                        'visibility': item_copy.get('visibility', 'public'),
                        'analysis_completed': item_copy.get('analysis_completed', True),
                        'collaborators': item_copy.get('collaborators', '[]'),
                        'created_at': item_copy.get('created_at', current_time),
                        'last_updated_at': item_copy.get('last_updated_at', current_time),
                        'related_url': item_copy.get('related_url', '')
                    }
                    
                    # Update item_copy with required fields to ensure consistency
                    item_copy.update(required_fields)
                    
                    processed_items.append(item_copy)
                    logger.debug(f"Successfully processed {content_type}: {item_copy.get('title')}")
                    
                except Exception as e:
                    logger.error(f"Error processing {content_type} #{idx+1}: {e}")
                    try:
                        logger.error(f"{content_type.title()} that caused error: title={item.get('title', 'unknown')}")
                    except:
                        pass
            
            if not processed_items:
                logger.warning(f"No valid {content_type} items to save after processing")
                return            # Create DataFrame with only universal schema columns
            # Define universal table column order to ensure consistent CSV headers
            universal_columns = [
                'id', 'content_type', 'title', 'description', 'tags', 'purpose', 'location', 
                'origin_url', 'related_url', 'status', 'creator_id', 'group_id', 'visibility', 
                'analysis_completed', 'collaborators', 'created_at', 'last_updated_at',
                # Content-specific fields that are part of universal schema (no backward compatibility)
                'steps', 'goals', 'achievement', 'documentation_url', 'file_path'
            ]
            
            # Filter items to only include universal schema fields and exclude legacy fields
            filtered_items = []
            for item in processed_items:
                filtered_item = {}
                for column in universal_columns:
                    if column in item:
                        filtered_item[column] = item[column]
                filtered_items.append(filtered_item)
            
            df = pd.DataFrame(filtered_items)
            
            # Ensure column order matches universal schema
            existing_cols = [col for col in universal_columns if col in df.columns]
            df = df[existing_cols]
            
            # Handle merging with existing data if file exists
            file_exists = os.path.isfile(csv_path)
            logger.info(f"Universal save for {content_type}: File {csv_path} exists? {file_exists}")
            
            if file_exists:
                try:
                    # Read existing file
                    existing_df = pd.read_csv(csv_path)
                    logger.info(f"Loaded existing {content_type} file with {len(existing_df)} rows")
                    
                    # Use ID-based merging if both have IDs
                    if 'id' in existing_df.columns and 'id' in df.columns:
                        existing_ids = set(existing_df['id'].astype(str))
                        updates = df[df['id'].astype(str).isin(existing_ids)]
                        inserts = df[~df['id'].astype(str).isin(existing_ids)]
                        
                        # Remove existing rows that will be updated
                        if not updates.empty:
                            existing_df = existing_df[~existing_df['id'].astype(str).isin(updates['id'].astype(str))]
                        
                        # Combine all data
                        combined_df = pd.concat([existing_df, updates, inserts], ignore_index=True)
                        logger.info(f"Updated {len(updates)} existing {content_type}s and added {len(inserts)} new {content_type}s")
                    
                    else:
                        # Fallback to title-based deduplication
                        combined_df = pd.concat([existing_df, df], ignore_index=True)
                        if 'title' in combined_df.columns:
                            if 'origin_url' in combined_df.columns:
                                combined_df = combined_df.drop_duplicates(subset=['title', 'origin_url'], keep='last')
                            else:
                                combined_df = combined_df.drop_duplicates(subset=['title'], keep='last')
                            logger.info(f"Deduplicated {content_type}s by title")
                        else:
                            logger.warning(f"No good identifiers for {content_type} deduplication - may create duplicates")
                    
                    # Save combined data
                    combined_df.to_csv(csv_path, index=False)
                    logger.info(f"Saved {len(combined_df)} total {content_type}s to {csv_path}")
                    
                except Exception as merge_err:
                    logger.error(f"Error merging {content_type}s with existing CSV: {merge_err}")
                    # Fallback to overwriting
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Created {csv_path} with {len(df)} {content_type}s (overwrite after merge error)")
            else:
                # No existing file, create new
                df.to_csv(csv_path, index=False)
                logger.info(f"Created new {csv_path} with {len(df)} {content_type}s")
              # Verify file was written
            if os.path.exists(csv_path):
                file_size = os.path.getsize(csv_path)
                logger.info(f"Universal save complete: {csv_path} exists with {file_size} bytes")
            else:
                logger.error(f"Universal save failed: {csv_path} does not exist after write!")
                
        except Exception as e:
            logger.error(f"Error in universal save for {content_type}: {e}", exc_info=True)
    
    # Backward compatibility wrapper functions
    def save_resources(self, resources: List[Dict], csv_path: str = None) -> None:
        """Wrapper for backward compatibility - uses universal save function."""
        self.save_universal_content(resources, 'resource', csv_path)

    def save_definitions(self, definitions: List[Dict], csv_path: str = None) -> None:
        """Wrapper for backward compatibility - uses universal save function."""
        self.save_universal_content(definitions, 'definition', csv_path)
    
    def save_projects(self, projects: List[Dict], csv_path: str = None) -> None:
        """Wrapper for backward compatibility - uses universal save function."""
        self.save_universal_content(projects, 'project', csv_path)
    
    def save_methods(self, methods: List[Dict], csv_path: str = None) -> None:
        """Wrapper for backward compatibility - uses universal save function."""
        self.save_universal_content(methods, 'method', csv_path)
    
    def save_references(self, references: List[Dict], csv_path: str = None) -> None:
        """Wrapper for backward compatibility - uses universal save function."""
        self.save_universal_content(references, 'reference', csv_path)
    
    def save_single_resource(self, resource: Dict, csv_path: str, idx: int) -> None:
        """
        Update a single resource in the CSV file without loading the entire file.
        Uses universal schema format.
        
        Args:
            resource: Resource dictionary to save
            csv_path: Path to the CSV file
            idx: Index of the resource in the CSV
        """
        try:
            # Process the resource using universal schema first
            processed_resources = []
            self.save_universal_content([resource], 'resource', None)
            
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