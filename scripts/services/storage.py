import json
from pathlib import Path
import logging
import shutil
from utils.config import BACKEND_CONFIG
import csv
from datetime import datetime
import os
import pandas as pd  # Add pandas import
from fastapi import UploadFile

logger = logging.getLogger(__name__)

class DataSaver:
    def save_definition(self, definition):
        """
        Save a definition to CSV storage. Handles both single definition or a list of definitions.

        Args:
            definition: A dictionary or list of dictionaries with definition data

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Handle both single definition and lists of definitions
            definitions_to_save = []
            if isinstance(definition, list):
                definitions_to_save = definition
            else:
                definitions_to_save = [definition]

            # Skip if nothing to save
            if not definitions_to_save:
                return True

            data_path = Path(BACKEND_CONFIG.get('data_path', './data'))
            data_path.mkdir(parents=True, exist_ok=True)

            csv_file = data_path / "definitions.csv"
            is_new_file = not csv_file.exists()

            # Prepare CSV fields
            fields = ['term', 'definition', 'tags', 'source', 'created_at']

            # Add all definitions
            for definition_item in definitions_to_save:
                # Skip invalid definitions
                if not isinstance(definition_item, dict):
                    logger.warning(f"Skipping invalid definition (not a dict): {type(definition_item)}")
                    continue

                if not definition_item.get('term') or not definition_item.get('definition'):
                    logger.warning(f"Skipping incomplete definition: {definition_item.get('term', 'Unknown')}")
                    continue

                # Prepare row data with proper tag handling
                tags = definition_item.get('tags', [])
                # Convert list to comma-separated string if needed
                tags_str = tags if isinstance(tags, str) else ','.join(tags) if isinstance(tags, list) else ''

                row = {
                    'term': definition_item.get('term', ''),
                    'definition': definition_item.get('definition', ''),
                    'tags': tags_str,
                    'source': definition_item.get('source', ''),
                    'created_at': definition_item.get('created_at') or datetime.now().isoformat()
                }

                try:
                    # Always save the definition, even if it's a duplicate
                    # Append row to CSV
                    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fields)

                        # Write header only for new files
                        if is_new_file:
                            writer.writeheader()
                            is_new_file = False

                        writer.writerow(row)
                        logger.info(f"Definition saved to CSV: {row['term']} (may be duplicate)")
                except Exception as row_error:
                    logger.error(f"Error saving definition row: {row_error}")

            return True

        except Exception as e:
            logger.error(f"Error saving definition(s): {e}")
            return False

    def save_project(self, project):
        """
        Save a project to CSV storage. Handles both single project or a list of projects.

        Args:
            project: A dictionary or list of dictionaries with project data

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Handle both single project and lists of projects
            projects_to_save = []
            if isinstance(project, list):
                projects_to_save = project
            else:
                projects_to_save = [project]

            # Skip if nothing to save
            if not projects_to_save:
                return True

            data_path = Path(BACKEND_CONFIG.get('data_path', './data'))
            data_path.mkdir(parents=True, exist_ok=True)

            csv_file = data_path / "projects.csv"
            is_new_file = not csv_file.exists()

            # Get existing projects to avoid duplicates
            existing_titles = set()
            if not is_new_file:
                try:
                    with open(csv_file, mode='r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        existing_titles = {r.get('title', '').lower() for r in reader}
                except Exception as e:
                    logger.error(f"Error reading existing projects: {e}")

            # Define standard fields - using only tags for consistency
            fields = [
                'title', 'description', 'goal', 'achievement',
                'tags', 'collaborators', 'documentation_url', 'created_at'
            ]

            # Process each project
            saved_count = 0
            for project_item in projects_to_save:
                # Skip invalid projects
                if not isinstance(project_item, dict):
                    logger.warning(f"Skipping invalid project (not a dict): {type(project_item)}")
                    continue

                if not project_item.get('title'):
                    logger.warning(f"Skipping project with missing title")
                    continue

                # Process tags (using fields for backward compatibility if needed)
                tags_value = project_item.get('tags', project_item.get('fields', []))  # Fallback to fields if tags not present
                tags_str = tags_value if isinstance(tags_value, str) else ','.join(tags_value) if isinstance(tags_value, list) else ''

                collaborators_value = project_item.get('collaborators', [])
                collaborators_str = collaborators_value if isinstance(collaborators_value, str) else ','.join(collaborators_value) if isinstance(collaborators_value, list) else ''

                # Prepare row data - use only tags for consistency
                row = {
                    'title': project_item.get('title', ''),
                    'description': project_item.get('description', ''),
                    'goal': project_item.get('goal', ''),
                    'achievement': project_item.get('achievement', ''),
                    'tags': tags_str,
                    'collaborators': collaborators_str,
                    'documentation_url': project_item.get('documentation_url', ''),
                    'created_at': project_item.get('created_at') or datetime.now().isoformat()
                }

                try:
                    # Write to CSV
                    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fields)

                        # Write header only for new files
                        if is_new_file:
                            writer.writeheader()
                            is_new_file = False

                        writer.writerow(row)
                        saved_count += 1

                    # Add to existing titles to prevent duplicates in this batch
                    existing_titles.add(row['title'].lower())

                except Exception as row_error:
                    logger.error(f"Error saving project row: {row_error}")

            logger.info(f"Saved {saved_count} projects to CSV")
            return True

        except Exception as e:
            logger.error(f"Error saving project(s): {e}")
            return False

    def save_method(self, method):
        """
        Save a method to CSV storage. Handles both single method or a list of methods.

        Args:
            method: A dictionary or list of dictionaries with method data

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Handle both single method and lists of methods
            methods_to_save = []
            if isinstance(method, list):
                methods_to_save = method
            else:
                methods_to_save = [method]

            # Skip if nothing to save
            if not methods_to_save:
                return True

            data_path = Path(BACKEND_CONFIG.get('data_path', './data'))
            data_path.mkdir(parents=True, exist_ok=True)

            csv_file = data_path / "methods.csv"
            is_new_file = not csv_file.exists()

            # Get existing methods to avoid duplicates
            existing_titles = set()
            if not is_new_file:
                try:
                    with open(csv_file, mode='r', newline='', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        existing_titles = {r.get('title', '').lower() for r in reader}
                except Exception as e:
                    logger.error(f"Error reading existing methods: {e}")

            # Define fields - added tags field
            fields = ['title', 'usecase', 'steps', 'tags', 'created_at']

            # Process each method
            saved_count = 0
            for method_item in methods_to_save:
                # Skip invalid methods
                if not isinstance(method_item, dict):
                    logger.warning(f"Skipping invalid method (not a dict): {type(method_item)}")
                    continue

                if not method_item.get('title'):
                    logger.warning(f"Skipping method with missing title")
                    continue

                # Process steps to ensure they're stored properly
                steps = method_item.get('steps', [])
                steps_str = steps if isinstance(steps, str) else '|'.join(steps) if isinstance(steps, list) else ''
                
                # Process tags to ensure they're stored properly
                tags = method_item.get('tags', [])
                tags_str = tags if isinstance(tags, str) else ','.join(tags) if isinstance(tags, list) else ''

                # Prepare row data - fix field name mismatch between 'usecase' and 'description'
                row = {
                    'title': method_item.get('title', ''),
                    'usecase': method_item.get('description', method_item.get('usecase', '')),  # Get description first, fall back to usecase
                    'steps': steps_str,
                    'tags': tags_str,
                    'created_at': method_item.get('created_at') or datetime.now().isoformat()
                }

                try:
                    # Write to CSV
                    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=fields)

                        # Write header only for new files
                        if is_new_file:
                            writer.writeheader()
                            is_new_file = False

                        writer.writerow(row)
                        saved_count += 1

                    # Add to existing titles to prevent duplicates in this batch
                    existing_titles.add(row['title'].lower())

                except Exception as row_error:
                    logger.error(f"Error saving method row: {row_error}")

            logger.info(f"Saved {saved_count} methods to CSV")
            return True

        except Exception as e:
            logger.error(f"Error saving method(s): {e}")
            return False

    def save_resource(self, resource, csv_path=None, idx=None):
        """
        Save a resource to CSV storage, with support for saving at specific index.

        Args:
            resource: A dictionary with resource data
            csv_path: Optional path to CSV file (defaults to resources.csv in data folder)
            idx: Optional row index in the CSV file to update

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Handle the case where a list of resources is passed
            if isinstance(resource, list):
                success = True
                for r in resource:
                    if not self.save_resource(r, csv_path):
                        success = False
                return success

            data_path = Path(BACKEND_CONFIG.get('data_path', './data'))
            data_path.mkdir(parents=True, exist_ok=True)

            # Use provided csv_path or default
            if csv_path is None:
                csv_path = data_path / "resources.csv"
            else:
                csv_path = Path(csv_path)

            # Make a copy of the resource to avoid modifying the original
            resource_copy = resource.copy()

            # Convert analysis_completed to explicit boolean to ensure it's handled correctly
            if 'analysis_completed' in resource_copy:
                resource_copy['analysis_completed'] = bool(resource_copy['analysis_completed'])
                logger.info(f"Setting analysis_completed to: {resource_copy['analysis_completed']}")

            # If index is provided, update existing row
            if idx is not None and csv_path.exists():
                try:
                    # Read the CSV file
                    df = pd.read_csv(csv_path)

                    # Validate index
                    if idx < 0 or idx >= len(df):
                        logger.warning(f"Invalid row index {idx} for CSV with {len(df)} rows")
                        return False

                    # Log before update
                    logger.info(f"Updating resource at index {idx}: analysis_completed={resource_copy.get('analysis_completed')}")
                    
                    # Properly handle analysis_results
                    if 'analysis_results' in resource_copy and resource_copy['analysis_results']:
                        try:
                            if isinstance(resource_copy['analysis_results'], dict):
                                resource_copy['analysis_results'] = json.dumps(resource_copy['analysis_results'])
                            elif isinstance(resource_copy['analysis_results'], str):
                                # Validate if it's already JSON
                                try:
                                    json.loads(resource_copy['analysis_results'])
                                except json.JSONDecodeError:
                                    # Not valid JSON, make it a clean JSON string
                                    resource_copy['analysis_results'] = json.dumps({"data": resource_copy['analysis_results']})
                        except Exception as json_error:
                            logger.error(f"Error serializing analysis_results: {json_error}")
                            resource_copy['analysis_results'] = json.dumps({"error": "Serialization failed"})
                    
                    # Fix: Update the DataFrame one column at a time
                    for key, value in resource_copy.items():
                        if key in df.columns:
                            # Handle tags specially
                            if key == 'tags' and isinstance(value, list):
                                try:
                                    # First convert column to string type if needed
                                    if df[key].dtype != 'object':
                                        df[key] = df[key].astype('object')
                                    
                                    if all(isinstance(tag, str) for tag in value):
                                        df.loc[idx, key] = ','.join(value)
                                    else:
                                        df.loc[idx, key] = json.dumps(value)
                                except Exception as tag_error:
                                    logger.error(f"Error updating tags: {tag_error}")
                                    # Fall back to basic string
                                    df.loc[idx, key] = str(value)
                            
                            # Handle analysis_completed specially - ensure it's boolean
                            elif key == 'analysis_completed':
                                # Make sure the column is boolean
                                try:
                                    if df[key].dtype != 'bool':
                                        df[key] = df[key].astype('bool')
                                    df.loc[idx, key] = bool(value)
                                    logger.info(f"Setting analysis_completed at idx {idx} to {bool(value)}")
                                except Exception as ac_error:
                                    logger.error(f"Error setting analysis_completed: {ac_error}")
                                    # Even if conversion fails, still try to set the value
                                    df.loc[idx, key] = bool(value)
                            
                            else:
                                # Handle other fields - convert column type if needed
                                try:
                                    # If value type doesn't match column type, convert column first
                                    if isinstance(value, str) and df[key].dtype != 'object':
                                        df[key] = df[key].astype('object')
                                    elif isinstance(value, bool) and df[key].dtype != 'bool':
                                        df[key] = df[key].astype('bool')
                                    elif isinstance(value, (int, float)) and not pd.api.types.is_numeric_dtype(df[key]):
                                        df[key] = df[key].astype('float64')
                                    
                                    # Use loc instead of at for more reliable indexing
                                    df.loc[idx, key] = value
                                except Exception as type_error:
                                    logger.error(f"Error converting types for column {key}: {type_error}")
                                    # Fall back to string conversion
                                    if df[key].dtype != 'object':
                                        df[key] = df[key].astype('object')
                                    df.loc[idx, key] = str(value)
                        else:
                            # Add new column if needed - use correct data type
                            if isinstance(value, bool):
                                df[key] = False  # Default for new boolean column
                                df.loc[idx, key] = bool(value)
                            elif isinstance(value, (int, float)):
                                df[key] = 0.0  # Default for new numeric column
                                df.loc[idx, key] = value
                            else:
                                df[key] = None  # Default for other types
                                df.loc[idx, key] = value
                            
                            logger.debug(f"Added new column '{key}' to resources CSV")
                    
                    # Verify the update before saving
                    analysis_completed_value = df.loc[idx, 'analysis_completed'] if 'analysis_completed' in df.columns else None
                    logger.info(f"Before saving CSV, analysis_completed at idx {idx} is: {analysis_completed_value}")
                    
                    # Save the updated DataFrame
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Updated resource at index {idx} in {csv_path.name} (analysis_completed={resource_copy.get('analysis_completed', False)})")
                    
                    # Verify the data was actually saved correctly by reading it back
                    try:
                        verification_df = pd.read_csv(csv_path)
                        value_after_save = verification_df.loc[idx, 'analysis_completed'] if 'analysis_completed' in verification_df.columns else None
                        logger.info(f"After CSV save, verified analysis_completed at idx {idx} is: {value_after_save}")
                    except Exception as verify_error:
                        logger.error(f"Error verifying saved data: {verify_error}")
                    
                    return True

                except Exception as e:
                    logger.error(f"Error updating resource at index {idx}: {e}")
                    return False

            # Otherwise add new row
            is_new_file = not csv_path.exists()

            # Get existing URLs to avoid duplicates
            existing_urls = set()
            if not is_new_file:
                try:
                    df = pd.read_csv(csv_path)
                    if 'url' in df.columns:
                        existing_urls = set(url.lower() for url in df['url'].dropna())
                except Exception as e:
                    logger.error(f"Error reading existing resources: {e}")

            # Skip duplicate URLs
            if resource.get('url', '').lower() in existing_urls:
                logger.info(f"Resource with URL '{resource.get('url')}' already exists, skipping")
                return True  # Not an error

            # Process tags
            tags_value = resource.get('tags', [])
            if isinstance(tags_value, list):
                if all(isinstance(tag, str) for tag in tags_value):
                    tags_str = ','.join(tags_value)
                else:
                    tags_str = json.dumps(tags_value)
            else:
                tags_str = str(tags_value) if tags_value else ''

            # Prepare row data
            row = {
                'title': resource.get('title', ''),
                'url': resource.get('url', ''),
                'description': resource.get('description', ''),
                'type': resource.get('type', 'other'),
                'category': resource.get('category', ''),
                'tags': tags_str,
                'created_at': resource.get('created_at') or datetime.now().isoformat(),
                'analysis_completed': resource.get('analysis_completed', False)
            }

            try:
                # Standard fields
                default_fields = ['title', 'url', 'description', 'type', 'category', 'tags', 'created_at', 'analysis_completed']

                # Add any extra fields that might be present
                for key, value in resource.items():
                    if key not in default_fields and key != 'analysis_results':
                        row[key] = value

                # Handle analysis_results specially (store as JSON string)
                if 'analysis_results' in resource and resource['analysis_results']:
                    try:
                        if isinstance(resource['analysis_results'], dict):
                            row['analysis_results'] = json.dumps(resource['analysis_results'])
                        elif isinstance(resource['analysis_results'], str):
                            # Check if it's already valid JSON
                            json.loads(resource['analysis_results'])
                            row['analysis_results'] = resource['analysis_results']
                        else:
                            row['analysis_results'] = json.dumps({"error": "Invalid analysis results format"})
                    except Exception:
                        row['analysis_results'] = json.dumps({"error": "Failed to serialize analysis results"})

                # Determine fields to use in CSV
                fields_to_use = list(row.keys())

                # Write to CSV
                with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fields_to_use)

                    # Write header only for new files
                    if is_new_file:
                        writer.writeheader()

                    writer.writerow(row)
                    logger.info(f"Added resource to CSV: {row['title']}")

                return True

            except Exception as row_error:
                logger.error(f"Error saving resource row: {row_error}")
                return False

        except Exception as e:
            logger.error(f"Error saving resource: {e}")
            return False

    def save_reading_material(self, material: dict, file: UploadFile) -> bool:
        """
        Save reading material metadata to CSV and PDF to storage

        Args:
            material: Dictionary with material metadata
            file: Uploaded file object

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            data_path = Path(BACKEND_CONFIG.get('data_path', './data'))
            materials_path = data_path / "materials"
            materials_path.mkdir(parents=True, exist_ok=True)

            # Save PDF file
            file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            file_path = materials_path / file_name

            with file_path.open("wb") as f:
                contents = file.file.read()
                f.write(contents)

            # Update material with file path
            material['file_path'] = str(file_path)

            # Save metadata to CSV
            csv_file = data_path / "materials.csv"
            is_new_file = not csv_file.exists()

            # Parse fields
            fields_value = material.get('fields', [])
            fields_str = fields_value if isinstance(fields_value, str) else ','.join(fields_value) if isinstance(fields_value, list) else ''

            row = {
                'title': material.get('title', ''),
                'description': material.get('description', ''),
                'fields': fields_str,
                'file_path': material['file_path'],
                'created_at': material.get('created_at') or datetime.now().isoformat()
            }

            # Write to CSV
            try:
                with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if is_new_file:
                        writer.writeheader()
                    writer.writerow(row)

                logger.info(f"Reading material saved: {row['title']} -> {file_path}")
                return True
            except Exception as csv_error:
                logger.error(f"Error saving material metadata to CSV: {csv_error}")
                return False

        except Exception as e:
            logger.error(f"Error saving reading material: {e}")
            return False

    def update_storage_path(self, new_path: Path) -> dict:
        """
        Update the storage path and migrate data if needed

        Args:
            new_path: New path for data storage

        Returns:
            dict: Status information
        """
        try:
            new_path = new_path.expanduser().resolve()
            new_path.mkdir(parents=True, exist_ok=True)

            # Load and update config
            config_path = Path.home() / '.blob' / 'config.json'
            config = json.loads(config_path.read_text()) if config_path.exists() else {}
            old_path = Path(config.get('data_path', BACKEND_CONFIG['data_path']))

            # Update config
            config['data_path'] = str(new_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, indent=2))

            # Update runtime config
            BACKEND_CONFIG['data_path'] = str(new_path)

            # Migrate data if needed
            if old_path.exists() and old_path != new_path:
                # Copy all CSV files
                csv_files = ['definitions.csv', 'methods.csv', 'projects.csv', 'resources.csv', 'materials.csv', 'goals.csv']

                for csv_file in csv_files:
                    old_csv = old_path / csv_file
                    if old_csv.exists():
                        new_csv = new_path / csv_file
                        shutil.copy2(str(old_csv), str(new_csv))
                        logger.info(f"Migrated {csv_file} to new location")

                # Migrate materials directory if it exists
                old_materials = old_path / "materials"
                if old_materials.exists() and old_materials.is_dir():
                    new_materials = new_path / "materials"
                    new_materials.mkdir(exist_ok=True)

                    # Copy files only, not recursive
                    for file in old_materials.glob("*.*"):
                        if file.is_file():
                            shutil.copy2(str(file), str(new_materials / file.name))
                    logger.info(f"Migrated materials to new location")

                # Migrate stats directory
                old_stats = old_path / "stats"
                if old_stats.exists() and old_stats.is_dir():
                    new_stats = new_path / "stats"
                    new_stats.mkdir(exist_ok=True)

                    for file in old_stats.glob("*.*"):
                        if file.is_file():
                            shutil.copy2(str(file), str(new_stats / file.name))
                    logger.info(f"Migrated stats to new location")

                logger.info(f"Data migration completed from {old_path} to {new_path}")

            return {"status": "success", "path": str(new_path)}

        except Exception as e:
            logger.error(f"Error updating storage path: {e}")
            raise

    def save_resources(self, resources: list, csv_path: str = None) -> bool:
        """Save resources to CSV file"""
        try:
            if not resources:
                logger.warning("No resources to save")
                return False

            # Get the resources file path
            if csv_path is None:
                file_path = Path(BACKEND_CONFIG.get('data_path', './data')) / "resources.csv"
            else:
                file_path = Path(csv_path)

            # If file doesn't exist, create it with headers
            if not file_path.exists():
                logger.info(f"Creating new resources file at {file_path}")
                df = pd.DataFrame(resources)
                df.to_csv(file_path, index=False)
                return True

            # If file exists, read it and update
            try:
                original_df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(original_df)} existing resources")

                # Handle each resource separately to avoid column mismatch
                for i, resource in enumerate(resources):
                    try:
                        # Find the matching row(s)
                        match_found = False
                        if 'id' in resource and resource['id']:
                            matches = original_df.loc[original_df['id'] == resource['id']].index.tolist()
                            if matches:
                                match_idx = matches[0]
                                match_found = True

                        if not match_found and 'url' in resource and resource['url']:
                            matches = original_df.loc[original_df['url'] == resource['url']].index.tolist()
                            if matches:
                                match_idx = matches[0]
                                match_found = True

                        if match_found:
                            # Update only the columns that exist in the resource
                            for col, value in resource.items():
                                # Add column if it doesn't exist
                                if col not in original_df.columns:
                                    original_df[col] = None

                                # Update the value
                                original_df.at[match_idx, col] = value

                            logger.info(f"Updated resource {resource.get('id', resource.get('url', 'unknown'))}")
                        else:
                            # New resource, make sure all columns are present
                            temp_df = pd.DataFrame([resource])
                            for col in original_df.columns:
                                if col not in temp_df.columns:
                                    temp_df[col] = None

                            # Append to the original df
                            original_df = pd.concat([original_df, temp_df], ignore_index=True)
                            logger.info(f"Added new resource {resource.get('id', resource.get('url', 'unknown'))}")

                    except Exception as e:
                        logger.error(f"Error updating resource at index {i}: {e}")

                # Save the updated dataframe
                original_df.to_csv(file_path, index=False)
                logger.info(f"Saved {len(original_df)} resources to {file_path}")
                return True

            except Exception as e:
                logger.error(f"Error reading/updating existing resources: {e}")
                # Fallback: Save as new file
                df = pd.DataFrame(resources)
                df.to_csv(file_path, index=False)
                logger.info(f"Created new resources file with {len(resources)} entries (after error)")
                return True

        except Exception as e:
            logger.error(f"Error saving resources: {e}")
            return False
