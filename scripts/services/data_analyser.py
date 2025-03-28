#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd

from utils.config import get_data_path
# Replace ResourceProcessor with MainProcessor
from scripts.analysis.main_processor import MainProcessor
from scripts.analysis.content_fetcher import ContentFetcher
from scripts.analysis.llm_analyzer import LLMAnalyzer
# Import DataSaver directly from storage.py, not from a separate module
from scripts.services.storage import DataSaver

logger = logging.getLogger(__name__)

class DataAnalyser:
    """
    Main interface for analyzing resource URLs to extract definitions, enhance tags, 
    and identify projects. This class coordinates the various components of the analysis system.
    """
    
    def __init__(self):
        """Initialize the data analyzer with necessary components"""
        self.data_folder = get_data_path()
        
        # Initialize component classes - use MainProcessor instead of ResourceProcessor
        self.main_processor = MainProcessor(self.data_folder)
        self.content_fetcher = ContentFetcher()
        self.llm_analyzer = LLMAnalyzer()
        self.data_saver = DataSaver()
        
        # Add LLM provider property for logging
        self.provider = self.llm_analyzer.provider if hasattr(self.llm_analyzer, 'provider') else "Unknown"
    
    def analyze_resources(self, csv_path: str = None, batch_size: int = 1, max_resources: int = None, force_reanalysis: bool = False) -> Tuple[List[Dict], List[Dict]]:
        """
        Analyzes resources from CSV file, extracting definitions and identifying projects.
        
        This implementation delegates to the MainProcessor to handle the full processing pipeline.
        
        Args:
            csv_path: Path to the resources CSV file
            batch_size: Number of resources to process at once (default: 1 for step-by-step)
            max_resources: Maximum number of resources to process (None for all)
            force_reanalysis: If True, analyze all resources even if they already have analysis
            
        Returns:
            Tuple of (updated_resources, identified_projects)
        """
        if csv_path is None:
            csv_path = os.path.join(self.data_folder, 'resources.csv')
        
        logger.info(f"Starting resource analysis with path={csv_path}, max_resources={max_resources}, force_reanalysis={force_reanalysis}")
        
        # Before processing, ensure references dir exists
        references_dir = os.path.join(self.data_folder, 'references')
        os.makedirs(references_dir, exist_ok=True)
        logger.info(f"Ensuring references directory exists: {references_dir}")
        
        # Use MainProcessor to process resources - this handles individual resource processing
        # and saves definitions, projects, and methods as it goes
        processing_result = self.main_processor.process_resources(
            csv_path=csv_path,
            max_resources=max_resources,
            force_reanalysis=force_reanalysis
        )
        
        # Extract results from processing_result
        updated_resources = []
        projects = []
        definitions = []
        methods = []
        
        if processing_result.get("status") == "completed":
            # Get resources from successful processing
            resources_processed = processing_result.get("resources_processed", 0)
            success_count = processing_result.get("success_count", 0)
            
            # Collect results from processing
            if "processed_resources" in processing_result:
                updated_resources = processing_result.get("processed_resources", [])
                
                # Save raw content from resources to the references directory
                saved_content_count = 0
                for resource in updated_resources:
                    if resource.get('content') and len(resource.get('content', '')) > 100:
                        try:
                            # Create a clean filename from resource ID or URL
                            import hashlib
                            
                            if resource.get('id'):
                                safe_name = f"{resource['id']}"
                            else:
                                # Generate a hash from the URL
                                url_hash = hashlib.md5(resource.get('url', '').encode()).hexdigest()[:10]
                                safe_name = f"resource_{url_hash}"
                                
                            # Save content to permanent storage before it's deleted
                            content_file = os.path.join(references_dir, f"{safe_name}.txt")
                            with open(content_file, 'w', encoding='utf-8') as f:
                                f.write(resource['content'])
                                
                            # Add content_file path to resource metadata
                            resource['content_file'] = content_file
                            saved_content_count += 1
                        except Exception as e:
                            logger.error(f"Error saving content file for resource {resource.get('id', 'unknown')}: {e}")
                
                logger.info(f"Saved raw content for {saved_content_count} resources to {references_dir}")
                
            if "projects" in processing_result:
                projects = processing_result.get("projects", [])
            if "definitions" in processing_result:
                definitions = processing_result.get("definitions", [])
            if "methods" in processing_result:
                methods = processing_result.get("methods", [])
                
            logger.info(f"Analysis complete: {resources_processed} resources processed, {success_count} successful")
            logger.info(f"Found {len(definitions)} definitions, {len(projects)} projects, and {len(methods)} methods")
        else:
            # Handle processing error
            logger.error(f"Error in resource processing: {processing_result.get('error', 'Unknown error')}")
        
        return updated_resources, projects
    
    def save_updated_resources(self, resources: List[Dict], csv_path: str = None) -> None:
        """
        Saves updated resources back to CSV.
        
        This is needed to save the final state of resources with their enhanced metadata
        after all individual processing is complete.
        
        Args:
            resources: List of resource dictionaries
            csv_path: Path to save CSV file (defaults to original location)
        """
        # Check if we need to persist any content before saving
        references_dir = os.path.join(self.data_folder, 'references')
        os.makedirs(references_dir, exist_ok=True)
        
        # Save any content not already saved
        for resource in resources:
            # Don't override existing analysis_completed flags
            logger.info(f"Saving resource: {resource.get('title', 'Unnamed')} with analysis_completed={resource.get('analysis_completed', False)}")
            
            # Check if content needs to be saved (exists but no content_file path)
            if resource.get('content') and len(resource.get('content', '')) > 100 and not resource.get('content_file'):
                try:
                    # Create a clean filename from resource ID or URL
                    import hashlib
                    
                    if resource.get('id'):
                        safe_name = f"{resource['id']}"
                    else:
                        # Generate a hash from the URL
                        url_hash = hashlib.md5(resource.get('url', '').encode()).hexdigest()[:10]
                        safe_name = f"resource_{url_hash}"
                        
                    # Save content to permanent storage
                    content_file = os.path.join(references_dir, f"{safe_name}.txt")
                    with open(content_file, 'w', encoding='utf-8') as f:
                        f.write(resource['content'])
                        
                    # Add content_file path to resource metadata
                    resource['content_file'] = content_file
                    logger.info(f"Saved content for resource {resource.get('id', 'unknown')} to {content_file}")
                except Exception as e:
                    logger.error(f"Error saving content file for resource {resource.get('id', 'unknown')}: {e}")
                
            # Remove raw content from CSV to keep it smaller, since we've saved it to a file
            if resource.get('content_file') and resource.get('content'):
                # Only remove content if it was successfully saved to a file
                if os.path.exists(resource['content_file']):
                    logger.info(f"Removing raw content from resource {resource.get('id', 'unknown')} after saving to file")
                    resource['content'] = None  # Set to None to indicate it was intentionally removed
                
        # Now save the resources with updated content_file paths
        self.data_saver.save_resources(resources, csv_path)
        logger.info(f"Saved {len(resources)} resources to CSV after batch processing")
    
    def save_projects_csv(self, projects: List[Dict], csv_path: str = None) -> None:
        """
        Saves identified projects to CSV.
        
        This handles saving projects that were identified across multiple resources
        and might not have been properly saved by the individual resource processors.
        
        Args:
            projects: List of project dictionaries
            csv_path: Path to save CSV file
        """
        # We still need to save projects that were aggregated across multiple resources
        self.data_saver.save_projects(projects, csv_path)
        logger.info(f"Saved {len(projects)} aggregated projects to projects.csv")
    
    def _get_definitions_from_resources(self, resources: List[Dict]) -> List[Dict]:
        """
        Extract definitions from resources based on analysis results
        
        Args:
            resources: List of resource dictionaries
            
        Returns:
            List of extracted definition dictionaries
        """
        definitions = []
        for resource in resources:
            # If the resource has been analyzed and has definitions
            if resource.get('analysis_results') and isinstance(resource.get('analysis_results'), dict):
                analysis = resource.get('analysis_results', {})
                for def_item in analysis.get('definitions', []):
                    if 'term' in def_item and 'definition' in def_item:
                        definition = {
                            'term': def_item['term'],
                            'definition': def_item['definition'],
                            'tags': analysis.get('tags', [])[:5],  # Use up to 5 tags from analysis
                            'source': resource.get('url', '')
                        }
                        definitions.append(definition)
        
        return definitions
    
    def process_all(self) -> Dict[str, int]:
        """
        Main method to process resources, extract definitions, and identify projects.
        
        Returns:
            Dictionary with statistics about the processing
        """
        # Analyze resources
        updated_resources, projects = self.analyze_resources()
        
        # Extract definitions directly
        definitions = self._get_definitions_from_resources(updated_resources)
        
        # Results are already saved by the MainProcessor
        # No need to save them again
        
        return {
            "resources_processed": len(updated_resources),
            "projects_identified": len(projects),
            "definitions_extracted": len(definitions)
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyser = DataAnalyser()
    results = analyser.process_all()
    print(f"Processing complete: {json.dumps(results, indent=2)}")