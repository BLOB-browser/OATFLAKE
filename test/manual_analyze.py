#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Manual analysis script to process resources without using LLM.
This script will generate placeholder data for resources to enable vector embedding.
"""

import pandas as pd
import json
import logging
import asyncio
import os
from pathlib import Path
from datetime import datetime

from scripts.data.data_processor import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from user's .blob directory"""
    config_path = Path.home() / '.blob' / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default config
        return {"data_path": str(Path.cwd() / "data")}

def manually_update_resources(csv_path):
    """
    Manually update resources with placeholder data to enable vector embedding.
    """
    logger.info(f"Loading resources from {csv_path}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Found {len(df)} resources")
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return False
    
    # Count resources that need updates
    needs_description = sum(1 for _, row in df.iterrows() 
                         if not row.get('description') or str(row.get('description', '')).strip() == '')
    
    needs_tags = sum(1 for _, row in df.iterrows() 
                  if not row.get('tags') or not isinstance(row.get('tags'), list) or len(row.get('tags', [])) == 0)
    
    needs_analysis = sum(1 for _, row in df.iterrows() 
                      if not row.get('analysis_results') or not isinstance(row.get('analysis_results'), dict))
    
    logger.info(f"Resources needing updates:")
    logger.info(f"  - Missing description: {needs_description}")
    logger.info(f"  - Missing tags: {needs_tags}")
    logger.info(f"  - Missing analysis: {needs_analysis}")
    
    # Update each resource
    updated_count = 0
    for idx, row in df.iterrows():
        resource = row.to_dict()
        resource_id = idx + 1
        title = resource.get('title', f'Resource {resource_id}')
        url = resource.get('url', '')
        
        updated = False
        
        # Add description if missing
        if not resource.get('description') or str(resource.get('description', '')).strip() == '':
            resource['description'] = f"Resource about {title} from {url}"
            updated = True
            
        # Add tags if missing
        if not resource.get('tags') or not isinstance(resource.get('tags'), list) or len(resource.get('tags', [])) == 0:
            resource['tags'] = ["resource", "website", "information", "content", "data"]
            updated = True
            
        # Add analysis_results if missing
        if not resource.get('analysis_results') or not isinstance(resource.get('analysis_results'), dict):
            resource['analysis_results'] = {
                "definitions": [],
                "projects": []
            }
            updated = True
            
        # Update dataframe if changes were made
        if updated:
            for key, value in resource.items():
                df.at[idx, key] = value
            updated_count += 1
            
    logger.info(f"Updated {updated_count} resources with placeholder data")
    
    # Save the updated CSV
    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved updated resources to {csv_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
        return False

async def generate_vector_embeddings(data_path):
    """Generate vector embeddings for all resources"""
    logger.info(f"Generating vector embeddings for resources in {data_path}")
    
    try:
        # Create DataProcessor instance
        processor = DataProcessor(data_path)
        
        # Process knowledge base with incremental=False to force processing of all documents
        result = await processor.process_knowledge_base(incremental=False)
        
        logger.info(f"Vector embedding generation complete:")
        logger.info(f"  - Total documents: {result.get('total_documents', 0)}")
        logger.info(f"  - Reference documents: {result.get('reference_documents', 0)}")
        logger.info(f"  - Content documents: {result.get('content_documents', 0)}")
        
        return result
    except Exception as e:
        logger.error(f"Error generating vector embeddings: {e}")
        return None

async def main():
    """Main function to run the manual analysis"""
    try:
        # Load configuration
        config = load_config()
        data_path = config.get('data_path', '')
        
        if not data_path:
            logger.error("No data path configured. Please set up config.json")
            return
            
        data_path = Path(data_path)
        resources_path = data_path / "resources.csv"
        
        if not resources_path.exists():
            logger.error(f"Resources file not found at {resources_path}")
            return
            
        # Step 1: Manually update resources with placeholder data
        logger.info("STEP 1: MANUALLY UPDATING RESOURCES")
        logger.info("=================================")
        
        success = manually_update_resources(str(resources_path))
        
        if not success:
            logger.error("Failed to update resources.")
            return
            
        # Step 2: Generate vector embeddings
        logger.info("\nSTEP 2: GENERATING VECTOR EMBEDDINGS")
        logger.info("=================================")
        
        result = await generate_vector_embeddings(data_path)
        
        if result:
            logger.info("\nMANUAL ANALYSIS COMPLETE")
            logger.info("======================")
            logger.info("Resources have been updated with placeholder data and vector embeddings have been generated.")
            logger.info("You can now use the RAG system for querying.")
        else:
            logger.error("Failed to complete vector embedding generation.")
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    logger.info("Starting manual resource analysis...")
    asyncio.run(main())