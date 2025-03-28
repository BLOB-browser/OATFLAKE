#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import logging
import sys
import time
import json
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def trigger_full_processing(host="localhost", port=8999):
    """
    Trigger the complete knowledge processing pipeline via the API endpoint:
    1. Process PDFs from materials.csv and methods from methods.csv (highest priority)
    2. Process markdown files 
    3. Analyze resources with LLM
    4. Generate remaining vector embeddings
    5. Generate questions
    6. Rebuild all FAISS indexes to ensure consistency
    """
    start_time = time.time()
    
    # Construct the API URL
    url = f"http://{host}:{port}/api/data/stats/knowledge/process"
    
    # Set parameters explicitly to make sure they're passed correctly
    params = {
        "skip_markdown_scraping": "false",    # Process and scrape markdown files
        "analyze_resources": "true",          # Analyze resources with LLM
        "analyze_all_resources": "false",     # Only analyze resources that need it
        "batch_size": "1",                    # Process 1 resource at a time for stability
        "resource_limit": "",                 # Process all resources
        "force_update": "false"               # Use incremental updates unless explicitly forced
    }
    
    logger.info(f"Triggering full knowledge processing at: {url}")
    logger.info(f"Parameters: {params}")
    
    processing_success = False
    
    # Make the API request for processing
    try:
        response = requests.post(url, params=params)
        
        # Check for successful response
        if response.status_code == 200:
            result = response.json()
            
            if result.get("status") == "success":
                processing_success = True
                logger.info("🎉 Knowledge processing completed successfully!")
                
                # Extract processing statistics
                data = result.get("data", {})
                
                # Markdown processing results
                markdown = data.get("markdown_processing", {})
                if markdown.get("status") != "skipped":
                    markdown_data = markdown.get("data_extracted", {})
                    logger.info("\n📄 Markdown Processing Results:")
                    logger.info(f"- Resources extracted: {markdown_data.get('resources', 0)}")
                    logger.info(f"- Definitions extracted: {markdown_data.get('definitions', 0)}")
                    logger.info(f"- Projects extracted: {markdown_data.get('projects', 0)}")
                    logger.info(f"- Methods extracted: {markdown_data.get('methods', 0)}")
                
                # Resource analysis results
                analysis = data.get("resource_analysis", {})
                if analysis.get("status") != "skipped":
                    logger.info("\n🔍 Resource Analysis Results:")
                    logger.info(f"- Resources processed: {analysis.get('resources_processed', 0)}")
                    logger.info(f"- Resources analyzed: {analysis.get('resources_analyzed', 0)}")
                    logger.info(f"- Resources updated: {analysis.get('resources_updated', 0)}")
                    logger.info(f"- Definitions extracted: {analysis.get('definitions_extracted', 0)}")
                    logger.info(f"- Projects identified: {analysis.get('projects_identified', 0)}")
                
                # Vector processing results
                vector = data.get("stats", {})
                logger.info("\n🧠 Vector Processing Results:")
                logger.info(f"- Total documents: {vector.get('total', 0)}")
                logger.info(f"- Documents processed: {vector.get('processed', 0)}")
                logger.info(f"- Documents skipped: {vector.get('skipped', 0)}")
                logger.info(f"- Failed documents: {vector.get('failed', 0)}")
                
                # Document counts by type
                by_type = vector.get("by_type", {})
                logger.info("\n📊 Document Counts by Type:")
                logger.info(f"- Definitions: {by_type.get('definitions', 0)}")
                logger.info(f"- Methods: {by_type.get('methods', 0)}")
                logger.info(f"- Materials: {by_type.get('materials', 0)}")
                logger.info(f"- Projects: {by_type.get('projects', 0)}")
                logger.info(f"- Resources: {by_type.get('resources', 0)}")
                
                # Question generation results
                questions = data.get("questions", {})
                logger.info("\n❓ Question Generation Results:")
                logger.info(f"- Questions generated: {questions.get('questions_generated', 0)}")
                logger.info(f"- Questions saved: {questions.get('questions_saved', False)}")
            else:
                logger.error(f"Processing failed: {result.get('message')}")
        else:
            logger.error(f"API request failed with status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
    except Exception as e:
        logger.error(f"Error making API request for processing: {e}")
    
    # CRITICAL STEP: Rebuild all FAISS indexes for consistency
    logger.info("\n🔄 Rebuilding all FAISS indexes to ensure consistency...")
    rebuild_url = f"http://{host}:{port}/api/rebuild-faiss-indexes"
    
    rebuild_success = False
    try:
        rebuild_response = requests.post(rebuild_url)
        
        if rebuild_response.status_code == 200:
            rebuild_result = rebuild_response.json()
            
            if rebuild_result.get("status") == "success":
                rebuild_success = True
                logger.info("✅ FAISS index rebuild completed successfully!")
                
                # Extract rebuild statistics
                stores_rebuilt = rebuild_result.get("stores_rebuilt", [])
                total_documents = rebuild_result.get("total_documents", 0)
                processing_time = rebuild_result.get("processing_time_seconds", 0)
                
                logger.info(f"- Stores rebuilt: {', '.join(stores_rebuilt)}")
                logger.info(f"- Total documents indexed: {total_documents}")
                logger.info(f"- Rebuild time: {processing_time:.2f} seconds")
            else:
                logger.error(f"FAISS rebuild failed: {rebuild_result.get('error', 'Unknown error')}")
        else:
            logger.error(f"FAISS rebuild API request failed: {rebuild_response.status_code}")
            logger.error(f"Response: {rebuild_response.text}")
    except Exception as e:
        logger.error(f"Error making API request for FAISS rebuild: {e}")
    
    # Calculate total processing time
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"\n⏱️ Total processing time: {duration:.2f} seconds")
    
    # Return combined success status
    return processing_success and rebuild_success

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description="Trigger full knowledge processing pipeline")
    parser.add_argument("--host", default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=8999, help="API port (default: 8999)")
    
    args = parser.parse_args()
    
    # Run the processing
    trigger_full_processing(host=args.host, port=args.port)

if __name__ == "__main__":
    main()