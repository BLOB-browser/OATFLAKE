#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import logging
import sys
import json
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def run_resource_analysis(host="localhost", port=8999, batch_size=5, max_resources=None, analyze_all=False):
    """
    Run resource analysis using the LLM to analyze website content from resources.csv
    """
    url = f"http://{host}:{port}/api/data/stats/knowledge/analyze-resources"
    
    # Set parameters
    params = {
        'batch_size': batch_size,
        'max_resources': max_resources,
        'analyze_all': 'true' if analyze_all else 'false'
    }
    
    logger.info(f"Making POST request to {url} with params: {params}")
    response = requests.post(url, params=params)
    
    if response.status_code == 200:
        result = response.json()
        
        if result.get('status') == 'success':
            data = result.get('data', {})
            logger.info(f"Resource analysis succeeded!")
            logger.info(f"- Resources processed: {data.get('resources_processed', 0)}")
            logger.info(f"- Resources analyzed: {data.get('resources_analyzed', 0)}")
            logger.info(f"- Resources updated: {data.get('resources_updated', 0)}")
            logger.info(f"- Definitions extracted: {data.get('definitions_extracted', 0)}")
            logger.info(f"- Projects identified: {data.get('projects_identified', 0)}")
            
            return result
        else:
            logger.error(f"Error: {result.get('message')}")
            return None
    else:
        logger.error(f"Request failed with status code: {response.status_code}")
        logger.error(f"Response: {response.text}")
        return None

def run_knowledge_processing(host="localhost", port=8999, analyze_resources=True, batch_size=5, max_resources=None, analyze_all=False):
    """
    Run the full knowledge processing pipeline including resource analysis and embedding generation
    """
    url = f"http://{host}:{port}/api/data/stats/knowledge/process"
    
    # Set parameters
    params = {
        'skip_markdown_scraping': 'true',
        'analyze_resources': 'true' if analyze_resources else 'false',
        'analyze_all_resources': 'true' if analyze_all else 'false',
        'batch_size': batch_size,
        'resource_limit': max_resources if max_resources else ''
    }
    
    logger.info(f"Making POST request to {url} with params: {params}")
    
    response = requests.post(url, params=params)
    
    if response.status_code == 200:
        result = response.json()
        
        if result.get('status') == 'success':
            data = result.get('data', {})
            
            # Show markdown processing results
            markdown_result = data.get('markdown_processing', {})
            if markdown_result.get('status') != 'skipped':
                logger.info("\nMarkdown Processing Results:")
                data_extracted = markdown_result.get('data_extracted', {})
                logger.info(f"- Resources extracted: {data_extracted.get('resources', 0)}")
                logger.info(f"- Definitions extracted: {data_extracted.get('definitions', 0)}")
                logger.info(f"- Projects extracted: {data_extracted.get('projects', 0)}")
                logger.info(f"- Methods extracted: {data_extracted.get('methods', 0)}")
            
            # Show resource analysis results
            resource_result = data.get('resource_analysis', {})
            if resource_result.get('status') != 'skipped' and resource_result.get('status') != 'error':
                logger.info("\nResource Analysis Results:")
                logger.info(f"- Resources processed: {resource_result.get('resources_processed', 0)}")
                logger.info(f"- Resources analyzed: {resource_result.get('resources_analyzed', 0)}")
                logger.info(f"- Definitions extracted: {resource_result.get('definitions_extracted', 0)}")
                logger.info(f"- Projects identified: {resource_result.get('projects_identified', 0)}")
            elif resource_result.get('status') == 'error':
                logger.error(f"Resource Analysis Error: {resource_result.get('error')}")
            
            # Show vector processing results
            vector_stats = data.get('vector_stats', {})
            logger.info("\nVector Processing Results:")
            logger.info(f"- Total documents: {vector_stats.get('total', 0)}")
            logger.info(f"- Documents vectorized: {vector_stats.get('vectorized', 0)}")
            logger.info(f"- Reference documents: {vector_stats.get('reference_docs', 0)}")
            logger.info(f"- Content documents: {vector_stats.get('content_docs', 0)}")
            
            logger.info("\nProcessing completed successfully!")
            return result
        else:
            logger.error(f"Error: {result.get('message')}")
            return None
    else:
        logger.error(f"Request failed with status code: {response.status_code}")
        logger.error(f"Response: {response.text}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Run resource analysis and knowledge processing')
    parser.add_argument('--host', default='localhost', help='Server host (default: localhost)')
    parser.add_argument('--port', type=int, default=8999, help='Server port (default: 8999)')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for analysis (default: 5)')
    parser.add_argument('--limit', type=int, default=None, help='Max resources to process (default: all)')
    parser.add_argument('--analyze-all', action='store_true', help='Analyze all resources, even if already analyzed')
    parser.add_argument('--full', action='store_true', help='Run full knowledge processing (extract, analyze, embed)')
    
    args = parser.parse_args()
    
    if args.full:
        logger.info("Running FULL knowledge processing pipeline...")
        run_knowledge_processing(
            host=args.host,
            port=args.port,
            analyze_resources=True,  # Always analyze resources in full mode
            batch_size=args.batch_size,
            max_resources=args.limit,
            analyze_all=args.analyze_all
        )
    else:
        logger.info("Running RESOURCE ANALYSIS only...")
        run_resource_analysis(
            host=args.host,
            port=args.port,
            batch_size=args.batch_size,
            max_resources=args.limit,
            analyze_all=args.analyze_all
        )

if __name__ == "__main__":
    main()