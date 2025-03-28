#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import logging
import sys
import json
import time
import argparse
from pathlib import Path
import pandas as pd
from datetime import datetime

from scripts.data.markdown_processor import MarkdownProcessor
from scripts.data.data_processor import DataProcessor
from scripts.services.data_analyser import DataAnalyser

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_config_path():
    """Get the path to the config file"""
    local_config = Path.home() / '.blob' / 'config.json'
    if local_config.exists():
        return local_config
    
    # If it doesn't exist, use a default location
    return Path("config.json")

async def process_knowledge_base(
    analyze_resources=True, 
    analyze_all=False,
    batch_size=5,
    resource_limit=None,
    skip_scraping=True,
    group_id="default"
):
    """
    Process the entire knowledge base pipeline:
    1. Process markdown files to extract resources
    2. Analyze resources with LLM
    3. Generate vector embeddings
    
    Args:
        analyze_resources: Whether to analyze resources with LLM
        analyze_all: Whether to analyze all resources (even previously analyzed ones)
        batch_size: Number of resources to process in each batch
        resource_limit: Maximum number of resources to process
        skip_scraping: Whether to skip web scraping during markdown processing
        group_id: Group ID for vector store organization
    """
    start_time = time.time()
    logger.info("Starting knowledge base processing")
    
    try:
        # Get data path from config
        config_path = get_config_path()
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        data_path = Path(config.get('data_path', ''))
        logger.info(f"Using data path: {data_path}")
        
        # Step 1: Process markdown files
        logger.info("\n==== STEP 1: PROCESSING MARKDOWN FILES ====")
        markdown_path = data_path / "markdown"
        markdown_files = list(markdown_path.glob("**/*.md")) if markdown_path.exists() else []
        
        if markdown_files:
            logger.info(f"Found {len(markdown_files)} markdown files to process")
            markdown_processor = MarkdownProcessor(data_path, group_id)
            
            # Process markdown files to extract resources
            # We set analyze_resources=False to handle analysis in a separate step
            markdown_result = await markdown_processor.process_markdown_files(
                skip_scraping=skip_scraping,
                analyze_resources=False
            )
            
            logger.info(f"Markdown processing completed:")
            logger.info(f"- Files processed: {markdown_result.get('files_processed', 0)}")
            data_extracted = markdown_result.get('data_extracted', {})
            logger.info(f"- Resources: {data_extracted.get('resources', 0)}")
            logger.info(f"- Definitions: {data_extracted.get('definitions', 0)}")
            logger.info(f"- Projects: {data_extracted.get('projects', 0)}")
            logger.info(f"- Methods: {data_extracted.get('methods', 0)}")
        else:
            logger.info("No markdown files found to process")
            markdown_result = {"status": "skipped", "data_extracted": {}}
        
        # Step 2: Analyze resources with LLM
        logger.info("\n==== STEP 2: ANALYZING RESOURCES WITH LLM ====")
        resources_path = data_path / "resources.csv"
        
        resource_analysis_result = {}
        if resources_path.exists():
            # Count resources for reporting
            try:
                df = pd.read_csv(resources_path)
                total_resources = len(df)
                
                # Find resources that need analysis
                needs_analysis = []
                incomplete_with_results = 0
                
                for _, row in df.iterrows():
                    # Check if resource has analysis_results but not marked as completed
                    has_results = bool(row.get('analysis_results'))
                    is_completed = row.get('analysis_completed', False)
                    
                    if has_results and not is_completed:
                        incomplete_with_results += 1
                        # These resources have results but processing wasn't completed properly
                        logger.warning(f"Resource '{row.get('title', 'Unnamed')}' has analysis_results but analysis_completed=False")
                    
                    # Add to processing list if:
                    # 1. Force reanalysis is enabled OR
                    # 2. Missing tags OR
                    # 3. Missing analysis_results OR
                    # 4. Has results but analysis_completed=False (crashed during processing)
                    if analyze_all or not row.get('tags') or not row.get('analysis_results') or (has_results and not is_completed):
                        needs_analysis.append(row)
                
                logger.info(f"Found {total_resources} total resources")
                logger.info(f"Found {len(needs_analysis)} resources that need analysis")
                if incomplete_with_results > 0:
                    logger.warning(f"Found {incomplete_with_results} resources with results but incomplete processing (will be reprocessed)")
                
                if analyze_resources and (len(needs_analysis) > 0 or analyze_all):
                    # Initialize data analyzer
                    analyzer = DataAnalyser()
                    
                    # Determine how many resources to process
                    if resource_limit is not None and resource_limit < len(needs_analysis):
                        actual_limit = resource_limit
                        logger.info(f"Limiting analysis to {actual_limit} resources as requested")
                    else:
                        actual_limit = None  # Process all that need analysis
                    
                    # Process resources with LLM
                    logger.info(f"Analyzing resources with batch_size={batch_size}, force_reanalysis={analyze_all}")
                    updated_resources, projects = analyzer.analyze_resources(
                        csv_path=str(resources_path),
                        batch_size=batch_size,
                        max_resources=actual_limit,
                        force_reanalysis=analyze_all
                    )
                    
                    # Get definitions extracted during analysis
                    definitions = analyzer._get_definitions_from_resources(updated_resources)
                    
                    # Count resources that were actually processed by the LLM and properly completed
                    resources_with_llm = sum(1 for r in updated_resources 
                                        if r.get('analysis_results') and 
                                        isinstance(r.get('analysis_results'), dict))
                    
                    # Count resources that were fully completed (not just having results)
                    resources_fully_completed = sum(1 for r in updated_resources 
                                                if r.get('analysis_completed', False) and
                                                r.get('analysis_results') and 
                                                isinstance(r.get('analysis_results'), dict))
                    
                    # Count resources with results but not marked as completed (processing crashed)
                    resources_incomplete = resources_with_llm - resources_fully_completed
                    
                    logger.info(f"Resources analyzed: {resources_with_llm}")
                    logger.info(f"Resources fully completed: {resources_fully_completed}")
                    if resources_incomplete > 0:
                        logger.warning(f"Resources with partial results (crashed): {resources_incomplete}")
                    logger.info(f"Definitions extracted: {len(definitions)}")
                    logger.info(f"Projects identified: {len(projects)}")
                    
                    # Display sample of the processed resources
                    if updated_resources and len(updated_resources) > 0:
                        sample = updated_resources[0]
                        logger.info(f"\nSample resource: {sample.get('title')}")
                        logger.info(f"URL: {sample.get('url')}")
                        logger.info(f"Description: {sample.get('description')}")
                        logger.info(f"Tags: {sample.get('tags')}")
                    
                    resource_analysis_result = {
                        "resources_processed": len(updated_resources),
                        "resources_analyzed": resources_with_llm,
                        "resources_fully_completed": resources_fully_completed,
                        "resources_incomplete": resources_incomplete,
                        "resources_updated": sum(1 for r in updated_resources 
                                            if r.get('tags') and 
                                            isinstance(r.get('tags'), list) and 
                                            len(r.get('tags')) > 0),
                        "definitions_extracted": len(definitions),
                        "projects_identified": len(projects)
                    }
                else:
                    if not analyze_resources:
                        logger.info("Resource analysis skipped (not requested)")
                    elif len(needs_analysis) == 0:
                        logger.info("Resource analysis skipped (no resources need analysis)")
                    
                    resource_analysis_result = {"status": "skipped"}
            except Exception as resource_error:
                logger.error(f"Error during resource analysis: {resource_error}", exc_info=True)
                resource_analysis_result = {"status": "error", "error": str(resource_error)}
        else:
            logger.info("No resources.csv found, skipping analysis")
            resource_analysis_result = {"status": "skipped", "reason": "no resources.csv found"}
        
        # Step 3: Generate embeddings
        logger.info("\n==== STEP 3: GENERATING VECTOR EMBEDDINGS ====")
        data_processor = DataProcessor(data_path, group_id)
        
        # Process all knowledge base data and generate embeddings
        embedding_result = await data_processor.process_knowledge_base(incremental=True)
        
        logger.info(f"Embedding generation completed:")
        logger.info(f"- Total documents: {embedding_result.get('stats', {}).get('total', 0)}")
        logger.info(f"- Documents processed: {embedding_result.get('stats', {}).get('processed', 0)}")
        logger.info(f"- Documents skipped: {embedding_result.get('stats', {}).get('skipped', 0)}")
        
        # Calculate total time
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"\nTotal processing time: {duration:.2f} seconds")
        
        # Return combined results
        return {
            "status": "success",
            "markdown_processing": markdown_result,
            "resource_analysis": resource_analysis_result,
            "vector_processing": embedding_result,
            "processing_time": f"{duration:.2f} seconds"
        }
        
    except Exception as e:
        logger.error(f"Error in knowledge base processing: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e)
        }

async def main():
    """Command-line interface to run knowledge base processing"""
    parser = argparse.ArgumentParser(description='Process the knowledge base')
    parser.add_argument('--no-analysis', action='store_true', help='Skip resource analysis')
    parser.add_argument('--analyze-all', action='store_true', help='Reanalyze all resources')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size for resource analysis')
    parser.add_argument('--limit', type=int, default=None, help='Max resources to analyze')
    parser.add_argument('--scrape', action='store_true', help='Scrape websites from markdown links')
    parser.add_argument('--group-id', type=str, default="default", help='Group ID for vector stores')
    
    args = parser.parse_args()
    
    # Run with the provided arguments
    logger.info("Starting knowledge base processing with the following options:")
    logger.info(f"- analyze_resources: {not args.no_analysis}")
    logger.info(f"- analyze_all: {args.analyze_all}")
    logger.info(f"- batch_size: {args.batch_size}")
    logger.info(f"- resource_limit: {args.limit}")
    logger.info(f"- skip_scraping: {not args.scrape}")
    logger.info(f"- group_id: {args.group_id}")
    
    result = await process_knowledge_base(
        analyze_resources=not args.no_analysis,
        analyze_all=args.analyze_all,
        batch_size=args.batch_size,
        resource_limit=args.limit,
        skip_scraping=not args.scrape,
        group_id=args.group_id
    )
    
    if result["status"] == "success":
        logger.info("Knowledge base processing completed successfully!")
    else:
        logger.error(f"Knowledge base processing failed: {result.get('error')}")
    
    return result

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())