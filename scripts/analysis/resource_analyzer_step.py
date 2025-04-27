#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

class ResourceAnalyzerStep:
    """
    Analyzes resources extracted from markdown files or other sources.
    This component handles STEP 3 of the knowledge processing workflow.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the resource analyzer step.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = Path(data_folder)
        self.resources_path = self.data_folder / "resources.csv"
    
    async def analyze_resources(
        self, 
        analyze_all: bool = False,
        batch_size: int = 1, 
        limit: int = None,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze resources with LLM to extract definitions, projects, and other insights.
        
        Args:
            analyze_all: If True, analyze all resources even if already analyzed
            batch_size: Number of resources to process at once
            limit: Maximum number of resources to process (None for all)
            force_update: If True, force reanalysis even of already analyzed resources
            
        Returns:
            Dictionary with analysis results
        """
        logger.info("STEP 3: ANALYZING RESOURCES WITH LLM")
        logger.info("===================================")
        
        try:
            from scripts.services.data_analyser import DataAnalyser
            
            # Check if resources file exists
            if not self.resources_path.exists():
                logger.info("No resources.csv file found - skipping resource analysis")
                return {
                    "status": "skipped",
                    "reason": "no resources.csv file found",
                    "resources_analyzed": 0
                }
            
            # Read resources to determine if analysis is needed
            try:
                df = pd.read_csv(self.resources_path)
                total_resources = len(df)
                
                logger.info(f"Found {total_resources} total resources")
                
                # Find resources that need analysis if not analyzing all
                needs_analysis_count = 0
                if not analyze_all:
                    # Check how many resources need analysis (analysis_completed is False)
                    needs_analysis = []
                    for _, row in df.iterrows():
                        # Convert to boolean and check if analysis_completed is False
                        if pd.isna(row.get('analysis_completed')) or row.get('analysis_completed') == False:
                            needs_analysis.append(row)
                    needs_analysis_count = len(needs_analysis)
                    logger.info(f"Found {needs_analysis_count} resources with analysis_completed=False that need analysis")
                
                # Skip analysis ONLY if nothing needs analysis AND we're not forcing reanalysis
                if needs_analysis_count == 0 and not analyze_all and not force_update:
                    logger.info("All resources already have analysis results - skipping resource analysis")
                    return {
                        "status": "skipped",
                        "reason": "all resources already analyzed",
                        "resources_analyzed": 0
                    }
                
            except Exception as e:
                logger.error(f"Error reading resources.csv: {e}", exc_info=True)
                return {
                    "status": "error",
                    "error": f"Error reading resources.csv: {str(e)}",
                    "resources_analyzed": 0
                }
                
            # Initialize data analyzer
            analyzer = DataAnalyser()
            
            try:
                logger.info(f"Starting resource analysis with LLM analyzer")
                logger.info(f"Processing options: batch_size={batch_size}, limit={limit}, analyze_all={analyze_all}")
                
                # Analyze resources using LLM
                updated_resources, projects = analyzer.analyze_resources(
                    csv_path=str(self.resources_path),
                    batch_size=batch_size,
                    max_resources=limit,
                    force_reanalysis=analyze_all or force_update
                )
                
                # Get extracted definitions
                definitions = analyzer._get_definitions_from_resources(updated_resources)
                
                # We can't get methods directly from processing_result since analyze_resources doesn't return it
                # Instead, extract methods from CSV or create an empty list
                methods_path = os.path.join(self.data_folder, 'methods.csv')
                methods = []
                
                # If methods CSV exists, read it
                if os.path.exists(methods_path):
                    try:
                        methods_df = pd.read_csv(methods_path)
                        # Get only recent methods (last 24 hours)
                        recent_time = (datetime.now() - pd.Timedelta(days=1)).isoformat()
                        if 'created_at' in methods_df.columns:
                            recent_methods = methods_df[methods_df['created_at'] > recent_time]
                            methods = recent_methods.to_dict('records')
                            logger.info(f"Retrieved {len(methods)} recently added methods from {methods_path}")
                    except Exception as m_err:
                        logger.error(f"Error retrieving methods from CSV: {m_err}")
                
                # Count resources that were actually processed by the LLM
                resources_with_llm = sum(1 for r in updated_resources 
                                     if r.get('analysis_results') and 
                                     isinstance(r.get('analysis_results'), dict))
                
                logger.info(f"Successfully processed {resources_with_llm} resources with LLM analysis")
                
                # Save processed data
                analyzer.save_updated_resources(updated_resources)
                logger.info(f"✅ Resources saved successfully, moving to next step in pipeline")
                
                # Save projects separately if any were found
                if projects:
                    analyzer.save_projects_csv(projects)
                    logger.info(f"Saved {len(projects)} identified projects to projects.csv")
                
                # Save methods separately if any were found
                if methods:
                    # Use data_saver directly to ensure methods get saved
                    from scripts.analysis.data_saver import DataSaver
                    from utils.config import BACKEND_CONFIG
                    data_saver = DataSaver(BACKEND_CONFIG['data_path'])
                    data_saver.save_methods(methods)
                    logger.info(f"Saved {len(methods)} extracted methods to methods.csv")
                
                logger.info(f"Found {len(definitions)} definitions, {len(projects)} projects, and {len(methods)} methods")
                logger.info(f"✅ Resource analysis phase complete - continuing to vector embedding phase")
                
                return {
                    "status": "completed",
                    "resources_processed": len(updated_resources),
                    "resources_analyzed": resources_with_llm,
                    "resources_updated": sum(1 for r in updated_resources 
                                        if r.get('tags') and 
                                        isinstance(r.get('tags'), list) and 
                                        len(r.get('tags')) > 0),
                    "definitions_extracted": len(definitions),
                    "projects_identified": len(projects)
                }
                
            except Exception as analyze_error:
                logger.error(f"Error during resource analysis: {analyze_error}", exc_info=True)
                return {
                    "status": "error",
                    "error": str(analyze_error),
                    "resources_analyzed": 0
                }
                
        except Exception as e:
            logger.error(f"Error in resource analyzer step: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "resources_analyzed": 0
            }


# Standalone function for easy import
async def analyze_resources(
    data_folder: str, 
    analyze_all: bool = False, 
    batch_size: int = 1, 
    limit: int = None,
    force_update: bool = False
) -> Dict[str, Any]:
    """
    Analyze resources to extract definitions, projects, and other insights.
    
    Args:
        data_folder: Path to data folder
        analyze_all: If True, analyze all resources even if already analyzed
        batch_size: Number of resources to process at once
        limit: Maximum number of resources to process (None for all)
        force_update: If True, force reanalysis even of already analyzed resources
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = ResourceAnalyzerStep(data_folder)
    return await analyzer.analyze_resources(
        analyze_all=analyze_all, 
        batch_size=batch_size, 
        limit=limit,
        force_update=force_update
    )
