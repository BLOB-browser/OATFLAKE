#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class GoalExtractorStep:
    """
    Extracts learning goals from vector stores.
    This component handles STEP 6 of the knowledge processing workflow.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the goal extractor step.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        
    async def extract_goals(self, ollama_client=None) -> Dict[str, Any]:
        """
        Extract learning goals from vector stores.
        
        Args:
            ollama_client: Optional ollama client for LLM calls
            
        Returns:
            Dictionary with processing results
        """
        logger.info("STEP 6: EXTRACTING GOALS FROM VECTOR STORES")
        logger.info("=========================================")
        
        try:
            from scripts.analysis.goal_extractor import GoalExtractor
            
            # Initialize goal extractor
            goal_extractor = GoalExtractor(self.data_folder)
            
            # Extract goals using the provided ollama client
            goals_result = await goal_extractor.extract_goals(ollama_client=ollama_client)
            
            if goals_result.get("status") == "success":
                logger.info(f"Successfully extracted {goals_result.get('stats', {}).get('goals_extracted', 0)} goals")
                return {
                    "status": "success",
                    "goals_extracted": goals_result.get("stats", {}).get("goals_extracted", 0),
                    "stores_analyzed": goals_result.get("stats", {}).get("stores_analyzed", []),
                    "duration_seconds": goals_result.get("stats", {}).get("duration_seconds", 0)
                }
            else:
                logger.warning(f"Goal extraction issue: {goals_result.get('message')}")
                return {
                    "status": "warning",
                    "message": goals_result.get("message", "Unknown issue during goal extraction")
                }
                
        except Exception as e:
            logger.error(f"Error during goal extraction: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

# Standalone function for easy import
async def extract_goals(data_folder: str, ollama_client=None) -> Dict[str, Any]:
    """
    Extract learning goals from vector stores.
    
    Args:
        data_folder: Path to data folder
        ollama_client: Optional ollama client for LLM calls
        
    Returns:
        Dictionary with processing results
    """
    extractor = GoalExtractorStep(data_folder)
    return await extractor.extract_goals(ollama_client)