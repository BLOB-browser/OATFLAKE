#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Dict, Any, List
import asyncio

logger = logging.getLogger(__name__)

class QuestionGeneratorStep:
    """
    Generates questions based on processed knowledge.
    This component handles STEP 7 of the knowledge processing workflow.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the question generator step.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = Path(data_folder)
    
    async def generate_questions(self, num_questions: int = 15) -> Dict[str, Any]:
        """
        Generate questions based on the processed knowledge base.
        
        Args:
            num_questions: Number of questions to generate
            
        Returns:
            Dictionary with question generation results
        """
        logger.info("STEP 7: GENERATING QUESTIONS FROM PROCESSED KNOWLEDGE")
        logger.info("===================================================")
        
        try:
            from scripts.services.question_generator import generate_questions, save_questions
            
            # Generate questions with more than requested to ensure variety
            questions = await generate_questions(num_questions=num_questions)
            
            if not questions:
                logger.warning("No questions could be generated")
                return {
                    "questions_generated": 0,
                    "questions_saved": False,
                    "error": "No questions generated - possible timeout or empty context"
                }
            
            # Save generated questions
            questions_saved = await save_questions(questions)
            
            result = {
                "questions_generated": len(questions),
                "questions_saved": questions_saved
            }
            
            logger.info(f"Successfully generated {len(questions)} new questions from knowledge base")
            return result
            
        except Exception as question_error:
            logger.error(f"Error generating questions: {question_error}", exc_info=True)
            return {
                "questions_generated": 0,
                "questions_saved": False,
                "error": str(question_error)
            }


# Standalone function for easy import
async def generate_questions(data_folder: str, num_questions: int = 15) -> Dict[str, Any]:
    """
    Generate questions based on the processed knowledge base.
    
    Args:
        data_folder: Path to data folder
        num_questions: Number of questions to generate
        
    Returns:
        Dictionary with question generation results
    """
    generator = QuestionGeneratorStep(data_folder)
    return await generator.generate_questions(num_questions=num_questions)
