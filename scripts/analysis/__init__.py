#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Knowledge processing components for analyzing and extracting information from content.
This package provides a modular approach to the knowledge processing workflow.
"""

# Import main orchestrator for easy access
from scripts.analysis.knowledge_orchestrator import KnowledgeOrchestrator

# Import individual step components
from scripts.analysis.change_detector import ChangeDetector, check_for_processing_needs
from scripts.analysis.critical_content_processor import CriticalContentProcessor, process_critical_content
from scripts.analysis.markdown_processor_step import MarkdownProcessingStep, process_markdown_files
# from scripts.analysis.resource_analyzer_step import ResourceAnalyzerStep, analyze_resources  # File not found
from scripts.analysis.knowledge_base_processor import KnowledgeBaseProcessor, process_remaining_knowledge
from scripts.analysis.url_processor_step import URLProcessorStep, process_pending_urls
from scripts.analysis.vector_store_generator import VectorStoreGenerator, generate_vector_stores
from scripts.analysis.goal_extractor_step import GoalExtractorStep, extract_goals
from scripts.analysis.question_generator_step import QuestionGeneratorStep, generate_questions

__all__ = [
    # Main orchestrator
    'KnowledgeOrchestrator',
    'process_knowledge_base',
    
    # Individual step components
    'ChangeDetector',
    'check_for_processing_needs',
    'CriticalContentProcessor',
    'process_critical_content',
    'MarkdownProcessingStep',
    'process_markdown_files',
    'ResourceAnalyzerStep',
    'analyze_resources',
    'KnowledgeBaseProcessor',
    'process_remaining_knowledge',
    'URLProcessorStep', 
    'process_pending_urls',
    'VectorStoreGenerator',
    'generate_vector_stores',
    'GoalExtractorStep',
    'extract_goals',
    'QuestionGeneratorStep',
    'generate_questions'
]