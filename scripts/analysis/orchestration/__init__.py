#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Knowledge orchestration components for coordinating the knowledge processing workflow.
This package provides a modular approach to knowledge orchestration.
"""

from scripts.analysis.orchestration.main import KnowledgeOrchestrator, process_knowledge_base
from scripts.analysis.orchestration.base_orchestrator import BaseOrchestrator
from scripts.analysis.orchestration.url_processor import URLProcessor
from scripts.analysis.orchestration.knowledge_processor import KnowledgeProcessor
from scripts.analysis.orchestration.phased_processor import PhasedProcessor

__all__ = [
    # Main orchestrator
    'KnowledgeOrchestrator',
    'process_knowledge_base',
    
    # Component classes
    'BaseOrchestrator',
    'URLProcessor',
    'KnowledgeProcessor',
    'PhasedProcessor'
]
