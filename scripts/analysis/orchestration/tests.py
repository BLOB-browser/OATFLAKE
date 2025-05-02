#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the refactored KnowledgeOrchestrator modules.
"""

import unittest
import os
import sys
import logging
from unittest.mock import patch, MagicMock

# Import the components to test
from scripts.analysis.orchestration import (
    KnowledgeOrchestrator,
    BaseOrchestrator,
    URLProcessor,
    KnowledgeProcessor,
    PhasedProcessor
)

class TestKnowledgeOrchestrator(unittest.TestCase):
    """Test the KnowledgeOrchestrator class and its components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_folder = "/tmp/test_knowledge_orchestrator"
        os.makedirs(self.test_data_folder, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up test data if needed
        pass
    
    def test_base_orchestrator_initialization(self):
        """Test that BaseOrchestrator initializes correctly."""
        base = BaseOrchestrator(self.test_data_folder)
        self.assertEqual(base.data_folder, self.test_data_folder)
        self.assertFalse(base.processing_active)
        self.assertFalse(base.cancel_requested)
        self.assertFalse(base.force_url_fetch)
    
    def test_url_processor_initialization(self):
        """Test that URLProcessor initializes correctly."""
        processor = URLProcessor(self.test_data_folder)
        self.assertEqual(processor.data_folder, self.test_data_folder)
        self.assertFalse(processor.processing_active)
    
    def test_knowledge_processor_initialization(self):
        """Test that KnowledgeProcessor initializes correctly."""
        processor = KnowledgeProcessor(self.test_data_folder)
        self.assertEqual(processor.data_folder, self.test_data_folder)
        self.assertFalse(processor.processing_active)
    
    def test_phased_processor_initialization(self):
        """Test that PhasedProcessor initializes correctly."""
        processor = PhasedProcessor(self.test_data_folder)
        self.assertEqual(processor.data_folder, self.test_data_folder)
        self.assertFalse(processor.processing_active)
    
    def test_knowledge_orchestrator_initialization(self):
        """Test that KnowledgeOrchestrator initializes correctly."""
        orchestrator = KnowledgeOrchestrator(self.test_data_folder)
        self.assertEqual(orchestrator.data_folder, self.test_data_folder)
        self.assertFalse(orchestrator.processing_active)
        self.assertFalse(orchestrator.cancel_requested)
        self.assertFalse(orchestrator.force_url_fetch)
    
    @patch('scripts.analysis.interruptible_llm.request_interrupt')
    def test_cancel_processing(self, mock_request_interrupt):
        """Test the cancel_processing method."""
        orchestrator = KnowledgeOrchestrator(self.test_data_folder)
        
        # When processing is not active
        result = orchestrator.cancel_processing()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "No active processing to cancel")
        self.assertFalse(mock_request_interrupt.called)
        
        # When processing is active
        orchestrator.processing_active = True
        result = orchestrator.cancel_processing()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["message"], "Cancellation request sent")
        self.assertTrue(orchestrator.cancel_requested)
        mock_request_interrupt.assert_called_once()
    
if __name__ == "__main__":
    unittest.main()
