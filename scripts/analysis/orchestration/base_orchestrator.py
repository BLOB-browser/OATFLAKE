#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class BaseOrchestrator:
    """
    Base class for knowledge orchestration.
    Provides initialization and cancellation functionality.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the knowledge orchestrator.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        self.processing_active = False
        self.cancel_requested = False
        self.force_url_fetch = False  # Default to not forcing URL fetching
        
    def cancel_processing(self) -> Dict[str, Any]:
        """
        Cancel any active processing.
        
        Returns:
            Status dictionary
        """
        if not self.processing_active:
            return {
                "status": "success",
                "message": "No active processing to cancel"
            }
        
        self.cancel_requested = True
        logger.info("Cancellation requested for knowledge processing")
        
        # Also cancel processing in underlying components
        from scripts.analysis.interruptible_llm import request_interrupt
        request_interrupt()
        
        return {
            "status": "success",
            "message": "Cancellation request sent"
        }
