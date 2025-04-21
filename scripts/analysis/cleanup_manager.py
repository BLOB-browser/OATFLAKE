#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

class CleanupManager:
    """
    Handles cleanup operations for temporary files and directories created during processing.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the CleanupManager.
        
        Args:
            data_folder: Path to the data directory
        """
        self.data_folder = data_folder
        
        # Import storage services
        from scripts.storage.content_storage_service import ContentStorageService
        from scripts.storage.temporary_storage_service import TemporaryStorageService
        
        # Initialize services
        self.content_storage = ContentStorageService(data_folder)
        self.temp_storage = TemporaryStorageService(data_folder)
    
    def cleanup_temporary_files(self):
        """Clean up all temporary files and directories after processing."""
        try:
            logger.info("Performing comprehensive cleanup of temporary files")
            
            # 1. Clean up any remaining resource temp files
            temp_files = list(Path(self.temp_storage.temp_dir).glob("*"))
            if temp_files:
                logger.info(f"Cleaning up {len(temp_files)} temporary resource files")
                for temp_file in temp_files:
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file, ignore_errors=True)
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
            
            # 2. Clean up any remaining JSONL files in multiple locations
            # 2a. Clean JSONL files in content storage temp path
            jsonl_files = list(self.content_storage.temp_storage_path.glob("*.jsonl"))
            if jsonl_files:
                logger.info(f"Cleaning up {len(jsonl_files)} remaining JSONL files in content storage")
                for jsonl_file in jsonl_files:
                    try:
                        jsonl_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove JSONL file {jsonl_file}: {e}")
            
            # 2b. Clean JSONL files in general temp directory
            temp_dir = Path(self.data_folder) / "temp"
            if temp_dir.exists():
                temp_jsonl_files = list(temp_dir.glob("*.jsonl"))
                if temp_jsonl_files:
                    logger.info(f"Cleaning up {len(temp_jsonl_files)} remaining JSONL files in temp directory")
                    for jsonl_file in temp_jsonl_files:
                        try:
                            jsonl_file.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to remove JSONL file {jsonl_file}: {e}")
            
            # 3. Log completion of cleanup
            logger.info("Temporary file cleanup completed")
        except Exception as e:
            logger.error(f"Error during temporary file cleanup: {e}")
            
    def cleanup_specific_files(self, file_patterns=None):
        """
        Clean up specific files matching the given patterns.
        
        Args:
            file_patterns: List of glob patterns to match files for cleanup
        """
        if file_patterns is None:
            file_patterns = ["*.jsonl", "*.tmp"]
            
        try:
            logger.info(f"Cleaning up files matching patterns: {file_patterns}")
            
            # Clean in temp storage
            for pattern in file_patterns:
                temp_files = list(Path(self.temp_storage.temp_dir).glob(pattern))
                if temp_files:
                    logger.info(f"Cleaning up {len(temp_files)} {pattern} files in temp storage")
                    for temp_file in temp_files:
                        try:
                            if temp_file.is_file():
                                temp_file.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to remove file {temp_file}: {e}")
                            
            # Clean in content storage temp path
            for pattern in file_patterns:
                content_files = list(self.content_storage.temp_storage_path.glob(pattern))
                if content_files:
                    logger.info(f"Cleaning up {len(content_files)} {pattern} files in content storage")
                    for content_file in content_files:
                        try:
                            if content_file.is_file():
                                content_file.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to remove file {content_file}: {e}")
                            
            logger.info("Specific file cleanup completed")
        except Exception as e:
            logger.error(f"Error during specific file cleanup: {e}")
