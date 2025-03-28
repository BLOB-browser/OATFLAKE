#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class TemporaryStorageService:
    """
    Service for managing temporary files during processing.
    Handles creation, writing, reading, and cleanup of temporary files.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the temporary storage service.
        
        Args:
            data_folder: Base data folder path
        """
        self.data_folder = data_folder
        self.temp_dir = Path(data_folder) / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.active_files = {}  # Track active temporary files
        
        logger.info(f"TemporaryStorageService initialized at {self.temp_dir}")
        
        # Cleanup old files on initialization
        self.cleanup_old_files()
        
    def create_temp_file(self, prefix: str = "", suffix: str = ".txt") -> Path:
        """
        Create a new temporary file with a unique name.
        
        Args:
            prefix: Optional prefix for the filename
            suffix: File extension to use (default: .txt)
            
        Returns:
            Path object for the created temporary file
        """
        # Create a unique filename based on timestamp and optional prefix
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_part = str(int(time.time() * 1000) % 1000)
        
        if prefix:
            prefix = prefix.replace(" ", "_").lower()[:30]
            filename = f"{prefix}_{timestamp}_{random_part}{suffix}"
        else:
            filename = f"temp_{timestamp}_{random_part}{suffix}"
            
        temp_path = self.temp_dir / filename
        
        # Track this file as active
        self.active_files[str(temp_path)] = {
            "created_at": datetime.now(),
            "size": 0,
            "last_access": datetime.now(),
            "path": temp_path
        }
        
        logger.info(f"Created temporary file at {temp_path}")
        return temp_path
        
    def write_to_file(self, file_path: Union[str, Path], content: str, mode: str = "a") -> int:
        """
        Write content to a temporary file.
        
        Args:
            file_path: Path to the file to write to
            content: Content to write
            mode: File mode ('w' for overwrite, 'a' for append)
            
        Returns:
            Number of bytes written
        """
        file_path = Path(file_path)
        try:
            with open(file_path, mode, encoding='utf-8') as f:
                f.write(content)
                bytes_written = len(content.encode('utf-8'))
                
            # Update tracking information
            path_str = str(file_path)
            if path_str in self.active_files:
                self.active_files[path_str]["size"] += bytes_written
                self.active_files[path_str]["last_access"] = datetime.now()
                
            return bytes_written
            
        except Exception as e:
            logger.error(f"Error writing to {file_path}: {e}")
            return 0
            
    def read_from_file(self, file_path: Union[str, Path], max_size: int = None) -> str:
        """
        Read content from a temporary file.
        
        Args:
            file_path: Path to the file to read from
            max_size: Maximum number of characters to read (None for all)
            
        Returns:
            File content as string
        """
        file_path = Path(file_path)
        try:
            # Update last access time
            path_str = str(file_path)
            if path_str in self.active_files:
                self.active_files[path_str]["last_access"] = datetime.now()
                
            if not file_path.exists():
                logger.warning(f"Temporary file not found: {file_path}")
                return ""
                
            if max_size is None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read(max_size)
                    
        except Exception as e:
            logger.error(f"Error reading from {file_path}: {e}")
            return ""
            
    def append_to_file(self, file_path: Union[str, Path], content: str) -> int:
        """Alias for write_to_file with append mode."""
        return self.write_to_file(file_path, content, mode="a")
        
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """
        Delete a temporary file.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        file_path = Path(file_path)
        path_str = str(file_path)
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted temporary file: {file_path}")
                
                # Remove from tracking
                if path_str in self.active_files:
                    del self.active_files[path_str]
                    
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {e}")
            return False
            
    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up old temporary files based on creation time.
        
        Args:
            max_age_hours: Maximum age in hours before a file is deleted
            
        Returns:
            Number of files cleaned up
        """
        try:
            # Count files before cleanup
            all_files = list(self.temp_dir.glob("*.txt"))
            if not all_files:
                return 0
                
            logger.info(f"Found {len(all_files)} temporary files for cleanup")
            
            # Clean up old files
            current_time = datetime.now().timestamp()
            max_age_seconds = max_age_hours * 3600  # Convert hours to seconds
            cleaned_count = 0
            
            for old_file in all_files:
                # Check file age
                try:
                    file_age = current_time - os.path.getmtime(old_file)
                    if file_age > max_age_seconds:
                        os.unlink(old_file)
                        cleaned_count += 1
                        
                        # Remove from tracking if present
                        path_str = str(old_file)
                        if path_str in self.active_files:
                            del self.active_files[path_str]
                            
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary file {old_file}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} old temporary files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")
            return 0
            
    def get_active_files_info(self) -> Dict[str, Any]:
        """Get information about currently active temporary files."""
        return {
            "active_file_count": len(self.active_files),
            "total_size_bytes": sum(f["size"] for f in self.active_files.values()),
            "files": self.active_files
        }
        
    def cleanup_all(self) -> int:
        """Delete all temporary files in the temp directory."""
        try:
            all_files = list(self.temp_dir.glob("*.txt"))
            count = len(all_files)
            
            for file_path in all_files:
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.warning(f"Error deleting {file_path}: {e}")
                    
            # Clear tracking
            self.active_files = {}
            
            logger.info(f"Cleaned up all {count} temporary files")
            return count
            
        except Exception as e:
            logger.error(f"Error during complete cleanup: {e}")
            return 0