"""
Conversation logging service to store search conversations in multiple formats.

This module logs all search conversations to txt, json, and csv files in the base data folder
from config.json for analysis and record-keeping purposes.
"""

import json
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class ConversationLogger:
    """Logger for storing search conversations in multiple formats."""
    
    def __init__(self, base_data_path: str = None):
        """
        Initialize the conversation logger.
        
        Args:
            base_data_path: Base directory path from config.json, falls back to './data'
        """
        # Set up base conversation directory
        if base_data_path:
            self.base_path = Path(base_data_path)
        else:
            # Try to read from config.json
            try:
                config_path = Path("config.json")
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    self.base_path = Path(config.get('data_path', './data'))
                else:
                    self.base_path = Path('./data')
            except Exception as e:
                logger.warning(f"Could not read config, using default data path: {e}")
                self.base_path = Path('./data')
                
        # Create conversations subdirectory
        self.conversations_dir = self.base_path / 'conversations'
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up file paths for different formats
        today = datetime.now().strftime('%Y-%m-%d')
        self.txt_file = self.conversations_dir / f'conversations_{today}.txt'
        self.json_file = self.conversations_dir / f'conversations_{today}.json'
        self.csv_file = self.conversations_dir / f'conversations_{today}.csv'
        
        # Initialize CSV file with headers if it doesn't exist
        self._init_csv_file()
        
        # Load existing JSON conversations
        self.conversations_data = self._load_json_conversations()
        
        logger.info(f"Initialized conversation logger in: {self.conversations_dir}")
    
    def _init_csv_file(self) -> None:
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_file.exists():
            headers = [
                'timestamp', 'request_id', 'query', 'response_preview', 
                'word_count', 'provider', 'model', 'skip_search', 
                'retrieval_time_ms', 'generation_time_ms', 'total_time_ms',
                'references_count', 'content_count'
            ]
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def _load_json_conversations(self) -> List[Dict[str, Any]]:
        """Load existing conversations from JSON file."""
        if self.json_file.exists():
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing JSON conversations: {e}")
        return []
    
    def _save_json_conversations(self) -> None:
        """Save conversations to JSON file."""
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversations_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving JSON conversations: {e}")
    
    def log_conversation(self, conversation_data: Dict[str, Any]) -> None:
        """
        Log a conversation to all three formats (txt, json, csv).
        
        Args:
            conversation_data: Dictionary containing conversation details
        """
        try:
            # Extract key information
            timestamp = conversation_data.get('timestamp', datetime.now().isoformat())
            request_id = conversation_data.get('request_id', 'unknown')
            query = conversation_data.get('query', '')
            response = conversation_data.get('response', '')
            word_count = conversation_data.get('word_count', len(response.split()) if response else 0)
            model_info = conversation_data.get('model_info', {})
            provider = model_info.get('provider', 'unknown')
            model = model_info.get('model_name', 'unknown')
            skip_search = conversation_data.get('skip_search', False)
            
            # Extract timing information
            timing = conversation_data.get('timing', {})
            retrieval_time = timing.get('retrieval_seconds', 0) * 1000  # Convert to ms
            generation_time = timing.get('generation_seconds', 0) * 1000  # Convert to ms
            total_time = timing.get('total_seconds', 0) * 1000  # Convert to ms
            
            # Extract reference counts
            references_count = len(conversation_data.get('references', []))
            content_count = len(conversation_data.get('content', []))
            
            # Create response preview (first 100 characters)
            response_preview = (response[:100] + '...') if len(response) > 100 else response
            response_preview = response_preview.replace('\n', ' ').strip()
            
            # Log to TXT file (human-readable format)
            self._log_to_txt(timestamp, request_id, query, response, word_count, provider, model, skip_search)
            
            # Log to JSON file (structured format)
            self._log_to_json(conversation_data)
            
            # Log to CSV file (tabular format for analysis)
            self._log_to_csv([
                timestamp, request_id, query, response_preview, word_count,
                provider, model, skip_search, retrieval_time, generation_time, 
                total_time, references_count, content_count
            ])
            
            logger.debug(f"Logged conversation {request_id} to all formats")
            
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
    
    def _log_to_txt(self, timestamp: str, request_id: str, query: str, response: str, 
                    word_count: int, provider: str, model: str, skip_search: bool) -> None:
        """Log conversation to TXT file in human-readable format."""
        try:
            with open(self.txt_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Conversation ID: {request_id}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Provider: {provider} | Model: {model} | Skip Search: {skip_search}\n")
                f.write(f"Word Count: {word_count}\n")
                f.write(f"\nQUERY:\n{query}\n")
                f.write(f"\nRESPONSE:\n{response}\n")
                f.write(f"{'='*80}\n")
        except Exception as e:
            logger.error(f"Error writing to TXT file: {e}")
    
    def _log_to_json(self, conversation_data: Dict[str, Any]) -> None:
        """Log conversation to JSON file in structured format."""
        try:
            # Add to conversations list
            self.conversations_data.append(conversation_data)
            
            # Keep only last 1000 conversations to prevent file from growing too large
            if len(self.conversations_data) > 1000:
                self.conversations_data = self.conversations_data[-1000:]
            
            # Save to file
            self._save_json_conversations()
        except Exception as e:
            logger.error(f"Error writing to JSON file: {e}")
    
    def _log_to_csv(self, row_data: List[Any]) -> None:
        """Log conversation to CSV file in tabular format."""
        try:
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
        except Exception as e:
            logger.error(f"Error writing to CSV file: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about logged conversations."""
        try:
            # Count conversations from JSON file (most reliable)
            total_conversations = len(self.conversations_data)
            
            # Get file sizes
            txt_size = self.txt_file.stat().st_size if self.txt_file.exists() else 0
            json_size = self.json_file.stat().st_size if self.json_file.exists() else 0
            csv_size = self.csv_file.stat().st_size if self.csv_file.exists() else 0
            
            return {
                'conversations_dir': str(self.conversations_dir),
                'total_conversations': total_conversations,
                'files': {
                    'txt': {'path': str(self.txt_file), 'size_bytes': txt_size},
                    'json': {'path': str(self.json_file), 'size_bytes': json_size},
                    'csv': {'path': str(self.csv_file), 'size_bytes': csv_size}
                },
                'today_date': datetime.now().strftime('%Y-%m-%d')
            }
        except Exception as e:
            logger.error(f"Error getting conversation stats: {e}")
            return {'error': str(e)}

# Global instance to be used across the application
_conversation_logger = None

def get_conversation_logger() -> ConversationLogger:
    """Get or create the global conversation logger instance."""
    global _conversation_logger
    if _conversation_logger is None:
        _conversation_logger = ConversationLogger()
    return _conversation_logger

def log_search_conversation(conversation_data: Dict[str, Any]) -> None:
    """
    Convenience function to log a search conversation.
    
    Args:
        conversation_data: Dictionary containing conversation details
    """
    logger_instance = get_conversation_logger()
    logger_instance.log_conversation(conversation_data)
