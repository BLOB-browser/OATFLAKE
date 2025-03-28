from pathlib import Path
import logging
from typing import Dict, List
from datetime import datetime
import json
import os
from dataclasses import dataclass, asdict
from collections import defaultdict

from langchain.schema import Document

logger = logging.getLogger(__name__)

@dataclass
class ProcessedDocument:
    content: str
    metadata: Dict
    type: str
    topics: List[str]
    created_at: str

class DirectoryProcessor:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.processed: Dict[str, List[ProcessedDocument]] = defaultdict(list)

    def process_by_type(self) -> Dict[str, List[ProcessedDocument]]:
        """Split documents by type"""
        for doc in self.documents:
            doc_type = doc.metadata.get('type', 'general')
            topics = doc.metadata.get('topics', ['general'])
            
            processed_doc = ProcessedDocument(
                content=doc.page_content,
                metadata=doc.metadata,
                type=doc_type,
                topics=topics if isinstance(topics, list) else [topics],
                created_at=doc.metadata.get('created_at', datetime.now().isoformat())
            )
            
            self.processed[doc_type].append(processed_doc)
            
        return self.processed

    def process_by_topic(self) -> Dict[str, List[ProcessedDocument]]:
        """Split documents by topic"""
        topic_docs: Dict[str, List[ProcessedDocument]] = defaultdict(list)
        
        for doc in self.documents:
            topics = doc.metadata.get('topics', ['general'])
            if isinstance(topics, str):
                topics = [topics]
                
            for topic in topics:
                processed_doc = ProcessedDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    type=doc.metadata.get('type', 'general'),
                    topics=topics,
                    created_at=doc.metadata.get('created_at', datetime.now().isoformat())
                )
                topic_docs[topic].append(processed_doc)
                
        return topic_docs

    def save_processed_data(self, output_path: Path) -> bool:
        """Save processed documents to JSON files"""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save by type
            type_data = self.process_by_type()
            with open(output_path / "by_type.json", "w") as f:
                json.dump({k: [asdict(doc) for doc in v] 
                          for k, v in type_data.items()}, f)
            
            # Save by topic
            topic_data = self.process_by_topic()
            with open(output_path / "by_topic.json", "w") as f:
                json.dump({k: [asdict(doc) for doc in v] 
                          for k, v in topic_data.items()}, f)
                
            return True
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            return False
