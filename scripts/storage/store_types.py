from enum import Enum
from typing import Dict, List
from pydantic import BaseModel
from datetime import datetime

class StoreType(Enum):
    RULES = "rules"
    GOALS = "goals"
    DEFINITIONS = "definitions"
    CONTENT = "content"
    SUMMARY = "summary"

class BucketMetadata(BaseModel):
    name: str
    description: str
    topics: List[str]
    fields: List[str]
    created_at: datetime
    updated_at: datetime
    source_type: str  # 'pdf', 'csv', 'web'
    source_path: str
    tags: List[str]
