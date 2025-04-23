from pydantic import BaseModel, HttpUrl, validator, Field
from datetime import datetime
from typing import Optional, List, Dict
import uuid

class Definition(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    term: str  # Title equivalent for definitions
    definition: str  # Description equivalent for definitions
    content_type: str = "definition"  # Type of content
    origin_url: Optional[str] = None  # Primary source URL for the content
    tags: List[str] = []  # Array of keywords or tags
    purpose: Optional[str] = None  # Purpose or usecase of data
    related_url: Optional[str] = None  # Additional related URLs
    status: str = "active"  # Current status of the item
    creator_id: Optional[str] = None  # ID of the content creator
    collaborators: Optional[str] = None  # IDs of collaborators
    group_id: Optional[str] = None  # ID of the associated group
    created_at: datetime = Field(default_factory=datetime.now)  # Timestamp of creation
    last_updated_at: Optional[datetime] = None  # Timestamp of last update
    analysis_completed: bool = False  # Boolean indicating if analysis is complete
    visibility: str = "public"  # Status of visibility
    
    # Fields for backward compatibility
    source: Optional[str] = None
    resource_url: Optional[str] = None
    
    @validator('content_type')
    def validate_content_type(cls, v):
        if v != "definition":
            raise ValueError('content_type must be "definition"')
        return v
        
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['active', 'archive', 'draft', 'deprecated']
        if v not in valid_statuses:
            raise ValueError(f'status must be one of: {", ".join(valid_statuses)}')
        return v
        
    @validator('visibility')
    def validate_visibility(cls, v):
        valid_visibility = ['public', 'private', 'group']
        if v not in valid_visibility:
            raise ValueError(f'visibility must be one of: {", ".join(valid_visibility)}')
        return v

class JsonRequest(BaseModel):
    path: str

class ConnectionRequest(BaseModel):
    group_id: str
    client_version: str = "0.1.0"

class ConnectionResponse(BaseModel):
    status: str
    message: str
    server_version: str = "0.1.0"

class Project(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    content_type: str = "project"  # Type of content (method, definition, material, project, resource)
    origin_url: Optional[str] = None  # Primary source URL for the content
    tags: List[str] = []  # Array of keywords or tags - replacing fields for consistency
    purpose: Optional[str] = None  # Purpose or usecase of data
    related_url: Optional[str] = None  # Additional related URLs
    status: str = "active"  # Current status of the item (active, archive, etc.)
    creator_id: str  # ID of the content creator
    collaborators: Optional[str] = None  # IDs of collaborators
    group_id: Optional[str] = None  # ID of the associated group
    created_at: datetime = Field(default_factory=datetime.now)  # Timestamp of creation
    last_updated_at: Optional[datetime] = None  # Timestamp of last update
    analysis_completed: bool = False  # Boolean indicating if analysis is complete
    visibility: str = "public"  # Status of visibility (public, private, etc.)
    
    # Project-specific fields
    goals: str
    achievement: str
    documentation_url: Optional[str] = None
    
    # Fields for backward compatibility
    resource_url: Optional[str] = None
    source: Optional[str] = None
    
    @validator('content_type')
    def validate_content_type(cls, v):
        if v != "project":
            raise ValueError('content_type must be "project"')
        return v
        
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['active', 'archive', 'draft', 'deprecated']
        if v not in valid_statuses:
            raise ValueError(f'status must be one of: {", ".join(valid_statuses)}')
        return v
        
    @validator('visibility')
    def validate_visibility(cls, v):
        valid_visibility = ['public', 'private', 'group']
        if v not in valid_visibility:
            raise ValueError(f'visibility must be one of: {", ".join(valid_visibility)}')
        return v
        
    class Config:
        # Allow privacy to be provided as visibility for backward compatibility
        fields = {
            'visibility': {'alias': 'privacy'}
        }

class Method(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    content_type: str = "method"  # Type of content (method, definition, material, project, resource)
    origin_url: Optional[str] = None  # Primary source URL for the content
    tags: List[str] = []  # Array of keywords or tags
    purpose: Optional[str] = None  # Purpose or usecase of data
    related_url: Optional[str] = None  # Additional related URLs
    status: str = "active"  # Current status of the item (active, archive, etc.)
    creator_id: str  # ID of the content creator
    collaborators: Optional[str] = None  # IDs of collaborators
    group_id: str  # ID of the associated group
    created_at: datetime = Field(default_factory=datetime.now)  # Timestamp of creation
    last_updated_at: Optional[datetime] = None  # Timestamp of last update
    analysis_completed: bool = False  # Boolean indicating if analysis is complete
    visibility: str = "public"  # Status of visibility (public, private, etc.)
    
    # Method-specific fields
    steps: List[str] = []  # Steps for the method
    
    @validator('content_type')
    def validate_content_type(cls, v):
        if v != "method":
            raise ValueError('content_type must be "method"')
        return v
    
    @validator('steps')
    def validate_steps(cls, v):
        if not v:
            return []  # Return empty list if None
        return v

    @validator('tags')
    def validate_tags(cls, v):
        if not v:
            return []  # Return empty list if None
        return v
        
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['active', 'archive', 'draft', 'deprecated']
        if v not in valid_statuses:
            raise ValueError(f'status must be one of: {", ".join(valid_statuses)}')
        return v
        
    @validator('visibility')
    def validate_visibility(cls, v):
        valid_visibility = ['public', 'private', 'group']
        if v not in valid_visibility:
            raise ValueError(f'visibility must be one of: {", ".join(valid_visibility)}')
        return v

class Resource(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    content_type: str = "resource"  # Type of content (method, definition, material, project, resource)
    origin_url: HttpUrl  # Primary source URL for the content
    tags: List[str] = []  # Array of keywords or tags
    purpose: Optional[str] = None  # Purpose or usecase of data
    related_url: Optional[str] = None  # Additional related URLs
    status: str = "active"  # Current status of the item (active, archive, etc.)
    creator_id: Optional[str] = None  # ID of the content creator
    collaborators: Optional[str] = None  # IDs of collaborators
    group_id: Optional[str] = None  # ID of the associated group
    created_at: datetime = Field(default_factory=datetime.now)  # Timestamp of creation
    last_updated_at: Optional[datetime] = None  # Timestamp of last update
    analysis_completed: bool = False  # Boolean indicating if analysis is complete
    visibility: str = "public"  # Status of visibility (public, private, etc.)
    category: Optional[str] = None  # Category for backward compatibility
    
    @validator('content_type')
    def validate_content_type(cls, v):
        valid_types = ['method', 'definition', 'material', 'project', 'resource']
        if v not in valid_types:
            raise ValueError(f'content_type must be one of: {", ".join(valid_types)}')
        return v
        
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['active', 'archive', 'draft', 'deprecated']
        if v not in valid_statuses:
            raise ValueError(f'status must be one of: {", ".join(valid_statuses)}')
        return v
        
    @validator('visibility')
    def validate_visibility(cls, v):
        valid_visibility = ['public', 'private', 'group']
        if v not in valid_visibility:
            raise ValueError(f'visibility must be one of: {", ".join(valid_visibility)}')
        return v
        
    class Config:
        # Allow URL to be provided as origin_url for backward compatibility
        fields = {
            'origin_url': {'alias': 'url'}
        }

class ReadingMaterial(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: Optional[str] = None
    content_type: str = "material"  # Type of content
    origin_url: Optional[str] = None  # Primary source URL or file path
    tags: List[str] = []  # Array of keywords or tags (replacing fields)
    purpose: Optional[str] = None  # Purpose or usecase of data
    related_url: Optional[str] = None  # Additional related URLs
    status: str = "active"  # Current status of the item
    creator_id: Optional[str] = None  # ID of the content creator
    collaborators: Optional[str] = None  # IDs of collaborators
    group_id: Optional[str] = None  # ID of the associated group
    created_at: datetime = Field(default_factory=datetime.now)  # Timestamp of creation
    last_updated_at: Optional[datetime] = None  # Timestamp of last update
    analysis_completed: bool = False  # Boolean indicating if analysis is complete
    visibility: str = "public"  # Status of visibility
    
    # Reading material specific fields
    file_path: Optional[str] = None  # Will be set after file upload
    
    # For backward compatibility
    fields: List[str] = []  # Old name for tags
    
    @validator('content_type')
    def validate_content_type(cls, v):
        if v != "material":
            raise ValueError('content_type must be "material"')
        return v
        
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['active', 'archive', 'draft', 'deprecated']
        if v not in valid_statuses:
            raise ValueError(f'status must be one of: {", ".join(valid_statuses)}')
        return v
        
    @validator('visibility')
    def validate_visibility(cls, v):
        valid_visibility = ['public', 'private', 'group']
        if v not in valid_visibility:
            raise ValueError(f'visibility must be one of: {", ".join(valid_visibility)}')
        return v
        
    @validator('tags')
    def validate_tags(cls, v, values):
        # If tags is empty but fields exists (for backward compatibility), copy fields to tags
        if not v and 'fields' in values and values['fields']:
            return values['fields']
        return v

class WebRequest(BaseModel):
    prompt: str
    context: Optional[str] = None
    max_length: Optional[int] = 150

class WebResponse(BaseModel):
    response: str
    model: str
    status: str = "success"

class Answer(BaseModel):
    answer_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    answer_text: str
    answered_by: str
    answered_at: datetime = Field(default_factory=datetime.now)

class Question(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question_text: str
    created_by: str
    created_at: datetime = Field(default_factory=datetime.now)
    group_id: str
    answers: List[Answer] = Field(default_factory=list)

class AnswerCreate(BaseModel):
    answer_text: str
    group_id: str
    question_id: str

class SimpleAnswer(BaseModel):
    text: str
    answered_by: str

class SimpleQuestion(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    created_by: str
    answers: List[SimpleAnswer] = Field(default_factory=list)

class AnswerRequest(BaseModel):
    question_id: str
    answer: str

class QuestionGenerate(BaseModel):
    model: str = "llama3.2:1b"
    max_questions: int = 5
    temperature: float = 0.7

class SystemQuestion(BaseModel):
    question_text: str
    context_source: str
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str = "SYSTEM"
