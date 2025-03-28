from pydantic import BaseModel, HttpUrl, validator, Field
from datetime import datetime
from typing import Optional, List, Dict
import uuid

class Definition(BaseModel):
    term: str
    definition: str
    tags: List[str] = []
    source: Optional[str] = None
    resource_url: Optional[str] = None
    created_at: Optional[datetime] = None

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
    title: str
    description: str
    goals: str
    achievement: str
    documentation_url: Optional[str] = None
    fields: str
    privacy: str  # public or private
    status: str   # active or archived
    creator_id: str
    source: Optional[str] = None
    resource_url: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None

    @validator('privacy')
    def validate_privacy(cls, v):
        if v not in ['public', 'private']:
            raise ValueError('Privacy must be either public or private')
        return v

    @validator('status')
    def validate_status(cls, v):
        if v not in ['active', 'archived']:  # Fixed extra parenthesis here
            raise ValueError('Status must be either active or archived')
        return v

class Method(BaseModel):
    group_id: str
    title: str
    description: str
    steps: List[str] = []  # Make steps optional with default empty list
    tags: List[str] = []   # Make tags optional with default empty list
    creator_id: str
    created_at: Optional[datetime] = None

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

class Resource(BaseModel):
    title: str
    url: HttpUrl
    description: str
    type: str  # TODO: Could be made into Enum if types are fixed
    category: str
    tags: List[str] = []
    created_at: Optional[datetime] = None

class ReadingMaterial(BaseModel):
    title: str
    description: Optional[str] = None
    fields: List[str] = []
    created_at: Optional[datetime] = None
    file_path: Optional[str] = None  # Will be set after file upload

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
