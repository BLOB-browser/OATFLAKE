# Services Module

## Overview

The Services module provides core background services and utilities for the OATFLAKE platform. These services handle persistent operations, scheduled tasks, configuration management, and system-wide functionality that powers the entire application.

## Key Components

### Training Scheduler (`training_scheduler.py`)

Manages scheduled knowledge processing and training operations:

- **Automated Processing**: Schedule regular knowledge processing during off-peak hours
- **Configurable Schedule**: Set custom start and stop times for background processing
- **Incremental Processing**: Process content in batches to avoid resource exhaustion
- **Status Tracking**: Monitor processing status and completion metrics
- **Graceful Interruption**: Allow safe interruption of processing when needed

```python
from scripts.services.training_scheduler import start, stop, get_schedule

# Start the training scheduler
start()

# Get current schedule configuration
schedule = get_schedule()
print(f"Training runs from {schedule['start_time']} to {schedule['end_time']}")

# Stop the scheduler
stop()

# Set custom schedule
set_schedule(start_hour=2, start_minute=30, stop_hour=5, stop_minute=0)
```

### Settings Manager (`settings_manager.py`) 

Centralized management for application settings:

- **User Preferences**: Store and retrieve user-specific settings
- **System Configuration**: Manage system-wide configuration parameters
- **Persistent Storage**: Save settings to disk for persistence across restarts
- **Default Values**: Provide sensible defaults for unconfigured settings
- **Schema Validation**: Ensure settings adhere to expected format and values

```python
from scripts.services.settings_manager import SettingsManager

# Initialize settings manager
manager = SettingsManager()

# Get all settings
settings = manager.load_settings()

# Update a specific setting
manager.update_setting("model_name", "mistral:7b-instruct-v0.2-q4_0")

# Save settings
manager.save_settings()
```

### RAG Service (`rag_service.py`)

Provides retrieval-augmented generation capabilities across the application:

- **Content Retrieval**: Find relevant information from vector stores
- **Query Enhancement**: Improve search results with semantic understanding
- **Context Generation**: Create relevant context for LLM queries
- **Answer Generation**: Generate answers based on retrieved content
- **Reference Management**: Track sources used in generated responses

```python
from scripts.services.rag_service import RAGService

# Initialize RAG service
rag = RAGService()

# Get answer with relevant context
answer, sources = rag.answer_question("What is OATFLAKE?")

# Perform semantic search
results = rag.semantic_search("knowledge processing", top_k=5)
```

### Storage Service (`storage.py`)

Manages data persistence and storage operations:

- **File Management**: Handle saving and loading of various file types
- **Directory Structure**: Maintain organized data directory hierarchy
- **Backup Operations**: Create and restore backups of important data
- **File System Operations**: Abstract filesystem interactions for platform independence
- **Vector Store Integration**: Manage persistent vector storage operations

```python
from scripts.services.storage import StorageService

# Initialize storage service
storage = StorageService("data")

# Save data to a specific location
storage.save_json(data, "processed/results.json")

# Create backup of important data
storage.create_backup("knowledge_base")

# Get all files of a specific type
markdown_files = storage.get_files_by_extension(".md")
```

### Status Service (`status.py`)

Provides system status information and monitoring:

- **Health Checks**: Monitor overall system health and component status
- **Resource Usage**: Track CPU, memory, and disk usage
- **Service Status**: Report on the state of various services
- **Error Reporting**: Collect and organize error information
- **Performance Metrics**: Track system performance indicators

```python
from scripts.services.status import StatusService

# Get system status
status = StatusService()
health_info = status.get_health_status()

# Check specific component
ollama_status = status.check_component("ollama")

# Get resource usage
resources = status.get_resource_usage()
print(f"CPU: {resources['cpu']}%, Memory: {resources['memory']}%")
```

### Connection Service (`connection.py`)

Manages network connections and related operations:

- **Connection Pooling**: Efficiently manage HTTP connections
- **Request Handling**: Process HTTP requests with proper error handling
- **Rate Limiting**: Implement rate limiting for external APIs
- **Retry Logic**: Automatically retry failed requests with backoff
- **Connection Status**: Monitor connection health and availability

```python
from scripts.services.connection import ConnectionService

# Create connection service
conn = ConnectionService()

# Make HTTP request with retry logic
response = conn.request("GET", "https://api.example.com/data")

# Check connection status
is_connected = conn.check_connection("https://api.example.com")
```

### Data Analyzer (`data_analyser.py`)

Analyzes processed data to extract insights:

- **Content Analysis**: Extract meaningful information from text
- **Trend Detection**: Identify patterns and trends in data
- **Metadata Extraction**: Generate useful metadata from content
- **Statistical Analysis**: Compute statistics on processed data
- **Content Classification**: Categorize content by type and relevance

```python
from scripts.services.data_analyser import DataAnalyser

# Initialize analyzer
analyzer = DataAnalyser()

# Analyze a document
results = analyzer.analyze_document("path/to/document.md")

# Generate summary statistics
stats = analyzer.generate_statistics("project_name")
```

### Question Generator (`question_generator.py`)

Automatically generates relevant questions from content:

- **Question Extraction**: Generate questions based on document content
- **Educational Focus**: Create questions that test understanding
- **Relevance Scoring**: Rank questions by relevance and importance
- **Variety Generation**: Create different types of questions (MCQ, free form, etc.)
- **Answer Inclusion**: Generate questions with corresponding answers

```python
from scripts.services.question_generator import QuestionGenerator

# Initialize generator
generator = QuestionGenerator()

# Generate questions from text
questions = generator.generate_from_text("Sample text content...")

# Generate questions from processed documents
project_questions = generator.generate_for_project("project_id")
```

## Architecture

The Services module follows a modular architecture where each service:

1. Has a clear, focused responsibility
2. Operates independently with well-defined interfaces
3. Can be started, stopped, and monitored individually
4. Communicates with other services through standardized interfaces
5. Handles its own resource management and cleanup

### Service Lifecycle

Most services follow this standard lifecycle:
- **Initialization**: Load configuration and prepare resources
- **Start**: Begin operation, possibly in a background thread
- **Operation**: Perform ongoing work and respond to requests
- **Monitoring**: Track status and report health information
- **Shutdown**: Release resources and terminate gracefully

## Integration Points

The Services module integrates with other OATFLAKE components:

- **API Layer**: Exposes service functionality through API endpoints
- **LLM Module**: Provides orchestration for language model operations
- **Analysis Module**: Coordinates with analysis components for processing
- **Storage**: Manages persistent data across service operations

## Configuration

Services can be configured through several mechanisms:

- **Environment Variables**: Configure basic service parameters
- **Settings Files**: Detailed configuration through JSON settings
- **API Configuration**: Dynamic configuration through API calls
- **Database Settings**: Persistent settings stored in database

## Error Handling

Services implement robust error handling strategies:

- **Graceful Degradation**: Continue operation when possible, despite failures
- **Retry Logic**: Automatically retry operations with exponential backoff
- **Logging**: Comprehensive error logging for troubleshooting
- **Recovery**: Self-healing mechanisms to recover from failure states
- **Notification**: Alert system for critical failures requiring attention

## Performance Considerations

- Services are designed to minimize resource usage during idle periods
- Background operations schedule resource-intensive tasks during configured windows
- Thread pooling prevents excessive thread creation
- Services implement appropriate caching to reduce redundant operations

## Development Guidelines

When extending the Services module:

1. Follow the established service pattern for consistency
2. Implement proper startup and shutdown procedures
3. Use appropriate threading and asyncio patterns for background operations
4. Include comprehensive logging for operations and errors
5. Provide clear status reporting mechanisms
