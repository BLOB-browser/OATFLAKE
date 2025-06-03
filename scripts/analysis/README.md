# Analysis Module

## Overview

The Analysis module is the core processing engine of OATFLAKE, responsible for transforming raw content into structured knowledge. This module is primarily triggered through:

1. The `/api/knowledge/process` endpoint for on-demand processing
2. The training scheduler for automated batch processing

## Main Entry Point: MainProcessor

`main_processor.py` serves as the central orchestration component that manages the full knowledge extraction pipeline. It coordinates all aspects of content analysis including:

- Content retrieval and preprocessing
- Document parsing and text extraction
- LLM-based analysis and entity identification
- Storage of processed information
- Vector embedding generation

### Key Responsibilities

- **Pipeline Management**: Coordinates the end-to-end workflow from content fetching to storage
- **Incremental Processing**: Supports processing single resources or batches
- **Progress Tracking**: Provides status updates for frontend display
- **Resource Management**: Handles efficient allocation of computing resources
- **Error Handling**: Implements graceful failure recovery and logging
- **Cancellation Support**: Allows for safe interruption of processing

### Typical Flow

1. MainProcessor is invoked by API call or scheduler
2. Content is retrieved from specified sources (URLs, files, etc.)
3. Content is preprocessed and chunked appropriately
4. LLM analysis extracts structured information (definitions, methods, etc.)
5. Extracted information is saved to persistent storage
6. Vector embeddings are generated for RAG functionality
7. Status updates and results are returned

## Scalable Search with Level-Based Processing

OATFLAKE implements a sophisticated multi-level content processing system that operates in a breadth-first manner:

- **Level-Based URL Discovery**: The ContentFetcher discovers URLs in a breadth-first pattern, limiting discovery to specified depth levels to prevent going too deep too early
- **Hierarchical Processing**: LevelBasedProcessor ensures all URLs at level 1 are processed before moving to level 2, etc.
- **Batch Processing**: ResourceBatchProcessor handles resources in configurable batch sizes to prevent memory issues
- **Automatic Continuation**: When one batch completes, a new batch of resources is automatically processed until the training scheduler interrupts or no resources remain
- **Incremental Vector Store Generation**: Vector stores are built incrementally after processing N resources to maintain search functionality during long processing runs

This scalable approach ensures that the system can gracefully handle large websites and content repositories without exhausting system resources.

## Interruptible LLM Functionality

OATFLAKE implements a robust interruption mechanism in `interruptible_llm.py` that allows safely canceling long-running LLM operations:

- **Safe Cancellation**: When a user presses Ctrl+C, the system sets a cancellation flag rather than abruptly terminating
- **Graceful Cleanup**: Active requests are properly closed and resources released
- **Signal Handling**: Preserves and restores original signal handlers
- **Cross-Component Coordination**: Uses threading.Event to safely coordinate interrupt state
- **Support for Async**: Works with both synchronous and asynchronous code

The interruptible functionality is integrated throughout the analysis pipeline, allowing operations to be safely canceled at any point without corrupting data.

## LLM Implementations

OATFLAKE supports multiple LLM backends with specialized implementations:

### LLM Providers

- **Ollama Integration** (`ollama_client.py`):
  - Uses locally hosted models via Ollama
  - Default model: `mistral:7b-instruct-v0.2-q4_0`
  - Handles both generation and embedding requests
  - Manages local vector stores for retrieval-augmented generation

- **OpenRouter Integration** (`open_router_client.py`):
  - Connects to cloud-based models via OpenRouter API
  - Default model: `mistralai/mistral-nemo:free`
  - Provides access to various commercial models (Anthropic, OpenAI, etc.)
  - Falls back to local embeddings for RAG functionality

### Task-Specific LLMs

- **ResourceLLM**: Extracts structured data from resources
- **GoalLLM**: Specialized for educational goal extraction
- **MethodLLM**: Optimized for extracting step-by-step methodologies

### Configuration System

Model settings are managed through a unified configuration system that:
- Adapts to available hardware resources
- Provides sensible defaults for model parameters
- Allows overriding via environment variables or config files

## Core Components

### Content Processing

- **Resource Processing**: Handles individual web pages, PDFs, and other source documents
- **Text Extraction**: Extracts plain text from various document formats
- **Chunking**: Breaks content into appropriate segments for analysis

### LLM Analysis

- **Entity Extraction**: Identifies key concepts, definitions, and methodologies
- **Structured Information**: Converts unstructured text to structured data
- **Categorization**: Classifies content into appropriate categories

### Data Management

- **Storage Coordination**: Saves processed data to appropriate data stores
- **Vector Generation**: Creates embeddings for similarity search
- **Validation**: Ensures data quality and consistency

## Usage

### API Integration

The MainProcessor is typically invoked through API calls:

```python
# Example of how MainProcessor is triggered through API
@router.post("/knowledge/process")
async def process_knowledge(request: KnowledgeRequest):
    processor = MainProcessor()
    results = await processor.process_resources(request.urls)
    return {"status": "success", "processed": len(results)}
```

### Command Line Tools

The analysis system can be used through various command-line tools:

#### Process a Batch of Resources

```bash
# Process a batch with size 5
python -m scripts.tools.process_batch --batch-size 5

# Force reanalysis of already processed resources
python -m scripts.tools.process_batch --force-reanalysis
```

#### Process Resources by Level

```bash
# Process all URLs at level 1, then level 2
python -m scripts.tools.process_by_level --max-depth 2

# Process only level 1 URLs for all resources
python -m scripts.tools.process_by_level --level 1
```

#### Complete Incremental Processing

```bash
# Process all document types incrementally with regular FAISS rebuilds
python -m scripts.tools.run_incremental_processing

# Specify batch size for incremental processing
python -m scripts.tools.run_incremental_processing --batch-size 10
```

### Programmatic Usage

The analysis system can be used programmatically in Python:

```python
# Process a single resource
from scripts.analysis.single_resource_processor_universal import SingleResourceProcessorUniversal

processor = SingleResourceProcessorUniversal("data")
resource = {"url": "https://example.com", "title": "Example Site"}
result = processor.process_resource(resource)

# Process by levels
from scripts.analysis.level_processor import LevelBasedProcessor

level_proc = LevelBasedProcessor("data")
result = level_proc.process_level(level=1, max_resources=10)

# Batch processing with auto-continuation
from scripts.analysis.resource_batch_processor import ResourceBatchProcessor

batch_proc = ResourceBatchProcessor("data")
result = batch_proc.process_resources(max_resources=5)
```

### Interrupting Processing

To safely interrupt any processing operation, press `Ctrl+C`. The system will:

1. Set the interrupt flag
2. Clean up any active resources
3. Complete the current operation and exit gracefully
4. Save progress so processing can be resumed later

You can also interrupt programmatically:

```python
from scripts.analysis.interruptible_llm import is_interrupt_requested, clear_interrupt

# Check if interrupted
if is_interrupt_requested():
    # Handle interruption...

# Clear interrupt flag
clear_interrupt()
```