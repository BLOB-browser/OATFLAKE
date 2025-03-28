# Analysis Module

This module contains components for analyzing content, extracting structured information using LLMs, and storing processed data for use in the BLOB RAG platform.

## Overview

The analysis module is responsible for:

1. Fetching content from web resources and local files
2. Processing and extracting meaningful information using LLMs
3. Identifying key entities like definitions, projects, and methodologies
4. Saving processed data to vector stores for RAG retrieval
5. Providing structured data for the knowledge base

## Components

### ContentFetcher (`content_fetcher.py`)

Handles the retrieval of content from various sources:

- URL fetching with error handling and retries
- HTML parsing and text extraction
- Content chunking for processing
- Platform-specific optimizations

```python
# Example usage
from scripts.analysis.content_fetcher import ContentFetcher

fetcher = ContentFetcher()
content = fetcher.fetch_url("https://example.com")
```

### LLMAnalyzer (`llm_analyzer.py`)

Core component that orchestrates LLM analysis:

- Generates descriptions and summaries
- Extracts tags and categories
- Identifies definitions and concepts
- Extracts project information

```python
# Example usage
from scripts.analysis.llm_analyzer import LLMAnalyzer

analyzer = LLMAnalyzer()
analysis = analyzer.analyze_content("Text to analyze")
```

### ResourceLLM (`resource_llm.py`)

Specialized LLM interface for resource analysis:

- Optimized for JSON response generation
- Handles model fallbacks
- Threading and context window configuration
- Format-specific prompt templates

### MethodLLM (`method_llm.py`)

Specialized for extracting methodologies and procedures:

- Identifies step-by-step processes
- Extracts structured method information
- Formats steps with descriptions and metadata
- Handles complex nested structures

### DataSaver (`data_saver.py`)

Handles persistence of processed data:

- Saves to CSV files and vector stores
- Manages duplicates and versioning
- Converts between data formats
- Updates statistics and metadata

### ResourceProcessor (`resource_processor.py`)

High-level controller for the analysis workflow:

- Coordinates between fetching, analysis, and saving
- Manages resource batches and throttling
- Implements platform-specific optimizations
- Handles error recovery and logging

## Workflow

1. **Fetch Content**: Retrieve content from URLs or files
2. **Process Resource**: Analyze the content to extract structured information
3. **Extract Entities**: Identify definitions, methods, projects
4. **Generate Embeddings**: Create vector representations for RAG
5. **Save Data**: Persist processed information to storage

## Configuration

Analysis behavior can be configured through various settings:

- Model selection in `settings/model_settings.json`
- Processing parameters in resource_processor.py
- Hardware optimization through thread and batch size settings

## Hardware Considerations

- Default thread count: 4 (can be adjusted based on CPU)
- Batch processing for large datasets
- Platform detection for optimized settings
- Memory monitoring for long-running operations

## Usage Examples

### Processing a URL

```python
from scripts.analysis.resource_processor import ResourceProcessor

processor = ResourceProcessor()
result = processor.process_url("https://example.com")
```

### Analyzing Local Content

```python
from scripts.analysis.llm_analyzer import LLMAnalyzer

analyzer = LLMAnalyzer()
analysis = analyzer.analyze_local_content("path/to/content.md")
```

### Running Batch Analysis

```python
from scripts.analysis.resource_processor import ResourceProcessor

processor = ResourceProcessor()
processor.process_resource_batch(urls_list)
```

## Performance Monitoring

The module includes performance tracking with:

- Progress indicators for batch processing
- Time tracking for API calls
- Memory usage monitoring
- Error rate tracking and logging