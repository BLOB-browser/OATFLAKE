# Data Processing Module

## Overview

The Data Processing module is responsible for handling all document processing operations within the OATFLAKE platform. It provides functionality for loading, processing, chunking, and embedding various document types to prepare them for RAG (Retrieval-Augmented Generation) operations.

## Key Components

### Document Loading

The document loading subsystem handles the intake of various document formats:

- `document_loader.py`: Core loader that dispatches to format-specific loaders
  - Supports PDF files, Markdown, CSV, plain text, and other document formats
  - Uses LangChain's document loaders for standardized document handling
  - Implements automatic format detection based on file extensions

### Text Processing

Text processing components handle the conversion of raw documents into processable chunks:

- `document_processor.py`: Manages document chunking with consistent settings
  - Implements chunking strategies optimized for different hardware capabilities
  - Uses RecursiveCharacterTextSplitter for intelligent text chunking
  - Handles document metadata preservation during chunking
  
- `markdown_processor.py`: Specialized processing for Markdown documents
  - Extracts structured content from Markdown files
  - Preserves headers, lists, and other Markdown structures
  - Integrates with the markdown_scraper for additional metadata extraction

### Embedding Generation

The embedding subsystem converts text chunks into vector representations:

- `embedding_service.py`: Manages the creation of embeddings
  - Interfaces with embedding models (local via Ollama or remote)
  - Handles batching for efficient processing
  - Implements caching to avoid reprocessing identical content
  
- `faiss_builder.py`: Creates and manages FAISS vector indexes
  - Builds efficient vector stores for similarity search
  - Supports incremental index updates
  - Optimizes for memory efficiency on various hardware configurations

### Process Management

- `processing_manager.py`: Orchestrates the overall data processing workflow
  - Coordinates document loading, processing, and embedding generation
  - Implements processing queues and priority management
  - Handles error recovery and logging
  
- `data_processor.py`: High-level API for data processing operations
  - Provides simplified interfaces for common operations
  - Manages file system interactions and data storage
  - Coordinates between various processing components

## Usage

### Basic Document Processing

```python
from scripts.data.data_processor import DataProcessor
from pathlib import Path

# Initialize processor with data directory
processor = DataProcessor(Path("./data"), group_id="research")

# Process documents
processed_docs = processor.process_directory("documents/pdfs")

# Generate embeddings for processed documents
processor.generate_embeddings(processed_docs)
```

### Working with Markdown Documents

```python
from scripts.data.markdown_processor import MarkdownProcessor

# Initialize processor
md_processor = MarkdownProcessor()

# Process a markdown file
processed_content = md_processor.process_file("documents/example.md")

# Extract metadata and content
metadata = processed_content["metadata"]
chunks = processed_content["chunks"]
```

### Building Vector Stores

```python
from scripts.data.faiss_builder import FAISSBuilder
from pathlib import Path

# Initialize builder
builder = FAISSBuilder(Path("./vectors"))

# Build vector store from documents
documents = [...] # List of processed documents
builder.build_vector_store(documents, "research_papers")

# Search the vector store
results = builder.search("quantum computing applications", k=5)
```

## Configuration

The data processing module can be configured through environment variables and settings files:

- `CHUNK_SIZE`: Default chunk size for text splitting (default: 200)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 20)
- `EMBEDDING_MODEL`: Model to use for embeddings (default: determined by system)
- `MAX_THREADS`: Maximum threads for parallel processing

## Integration Points

The Data Processing module integrates with:

- **Analysis Module**: Provides processed documents for deep analysis
- **API Layer**: Exposes document processing capabilities through REST endpoints
- **Storage System**: Persists processed documents and vector stores
- **LLM Module**: Supplies context for RAG operations

## Performance Considerations

- Document processing is optimized for lower CPU systems by default
- Chunking settings are balanced for performance vs. quality
- Vector operations are memory-intensive; consider hardware limitations
- Batch processing is recommended for large document collections

## Future Development

- Implement adaptive chunking based on document semantics
- Add support for more document formats (DOCX, EPUB, etc.)
- Improve multilingual support for non-English documents
- Implement document deduplication strategies
