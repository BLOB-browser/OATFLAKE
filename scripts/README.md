# OATFLAKE Scripts

This directory contains the core processing scripts and business logic for the OATFLAKE platform. These scripts handle everything from document processing and analysis to model interaction and data management.

## Directory Structure

- `analysis/`: Core analysis engines for content processing
  - `main_processor.py`: Central processing coordinator
  - `entity_extractor.py`: Extracts entities from processed text
  - `text_analyzer.py`: Performs semantic analysis on text
  - `summarizer.py`: Creates summaries of documents
  - `content_classifier.py`: Classifies content into categories
  
- `embedding/`: Vector embedding generation and manipulation
  - `embedding_manager.py`: Manages embedding creation and storage
  - `vector_operations.py`: Vector math and manipulation functions
  - `chunking.py`: Text chunking strategies for optimal embedding
  
- `extraction/`: Document parsing and text extraction
  - `markdown_extractor.py`: Extracts content from markdown files
  - `pdf_extractor.py`: Extracts text from PDF documents
  - `web_extractor.py`: Scrapes and extracts content from web pages
  - `image_extractor.py`: Extracts text from images using OCR
  
- `services/`: Background services and scheduled tasks
  - `training_scheduler.py`: Schedules and manages training jobs
  - `notification_service.py`: Handles notifications and alerts
  - `storage.py`: Manages persistent storage operations
  - `cache_manager.py`: Handles caching for performance optimization
  
- `integration/`: External system connectors
  - `slack_connector.py`: Integration with Slack API
  - `openrouter_client.py`: Client for OpenRouter API
  - `ollama_interface.py`: Interface to Ollama for local models
  - `supabase_connector.py`: Integration with Supabase

## Key Components

### MainProcessor

The `MainProcessor` class in `analysis/main_processor.py` serves as the central orchestrator for all document processing. It coordinates:

1. Document intake and validation
2. Text extraction and preprocessing
3. Content analysis and entity extraction
4. Embedding generation and storage
5. Result persistence

### Model Interaction

The scripts handle interaction with various LLM backends:

- Local models via Ollama
- Remote models via OpenRouter
- Vector operations via FAISS

### Data Flow

1. Documents are received through the API
2. `extraction` scripts parse and extract the text
3. `analysis` scripts process the content for entities and insights
4. `embedding` scripts generate vector representations
5. `services` scripts handle storage and persistence

## Usage

Most scripts are not meant to be run directly but are imported by the API routes. However, some scripts can be run for testing or maintenance:

```bash
# Run a test extraction on a specific file
python -m scripts.extraction.markdown_extractor --input path/to/file.md

# Process a document through the full pipeline
python -m scripts.analysis.main_processor --input path/to/document.pdf

# Generate embeddings for a specific document
python -m scripts.embedding.embedding_manager --input path/to/text.txt