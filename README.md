# OATFLAKE

## Introduction

OATFLAKE is a no-code interface framework designed as a submodule of the BLOB browser. It enables community-governed intelligence training based on Retrieval-Augmented Generation (RAG). This repository serves as an easy-to-setup beta testing and development space for contributors to the system.

The unique value of OATFLAKE lies in its flexibility and autonomy. The backend can run entirely locally, without requiring external APIs or web access, by leveraging Ollama and local folder files. Additionally, it supports web scraping to gather resources and integrate them into the local vector space using FAISS. Many components are built with LangChain, and OpenRouter is included as an API for extended functionality.

OATFLAKE empowers communities, such as research groups and collectives, to maintain their local intelligence systems and easily swap out models. While currently tailored for extracting methods, definitions, resources, and materials, the system is evolving to support the collection and analysis of any type of text data. The framework is built with modularity in mind, offering small building blocks and adapters to handle diverse file inputs.

Our vision is to make as much of the system customizable through the interface as possible, enabling communities to adapt it to their unique needs without requiring coding expertise.

## Features
- Knowledge processing and extraction
- Goal-based analysis
- Slack integration for communication
- Vector-based search capabilities
- API endpoints for various functionalities
- Web-based user interface

## Architecture

OATFLAKE follows a modular architecture designed to provide flexibility and extensibility:

### Core Components

1. **FastAPI Backend**: Powers all API endpoints and server-side operations through organized route handlers
2. **Web Interface**: Modern, responsive JavaScript frontend with Tailwind CSS for interacting with the system
3. **RAG Pipeline**: End-to-end pipeline for document processing, analysis, and retrieval
4. **Vector Storage**: FAISS-based vector storage for efficient similarity searches
5. **Integrations**: Connections to external systems (Slack, OpenRouter, Ollama, etc.)

### Data Flow

1. **Input Sources** → Documents uploaded or URLs provided
2. **Content Processing** → Text extraction and chunking using format-specific processors
3. **Analysis** → LLM-powered analysis with entity extraction via MainProcessor
4. **Embedding Generation** → Vector embedding creation through local or remote models
5. **Storage** → Persistence to vector stores and databases with incremental updates
6. **Retrieval** → Context-sensitive document retrieval via FAISS similarity search
7. **Generation** → LLM-augmented response creation with local or cloud-based models

### Key Subsystems

#### Analysis Engine
- Orchestrates document processing through MainProcessor
- Implements level-based URL discovery for breadth-first processing
- Supports batched resource processing to prevent memory issues
- Provides interruptible LLM functionality for long-running tasks
- Extracts entities, methods, definitions, and other structured data

#### LLM Integration
- Abstracts model interactions through unified interfaces
- Supports local models via Ollama for privacy and cost efficiency
- Integrates with OpenRouter for access to powerful cloud models
- Provides embedding generation for vector search capabilities
- Implements configurable model parameters and context handling

#### Data Processing
- Handles diverse document formats (PDF, Markdown, HTML, etc.)
- Implements intelligent chunking strategies optimized for different hardware
- Preserves document structure and metadata during processing
- Generates efficient vector embeddings for similarity search

#### Services Layer
- Manages scheduled training and knowledge processing
- Handles system-wide configuration and settings
- Provides background tasks and automated operations
- Implements caching and optimization strategies

#### Frontend System
- Built with modern JavaScript and Tailwind CSS
- Features modular widget architecture for extensibility
- Provides interactive UI components for data visualization
- Implements responsive design for all device sizes

#### API Endpoints
- Organized into logical domains (knowledge, goals, analysis)
- Implements RESTful patterns for consistent interaction
- Provides authentication and permission management
- Offers comprehensive system management capabilities

## Prerequisites
- Python 3.10 or higher
- Poetry (for dependency management)
- Ollama (optional, for local model hosting)
- Slack workspace (for Slack integration)
- Supabase account (for data storage)
- OpenRouter account (for model access)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/blob/OATFLAKE.git
cd OATFLAKE
```

### 2. Run the setup and start script

#### On Windows:
Double-click the `start.bat` file in File Explorer, or run it from the command line:
```
# In Command Prompt:
start.bat

# In PowerShell:
.\start.bat
```

#### On Mac/Linux:
```bash
# First time only - make the startup script executable
chmod +x start.sh

# Then run the script
./start.sh
```

This will:
1. Start the FastAPI server
2. Set up a ngrok tunnel for external access (if configured)
3. Open the web interface in your default browser


### 3. Create a `.env` file
Create a `.env` file in the root directory with the following content (replace with your actual credentials):
```
# Server Configuration
LOCAL_HOST=127.0.0.1
LOCAL_PORT=8999
UI_PORT=3000

# Central Server
BASE_DOMAIN=your-base-domain
API_KEY=your-api-key-here

# Ollama Configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434

# Slack Configuration (Required)
SLACK_BOT_TOKEN=your-slack-bot-token
SLACK_SIGNING_SECRET=your-slack-signing-secret
SLACK_BOT_USER_ID=your-slack-bot-user-id

# Supabase Configuration
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-supabase-key

# OpenRouter Configuration
OPENROUTER_API_KEY=your-openrouter-api-key
```


## Project Structure
- `api/`: API endpoints and routes
  - `routes/`: RESTful endpoints organized by domain (knowledge, goals, analysis)
    - `auth.py`: Authentication and user management endpoints
    - `knowledge.py`: Knowledge base and document processing endpoints
    - `goals.py`: Goal tracking and management endpoints
    - `slack.py`: Slack integration endpoints
    - `ollama.py`: Ollama model interaction endpoints
    - `openrouter.py`: OpenRouter API integration endpoints
  - `middleware/`: Request processing and authentication middleware
  - `models/`: Pydantic data models for API requests and responses
  - `dependencies/`: Reusable API dependencies and injected services
  - `main.py`: Entry point and router registration
- `scripts/`: Core processing and analysis scripts
  - `analysis/`: LLM-powered content analysis modules
    - `main_processor.py`: Central orchestration for document processing
    - `content_extractor.py`: Entity extraction from documents
    - `llm_analyzer.py`: LLM-based document analysis
    - `goal_extractor.py`: Identification of goals in content
    - `level_processor.py`: Level-based URL discovery and processing
  - `data/`: Document processing and management
    - `document_loader.py`: Format-specific document loaders
    - `document_processor.py`: Document chunking and preprocessing
    - `embedding_service.py`: Creation of vector embeddings
    - `faiss_builder.py`: FAISS index creation and management
  - `llm/`: Language model integration
    - `ollama_client.py`: Local model inference via Ollama
    - `open_router_client.py`: Cloud model access via OpenRouter
    - `prompt_templates.py`: Reusable prompt templates
  - `services/`: Background services and scheduled tasks
    - `training_scheduler.py`: Scheduled knowledge processing
    - `settings_manager.py`: Application settings management
    - `cache_manager.py`: Performance optimization through caching
  - `integrations/`: External system connectors
    - `slack.py`: Slack messaging and event handling
    - `supabase_connector.py`: Supabase database integration
- `settings/`: Configuration files and environment settings
  - `config.py`: Central configuration management
  - `default_settings.json`: Default application settings
  - `model_settings.json`: Model-specific configuration
- `static/`: Static assets for the web interface
  - `js/`: JavaScript modules and UI components
    - `components/`: Reusable UI components
    - `widgets/`: Interactive widget implementations
    - `modals/`: Modal dialog implementations
  - `css/`: Styling with Tailwind CSS and custom styles
    - `main.css`: Custom styles beyond Tailwind
  - `icons/`: Icons and visual assets
- `templates/`: HTML templates for web rendering
  - `components/`: Reusable UI components
  - `pages/`: Full page templates
- `utils/`: Utility functions and helpers
  - `logging/`: Logging configuration and utilities
  - `helpers/`: Common utility functions
  - `security/`: Authentication and authorization utilities
- `data/`: Data storage and persistence
  - `vector_stores/`: FAISS and other vector index storage
  - `processed/`: Processed document outputs
- `tests/`: Testing infrastructure and test cases
- `run.py`: Main application entry point for running the server
- `start.sh/bat`: Platform-specific startup scripts
- `.env`: Environment configuration file (to be created by user)

## Future Development
- In-app token management through the interface (under development)
- Additional integration options
- Enhanced analysis capabilities

## License
This project is licensed under the MIT License (Modified for Non-Commercial Use) - with the following key restrictions:
1. The software cannot be used for commercial purposes
2. Proper attribution to the origin of the software stack is required

See the LICENSE file for full details.
