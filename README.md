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

1. **FastAPI Backend**: Powers all API endpoints and server-side operations
2. **Web Interface**: Modern, responsive JavaScript frontend for interacting with the system
3. **RAG Pipeline**: End-to-end pipeline for document processing, analysis, and retrieval
4. **Vector Storage**: FAISS-based vector storage for efficient similarity searches
5. **Integrations**: Connections to external systems (Slack, OpenRouter, etc.)

### Data Flow

1. **Input Sources** → Document uploaded or URL provided
2. **Content Processing** → Text extraction and chunking
3. **Analysis** → LLM-powered analysis with entity extraction
4. **Embedding Generation** → Vector embedding creation
5. **Storage** → Persistence to vector stores and databases
6. **Retrieval** → Context-sensitive document retrieval
7. **Generation** → LLM-augmented response creation

### Submodules

- **Analysis Engine**: LLM-powered content analysis and entity extraction
- **Storage Manager**: Vector store and file-based data persistence
- **Scheduler System**: Task scheduling and automated processing
- **UI Components**: Modular interface elements with independent functionality
- **API Endpoints**: RESTful API organized by functional domain

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
  - `middleware/`: Request processing and authentication middleware
  - `models/`: Pydantic data models for API requests and responses
  - `dependencies/`: Reusable API dependencies and injected services
  - `main.py`: Entry point and router registration
- `scripts/`: Core processing and analysis scripts
  - `analysis/`: LLM-powered content analysis modules
  - `extraction/`: Text extraction from various document formats
  - `embedding/`: Vector embedding generation and management
  - `services/`: Background services and scheduled tasks
  - `integration/`: External system connectors (Slack, OpenRouter)
- `settings/`: Configuration files and environment settings
  - `config.py`: Central configuration management
  - `default_settings.json`: Default application settings
- `static/`: Static assets for the web interface
  - `js/`: JavaScript modules and UI components
  - `css/`: Styling and theme definitions
  - `img/`: Images, icons and visual assets
  - `vendor/`: Third-party libraries and dependencies
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
