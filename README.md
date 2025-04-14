# OATFLAKE

## Overview
OATFLAKE is a RAG-based (Retrieval-Augmented Generation) assistant platform that helps process, analyze, and interact with your data. It features integrations with Slack, OpenRouter, Supabase, and other services to provide a comprehensive solution for knowledge management and AI assistance.

## Features
- Knowledge processing and extraction
- Goal-based analysis
- Slack integration for communication
- Vector-based search capabilities
- API endpoints for various functionalities
- Web-based user interface

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

### 2. Create a virtual environment and install dependencies using Poetry
```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
# .venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies using Poetry
poetry install
```

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

## Usage

### Running the Application
To start the application:
```bash
python run.py
```

This will:
1. Start the FastAPI server
2. Set up a ngrok tunnel for external access (if configured)
3. Open the web interface in your default browser

### Processing Knowledge
To run the knowledge processing workflow:
```bash
python run_complete_processing.py
```

### Extracting Goals
To run the goal extraction:
```bash
python run_goal_extraction.py
```

### Running Analysis
To run analysis on your data:
```bash
python run_analysis.py
```

## Project Structure
- `api/`: API endpoints and routes
- `scripts/`: Core processing and analysis scripts
- `settings/`: Configuration files
- `static/`: Static assets for the web interface
- `templates/`: HTML templates
- `utils/`: Utility functions

## Future Development
- In-app token management through the interface (under development)
- Additional integration options
- Enhanced analysis capabilities

## License
This project is licensed under the MIT License (Modified for Non-Commercial Use) - with the following key restrictions:
1. The software cannot be used for commercial purposes
2. Proper attribution to the origin of the software stack is required

See the LICENSE file for full details.
