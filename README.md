# BLOB - RAG Assistant Platform

A Retrieval-Augmented Generation (RAG) platform for building AI assistants with context from your own data.

## Overview

BLOB enables you to:
- Process and index various document formats
- Ask questions against your knowledge base
- Use both local and cloud LLM providers
- Interact via API, web interface, or Slack

## Features

- **Document Processing**: Upload and process documents to create a knowledge base
- **Vector Storage**: FAISS-based vector database for efficient similarity search
- **Multiple LLM Providers**: 
  - Local models via Ollama (default: llama3.2:1b)
  - Cloud models via OpenRouter (Claude, GPT-4, etc.)
- **Web Interface**: Simple management UI
- **Slack Integration**: Chat with your assistant directly in Slack
- **API Access**: FastAPI endpoints for programmatic interaction

## Setup

### Prerequisites

- Python 3.9+
- Ollama (for local models)
- Supabase account (for authentication and storage)
- OpenRouter API key (optional, for cloud models)
- Slack API credentials (optional, for Slack integration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/blob.git
cd blob/blob-test-backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with required environment variables:
```
# Required
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Optional
OPENROUTER_API_KEY=your_openrouter_api_key
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_SIGNING_SECRET=your_slack_signing_secret
SLACK_BOT_USER_ID=your_slack_bot_user_id
```

4. Configure your LLM provider in `settings/model_settings.json`

### Model Settings

The application's LLM behavior is configured in `settings/model_settings.json`:

```json
{
  "provider": "ollama",              // "ollama" for local, "openrouter" for cloud
  "model_name": "llama3.2:1b",       // Model name for Ollama
  "openrouter_model": "anthropic/claude-3-haiku", // Model for OpenRouter
  "system_prompt": "You are an assistant...",
  "temperature": 0.7,               // Higher = more creative, lower = more deterministic
  "max_tokens": 1000,               // Maximum tokens in LLM response
  "top_p": 0.9,                     // Nucleus sampling parameter
  "top_k": 40,                      // Top-k sampling parameter
  "num_ctx": 2048,                  // Context window size
  "num_thread": 4,                  // CPU threads to use
  "stop_sequences": null,           // Optional sequences to stop generation
  "custom_parameters": null         // Provider-specific parameters
}
```

### Hardware Optimization

#### CPU Configuration

The application is optimized for CPU usage with these guidelines:

- `num_thread` setting controls CPU thread allocation:
  - 2 threads: Low CPU usage, good for background processing
  - 4 threads: Balanced performance (default)
  - 8 threads: Faster but more CPU intensive

- For low-powered devices (like Raspberry Pi):
  - Reduce `num_thread` to 2
  - Set smaller `num_ctx` (e.g., 1024)
  - Use smaller models (e.g., tinyllama)

- For powerful workstations:
  - Increase `num_thread` up to your CPU core count
  - Use larger models for better responses

#### GPU Support

The application primarily uses CPU for inference. GPU acceleration depends on:

1. Using Ollama with GPU support enabled
2. Having compatible CUDA/ROCm drivers installed
3. Using models optimized for your specific GPU

### Running the Application

Start the server:
```bash
python run.py
```

The application will be available at http://localhost:8999

## Ollama Setup (for local models)

1. Install Ollama:
```bash
curl https://ollama.ai/install.sh | sh
```

2. Start Ollama service:
```bash
ollama serve
```

3. Pull the default model:
```bash
ollama pull llama3.2:1b
```

## Usage

### Processing Documents

Upload documents through the web interface or add them to your configured data directory.

```bash
# Process all knowledge in the configured directories
python run_complete_processing.py
```

### Asking Questions

Use the web interface or API to ask questions against your knowledge base:

1. Through the web UI at http://localhost:8999
2. Via API endpoints:
   ```
   POST /api/questions/ask
   {
     "question": "Your question here",
     "project_id": "your_project_id"
   }
   ```
3. Using the included test script:
   ```bash
   python test_questions.py
   ```

### Performance Monitoring

The application includes performance tracking for resource-intensive operations:

- Vector embedding generation shows progress and timing information
- Document processing displays batch processing metrics
- Retrieval operations provide timing and match score details

For long-running operations, check the console output for real-time performance data.

### Slack Integration

If configured, interact with your assistant directly in Slack using the `/ask` command:
```
/ask What information do you have about project X?
```

## Project Structure

- `api/`: FastAPI routes and endpoints
- `models/`: Pydantic schemas
- `scripts/groups/`: Core RAG implementation
- `services/`: Backend services and connections
- `settings/`: Configuration files


## Acknowledgements

Built with:
- FastAPI
- Supabase
- FAISS
- LangChain
- Ollama
- OpenRouter