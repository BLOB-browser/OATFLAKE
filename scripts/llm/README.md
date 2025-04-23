# LLM Module

## Overview

The LLM (Language Learning Model) module provides the interface between OATFLAKE and various language models. It abstracts away the complexity of working with different model providers and enables consistent interactions regardless of the underlying model implementation.

This module supports both local model inference via Ollama and cloud-based models via OpenRouter, allowing users to choose the approach that best fits their needs, hardware constraints, and privacy requirements.

## Key Components

### Model Clients

#### Ollama Integration (`ollama_client.py`)

Enables communication with locally-hosted models through Ollama:

- **Local Inference**: Run models entirely on your machine without external dependencies
- **Privacy-Focused**: Keep all data and prompts on your local hardware
- **Default Model**: `mistral:7b-instruct-v0.2-q4_0` (configurable through settings)
- **Embedding Generation**: Create vector embeddings for similarity search
- **Vector Store Management**: Load and interact with FAISS vector stores

```python
from scripts.llm.ollama_client import OllamaClient

# Initialize client
client = OllamaClient()

# Generate text with the model
response = client.generate("Explain retrieval-augmented generation")

# Generate with specific parameters
response = client.generate(
    "Summarize this article", 
    temperature=0.3, 
    max_tokens=150
)
```

#### OpenRouter Integration (`open_router_client.py`)

Provides access to cloud-hosted models through OpenRouter API:

- **Model Variety**: Access to various commercial models (Anthropic, OpenAI, etc.)
- **Fallback Mechanism**: Use when local resources are insufficient
- **Connection Caching**: Efficient management of API connections
- **Status Monitoring**: Track connection status and available models

```python
from scripts.llm.open_router_client import OpenRouterClient

# Initialize client with API key from environment
client = OpenRouterClient()

# Check connection status
is_connected, status = client.check_connection()

# Get available models
models = client.list_models()

# Generate text with default model
response = client.generate("Explain RAG systems")

# Generate with specific model
response = client.generate(
    "Explain RAG systems",
    model="anthropic/claude-3-opus:free"
)
```

### Embedding Services

#### Ollama Embeddings (`ollama_embeddings.py`)

Provides vector embeddings for RAG functionality:

- **Text-to-Vector Conversion**: Convert text chunks into vector representations
- **Batch Processing**: Efficiently process multiple documents
- **Compatible Interface**: Works with LangChain's vector stores
- **Customizable Dimensions**: Configure embedding dimensions based on needs

```python
from scripts.llm.ollama_embeddings import OllamaEmbeddings

# Initialize embeddings
embeddings = OllamaEmbeddings()

# Generate embeddings for text
vector = embeddings.embed_query("Sample text to embed")

# Generate embeddings for multiple texts
vectors = embeddings.embed_documents(["Text 1", "Text 2", "Text 3"])
```

### RAG Functionality

#### RAG Handler (`rag_handler.py`)

Handles retrieval-augmented generation operations:

- **Context Retrieval**: Find relevant documents based on queries
- **Content Augmentation**: Enhance LLM responses with retrieved information
- **Vector Search**: Perform similarity search across vector stores
- **Result Formatting**: Structure retrieved content for optimal LLM context

```python
from scripts.llm.rag_handler import RAGHandler

# Initialize handler
handler = RAGHandler()

# Get augmented response
response = handler.get_rag_response(
    "What are the key components of OATFLAKE?",
    max_tokens=500
)

# Retrieve relevant documents only
documents = handler.retrieve_relevant_documents(
    "Knowledge processing in OATFLAKE",
    k=5  # Number of documents to retrieve
)
```

### Configuration Utilities

#### LLM Configuration (`llm_config_utils.py`)

Provides utilities for configuring LLM behavior:

- **Parameter Management**: Standardized configuration across model providers
- **Hardware-Aware Settings**: Adapt settings based on available resources
- **Default Generation Parameters**: Pre-configured parameters for common tasks
- **Environment Variables**: Override settings through environment variables

#### Processor Configuration (`processor_config_utils.py`)

Configures processing behavior for optimal performance:

- **Thread Management**: Configure parallel processing capabilities
- **Batch Size Optimization**: Adjust batch sizes based on hardware
- **Memory Monitoring**: Track and manage memory usage during processing
- **CPU Optimization**: Adapt workload based on available CPU cores

## Usage Examples

### Basic LLM Interaction

```python
from scripts.llm.ollama_client import OllamaClient
from scripts.llm.open_router_client import OpenRouterClient

# Try OpenRouter first, fall back to Ollama
try:
    client = OpenRouterClient()
    if client.is_connected():
        response = client.generate("Explain OATFLAKE")
    else:
        # Fallback to local model
        client = OllamaClient()
        response = client.generate("Explain OATFLAKE")
except Exception as e:
    logger.error(f"Error with LLM: {e}")
```

### RAG-Enhanced Question Answering

```python
from scripts.llm.rag_handler import RAGHandler

handler = RAGHandler()

# Answer question with relevant context
answer = handler.answer_with_context(
    question="How does OATFLAKE handle knowledge processing?",
    k=3,  # Number of relevant documents to include
    temperature=0.2
)

print(answer)
```

### Creating and Using Vector Stores

```python
from scripts.llm.ollama_embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Create embeddings
embeddings = OllamaEmbeddings()

# Create vector store from documents
documents = [
    Document(page_content="OATFLAKE provides community-governed intelligence training", metadata={"source": "readme"}),
    Document(page_content="RAG systems enhance LLM responses with retrieved information", metadata={"source": "article"})
]

vector_store = FAISS.from_documents(documents, embeddings)

# Search the vector store
results = vector_store.similarity_search("How does OATFLAKE help communities?", k=1)
```

## Configuration

The LLM module relies on several configuration options that can be set through environment variables or settings files:

```
# Ollama Configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434

# OpenRouter Configuration
OPENROUTER_API_KEY=your-openrouter-api-key

# Model Settings
DEFAULT_MODEL=mistral:7b-instruct-v0.2-q4_0
EMBEDDING_MODEL=nomic-embed-text
```

Additional configuration is available through the settings interface in the OATFLAKE UI.

## Architecture

### Model Selection Flow

1. Check for OpenRouter API key and connectivity
2. If available and connected, use selected OpenRouter model
3. If unavailable or disconnected, fall back to local Ollama model
4. If specific model requested and available, use that model

### RAG Implementation

1. User query is received
2. Relevant documents are retrieved from vector stores
3. Retrieved content is formatted as context
4. Context and query are sent to the LLM
5. LLM generates a response based on the context and query
6. Response is returned to the user

## Performance Considerations

- Ollama models run on local hardware and performance depends on your machine's capabilities
- Vector operations can be memory-intensive during search operations
- Consider hardware limitations when selecting models and batch sizes
- OpenRouter models may have request latency but don't consume local resources

## Future Development

- Support for additional model providers
- Enhanced model caching for improved performance
- More sophisticated RAG strategies (multi-vector retrieval, etc.)
- Streaming responses for long-form content generation
