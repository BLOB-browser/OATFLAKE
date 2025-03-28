#!/usr/bin/env python3
"""
Test script for OpenRouter integration.
Set OPENROUTER_API_KEY environment variable before running.
"""

import asyncio
import os
from scripts.llm.open_router_client import OpenRouterClient

async def test_openrouter():
    # Check if API key is set
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY environment variable is not set")
        print("Run: export OPENROUTER_API_KEY=your_api_key")
        return
    
    # Create client
    client = OpenRouterClient(api_key=api_key)
    
    # Check connection
    print("Testing connection to OpenRouter...")
    ready, message = await client.check_connection()
    print(f"Connection status: {ready}, Message: {message}")
    
    if not ready:
        print("Failed to connect to OpenRouter. Check your API key.")
        return
    
    # List models
    print("\nListing available models...")
    models = await client.list_available_models()
    
    if not models:
        print("No models available. Check API key and connection.")
        return
    
    print(f"Found {len(models)} models:")
    for i, model in enumerate(models[:5]):  # Show first 5 models
        print(f"{i+1}. {model['id']} - {model['name']}")
    
    if len(models) > 5:
        print(f"...and {len(models) - 5} more")
    
    # Set default model to first one (usually gpt-3.5-turbo)
    if models:
        client.default_model = models[0]['id']
        print(f"\nSet default model to: {client.default_model}")
    
    # Test embedding and RAG
    print("\nTesting local embedding and retrieval...")
    
    # Check if we have vector stores loaded
    if client.reference_store is None and client.content_store is None:
        print("WARNING: No vector stores loaded. Context retrieval will be empty.")
        print("Upload documents through the API first to populate vector stores.")
    
    # Test RAG generation
    test_prompt = "What is the most important concept in the knowledge base?"
    print(f"\nTesting RAG with prompt: '{test_prompt}'")
    
    response = await client.generate_response(
        prompt=test_prompt,
        context_k=3,
        max_tokens=500
    )
    
    print("\nResponse:")
    print("-" * 80)
    print(response)
    print("-" * 80)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_openrouter())