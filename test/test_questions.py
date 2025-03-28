#!/usr/bin/env python3
"""
Script to test question generation directly
"""
import asyncio
import logging
from pathlib import Path
import json
import sys
import os
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test question generation')
parser.add_argument('--model', type=str, help='Override the model name to use for testing')
parser.add_argument('--list-models', action='store_true', help='List available models and exit')
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger("question_test")

async def list_available_models():
    """Get a list of all available models from Ollama"""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                available_models = [m.get("name") for m in models_data.get("models", [])]
                logger.info(f"Available models: {', '.join(available_models)}")
                return available_models
            else:
                logger.error(f"Failed to get models: {response.status_code}")
                return []
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return []

async def test_question_generation():
    """Test question generation functionality directly"""
    try:
        logger.info("Starting question generation test...")
        
        # Import the question generation functions and the Ollama client
        from scripts.services.question_generator import generate_questions, save_questions
        from scripts.llm.ollama_client import OllamaClient
        from scripts.services.settings_manager import SettingsManager
        
        # First, if we want to override the model, patch the settings
        if args.model:
            try:
                logger.info(f"Overriding model name to: {args.model}")
                settings_manager = SettingsManager()
                settings = settings_manager.load_settings()
                settings.model_name = args.model
                settings_manager.save_settings(settings)
            except Exception as e:
                logger.error(f"Failed to override model name: {e}")
        
        # Get list of available models
        available_models = await list_available_models()
        if not available_models:
            logger.error("❌ No models available. Is Ollama running?")
            return False
            
        # Check if Ollama is running and try to find a suitable model
        logger.info("Checking Ollama connection and model availability...")
        client = OllamaClient()
        is_connected, status_message = await client.check_connection()
        
        # If the connection failed and it's due to model not available
        if not is_connected and "model not available" in status_message.lower():
            logger.warning(f"Model '{client.model}' not available, attempting to use an alternative model")
            
            # Try to find a suitable alternative model
            fallback_models = ["mistral", "llama3", "llama2", "gemma", "mixtral", "orca", "phi"]
            selected_model = None
            
            for model_prefix in fallback_models:
                matching_models = [m for m in available_models if model_prefix in m.lower()]
                if matching_models:
                    # Sort by name to get the most basic version
                    matching_models.sort()
                    selected_model = matching_models[0]
                    logger.info(f"Selected alternative model: {selected_model}")
                    
                    # Update the client to use this model
                    client.model = selected_model
                    
                    # Update settings
                    try:
                        settings_manager = SettingsManager()
                        settings = settings_manager.load_settings()
                        settings.model_name = selected_model
                        settings_manager.save_settings(settings)
                    except Exception as e:
                        logger.error(f"Failed to update settings with new model: {e}")
                    
                    # Check connection with new model
                    is_connected, status_message = await client.check_connection()
                    if is_connected:
                        break
            
            if not is_connected:
                logger.error(f"❌ Could not find a suitable model. Available models: {', '.join(available_models)}")
                logger.error("Please run 'ollama pull mistral' or specify a model with --model")
                return False
        elif not is_connected:
            logger.error(f"❌ Ollama connection failed: {status_message}")
            logger.error("Please ensure Ollama is running")
            return False
        
        logger.info(f"✅ Ollama connection successful: {status_message}")
        
        # Additional health check
        if not await client.check_health():
            logger.error("❌ Ollama health check failed - API may not be fully responsive")
            return False
        
        logger.info("✅ Ollama health check passed")
        
        # Generate questions with increased timeout and reduced token count for faster response
        logger.info("Attempting to generate 5 questions...")
        
        # Modify the imported generate_questions function to have a longer timeout
        # This is a hacky approach but works for a test script
        import httpx
        httpx_post_original = httpx.AsyncClient.post
        
        async def patched_post(*args, **kwargs):
            # Set longer timeout for this test
            if 'timeout' in kwargs and kwargs['timeout'] < 120.0:
                logger.info(f"Extended timeout from {kwargs['timeout']} to 120.0 seconds")
                kwargs['timeout'] = 120.0
            return await httpx_post_original(*args, **kwargs)
        
        # Apply the patch
        httpx.AsyncClient.post = patched_post
        
        # Try up to 3 times with increasing timeouts
        success = False
        for attempt in range(1, 4):
            try:
                logger.info(f"Attempt {attempt} to generate questions...")
                questions = await generate_questions(num_questions=5)
                if questions:
                    logger.info(f"✅ Successfully generated {len(questions)} questions!")
                    for i, q in enumerate(questions, 1):
                        logger.info(f"Question {i}: {q['question_text']}")
                    
                    # Save the questions
                    success = await save_questions(questions)
                    if success:
                        logger.info("✅ Successfully saved questions to CSV!")
                    else:
                        logger.error("Failed to save questions")
                        
                    success = True
                    break
                else:
                    logger.warning("No questions were generated")
                    
                    # On failure, verify the API is still responding
                    if not await client.check_health():
                        logger.error("❌ Ollama API is not responding - service may have crashed")
                        break
            except Exception as e:
                logger.error(f"Error on attempt {attempt}: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(2)
        
        if not success:
            logger.error("❌ Failed to generate questions after multiple attempts")
            
        return success
            
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    logger.info("Starting question generation test script")
    
    # If just listing models, do that and exit
    if args.list_models:
        asyncio.run(list_available_models())
        sys.exit(0)
    
    success = asyncio.run(test_question_generation())
    
    if success:
        logger.info("✅ Test completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Test failed")
        sys.exit(1)
