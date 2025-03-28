import os
import logging
import re
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler
from dotenv import load_dotenv
from fastapi import Request
from scripts.services.settings_manager import SettingsManager
import json

load_dotenv()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

BOT_ID = os.environ.get("SLACK_BOT_USER_ID")
settings_manager = SettingsManager()

slack_app = AsyncApp(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)
handler = AsyncSlackRequestHandler(slack_app)

# Remove the direct OllamaClient initialization
# Instead, we'll get it from the app state

# Add URL detection function
def extract_urls(text: str) -> list:
    """Extract URLs from text"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

@slack_app.event("message")
async def handle_message(body, say, client):
    """Handle direct messages to the bot"""
    try:
        event = body.get('event', {})
        
        # Log full event data for debugging
        logger.debug(f"Received message event: {json.dumps(event, indent=2)}")
        
        # Ignore bot messages, thread replies, and non-DM messages
        if (event.get("bot_id") or 
            event.get("subtype") or 
            event.get("user") == BOT_ID or 
            event.get("channel_type") != "im"):
            return

        text = event.get('text', '')
        channel = event.get('channel')
        thread_ts = event.get('ts')
        
        # Check for URLs in the message
        urls = extract_urls(text)
        if urls:
            url_response = "I found these URLs in your message:\n"
            for url in urls:
                if url.lower().endswith('.pdf'):
                    url_response += f"• PDF file: {url}\n"
                else:
                    url_response += f"• Website: {url}\n"
            await say(text=url_response, channel=channel, thread_ts=thread_ts)
            
        # Check for files in the message
        files = event.get('files', [])
        if files:
            file_response = "I found these files in your message:\n"
            for file in files:
                file_type = file.get('filetype', 'unknown').upper()
                file_name = file.get('name', 'unnamed')
                file_url = file.get('url_private', 'no url')
                file_response += f"• {file_type} file: {file_name} ({file_url})\n"
            await say(text=file_response, channel=channel, thread_ts=thread_ts)

        # Get settings for context
        settings = settings_manager.load_settings()
        
        # Command handling
        if text.lower().startswith(("hello", "hi", "hey")):
            response = f"Hi <@{event['user']}>! 👋 How can I help you?"
        elif text.lower() == "help":
            response = """Here's what I can do:
• Say hello
• Share URLs or files - I'll analyze them
• Ask me anything about MDEF
• Type `help` to see this message again"""
        else:
            # Get client based on settings
            from app import app
            from scripts.models.settings import LLMProvider
            
            # Determine which LLM provider to use based on settings
            if settings.provider == LLMProvider.OPENROUTER:
                # Check if OpenRouter client exists and has API key
                if hasattr(app.state, "openrouter_client") and app.state.openrouter_client.api_key:
                    llm_client = app.state.openrouter_client
                    model = settings.openrouter_model
                    logger.info(f"Using OpenRouter with model {model} for response")
                else:
                    # Fallback to Ollama if OpenRouter not configured
                    llm_client = app.state.ollama_client
                    model = settings.model_name
                    logger.info(f"OpenRouter selected but not configured, falling back to Ollama with model {model}")
            else:
                # Use Ollama
                llm_client = app.state.ollama_client
                model = settings.model_name
                logger.info(f"Using Ollama with model {model} for response")
            
            try:
                context_prompt = f"""Using the MDEF knowledge base, help with: {text}
                Be helpful and educational while following these guidelines:
                {settings.system_prompt}"""
                
                # For OpenRouter, pass the model explicitly
                if settings.provider == LLMProvider.OPENROUTER and hasattr(app.state, "openrouter_client"):
                    response = await llm_client.generate_response(
                        prompt=context_prompt,
                        model=model
                    )
                else:
                    # For Ollama, use default method
                    response = await llm_client.generate_response(context_prompt)
                    
            except Exception as e:
                logger.error(f"LLM error with {settings.provider}: {e}")
                response = "Sorry, I had trouble processing that. Could you try rephrasing?"

        await say(text=response, channel=channel, thread_ts=thread_ts)

    except Exception as e:
        logger.error(f"Error in message handler: {e}", exc_info=True)
        await say(
            text="Sorry, something went wrong! Please try again.",
            channel=channel,
            thread_ts=thread_ts
        )

@slack_app.event("app_mention")
async def handle_mention(body, say, client):
    """Handle mentions in channels"""
    try:
        event = body.get('event', {})
        logger.debug(f"Received mention event: {json.dumps(event, indent=2)}")
        text = event.get('text', '').replace(f'<@{BOT_ID}>', '').strip()
        channel = event.get('channel')
        thread_ts = event.get('ts')

        # Get settings for context
        settings = settings_manager.load_settings()

        if text.lower() in ["hello", "hi", "hey"]:
            response = f"Hi <@{event['user']}>! 👋"
        elif text.lower() == "help":
            response = "DM me for a private chat or mention me with your questions!"
        else:
            # Get client based on settings
            from app import app
            from scripts.models.settings import LLMProvider
            
            # Determine which LLM provider to use based on settings
            if settings.provider == LLMProvider.OPENROUTER:
                # Check if OpenRouter client exists and has API key
                if hasattr(app.state, "openrouter_client") and app.state.openrouter_client.api_key:
                    llm_client = app.state.openrouter_client
                    model = settings.openrouter_model
                    logger.info(f"Using OpenRouter with model {model} for mention response")
                else:
                    # Fallback to Ollama if OpenRouter not configured
                    llm_client = app.state.ollama_client
                    model = settings.model_name
                    logger.info(f"OpenRouter selected but not configured, falling back to Ollama with model {model}")
            else:
                # Use Ollama
                llm_client = app.state.ollama_client
                model = settings.model_name
                logger.info(f"Using Ollama with model {model} for mention response")
            
            try:
                prompt = f"""Someone in MDEF asks: {text}
                Provide a helpful response following these guidelines:
                {settings.system_prompt}"""
                
                # For OpenRouter, pass the model explicitly
                if settings.provider == LLMProvider.OPENROUTER and hasattr(app.state, "openrouter_client"):
                    response = await llm_client.generate_response(
                        prompt=prompt,
                        model=model
                    )
                else:
                    # For Ollama, use default method
                    response = await llm_client.generate_response(prompt)
                    
            except Exception as e:
                logger.error(f"LLM error with {settings.provider}: {e}")
                response = "Sorry, I'm having trouble right now. Try again later?"

        await say(text=response, channel=channel, thread_ts=thread_ts)

    except Exception as e:
        logger.error(f"Error in mention handler: {e}")
        await say(text="Sorry, I encountered an error! 😅", channel=channel, thread_ts=thread_ts)