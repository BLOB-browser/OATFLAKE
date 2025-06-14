#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Universal Analysis LLM for content extraction following the universal table structure.
This implementation can process various content types using configurable prompts.
"""

import logging
import json
import re
import requests
import httpx
import time
import os
from pathlib import Path
import uuid
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import concurrent.futures

# Add import for interruptible requests
from scripts.analysis.interruptible_llm import (
    interruptible_post, is_interrupt_requested, 
    setup_interrupt_handling, restore_interrupt_handling,
    clear_interrupt
)

# Import only the UniversalTable schema for validation
from scripts.models.schemas import UniversalTable

# Import text splitter for chunking functionality
from langchain.text_splitter import RecursiveCharacterTextSplitter
from scripts.models.settings import LLMProvider

logger = logging.getLogger(__name__)

class UniversalAnalysisLLM:
    """
    A universal LLM interface for extracting various content types following
    the universal table structure. Uses configurable prompts from analysis-tasks-config.json.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = None, data_folder: str = None):
        """
        Initialize UniversalAnalysisLLM with static model settings.
        
        Args:
            base_url: Ollama API base URL
            model: Override model name
            data_folder: Path to data directory
        """
        # Set up data folder
        if data_folder is None:
            from utils.config import get_data_path
            data_folder = get_data_path()
        self.data_folder = data_folder
        
        # Set base URL
        self.base_url = base_url
        
        # Initialize DataSaver for CSV storage operations
        from scripts.analysis.data_saver import DataSaver
        self.data_saver = DataSaver(self.data_folder)
        
        # Load analysis-specific model settings
        self.settings = self._load_analysis_settings()
        
        # Set model based on settings or parameters
        if model:
            # Override with provided model
            if self.settings["provider"] == "ollama":
                self.model = model
            else:
                self.openrouter_model = model
        else:
            # Use settings
            if self.settings["provider"] == "ollama":
                self.model = self.settings["model_name"]
            else:
                self.openrouter_model = self.settings["openrouter_model"]
        
        # Load task configuration with prompts
        self.task_config = self._load_task_config()
        
        # Get OpenRouter API key if OpenRouter is the provider
        self.openrouter_api_key = None
        if self.settings["provider"] == "openrouter":
            self._load_openrouter_api_key()
        
        # Log initialization
        logger.info(f"🚀 Initialized UniversalAnalysisLLM with static model settings")
        logger.info(f"Provider: {self.settings['provider']}")
        if self.settings["provider"] == "ollama":
            logger.info(f"Using Ollama model: {self.model}")
        else:
            logger.info(f"Using OpenRouter model: {self.openrouter_model}")
        
        logger.info(f"Model config: {self.settings['num_thread']} threads, {self.settings['num_ctx']} context")
    
    def _load_analysis_settings(self) -> Dict:
        """
        Load settings from analysis-model-settings.json.
        If the file doesn't exist, create it with default settings.
        
        Returns:
            Dictionary of analysis model settings
        """
        # Default settings
        default_settings = {
            "provider": "ollama",
            "model_name": "mistral:7b-instruct-v0.2-q4_0",
            "openrouter_model": "anthropic/claude-3-haiku",
            "system_prompt": "You are an AI assistant helping with content analysis and knowledge extraction.",
            "temperature": 0.3,
            "max_tokens": 2000,
            "top_p": 0.9,
            "top_k": 40,
            "num_ctx": 256,
            "num_thread": 4,
            "stop_sequences": None,
            "custom_parameters": None
        }
        
        # Path to analysis model settings in scripts/settings directory
        settings_path = Path(__file__).parent.parent / "settings" / "analysis-model-settings.json"
        
        try:
            # Try to load the settings if the file exists
            if settings_path.exists():
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                logger.info(f"Loaded analysis model settings from {settings_path}")
                return settings
            else:
                # If file doesn't exist, create it with default settings
                os.makedirs(settings_path.parent, exist_ok=True)
                with open(settings_path, "w", encoding="utf-8") as f:
                    json.dump(default_settings, f, indent=4)
                logger.info(f"Created new analysis model settings at {settings_path}")
                return default_settings
                
        except Exception as e:
            logger.error(f"Error loading analysis model settings: {e}")
            logger.warning("Using default analysis model settings")
            return default_settings
            
    def _load_openrouter_api_key(self):
        """Load OpenRouter API key from environment or .env files"""
        # First try from environment
        api_key = os.environ.get("OPENROUTER_API_KEY")
        
        # If not in environment, try ~/.blob/.env
        if not api_key:
            blob_env = Path.home() / ".blob" / ".env"
            if blob_env.exists():
                try:
                    with open(blob_env) as f:
                        for line in f:
                            if line.strip() and not line.startswith('#'):
                                try:
                                    key, value = line.strip().split('=', 1)
                                    if key == "OPENROUTER_API_KEY":
                                        api_key = value.strip().strip('"\'')
                                        break
                                except ValueError:
                                    pass
                except Exception as e:
                    logger.warning(f"Error reading from {blob_env}: {e}")
        
        self.openrouter_api_key = api_key
        if not api_key and self.settings["provider"] == "openrouter":
            logger.warning("OpenRouter selected as provider but no API key found. Analysis may fail.")
    
    def _load_task_config(self) -> Dict:
        """
        Load task configuration from analysis-tasks-config.json, containing prompts for different content types.
        Falls back to empty default configuration if the file doesn't exist.
        
        Returns:
            Dictionary of task configurations
        """
        # Empty default configuration
        default_config = {}
        
        # Look in scripts/settings directory
        config_path = Path(__file__).parent.parent / "settings" / "analysis-tasks-config.json"
        
        try:
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                logger.info(f"Loaded task configuration from {config_path}")
                return config
            else:
                logger.warning(f"Task configuration file not found at {config_path}")
                return default_config
                    
        except Exception as e:
            logger.error(f"Error loading task configuration: {e}")
            logger.warning("Using empty default task configuration")
            return default_config
    
    def generate_structured_response(self, prompt: str, format_type: str = "json", 
                                   temperature: float = None, max_tokens: int = None) -> Any:
        """
        Generate a structured response from the LLM (either Ollama or OpenRouter).
        
        Args:
            prompt: The prompt to send to the model
            format_type: Type of formatting to expect ("json", "list", "text")
            temperature: Temperature setting (lower for more deterministic output)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed structured response (JSON object, list, or text)
        """
        # Check for interruption
        if is_interrupt_requested():
            logger.warning("Skipping analysis due to interrupt request")
            return None
        
        # Use settings for parameters if not explicitly specified
        temperature_value = temperature if temperature is not None else float(self.settings["temperature"])
        max_tokens_value = max_tokens if max_tokens is not None else int(self.settings["max_tokens"])
        
        # Use appropriate provider based on settings
        if self.settings["provider"] == "ollama":
            return self._generate_with_ollama(prompt, format_type, temperature_value, max_tokens_value)
        else:
            return self._generate_with_openrouter(prompt, format_type, temperature_value, max_tokens_value)
    
    def _generate_with_ollama(self, prompt: str, format_type: str, temperature: float, max_tokens: int) -> Any:
        """Generate response using Ollama"""
        # Verify model exists
        try:
            model_check_response = requests.get(f"{self.base_url}/api/tags")
            if model_check_response.status_code == 200:
                models_data = model_check_response.json()
                available_models = [m.get("name") for m in models_data.get("models", [])]
                
                if self.model not in available_models:
                    logger.warning(f"Model {self.model} not found, checking for alternatives")
                    # Fall back to any available instruction model
                    available_models = [m for m in available_models if 'instruct' in m.lower()]
                    if available_models:
                        self.model = available_models[0]
                        logger.warning(f"Falling back to available model: {self.model}")
                    else:
                        logger.error("No instruction models available")
                        return None
        except Exception as e:
            logger.warning(f"Error checking model availability: {e}")
        
        # Add formatting guidance for Mistral model
        if format_type == "json":                    
            # Use system prompt from config with JSON-specific formatting for Mistral
            base_system_prompt = self.settings.get("system_prompt", "You are an expert content analyzer.")
            prompt = f"""<s>[INST] {base_system_prompt}

{prompt.strip()}

CRITICAL: Your response MUST be a JSON array starting with [ and ending with ]. Never return a single object. Always return an array format like: [{{...}}]

Output ONLY the JSON array with no additional text. [/INST]</s>"""
        else:
            # For non-JSON responses, still use the Mistral instruction format with config system prompt
            base_system_prompt = self.settings.get("system_prompt", "You are an expert content analyzer.")
            prompt = f"""<s>[INST] {base_system_prompt}

{prompt.strip()} [/INST]</s>"""
        
        # Make API call to generate response
        try:
            logger.info(f"Generating structured response for content analysis using Ollama: {self.model}")
            
            # Debug: Print the full prompt being sent to the LLM
            print(f"\n📋 DEBUG - FULL PROMPT SENT TO LLM:")
            print("=" * 80)
            print(f"Length: {len(prompt)} characters")
            print("First 1000 chars:")
            print(prompt[:1000])
            print("...")
            print("Last 1000 chars:")
            print(prompt[-1000:])
            print("=" * 80)
            
            start_time = time.time()
            
            api_params = {
                "model": self.model,
                "prompt": prompt.strip(),
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens,
                # Add model configuration parameters using settings
                "options": {
                    "num_thread": self.settings.get("num_thread", 4),
                    "num_ctx": self.settings.get("num_ctx", 256),
                    "top_p": self.settings.get("top_p", 0.9),
                    "top_k": self.settings.get("top_k", 40),
                }
            }
            
            # Use increased timeout for content analysis (5 minutes)
            timeout_seconds = 300
            
            # Use standard requests with timeout
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=api_params,
                timeout=timeout_seconds
            )
            
            # Process response normally
            if response.status_code != 200:
                logger.error(f"Error from Ollama API: {response.status_code}")
                logger.error(f"Response text: {response.text}")
                return None
            
            result = response.json()
            
            # Check result format
            if "response" not in result:
                logger.error(f"Invalid response format from Ollama: {result}")
                return None
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Generated response in {duration:.2f} seconds")
            
            if format_type == "json":
                try:
                    # Extract JSON from response using robust parsing
                    json_str = result["response"].strip()
                    print(f"\n🤖 RAW LLM RESPONSE:")
                    print("=" * 60)
                    print(f"Response length: {len(json_str)} characters")
                    print(f"First 500 chars: {json_str[:500]}")
                    print("=" * 60)
                    
                    structured_response = self._parse_json_response(json_str)
                    
                    print(f"\n📊 PARSED JSON RESPONSE:")
                    print("=" * 60)
                    if structured_response is not None:
                        print(f"Type: {type(structured_response)}")
                        if isinstance(structured_response, list):
                            print(f"Number of items: {len(structured_response)}")
                            for i, item in enumerate(structured_response):
                                print(f"Item {i+1}: {json.dumps(item, indent=2)}")
                        else:
                            print(f"Single item: {json.dumps(structured_response, indent=2)}")
                    else:
                        print("❌ PARSING FAILED - structured_response is None")
                    print("=" * 60)
                    
                    # Check if parsing failed
                    if structured_response is None:
                        logger.error("Failed to parse JSON response with robust parser")
                        return None
                    
                    # Ensure it's a list
                    if isinstance(structured_response, dict):
                        structured_response = [structured_response]
                    
                    # Apply item limits if json response has multiple items
                    if isinstance(structured_response, list):
                        # Limit to 5 items per chunk
                        structured_response = structured_response[:5]
                        
                        # Add universal table schema defaults
                        for item in structured_response:
                            item["location"] = item.get("location", "")
                            
                    print(f"\n🔧 AFTER PROCESSING:")
                    print("=" * 60)
                    if isinstance(structured_response, list):
                        print(f"Final number of items: {len(structured_response)}")
                        for i, item in enumerate(structured_response):
                            print(f"Final Item {i+1}:")
                            print(f"  - title: {item.get('title', 'NO TITLE')}")
                            print(f"  - description: {item.get('description', 'NO DESCRIPTION')[:100]}...")
                            print(f"  - location: '{item.get('location', 'NO LOCATION')}'")
                            print(f"  - tags: {item.get('tags', 'NO TAGS')}")
                            print(f"  - purpose: {item.get('purpose', 'NO PURPOSE')[:50]}...")
                    print("=" * 60)
                            
                    logger.info(f"Successfully parsed JSON response with {len(structured_response) if isinstance(structured_response, list) else 1} items")
                    return structured_response
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Raw response text: {result['response'][:200]}...")
                    return None
            else:
                # For non-JSON responses, return the raw text
                return result["response"].strip()
                
        except Exception as e:
            logger.error(f"Error in Ollama API request: {e}")
            return None
    
    def _generate_with_openrouter(self, prompt: str, format_type: str, temperature: float, max_tokens: int) -> Any:
        """Generate response using OpenRouter (synchronous version)"""
        if not self.openrouter_api_key:
            logger.error("OpenRouter API key not set. Cannot generate response.")
            return None
        
        openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        if format_type == "json":
            # Use comprehensive system prompt from config for JSON responses
            system_prompt = self.settings.get("system_prompt", "You are an expert content analyzer and JSON formatter.")
        else:
            # Use system prompt from config for non-JSON responses  
            system_prompt = self.settings.get("system_prompt", "You are an AI assistant helping with content analysis and knowledge extraction.")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.strip()}
        ]
        try:
            logger.info(f"Generating structured response using OpenRouter: {self.openrouter_model}")
            
            # Only setup interrupt handling if we're in the main thread
            import threading
            is_main_thread = threading.current_thread() is threading.main_thread()
            if is_main_thread:
                setup_interrupt_handling()
            
            try:
                start_time = time.time()
                headers = {
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://blob.iaac.net",
                    "X-Title": "Blob Analysis LLM"
                }
                with httpx.Client(timeout=120.0) as client:
                    response = client.post(
                        openrouter_url,
                        json={
                            "model": self.openrouter_model,
                            "messages": messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": float(self.settings.get("top_p", 0.9)),
                            "frequency_penalty": 0.0,
                            "presence_penalty": 0.0
                        },
                        headers=headers
                    )
                elapsed = time.time() - start_time
                logger.info(f"OpenRouter response time: {elapsed:.2f} seconds")
                if is_interrupt_requested():
                    logger.warning("Content analysis was interrupted")
                    return None
                if response.status_code != 200:
                    error_message = f"Error from OpenRouter API: {response.status_code}"
                    try:
                        error_details = response.json()
                        if 'error' in error_details:
                            error_detail = error_details['error']
                            if isinstance(error_detail, dict):
                                error_message += f", Details: {error_detail.get('message', error_detail)}"
                            else:
                                error_message += f", Details: {error_detail}"
                        logger.error(f"Full error response: {error_details}")
                    except:
                        if response.text:
                            error_message += f", Response: {response.text[:200]}"
                    logger.error(error_message)
                    return None
                result = response.json()
                if not result or 'choices' not in result or not result['choices']:
                    logger.error("Invalid response format from OpenRouter API")
                    return None
                raw_response = result['choices'][0]['message']['content']
                if format_type == "json":
                    try:
                        return self._parse_json_response(raw_response)
                    except Exception as e:
                        logger.error(f"Error parsing JSON response: {e}")
                        return None
                else:
                    return raw_response
            except KeyboardInterrupt:
                logger.warning("Content analysis interrupted by user")
                return None
            except httpx.ReadTimeout:
                logger.error("Request to OpenRouter timed out")
                return None
        except Exception as e:
            logger.error(f"Error generating structured response with OpenRouter: {e}")
            return None
        finally:
            # Only restore interrupt handling if we set it up (i.e., if we're in main thread)
            if is_main_thread:
                restore_interrupt_handling()
    
    def analyze_content(self, title: str, url: str, content: str, content_type: str = "resource") -> List[Dict]:
        """
        Analyze content and extract structured data based on the specified content type.
        
        Args:
            title: Resource title
            url: Resource URL
            content: Resource content text
            content_type: Type of content to extract (method, definition, project, resource, material)
            
        Returns:
            List of dictionaries containing structured data (varies based on content type)
        """
        try:
            # Validate content type - fall back to generic resource extraction if unknown
            if content_type not in self.task_config:
                logger.warning(f"Unknown content type: {content_type}, using generic resource extraction")
                content_type = "resource"
            
            # Get task prompt for this content type
            task_prompt = self.task_config.get(content_type, "")
            if not task_prompt:
                # Create a generic prompt for resource extraction
                task_prompt = "TASK: Extract key information from the content below.\nFind: important concepts, tools, methods, or resources mentioned.\nContent to analyze:"
                logger.warning(f"No task prompt found for content type: {content_type}, using generic prompt")
            
            # Use a large enough content size for analysis
            max_content_length = 6000
            truncated_content = content[:max_content_length] if content else ""
            
            # Build the complete prompt with task instruction and content
            prompt = f"{task_prompt}\n\nTitle: {title}\nURL: {url}\n\n{truncated_content}"
            
            # The system prompt already contains the JSON schema and formatting instructions
            
            # Use JSON format for structured data
            response = self.generate_structured_response(prompt, format_type="json")
            
            # Process response based on content type
            if not response:
                logger.warning(f"No response for {title} ({content_type})")
                return []
            
            # Convert to list for consistent processing
            items = []
            if isinstance(response, dict):
                items = [response]
            elif isinstance(response, list):
                items = response
            else:
                logger.warning(f"Unexpected response type for {title}: {type(response)}")
                return []
            
            # Check if response is essentially empty before processing
            if self._is_response_essentially_empty(items, title):
                logger.warning(f"LLM returned essentially empty response for {title} ({content_type})")
                return []
            
            # Validate and enrich items
            validated_items = self._validate_items(items, title, url, content_type)
            
            # Double-check that we have meaningful results after validation
            if not validated_items:
                logger.warning(f"No valid items extracted from {title} ({content_type}) after validation")
                return []
            
            logger.info(f"Extracted {len(validated_items)} {content_type} items from {title}")
            return validated_items
            
        except Exception as e:
            logger.error(f"Error analyzing content ({content_type}): {e}")
            return []
    
    def analyze_content_in_chunks(self, title: str, url: str, content: str, content_type: str = "resource") -> List[Dict]:
        """
        Analyze content by splitting it into chunks for better performance and more thorough analysis.
        
        Args:
            title: Resource title
            url: Resource URL
            content: Resource content text
            content_type: Type of content to extract
            
        Returns:
            List of dictionaries containing structured data from all chunks
        """
        try:
            # For content analysis (web content), bypass chunking since content is already limited
            # during HTML extraction (2000-4000 chars). Chunking creates unnecessary overhead.
            if len(content) <= 5000:  # Content from HTML extraction is typically pre-limited
                logger.info(f"Content is pre-limited ({len(content)} chars), using direct analysis to avoid unnecessary chunking")
                return self.analyze_content(title, url, content, content_type)
            
            # Only use chunking for very large content (e.g., full documents, PDFs)
            logger.info(f"Content is large ({len(content)} chars), proceeding with adaptive chunked analysis")
            
            # Use adaptive chunking method instead of duplicating logic
            content_chunks = self._split_content_into_chunks(content)
            
            # If content is small enough after splitting, use direct analysis
            if len(content_chunks) <= 1:
                logger.info("Content small enough for direct analysis")
                return self.analyze_content(title, url, content, content_type)
            
            # Process each chunk and collect results
            all_results = []
            for i, chunk in enumerate(content_chunks):
                logger.info(f"Analyzing chunk {i+1}/{len(content_chunks)} (size: {len(chunk)} chars)")
                
                # Check for interrupt request
                if is_interrupt_requested():
                    logger.info("Interrupt requested, stopping chunk analysis")
                    break
                
                try:
                    # Analyze this chunk
                    chunk_results = self.analyze_content(title, url, chunk, content_type)
                    
                    # Add chunk metadata to results
                    for result in chunk_results:
                        if isinstance(result, dict):
                            result['chunk_index'] = i
                            result['total_chunks'] = len(content_chunks)
                            result['chunk_size'] = len(chunk)
                        all_results.append(result)
                    
                    logger.info(f"Chunk {i+1} yielded {len(chunk_results)} results")
                    
                except Exception as e:
                    logger.error(f"Error analyzing chunk {i+1}: {e}")
                    continue
            
            # Deduplicate results based on content similarity
            unique_results = self._deduplicate_results(all_results, content_type)
            
            logger.info(f"Chunked analysis complete: {len(all_results)} total results, {len(unique_results)} unique results")
            return unique_results
            
        except Exception as e:
            logger.error(f"Error in chunked analysis for {title}: {e}")
            # Fall back to direct analysis if chunking fails
            logger.info("Falling back to direct analysis")
            return self.analyze_content(title, url, content, content_type)
    
    def _deduplicate_results(self, results: List[Dict], content_type: str) -> List[Dict]:
        """
        Remove duplicate results from chunk analysis based on content similarity.
        
        Args:
            results: List of results from all chunks
            content_type: Type of content being analyzed
            
        Returns:
            List of deduplicated results
        """
        if not results:
            return []
        
        unique_results = []
        seen_items = set()
        
        for result in results:
            if not isinstance(result, dict):
                continue
                
            # Create a normalized key for deduplication based on content type
            if content_type == "method":
                key_fields = [result.get("name", ""), result.get("description", "")]
            elif content_type == "definition":
                key_fields = [result.get("term", ""), result.get("definition", "")]
            elif content_type == "material":
                key_fields = [result.get("name", ""), result.get("type", "")]
            else:  # resource, project, or other
                key_fields = [result.get("title", ""), result.get("description", "")]
            
            # Create normalized key (lowercase, stripped)
            normalized_key = "|".join(str(field).lower().strip() for field in key_fields if field)
            
            # Skip if we've seen this item before (based on key similarity)
            if normalized_key and normalized_key not in seen_items:
                seen_items.add(normalized_key)
                unique_results.append(result)
        
        logger.info(f"Deduplication: {len(results)} -> {len(unique_results)} results")
        return unique_results
    
    def _get_field_requirements(self, content_type: str) -> str:
        """
        Get field requirements based on content type for the universal table structure.
        
        Args:
            content_type: Type of content (method, definition, material, project, resource)
            
        Returns:
            String describing required fields for the content type
        """
        base_fields = "title (required), description (required), tags (array of keywords), purpose (use case or purpose)"
        
        content_specific = {
            "method": f"{base_fields}, location (where method is used/applicable)",
            "definition": f"{base_fields}, location (context where definition applies)",
            "material": f"{base_fields}, location (where material is available/used)",
            "project": f"{base_fields}, location (project location or context)",
            "resource": f"{base_fields}, location (resource location or context)"
        }
        
        return content_specific.get(content_type, base_fields)
    
    def _validate_items(self, items: List[Dict], title: str, url: str, content_type: str) -> List[Dict]:
        """
        Validate and clean extracted items to ensure they follow the universal table structure.
        
        Args:
            items: List of extracted items
            title: Original title for fallback
            url: Original URL
            content_type: Type of content
            
        Returns:
            List of validated and cleaned items
        """
        validated_items = []
        
        for item in items:
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict item: {item}")
                continue
            
            # Check if this is essentially an empty response from LLM
            if self._is_empty_response(item):
                logger.warning(f"Skipping empty/meaningless LLM response for {title}: {item}")
                continue
                
            # Ensure required fields exist
            validated_item = {
                "title": item.get("title", title),
                "description": item.get("description", ""),
                "tags": item.get("tags", []) if isinstance(item.get("tags"), list) else [],
                "purpose": item.get("purpose", ""),
                "location": self._clean_location_field(item.get("location", "")),
                "origin_url": url,
                "content_type": content_type
            }
            
            # Clean and validate fields
            if not validated_item["title"]:
                validated_item["title"] = title
                
            if not validated_item["description"]:
                logger.warning(f"Empty description for item: {validated_item['title']}")
                
            # Ensure tags is a list
            if not isinstance(validated_item["tags"], list):
                validated_item["tags"] = []
                
            validated_items.append(validated_item)
            
        return validated_items
    
    def _clean_location_field(self, location) -> str:
        """
        Clean location field to ensure it's a simple string.
        Handle cases where LLM returns complex objects instead of simple strings.
        
        Args:
            location: Location data from LLM (could be string, dict, or other)
            
        Returns:
            Clean location string
        """
        if not location:
            return ""
        
        if isinstance(location, str):
            return location.strip()
        
        if isinstance(location, dict):
            # Try to extract meaningful location information from dict
            location_parts = []
            
            # Common location field names to check
            for field in ['name', 'address', 'street_address', 'city', 'state', 'country']:
                if field in location and location[field]:
                    location_parts.append(str(location[field]).strip())
            
            if location_parts:
                return ", ".join(location_parts)
            else:
                # If no recognizable fields, convert dict to string
                return str(location)
        
        # For other types, convert to string
        return str(location).strip() if location else ""
    
    def _is_empty_response(self, item: Dict) -> bool:
        """
        Check if an LLM response item is essentially empty or meaningless.
        
        Args:
            item: Dictionary representing an extracted item
            
        Returns:
            True if the response is considered empty/meaningless, False otherwise
        """
        # Get the main content fields
        title = item.get("title", "").strip()
        description = item.get("description", "").strip()
        purpose = item.get("purpose", "").strip()
        tags = item.get("tags", [])
        
        # Check if all main content fields are empty or trivial
        has_meaningful_title = title and len(title) > 3
        has_meaningful_description = description and len(description) > 10
        has_meaningful_purpose = purpose and len(purpose) > 5
        has_meaningful_tags = isinstance(tags, list) and len(tags) > 0 and any(tag.strip() for tag in tags)
        
        # If none of the main fields have meaningful content, it's an empty response
        if not (has_meaningful_title or has_meaningful_description or has_meaningful_purpose or has_meaningful_tags):
            logger.debug(f"Detected empty response - no meaningful content in title/description/purpose/tags")
            return True
        
        # Check for responses that only contain location (common LLM failure pattern)
        non_location_fields = [k for k in item.keys() if k not in ['location', 'origin_url', 'content_type', 'created_at']]
        meaningful_fields = []
        
        for field in non_location_fields:
            value = item.get(field, "")
            if isinstance(value, str) and value.strip() and len(value.strip()) > 2:
                meaningful_fields.append(field)
            elif isinstance(value, list) and len(value) > 0 and any(str(v).strip() for v in value):
                meaningful_fields.append(field)
        
        if len(meaningful_fields) == 0:
            logger.debug(f"Detected empty response - only location/metadata fields present")
            return True
            
        return False
    
    def _is_response_essentially_empty(self, items: List[Dict], title: str) -> bool:
        """
        Check if the entire LLM response is essentially empty or meaningless.
        
        Args:
            items: List of extracted items from LLM response
            title: Original title for context
            
        Returns:
            True if the entire response is considered empty/meaningless
        """
        if not items:
            return True
        
        # Check if all items are empty
        meaningful_items = [item for item in items if not self._is_empty_response(item)]
        
        if not meaningful_items:
            logger.debug(f"All {len(items)} response items are empty for {title}")
            return True
        
        # Check for single item responses that are just location/metadata
        if len(items) == 1:
            item = items[0]
            # Count non-empty fields excluding metadata
            content_fields = ['title', 'description', 'purpose', 'tags']
            non_empty_fields = []
            
            for field in content_fields:
                value = item.get(field, "")
                if isinstance(value, str) and value.strip():
                    non_empty_fields.append(field)
                elif isinstance(value, list) and len(value) > 0:
                    non_empty_fields.append(field)
            
            # If only location or no meaningful content fields
            if len(non_empty_fields) == 0:
                logger.debug(f"Single item response has no meaningful content fields for {title}")
                return True
                
        return False
    
    def _is_item_worth_saving(self, item: Dict) -> bool:
        """
        Check if an item has enough meaningful content to be worth saving to CSV.
        
        Args:
            item: Dictionary representing an item to potentially save
            
        Returns:
            True if the item is worth saving, False otherwise
        """
        # Check if it's an empty response
        if self._is_empty_response(item):
            return False
        
        # Check for auto-generated fallback content that indicates failed analysis
        title = item.get("title", "").strip()
        description = item.get("description", "").strip()
        
        # Don't save items with only auto-generated descriptions
        auto_generated_patterns = [
            "Auto-generated reference from content analysis",
            "Auto-generated",
            "NO DESCRIPTION",
            "NO TITLE"
        ]
        
        for pattern in auto_generated_patterns:
            if pattern.lower() in description.lower() or pattern.lower() in title.lower():
                logger.debug(f"Filtering out item with auto-generated content: {title}")
                return False
        
        # Check if we have at least one meaningful field
        has_meaningful_content = (
            (title and len(title) > 3) or
            (description and len(description) > 20) or
            (item.get("purpose", "").strip() and len(item.get("purpose", "").strip()) > 10) or
            (isinstance(item.get("tags", []), list) and len(item.get("tags", [])) > 0)
        )
        
        return has_meaningful_content
    
    def save_to_universal_csv(self, items: List[Dict], content_type: str = None) -> None:
        """
        Save analyzed items to appropriate CSV files based on content type.
        
        This method provides the missing link between LLM analysis and data persistence,
        delegating to the appropriate DataSaver methods based on content type.
        
        Args:
            items: List of analyzed items to save
            content_type: Optional content type override (defaults to items' content_type)
        """
        if not items:
            logger.info("No items to save to CSV")
            return
        
        # Filter out empty items before saving
        meaningful_items = [item for item in items if self._is_item_worth_saving(item)]
        
        if not meaningful_items:
            logger.warning(f"All {len(items)} items were filtered out as empty/meaningless - not saving to CSV")
            return
        
        if len(meaningful_items) < len(items):
            logger.info(f"Filtered out {len(items) - len(meaningful_items)} empty items, saving {len(meaningful_items)} meaningful items")
            
        print(f"\n💾 SAVING TO CSV:")
        print("=" * 60)
        print(f"Number of items to save: {len(meaningful_items)}")
        print(f"Content type: {content_type}")
        
        for i, item in enumerate(meaningful_items):
            print(f"\nItem {i+1} being saved:")
            print(f"  - title: {item.get('title', 'NO TITLE')}")
            print(f"  - description: {item.get('description', 'NO DESCRIPTION')[:100]}...")
            print(f"  - location: '{item.get('location', 'NO LOCATION')}'")
            print(f"  - tags: {item.get('tags', 'NO TAGS')}")
            print(f"  - purpose: {item.get('purpose', 'NO PURPOSE')[:50]}...")
            print(f"  - content_type: {item.get('content_type', 'NO CONTENT_TYPE')}")
        print("=" * 60)
            
        try:
            # Group items by content type if not explicitly provided
            if content_type:
                # All items are the same type
                self._save_items_by_type(meaningful_items, content_type)
            else:
                # Group by content_type field in each item
                from collections import defaultdict
                grouped_items = defaultdict(list)
                
                for item in meaningful_items:
                    item_type = item.get('content_type', 'resource')
                    grouped_items[item_type].append(item)
                
                # Save each group
                for item_type, type_items in grouped_items.items():
                    self._save_items_by_type(type_items, item_type)
                    
        except Exception as e:
            logger.error(f"Error saving items to universal CSV: {e}", exc_info=True)
    
    def save_items_by_type(self, content_type: str, items: List[Dict]) -> None:
        """
        Public method to save items of a specific content type using universal schema.
        
        Args:
            content_type: Content type determining which save method to use
            items: List of items to save
        """
        self._save_items_by_type(items, content_type)
    
    def _save_items_by_type(self, items: List[Dict], content_type: str) -> None:
        """
        Save items of a specific content type using the universal schema.
        
        Args:
            items: List of items to save
            content_type: Content type determining which save method to use
        """
        if not items:
            return
            
        logger.info(f"Saving {len(items)} items of type '{content_type}' to CSV using universal schema")
        
        try:
            # Use universal save method for all content types
            # This ensures consistent schema and preserves all fields including URL and resource_id
            self.data_saver.save_universal_content(items, content_type)
            
        except Exception as e:
            logger.error(f"Error saving {content_type} items: {e}", exc_info=True)

    def _parse_json_response(self, response_text: str) -> Any:
        """
        Parse JSON from LLM response with enhanced cleanup and validation.
        Handles cases where LLM includes extra text after valid JSON.
        
        Args:
            response_text: Raw response text from the LLM
            
        Returns:
            Parsed JSON object/array
        """
        # Check for empty responses
        if not response_text or response_text.strip() == "":
            logger.error("Received empty response from LLM")
            return None
            
        logger.debug(f"Parsing JSON from: {response_text[:100]}...")
        
        # First, handle markdown code blocks
        json_str = response_text.strip()
        if "```json" in json_str:
            json_str = json_str.split("```json")[1].split("```")[0].strip()
        elif "```" in json_str:
            json_str = json_str.split("```")[1].split("```")[0].strip()
        
        try:
            # Direct JSON parsing
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Advanced JSON extraction
            try:
                # Check for multiple JSON objects (invalid JSON) - reject these
                if json_str.count('{') > json_str.count('[') and '}{' in json_str:
                    logger.debug("Detected multiple concatenated JSON objects - rejecting")
                    return None
                
                # Find the first occurrence of '[' or '{'
                start = None
                for i, char in enumerate(json_str):
                    if (char in '[{'):
                        start = i
                        break
                
                if start is not None:
                    # Find the matching closing bracket/brace
                    opening_char = json_str[start]
                    closing_char = ']' if opening_char == '[' else '}'
                    
                    # Track nesting level
                    level = 0
                    end = None
                    in_string = False
                    escape_next = False
                    
                    for i in range(start, len(json_str)):
                        char = json_str[i]
                        
                        if escape_next:
                            escape_next = False
                            continue
                            
                        if char == '\\':
                            escape_next = True
                            continue
                            
                        if char == '"' and not escape_next:
                            in_string = not in_string
                            continue
                            
                        if not in_string:
                            if char == opening_char:
                                level += 1
                            elif char == closing_char:
                                level -= 1
                                if level == 0:
                                    end = i + 1
                                    break
                    
                    if end is not None:
                        # Extract the JSON portion
                        json_text = json_str[start:end]
                        logger.debug(f"Extracted JSON: {json_text[:100]}...")
                        
                        try:
                            return json.loads(json_text)
                        except json.JSONDecodeError as e:
                            # Try cleaning up common issues
                            logger.debug(f"Error with extracted JSON: {e}")
                            cleaned_json = self._clean_json_text(json_text)
                            try:
                                return json.loads(cleaned_json)
                            except json.JSONDecodeError:
                                logger.debug("Cleaned JSON still invalid")
                                return None
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                
            # If all parsing attempts fail, return None
            logger.error("All JSON parsing attempts failed")
            logger.debug(f"Failed to parse JSON: {response_text[:200]}...")
            return None

    def _clean_json_text(self, json_text: str) -> str:
        """
        Clean up common JSON formatting issues.
        
        Args:
            json_text: Raw JSON text with potential formatting issues
            
        Returns:
            Cleaned JSON text
        """
        try:
            # Log the original text for debugging
            logger.debug(f"Cleaning JSON text: {json_text[:50]}...")
            
            # Remove trailing commas before closing brackets/braces (more aggressive)
            cleaned = re.sub(r',\s*([}\]])', r'\1', json_text)
            
            # Handle trailing commas at the end of objects/arrays more specifically
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            
            # Remove any trailing commas after the last property in objects
            cleaned = re.sub(r',(\s*})', r'\1', cleaned)
            
            # Remove any trailing commas after the last element in arrays  
            cleaned = re.sub(r',(\s*])', r'\1', cleaned)
            
            # Ensure property names are quoted (but be careful not to quote already quoted names)
            cleaned = re.sub(r'([^"\w])(\w+)(\s*:)', r'\1"\2"\3', cleaned)
            
            # Replace single quotes with double quotes for strings
            cleaned = re.sub(r"'([^']*?)'", r'"\1"', cleaned)
            
            # Remove any control or non-printing characters
            cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)
            
            # Remove any extra whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned)
            cleaned = cleaned.strip()
            
            return cleaned
            
        except Exception as e:
            # If any cleanup fails, log it but return the original
            logger.error(f"Error during JSON cleanup: {e}")
            return json_text
    
    def _split_content_into_chunks(self, content: str) -> List[str]:
        """
        Split content into chunks using adaptive chunking settings based on system capabilities.
        
        Args:
            content: Content to split into chunks
            
        Returns:
            List of content chunks
        """
        try:
            # Use default chunking configuration
            chunk_size = 1500
            chunk_overlap = 200
            
            # Initialize text splitter with default settings
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    "\n\n",          # First try to split on double newlines
                    "\n",            # Then single newlines
                    ". ",            # Then sentences
                    ", ",            # Then clauses
                    " ",             # Then words
                    ""               # Finally characters
                ],
                length_function=len,
            )
            
            # Split content into chunks
            content_chunks = text_splitter.split_text(content)
            
            logger.info(f"Adaptive chunking: {len(content_chunks)} chunks using {chunk_size} size/{chunk_overlap} overlap (system tier: {self.system_info['performance_tier']})")
            
            return content_chunks
            
        except Exception as e:
            logger.error(f"Error splitting content into chunks: {e}")
            # Return original content as single chunk if splitting fails
            return [content]