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
        self.base_url = base_url
        
        # Store data folder for DataSaver initialization
        if data_folder is None:
            from utils.config import get_data_path
            data_folder = get_data_path()
        self.data_folder = data_folder
        
        # Initialize DataSaver for CSV storage operations
        from scripts.analysis.data_saver import DataSaver
        self.data_saver = DataSaver(self.data_folder)
        
        # Load analysis-specific model settings
        self.settings = self._load_analysis_settings()
        
        # Set model based on settings or parameter
        if model:
            # Override with provided model
            if self.settings["provider"] == "ollama":
                self.model = model
            else:
                self.openrouter_model = model
        else:
            # Use settings for model selection
            if self.settings["provider"] == "ollama":
                self.model = self.settings["model_name"]
            else:
                self.openrouter_model = self.settings["openrouter_model"]
        
        # Get adaptive model configuration from the utility
        from scripts.llm.processor_config_utils import get_adaptive_model_config
        self.model_config = get_adaptive_model_config()
        
        # Load task configuration with prompts
        self.task_config = self._load_task_config()
        
        # Get OpenRouter API key if OpenRouter is the provider
        self.openrouter_api_key = None
        if self.settings["provider"] == "openrouter":
            self._load_openrouter_api_key()
        
        # Log initialization
        logger.info(f"Initialized UniversalAnalysisLLM with provider: {self.settings['provider']}")
        if self.settings["provider"] == "ollama":
            logger.info(f"Using Ollama model: {self.model}")
        else:
            logger.info(f"Using OpenRouter model: {self.openrouter_model}")
        
        logger.info(f"Using {self.model_config['threads']} threads and {self.model_config['batch_size']} batch size")
    
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
            prompt = f"""<s>[INST] You are an expert JSON formatter with perfect accuracy.
When asked to generate JSON, you ONLY output valid, parseable JSON without any explanation or markdown formatting.

IMPORTANT LIMITS:
- Generate MAXIMUM 5 items per analysis to ensure quality and relevance
- Focus on the most important and distinct content items
- Avoid repetition or similar items

UNIVERSAL TABLE STRUCTURE REQUIREMENTS:
All content types must follow EXACTLY the same simplified schema with ONLY these fields:
- title: Clear and concise title
- description: Detailed explanation of the content
- purpose: Why this content exists or what problem it solves
- tags: Array of relevant keywords or categories (lowercase)
- location: Physical or geographical location related to this content (empty string if not applicable)

DO NOT include any other fields like steps, goals, term, etc. regardless of content type.
The following fields will be added automatically (don't include them): 
id, content_type, origin_url, creator_id, group_id, created_at, status, visibility

{prompt.strip()}

Output ONLY the JSON with no additional text. [/INST]</s>"""
        else:
            # For non-JSON responses, still use the Mistral instruction format
            prompt = f"""<s>[INST] {prompt.strip()} [/INST]</s>"""
        
        # Make API call to generate response
        try:
            logger.info(f"Generating structured response for content analysis using Ollama: {self.model}")
            
            start_time = time.time()
            
            api_params = {
                "model": self.model,
                "prompt": prompt.strip(),
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens,
                # Add all model configuration parameters
                "threads": self.model_config["threads"],
                "num_ctx": self.model_config["num_ctx"],
                "num_gpu": self.model_config["num_gpu"],
                "batch_size": self.model_config["batch_size"],
                "num_keep": self.model_config["num_keep"],
                "repeat_penalty": self.model_config["repeat_penalty"],
                "parallel": self.model_config["parallel"]
            }
            
            # Use standard requests with an extended timeout for complex analysis
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=api_params,
                timeout=600.0  # 10-minute timeout for complex document analysis
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
                    # Extract JSON from response
                    json_str = result["response"].strip()
                    # Handle case where response includes markdown code blocks
                    if "```json" in json_str:
                        json_str = json_str.split("```json")[1].split("```")[0].strip()
                    elif "```" in json_str:
                        json_str = json_str.split("```")[1].split("```")[0].strip()
                    
                    structured_response = json.loads(json_str)
                    
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
                            
                    logger.info(f"Successfully parsed JSON response with {len(structured_response) if isinstance(structured_response, list) else 1} items")
                    return structured_response
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Invalid JSON string: {json_str}")
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
            system_prompt = """You are an expert JSON formatter with perfect accuracy. Output ONLY valid, parseable JSON without any explanation or markdown formatting.
    
UNIVERSAL TABLE STRUCTURE REQUIREMENTS:
All content types must follow EXACTLY the same simplified schema with ONLY these fields:
- title: Clear and concise title
- description: Detailed explanation of the content
- purpose: Why this content exists or what problem it solves
- tags: Array of relevant keywords or categories (lowercase)
- location: Physical or geographical location related to this content (empty string if not applicable)

DO NOT include any other fields like steps, goals, term, etc. regardless of content type.
The following fields will be added automatically (don't include them): 
id, content_type, origin_url, creator_id, group_id, created_at, status, visibility"""
        else:
            system_prompt = self.settings.get("system_prompt", "You are an AI assistant helping with content analysis and knowledge extraction.")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt.strip()}
        ]
        try:
            logger.info(f"Generating structured response using OpenRouter: {self.openrouter_model}")
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
                            error_message += f", Details: {error_details['error'].get('message', '')}"
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
            # Validate content type
            if content_type not in self.task_config:
                logger.warning(f"Unknown content type: {content_type}, falling back to resource")
                content_type = "resource"
            
            # Get prompt template for this content type
            prompt_template = self.task_config[content_type].get("prompt_template", "")
            if not prompt_template:
                logger.error(f"No prompt template found for content type: {content_type}")
                return []
            
            # Use a large enough content size for analysis
            max_content_length = 6000
            truncated_content = content[:max_content_length] if content else ""
            
            # Format prompt with content and content_type explicitly included
            prompt = prompt_template.format(
                title=title,
                url=url,
                content=truncated_content.replace('{', '{{').replace('}', '}}'),
                content_type=content_type  # Pass content_type explicitly
            )
            
            # Add specific field requirements for this content type to prompt
            field_requirements = self._get_field_requirements(content_type)
            prompt = f"Content Type: {content_type}\nRequired fields: {field_requirements}\n\n{prompt}"
            
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
            
            # Validate and enrich items
            validated_items = self._validate_items(items, title, url, content_type)
            
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
            logger.info(f"Content is large ({len(content)} chars), proceeding with chunked analysis")
            
            # Initialize text splitter with optimized settings for performance
            # Using larger chunks to reduce excessive tiny chunks that create processing overhead
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,      # Optimized chunk size for better performance (7.5x larger than before)
                chunk_overlap=200,    # Optimized overlap for better context preservation (10x more overlap)
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
            logger.info(f"Split content into {len(content_chunks)} chunks for analysis")
            
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
                
            # Ensure required fields exist
            validated_item = {
                "title": item.get("title", title),
                "description": item.get("description", ""),
                "tags": item.get("tags", []) if isinstance(item.get("tags"), list) else [],
                "purpose": item.get("purpose", ""),
                "location": item.get("location", ""),
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
            
        try:
            # Group items by content type if not explicitly provided
            if content_type:
                # All items are the same type
                self._save_items_by_type(items, content_type)
            else:
                # Group by content_type field in each item
                from collections import defaultdict
                grouped_items = defaultdict(list)
                
                for item in items:
                    item_type = item.get('content_type', 'resource')
                    grouped_items[item_type].append(item)
                
                # Save each group
                for item_type, type_items in grouped_items.items():
                    self._save_items_by_type(type_items, item_type)
                    
        except Exception as e:
            logger.error(f"Error saving items to universal CSV: {e}", exc_info=True)
    
    def _save_items_by_type(self, items: List[Dict], content_type: str) -> None:
        """
        Save items of a specific content type using the appropriate DataSaver method.
        
        Args:
            items: List of items to save
            content_type: Content type determining which save method to use
        """
        if not items:
            return
            
        logger.info(f"Saving {len(items)} items of type '{content_type}' to CSV")
        
        try:
            # Map content types to appropriate save methods
            if content_type == "definition":
                # Transform items to match definition schema
                definitions = []
                for item in items:
                    definition = {
                        'term': item.get('title', ''),
                        'definition': item.get('description', ''),
                        'tags': item.get('tags', []),
                        'source': item.get('origin_url', ''),
                        'created_at': item.get('created_at'),
                    }
                    definitions.append(definition)
                self.data_saver.save_definitions(definitions)
                
            elif content_type == "project":
                # Transform items to match project schema
                projects = []
                for item in items:
                    project = {
                        'title': item.get('title', ''),
                        'description': item.get('description', ''),
                        'goal': item.get('purpose', ''),  # Map purpose to goal
                        'tags': item.get('tags', []),
                        'origin_url': item.get('origin_url', ''),
                        'created_at': item.get('created_at'),
                    }
                    projects.append(project)
                self.data_saver.save_projects(projects)
                
            elif content_type == "method":
                # Transform items to match method schema
                methods = []
                for item in items:
                    method = {
                        'title': item.get('title', ''),
                        'description': item.get('description', ''),
                        'steps': item.get('steps', []),  # Assume steps are extracted elsewhere
                        'tags': item.get('tags', []),
                        'source': item.get('origin_url', ''),
                        'created_at': item.get('created_at'),
                    }
                    methods.append(method)
                self.data_saver.save_methods(methods)
                
            elif content_type in ["resource", "material"]:
                # Save as resources (default case)
                resources = []
                for item in items:
                    resource = {
                        'title': item.get('title', ''),
                        'description': item.get('description', ''),
                        'tags': item.get('tags', []),
                        'origin_url': item.get('origin_url', ''),
                        'content_type': content_type,
                        'created_at': item.get('created_at'),
                        'analysis_completed': True,  # These items have been analyzed
                    }
                    resources.append(resource)
                self.data_saver.save_resources(resources)
                
            else:
                # Unknown content type, save as resources with type annotation
                logger.warning(f"Unknown content type '{content_type}', saving as resources")
                resources = []
                for item in items:
                    resource = {
                        'title': item.get('title', ''),
                        'description': item.get('description', ''),
                        'tags': item.get('tags', []),
                        'origin_url': item.get('origin_url', ''),
                        'content_type': content_type,
                        'created_at': item.get('created_at'),
                        'analysis_completed': True,
                    }
                    resources.append(resource)
                self.data_saver.save_resources(resources)
                
            logger.info(f"Successfully saved {len(items)} {content_type} items to CSV")
            
        except Exception as e:
            logger.error(f"Error saving {content_type} items: {e}", exc_info=True)