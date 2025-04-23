#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime

from scripts.llm.open_router_client import OpenRouterClient
from scripts.llm.ollama_client import OllamaClient
from scripts.services.settings_manager import SettingsManager
from scripts.llm.processor_config_utils import get_adaptive_model_config, get_best_available_model
from scripts.analysis.extraction_utils import ExtractionUtils

logger = logging.getLogger(__name__)

class LLMAnalyzer:
    """
    Handles interactions with LLM to analyze website content
    and extract useful information like descriptions, tags, etc.
    
    This class delegates extraction functions to ExtractionUtils to avoid redundancy.
    """
    
    def __init__(self):
        # For resource processing and data analysis, we always use the local Ollama
        # regardless of what global LLM provider is set, to ensure synchronous operation
        self.provider = "ollama"
        self.ollama_client = OllamaClient()
        
        # Store actual settings for reference
        self.settings = SettingsManager().load_settings()
        self.global_provider = str(self.settings.provider) if hasattr(self.settings, 'provider') else "ollama"
        
        # Get the best available model and adaptive config
        adaptive_config = get_adaptive_model_config()
        best_model = get_best_available_model()
        
        # Initialize the ResourceLLM with the best model and adaptive config
        from scripts.analysis.resource_llm import ResourceLLM
        self.resource_llm = ResourceLLM(model=best_model)
        self.resource_llm.model_config = adaptive_config
        
        # Initialize the extraction utils with our ResourceLLM instance
        self.extraction_utils = ExtractionUtils(self.resource_llm)
        
        logger.info(f"LLM Analyzer initialized - Using model {best_model} with adaptive processor configuration")
        logger.info(f"Using {adaptive_config['threads']} threads and {adaptive_config['batch_size']} batch size based on hardware")
        logger.info(f"Global provider setting is: {self.global_provider}")
    
    def generate_description(self, title: str, url: str, content: str) -> str:
        """Generate a concise description for a resource"""
        return self.resource_llm.generate_description(title, url, content)
    
    def generate_tags(self, title: str, url: str, content: str, description: str, existing_tags: List = None) -> List[str]:
        """Generate 3-5 relevant tags for a resource"""
        
        if existing_tags is None:
            existing_tags = []
            
        if existing_tags:
            logger.info(f"Using existing tags for {title}: {existing_tags}")
            return existing_tags
        
        try:
            logger.info(f"Generating tags for {title}")
            tags = self.resource_llm.generate_tags(title, url, content, description, existing_tags)
            logger.info(f"Generated tags for {title}: {tags}")
            return tags
        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            return []
            
    def generate_purpose(self, title: str, url: str, content: str, description: str) -> str:
        """
        Generate a concise statement of purpose for a resource.
        
        Args:
            title: Resource title
            url: Resource URL
            content: Resource content text
            description: Resource description
            
        Returns:
            Purpose statement or empty string if generation fails
        """
        try:
            logger.info(f"Generating purpose for {title}")
            # If ResourceLLM has generate_purpose method, use it
            if hasattr(self.resource_llm, 'generate_purpose'):
                purpose = self.resource_llm.generate_purpose(title, url, content, description)
                logger.info(f"Generated purpose for {title}: {purpose[:50]}...")
                return purpose
            else:
                # Otherwise create a default purpose based on title and description
                logger.warning(f"ResourceLLM doesn't have generate_purpose method, using default")
                if description and len(description) > 30:
                    # Extract first sentence from description as purpose
                    first_sentence = description.split('.')[0].strip()
                    if len(first_sentence) > 20:
                        return first_sentence
                
                # Fall back to generic purpose if description not usable
                return f"Resource for learning about {title}"
        except Exception as e:
            logger.error(f"Error generating purpose: {e}")
            return f"Resource related to {title}"

    # Delegate all extraction methods to the ExtractionUtils
    def extract_definitions(self, title: str, url: str, content: str) -> List[Dict]:
        """Extract definitions/terms from website content - delegates to extraction utils"""
        return self.extraction_utils.extract_definitions(title, url, content)
    
    def identify_projects(self, title: str, url: str, content: str) -> List[Dict]:
        """Identify projects/initiatives from website content - delegates to extraction utils"""
        return self.extraction_utils.identify_projects(title, url, content)
    
    def extract_methods(self, title: str, url: str, content: str) -> List[Dict]:
        """Extract methods and procedures from website content - delegates to extraction utils"""
        return self.extraction_utils.extract_methods(title, url, content)

    def parse_json_response(self, response: str) -> Any:
        """Utility method to parse JSON from LLM responses in various formats"""
        try:
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response to parse: {response}")
            
            # Remove any natural language wrappers that Ollama models often include
            for prefix in ["here is", "here's", "i've identified", "here are", "based on"]:
                if prefix in response.lower():
                    # Find where the text after the prefix starts
                    start_idx = response.lower().find(prefix) + len(prefix)
                    response = response[start_idx:].strip()
            
            # Handle common Ollama output formats: responses often have explanatory text
            # Try to find where the actual JSON data begins
            bracket_pos = response.find("[")
            brace_pos = response.find("{")
            
            if bracket_pos >= 0 or brace_pos >= 0:
                # Find the earlier of the two if both exist
                json_start = min(pos for pos in [bracket_pos, brace_pos] if pos >= 0)
                # Find the matching closing bracket/brace
                if response[json_start] == "[":
                    stack = 0
                    for i in range(json_start, len(response)):
                        if response[i] == "[":
                            stack += 1
                        elif response[i] == "]":
                            stack -= 1
                            if stack == 0:
                                json_end = i + 1
                                break
                else:  # Must be "{"
                    stack = 0
                    for i in range(json_start, len(response)):
                        if response[i] == "{":
                            stack += 1
                        elif response[i] == "}":
                            stack -= 1
                            if stack == 0:
                                json_end = i + 1
                                break
                
                # Extract just the JSON part if we found a complete JSON structure
                if 'json_end' in locals():
                    potential_json = response[json_start:json_end].strip()
                    logger.debug(f"Extracted potential JSON: {potential_json}")
                    try:
                        return json.loads(potential_json)
                    except:
                        # If that failed, we'll continue with other methods
                        pass
            
            # Try to find JSON within code blocks
            if "```" in response:
                # Handle various markdown code block formats
                code_blocks = re.findall(r'```(?:json)?(.*?)```', response, re.DOTALL)
                for block in code_blocks:
                    block = block.strip()
                    if block and (block.startswith('[') or block.startswith('{')):
                        try:
                            return json.loads(block)
                        except:
                            # Try cleaning
                            clean_block = re.sub(r',\s*([}\]])', r'\1', block)
                            try:
                                return json.loads(clean_block)
                            except:
                                continue
            
            # Try strict regex extraction based on balanced brackets/braces
            array_match = re.search(r'\[\s*(?:\{.*?\}\s*(?:,\s*\{.*?\}\s*)*)\]', response, re.DOTALL)
            if array_match:
                try:
                    return json.loads(array_match.group(0))
                except Exception as regex_err:
                    logger.debug(f"Regex extraction failed: {regex_err}")
            
            # Special case for Ollama: sometimes it returns malformed JSON with extra characters
            # Let's manually parse a response with the format "[{...}, {...}]"
            if '[' in response and ']' in response:
                # Find the outermost square brackets
                start_idx = response.find('[')
                end_idx = response.rfind(']')
                if start_idx >= 0 and end_idx > start_idx:
                    json_text = response[start_idx:end_idx+1]
                    # Clean up common JSON errors
                    json_text = re.sub(r',(\s*[\]}])', r'\1', json_text)  # Remove trailing commas
                    json_text = re.sub(r'(\w+):', r'"\1":', json_text)  # Quote unquoted keys
                    try:
                        return json.loads(json_text)
                    except:
                        # If that failed, we'll try one more approach
                        pass
            
            # Last resort: For Ollama's JSON-like responses that aren't valid JSON,
            # try to manually parse array of objects structure like this:
            # [{ "term": "foo", "definition": "bar" }, { "term": "baz", "definition": "qux" }]
            if '[{' in response and '}]' in response:
                start_idx = response.find('[{')
                end_idx = response.find('}]') + 2
                json_like = response[start_idx:end_idx]
                
                # Extract individual objects
                objects = []
                in_object = False
                object_start = 0
                bracket_count = 0
                
                for i, char in enumerate(json_like):
                    if char == '{':
                        if not in_object:
                            in_object = True
                            object_start = i
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0 and in_object:
                            obj_text = json_like[object_start:i+1]
                            # Clean and fix common issues
                            obj_text = re.sub(r'(\w+):', r'"\1":', obj_text)  # Quote unquoted keys
                            obj_text = re.sub(r':\s*"([^"]*)"', r': "\1"', obj_text)  # Ensure string values are properly quoted
                            try:
                                obj = json.loads(obj_text)
                                objects.append(obj)
                            except:
                                logger.debug(f"Failed to parse object: {obj_text}")
                            in_object = False
                
                if objects:
                    logger.info(f"Manually parsed {len(objects)} objects from malformed JSON")
                    return objects
            
            # If all else fails and response looks like it could be JSON, try direct parsing
            clean_response = response.strip()
            if (clean_response.startswith('[') and clean_response.endswith(']')) or \
               (clean_response.startswith('{') and clean_response.endswith('}')):
                try:
                    return json.loads(clean_response)
                except:
                    pass
            
            # If nothing worked, log and return empty array
            logger.warning("Could not extract valid JSON from LLM response")
            # If we expect an array but couldn't parse JSON, return empty array
            if "extract" in response.lower() or "identify" in response.lower():
                logger.info("Returning empty array as no valid JSON could be extracted")
                return []
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing JSON response: {e}")
            logger.debug(f"Response was: {response}")
            return []