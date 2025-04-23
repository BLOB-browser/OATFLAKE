#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Specialized LLM interface for extracting methods and procedures from content.
This implementation focuses on identifying step-by-step methods and processes.
"""

import logging
import json
import re
import requests
import time
from typing import Dict, List, Any, Optional, Union
import uuid
from datetime import datetime

# Add import for interruptible requests
from scripts.analysis.interruptible_llm import (
    interruptible_post, is_interrupt_requested, 
    setup_interrupt_handling, restore_interrupt_handling,
    clear_interrupt
)

# Import Method schema for validation
from scripts.models.schemas import Method

logger = logging.getLogger(__name__)

class MethodLLM:
    """
    A specialized LLM interface for extracting methodologies, procedures,
    and step-by-step processes from text content.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral:7b-instruct-v0.2-q4_0"):
        self.base_url = base_url
        self.model = model
        
        # Get adaptive model configuration from the utility - updated import path
        from scripts.llm.processor_config_utils import get_adaptive_model_config
        self.model_config = get_adaptive_model_config()
        
        logger.info(f"Initialized MethodLLM with model {model} ({self.model_config['threads']} threads, {self.model_config['num_ctx']} context, {self.model_config['batch_size']} batch size)")
    
    def generate_structured_response(self, prompt: str, format_type: str = "json", 
                                    temperature: float = 0.1, max_tokens: int = 1024) -> Any:
        """
        Generate a structured response from the LLM, with parameters
        optimized for extracting methods and procedures.
        
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
            logger.warning("Skipping method extraction due to interrupt request")
            return None
        
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
            # Add Mistral-specific system prompt for better JSON formatting
            prompt = f"""<s>[INST] You are an expert JSON formatter with perfect accuracy.
When asked to generate JSON, you ONLY output valid, parseable JSON without any explanation or markdown formatting.
            
{prompt.strip()}

Output ONLY the JSON with no additional text. [/INST]</s>"""
        else:
            # For non-JSON responses, still use the Mistral instruction format
            prompt = f"""<s>[INST] {prompt.strip()} [/INST]</s>"""
        
        # Make API call to generate response
        try:
            logger.info(f"Generating structured response for method extraction")
            
            # Set up interrupt handling
            setup_interrupt_handling()
            
            try:
                start_time = time.time()
                
                # Replace standard request with interruptible version - fix how we pass parameters
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
                
                response = interruptible_post(
                    f"{self.base_url}/api/generate",
                    json=api_params,
                    timeout=3600.0  # Extended to 1 hour for Raspberry Pi
                )
                
                elapsed = time.time() - start_time
                logger.info(f"LLM response time: {elapsed:.2f} seconds")
                
                # Check if interrupted
                if is_interrupt_requested():
                    logger.warning("Method extraction was interrupted")
                    return None
                
                if response.status_code != 200:
                    logger.error(f"Error from LLM API: {response.status_code}")
                    return None
                
                result = response.json()
                if not result or 'response' not in result:
                    logger.error("Invalid response format from API")
                    return None
                
                # Extract the raw response text
                raw_response = result['response']
                
                # Process based on expected format
                if format_type == "json":
                    try:
                        return self._parse_json_response(raw_response)
                    except Exception as e:
                        logger.error(f"Error parsing JSON response: {e}")
                        return None
                else:
                    return raw_response
                
            except KeyboardInterrupt:
                logger.warning("Method extraction interrupted by user")
                return None
            
        except Exception as e:
            logger.error(f"Error generating structured response: {e}")
            return None
        finally:
            # Restore interrupt handling
            restore_interrupt_handling()
    
    def extract_methods(self, title: str, url: str, content: str) -> List[Dict]:
        """
        Extract methodologies, processes, and step-by-step procedures from content.
        
        Args:
            title: Resource title
            url: Resource URL
            content: Resource content text
            
        Returns:
            List of method dictionaries containing steps, descriptions, etc.
        """
        try:
            # Use a large enough content size to find methods
            max_content_length = 4000
            truncated_content = content[:max_content_length] if content else ""
            
            prompt = f"""You are analyzing a web resource to extract explicit METHODS, PROCESSES, 
or STEP-BY-STEP PROCEDURES. Focus on finding clearly defined methodologies or workflows.

Title: {title}
URL: {url}

Content excerpt:
{truncated_content.replace('{', '{{').replace('}', '}}')}

Your task: Identify up to 3 distinct METHODS or PROCESSES that are described in this content.

ONLY INCLUDE ACTUAL METHODS/PROCESSES:
- Research methods or methodologies that can be followed as steps
- Design processes that have a clear sequence of actions
- Technical procedures with defined steps
- Workflows that others could implement or follow
- Frameworks with practical implementation steps

DO NOT INCLUDE:
- Single techniques that don't involve multiple steps
- Generic concepts without actionable steps
- Vague approaches without clear procedures
- Lists that aren't methodological in nature

For each method, extract:
- A descriptive title for the method
- A concise description explaining what the method accomplishes
- A list of clear, sequential steps that are part of the method
- Relevant tags that categorize the method

If no specific methods with clear steps are described, return an empty array.

EXTREMELY IMPORTANT FORMATTING INSTRUCTIONS:
1. Your response must be a valid JSON array of objects
2. Each object must have these exact fields:
   - "title": The name of the method (string)
   - "description": A brief description (string, 1-2 sentences)
   - "steps": An array of strings representing the steps
   - "tags": Array of strings with 2-5 relevant tags
3. Return ONLY the JSON array with no explanation text before or after
4. If no methods are found, return an empty array: []
5. The JSON MUST be valid and parseable

REQUIRED FORMAT:
[
  {{
    "title": "Method Name",
    "description": "Brief description of what this method accomplishes",
    "steps": ["Step 1: Do this", "Step 2: Then do that", "Step 3: Finally do this"],
    "tags": ["tag1", "tag2", "tag3"]
  }}
]
"""
            
            # Use JSON format for methods
            response = self.generate_structured_response(prompt, format_type="json", temperature=0.1)
            
            if response and isinstance(response, list):
                # Validate the structure of each method
                validated_methods = []
                for item in response:
                    try:
                        if isinstance(item, dict) and 'title' in item and 'description' in item and 'steps' in item:
                            # Convert all fields to appropriate types
                            title_str = str(item['title']).strip()
                            desc_str = str(item['description']).strip()
                            
                            # Ensure steps is a list of strings
                            steps_list = []
                            if 'steps' in item:
                                if isinstance(item['steps'], list):
                                    steps_list = [str(step).strip() for step in item['steps'] if step]
                                elif isinstance(item['steps'], str):
                                    # Try to parse string as JSON array if it looks like one
                                    if item['steps'].startswith('[') and item['steps'].endswith(']'):
                                        try:
                                            parsed_steps = json.loads(item['steps'])
                                            if isinstance(parsed_steps, list):
                                                steps_list = [str(step).strip() for step in parsed_steps if step]
                                        except:
                                            # If parsing fails, add it as a single step
                                            steps_list = [item['steps'].strip('[]').strip('"\'').strip()]
                                    else:
                                        # Add as a single step
                                        steps_list = [item['steps'].strip()]
                            
                            # Ensure tags is a list of strings
                            tags_list = []
                            if 'tags' in item:
                                if isinstance(item['tags'], list):
                                    tags_list = [str(tag).strip().lower() for tag in item['tags'] if tag]
                                elif isinstance(item['tags'], str):
                                    try:
                                        if item['tags'].startswith('[') and item['tags'].endswith(']'):
                                            parsed_tags = json.loads(item['tags'])
                                            if isinstance(parsed_tags, list):
                                                tags_list = [str(tag).strip().lower() for tag in parsed_tags if tag]
                                        else:
                                            tags_list = [item['tags'].strip().lower()]
                                    except:
                                        tags_list = [item['tags'].strip().lower()]
                            
                            # Only add if we have the required fields with content
                            if title_str and desc_str and steps_list:
                                try:
                                    # Create a Method object with all required fields using the schema
                                    method_model = Method(
                                        id=str(uuid.uuid4()),
                                        title=title_str,
                                        description=desc_str,
                                        content_type="method",
                                        tags=tags_list,
                                        steps=steps_list,
                                        creator_id="system",
                                        group_id="default",
                                        created_at=datetime.now(),
                                        status="active",
                                        visibility="public",
                                        analysis_completed=True,
                                        purpose="To document a methodology or process"
                                    )
                                    
                                    # Convert to dictionary
                                    valid_method = method_model.dict()
                                    
                                    # Add the validated method
                                    validated_methods.append(valid_method)
                                    logger.debug(f"Added validated method: {title_str}")
                                except Exception as validation_err:
                                    logger.warning(f"Method validation error for {title_str}: {validation_err}")
                                    # Fallback to the original dictionary if schema validation fails
                                    validated_methods.append({
                                        'title': title_str,
                                        'description': desc_str,
                                        'steps': steps_list,
                                        'tags': tags_list
                                    })
                                    logger.debug(f"Added unvalidated method: {title_str}")
                    except Exception as item_err:
                        logger.warning(f"Error validating method item: {item_err}")
                        continue
                
                logger.info(f"Extracted {len(validated_methods)} methods from {title}")
                return validated_methods
            else:
                # Return empty list if extraction fails
                logger.warning(f"No methods extracted for {title}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting methods: {e}")
            return []  # Return empty list in case of any error
    
    def _parse_json_response(self, response_text: str) -> Any:
        """
        Parse JSON from LLM response with enhanced cleanup and validation.
        
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
        
        try:
            # Direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Advanced JSON extraction
            try:
                # Find the first occurrence of '[' or '{'
                start = None
                for i, char in enumerate(response_text):
                    if (char in '[{'):
                        start = i
                        break
                
                if start is not None:
                    # Find the matching closing bracket/brace
                    opening_char = response_text[start]
                    closing_char = ']' if opening_char == '[' else '}'
                    
                    # Track nesting level
                    level = 0
                    end = None
                    
                    for i in range(start, len(response_text)):
                        if response_text[i] == opening_char:
                            level += 1
                        elif response_text[i] == closing_char:
                            level -= 1
                            if level == 0:
                                end = i + 1
                                break
                    
                    if end is not None:
                        # Extract the JSON portion
                        json_text = response_text[start:end]
                        logger.debug(f"Extracted JSON: {json_text[:100]}...")
                        
                        try:
                            return json.loads(json_text)
                        except json.JSONDecodeError as e:
                            # Try cleaning up common issues
                            logger.debug(f"Error with extracted JSON: {e}")
                            cleaned_json = self._clean_json_text(json_text)
                            return json.loads(cleaned_json)
            except Exception as e:
                logger.error(f"Error extracting JSON: {e}")
                
            # If all parsing attempts fail, return None
            logger.error("All JSON parsing attempts failed")
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
            
            # Remove trailing commas before closing brackets
            cleaned = re.sub(r',\s*([}\]])', r'\1', json_text)
            
            # Ensure property names are quoted
            cleaned = re.sub(r'(\w+)(?=\s*:)', r'"\1"', cleaned)
            
            # Fix unquoted string values
            cleaned = re.sub(r':\s*([^"{}\[\],\d][^{}\[\],]*?)([,}])', r': "\1"\2', cleaned)
            
            # Fix missing commas between array elements
            cleaned = re.sub(r'([\"\'\w\d])\s*\[', r'\1, [', cleaned)
            cleaned = re.sub(r'\]\s*([\"\'\w\d])', r'], \1', cleaned)
            
            # Replace single quotes with double quotes
            cleaned = re.sub(r'\'([^\']*?)\'', r'"\1"', cleaned)
            
            # Remove any control or non-printing characters
            cleaned = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', cleaned)
            
            return cleaned
            
        except Exception as e:
            # If any cleanup fails, log it but return the original
            logger.error(f"Error during JSON cleanup: {e}")
            return json_text