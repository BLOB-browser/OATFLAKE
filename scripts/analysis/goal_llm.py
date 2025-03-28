#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dedicated LLM interface for goal extraction.
This implementation is focused on reliable extraction of learning goals and objectives
from educational content, using a JSON structure for consistency.
"""

import logging
import json
import re
import requests
import time
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class GoalLLM:
    """
    A dedicated LLM interface for extracting learning goals and objectives.
    This ensures consistent extraction and format, separate from
    the chat-focused OllamaClient implementation.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral:7b-instruct-v0.2-q4_0"):
        self.base_url = base_url
        self.model = model
        
        # Get adaptive model configuration from the utility
        from scripts.llm.processor_config_utils import get_adaptive_model_config
        self.model_config = get_adaptive_model_config()
        
        logger.info(f"Initialized GoalLLM with model {model} ({self.model_config['threads']} threads, {self.model_config['num_ctx']} context)")
    
    def generate_structured_response(self, prompt: str, format_type: str = "json", 
                                    temperature: float = 0.1, max_tokens: int = 1024) -> Any:
        """
        Generate a structured response from the LLM, with parameters
        optimized for extracting goals.
        
        Args:
            prompt: The prompt to send to the model
            format_type: Type of formatting to expect ("json", "list", "text")
            temperature: Temperature setting (lower for more deterministic output)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed structured response (JSON object, list, or text)
        """
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
            logger.info(f"Generating structured response for goal extraction")
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
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
                },
                timeout=600.0  # 10 minutes should be enough
            )
            
            elapsed = time.time() - start_time
            logger.info(f"LLM response time: {elapsed:.2f} seconds")
            
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
                
        except Exception as e:
            logger.error(f"Error generating structured response: {e}")
            return None
    
    async def chat_completion_async(self, messages: List[Dict[str, str]], max_tokens: int = 1024, 
                             temperature: float = 0.2, model: str = None) -> Dict[str, Any]:
        """
        Generate a chat completion asynchronously.
        This method emulates the interface expected by the goal_extractor.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Temperature setting (lower for more deterministic output)
            model: Optional model override
            
        Returns:
            Dict with 'message' containing the response
        """
        import httpx
        import asyncio
        
        model_name = model or self.model
        logger.info(f"Generating chat completion with model {model_name}")
        
        # Extract prompt from messages
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt = msg["content"]
                break
        
        if not prompt:
            logger.error("No user message found in messages")
            return {"message": {"content": "Error: No prompt provided"}}
        
        # Process using an appropriate instruction format for Mistral
        instruction_prompt = f"""<s>[INST] {prompt.strip()} [/INST]</s>"""
        
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "prompt": instruction_prompt,
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
                )
                
                if response.status_code != 200:
                    logger.error(f"Error from LLM API: {response.status_code}")
                    return {"message": {"content": f"Error: API returned {response.status_code}"}}
                
                result = response.json()
                if not result or 'response' not in result:
                    logger.error("Invalid response format from API")
                    return {"message": {"content": "Error: Invalid response format"}}
                
                # Return in a format compatible with what goal_extractor expects
                return {"message": {"content": result['response']}}
                
        except Exception as e:
            logger.error(f"Error in chat_completion_async: {e}")
            return {"message": {"content": f"Error: {str(e)}"}}
    
    def extract_goals(self, content: str, store_type: str = "content") -> List[Dict[str, Any]]:
        """
        Extract educational goals from content.
        
        Args:
            content: Text content to analyze
            store_type: Type of store the content came from
            
        Returns:
            List of goal dictionaries
        """
        try:
            # Truncate content to a reasonable size
            max_content_length = 4000
            truncated_content = content[:max_content_length] if content else ""
            
            prompt = f"""You are an educational goals analyst. Your task is to extract clear, actionable learning goals from the content below.

Content from {store_type}:
{truncated_content}

Instructions:
1. Identify 2-3 specific learning goals implied or explicitly stated in the content
2. For each goal, assign an importance score from 1-10 (10 being highest)
3. Provide a brief justification for each goal (max 30 words)
4. Identify the relevant topic area for each goal
5. Format your response as a JSON array of objects with the following properties:
   - goal_text: The learning goal (should be clear and actionable)
   - importance: Numeric importance score (1-10)
   - justification: Brief explanation of why this goal matters
   - topic: The subject area or domain this goal belongs to
   - source: Where in the content you found evidence for this goal

Only output goals with clear educational or learning value. Format your response as valid parseable JSON without additional text.
For example:
[
  {{
    "goal_text": "Master the principles of responsive web design",
    "importance": 8,
    "justification": "Foundational for creating accessible websites across devices",
    "topic": "Web Development",
    "source": "Content mentioned responsive design principles extensively"
  }},
  {{
    "goal_text": "Learn to implement accessibility standards in web applications",
    "importance": 9,
    "justification": "Critical for inclusive design and legal compliance",
    "topic": "Web Accessibility",
    "source": "Referenced WCAG guidelines and their importance"
  }}
]
"""
            
            # Use JSON format for goals
            response = self.generate_structured_response(prompt, format_type="json", temperature=0.2)
            
            if response and isinstance(response, list):
                # Validate the structure of each goal
                validated_goals = []
                for item in response:
                    try:
                        if isinstance(item, dict) and 'goal_text' in item and 'importance' in item:
                            # Convert all fields to appropriate types
                            title_str = str(item['goal_text']).strip()
                            
                            # Ensure importance is a number between 1-10
                            try:
                                importance = float(item['importance'])
                                if importance < 1:
                                    importance = 1
                                elif importance > 10:
                                    importance = 10
                                item["importance"] = importance
                            except:
                                item["importance"] = 5  # Default if not parseable
                            
                            # Add store type as additional metadata
                            item["store_type"] = store_type
                            item["extracted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                            
                            validated_goals.append(item)
                    except Exception as item_err:
                        logger.warning(f"Error validating goal item: {item_err}")
                        continue
                
                logger.info(f"Extracted {len(validated_goals)} goals from content")
                return validated_goals
            else:
                # Return empty list if extraction fails
                logger.warning(f"No goals extracted from content")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting goals: {e}")
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
