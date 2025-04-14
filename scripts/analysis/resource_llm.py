#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dedicated LLM interface for resource processing.
This implementation is focused on reliable JSON generation for resource analysis,
independent of the web-focused OllamaClient.
"""

import logging
import json
import re
import requests
import time
from typing import Dict, List, Any, Optional, Union

# Add import for interruptible requests
from scripts.analysis.interruptible_llm import (
    interruptible_post, async_interruptible_post,
    is_interrupt_requested, setup_interrupt_handling, restore_interrupt_handling, clear_interrupt
)

logger = logging.getLogger(__name__)

class ResourceLLM:
    """
    A dedicated LLM interface for resource analysis tasks.
    This ensures reliable JSON generation and parsing separate from
    the chat-focused OllamaClient implementation.
    
    This class provides core generation capabilities, while extraction
    functions are now in ExtractionUtils.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral:7b-instruct-v0.2-q4_0"):
        self.base_url = base_url
        self.model = model
        
        # Get adaptive model configuration from the utility - updated import path
        from scripts.llm.processor_config_utils import get_adaptive_model_config
        self.model_config = get_adaptive_model_config()
        
        logger.info(f"Initialized ResourceLLM with model {model} ({self.model_config['threads']} threads, {self.model_config['num_ctx']} context, {self.model_config['batch_size']} batch size)")
    
    def call_ollama_api(self, prompt: str, model: str = None, **kwargs):
        """Call Ollama API with interruption handling"""
        # Set up interruptible handling
        setup_interrupt_handling()
        
        try:
            # Default to instance model if none provided
            if not model:
                model = self.model
                
            # Default arguments for the Ollama API
            args = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            }
            
            # Update with any additional kwargs
            args.update(kwargs)
            
            start_time = time.time()
            logger.debug(f"Calling Ollama API with model {model}")
            
            try:
                # Replace standard request with interruptible version - fix the API call
                response = interruptible_post(
                    "http://localhost:11434/api/generate",
                    json=args,
                    timeout=300  # 5 minute timeout for long processing
                )
                
                if is_interrupt_requested():
                    logger.warning("LLM processing was interrupted by user")
                    return {"response": "Processing interrupted by user", "interrupted": True}
                
                if response.status_code != 200:
                    logger.error(f"Error: Ollama API returned status code {response.status_code}")
                    logger.error(f"Error response: {response.text}")
                    return {"error": f"Ollama API error: {response.status_code}"}
                    
                result = response.json()
                
            except KeyboardInterrupt:
                logger.warning("LLM API call interrupted by keyboard interrupt")
                return {"response": "Processing interrupted by user", "interrupted": True}
            except Exception as e:
                logger.error(f"Error calling Ollama API: {e}")
                return {"error": f"API error: {e}"}
                
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Ollama API response time: {duration:.2f} seconds")
            
            return result
        finally:
            # Restore original handlers
            restore_interrupt_handling()

    def generate(self, prompt: str, temperature: float = None) -> str:
        """
        Generate a text response from the LLM without JSON parsing.
        This is a compatibility method needed by extraction_utils.py
        
        Args:
            prompt: The input prompt
            temperature: Optional temperature parameter
            
        Returns:
            Raw text response from the LLM
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Use appropriate internal method based on model type
            logger.info("Generating text response")
            
            # Use temperature if provided, otherwise use default
            temp = temperature if temperature is not None else self.model_config.get('temperature', 0.1)
            
            # Call the appropriate internal method based on model
            if hasattr(self, 'llm') and self.llm is not None:
                # Most LLM classes have a 'generate' or similar method
                if hasattr(self.llm, 'generate'):
                    return self.llm.generate(prompt, temperature=temp)
                elif hasattr(self.llm, '__call__'):
                    # Some LLMs can be called directly
                    return self.llm(prompt)
                    
            # Use Ollama API endpoint directly if needed 
            return self._generate_with_ollama(prompt, temperature=temp)
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""

    def _generate_with_ollama(self, prompt: str, temperature: float = 0.1) -> str:
        """Use Ollama API directly with increased timeout"""
        import httpx
        
        logger = logging.getLogger(__name__)
        try:
            # Default Ollama endpoint
            url = "http://localhost:11434/api/generate"
            
            # Create request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
                # Parameters that help with longer contexts and generation
                "num_predict": 1024,  # Request up to 1024 tokens in response
            }
            
            logger.info(f"Calling Ollama API with model {self.model}")
            
            # Use a much longer timeout (5 minutes) - LLMs should take the time they need
            with httpx.Client(timeout=300.0) as client:  # 5 minute timeout
                response = client.post(url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "")
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    return f"Error: Ollama API returned status {response.status_code}"
                    
        except httpx.TimeoutException:
            logger.error("Ollama API request timed out after 5 minutes")
            return "Error: The request to the language model timed out after 5 minutes."
        except Exception as e:
            logger.error(f"Error in Ollama API call: {e}")
            return ""

    def generate_structured_response(self, prompt: str, format_type: str = "json", temperature: float = None) -> Any:
        """
        Generate a structured response from the LLM with better error handling for JSON parsing.
        
        Args:
            prompt: Input prompt
            format_type: Type of format to extract ("json" or "yaml")
            temperature: Optional temperature to use
            
        Returns:
            Parsed structure or None on failure
        """
        logger.info(f"Generating structured response with format={format_type}")
        
        try:
            # Generate response with appropriate temp
            response_text = self.generate(prompt, temperature=temperature)
            
            if not response_text:
                logger.warning("No response received from LLM")
                return None
            
            # JSON extraction with improved robustness
            if format_type.lower() == "json":
                return self._extract_json_robustly(response_text)
            # ...existing code for other format types...
        
        except Exception as e:
            logger.error(f"Error in generate_structured_response: {e}")
            return None

    def generate_description(self, title: str, url: str, content: str) -> str:
        """
        Generate a comprehensive summary description for a resource.
        This provides an overview of what the website/resource is about.
        
        Args:
            title: Resource title
            url: Resource URL
            content: Resource content text
            
        Returns:
            Generated description string (paragraph summary)
        """
        # Use a larger content size for better understanding
        max_content_length = 4000
        truncated_content = content[:max_content_length] if content else ""
        
        prompt = f"""<s>[INST] You are analyzing a web resource to create a comprehensive summary.

        Title: {title}
        URL: {url}
        
        Content excerpt:
        {truncated_content}
        
        Task: Write a clear, informative paragraph (3-5 sentences) that summarizes what this resource is about.
        
        This is NOT a definition of a single term - this is a summary of the entire website/resource.
        
        For personal websites or portfolios:
        - Describe who the person is
        - Their field of study or expertise
        - The purpose of the website
        - Key projects or content featured
        
        For other websites:
        - The organization or entity behind it
        - The main purpose or topic
        - The type of content provided
        - The target audience
        
        Be specific and informative. Use factual details from the content.
        [/INST]</s>"""
        
        # Use text format for description
        response = self.generate_structured_response(prompt, format_type="text", temperature=0.3)
        
        if response:
            # Clean up any potential formatting artifacts
            description = response.strip()
            logger.info(f"Generated description for {title}: {description[:50]}...")
            return description
        else:
            # Return a default description if generation fails
            default = f"Resource about {title}."
            logger.warning(f"Using default description for {title}: {default}")
            return default
    
    def generate_tags(self, title: str, url: str, content: str, description: str, existing_tags: List = None) -> List[str]:
        """
        Generate tags for a resource.
        
        Args:
            title: Resource title
            url: Resource URL
            content: Resource content text
            description: Resource description
            existing_tags: Any existing tags to consider
            
        Returns:
            List of tags or empty list if generation fails
        """
        # If existing tags are provided, use them
        if existing_tags and isinstance(existing_tags, list) and len(existing_tags) >= 3:
            logger.info(f"Using existing tags for {title}: {existing_tags}")
            return existing_tags
        
        # Use a moderate content size to balance quality and performance
        max_content_length = 2500
        truncated_content = content[:max_content_length] if content else ""
        
        prompt = f"""
        You are analyzing a web resource to generate relevant tags. Based on the title, URL, and content, 
        generate 3-5 relevant tags that best categorize this resource.
        
        Title: {title}
        URL: {url}
        Description: {description}
        
        Content excerpt:
        {truncated_content}
        
        Your task: Generate 3-5 specific, relevant tags for this resource. Each tag should be a single lowercase 
        word or hyphenated phrase (e.g., "python", "web-development", "machine-learning").
        
        EXTREMELY IMPORTANT FORMATTING INSTRUCTIONS:
        1. Return ONLY a valid JSON array of tag strings
        2. Each tag must be a simple string with only lowercase letters, numbers, or hyphens
        3. No spaces, special characters, or punctuation in tags
        4. DO NOT include ANY explanation text before or after the JSON array
        5. The output MUST be valid JSON that can be parsed directly
        
        REQUIRED FORMAT:
        ["tag1", "tag2", "tag3"]
        
        BAD (DO NOT DO THIS):
        Here are some tags: ["tag1", "tag2", "tag3"]
        
        GOOD (DO THIS):
        ["web-development", "design", "programming", "technology"]
        """
        
        # Use JSON format for tags
        response = self.generate_structured_response(prompt, format_type="json")
        
        if response and isinstance(response, list) and len(response) > 0:
            # Ensure all tags are strings and lowercase
            # Add additional validation to ensure we have valid tag strings
            tags = []
            for tag in response:
                if tag and isinstance(tag, (str, int, float)):
                    # Convert to string and normalize
                    tag_str = str(tag).lower().strip()
                    # Only include non-empty tags without special characters
                    if tag_str and len(tag_str) > 1 and re.match(r'^[a-z0-9-]+$', tag_str):
                        tags.append(tag_str)
            
            tags = tags[:5]  # Limit to 5 tags maximum
            logger.info(f"Generated tags for {title}: {tags}")
            return tags
        else:
            # Return empty list if generation fails
            logger.warning(f"Failed to generate tags for {title}, returning empty list")
            return []
    
    def _extract_json_robustly(self, text: str) -> Any:
        """
        Extract JSON from text with multiple fallback approaches.
        
        Args:
            text: Text possibly containing JSON
            
        Returns:
            Parsed JSON object or None if extraction fails
        """
        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {e}")
        
        # Try to find JSON array/object boundaries with regex
        import re
        
        # Look for array pattern
        array_match = re.search(r'\[\s*{.*}\s*\]', text, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                logger.warning("Array pattern extraction failed")
        
        # Look for object pattern
        object_match = re.search(r'{.*}', text, re.DOTALL)
        if object_match:
            try:
                return json.loads(object_match.group(0))
            except json.JSONDecodeError:
                logger.warning("Object pattern extraction failed")
        
        # Try removing markdown code block markers
        clean_text = re.sub(r'```json|```|\[INST\]|\[/INST\]|<s>|</s>', '', text)
        try:
            return json.loads(clean_text.strip())
        except json.JSONDecodeError:
            logger.warning("Cleaned text JSON parsing failed")
        
        # Try line-by-line to find valid JSON
        lines = text.splitlines()
        
        # Look for lines that might start/end a JSON block
        for i in range(len(lines)):
            if lines[i].strip().startswith('['):
                # Try to find matching array end
                for j in range(i, len(lines)):
                    if lines[j].strip().endswith(']'):
                        json_block = '\n'.join(lines[i:j+1])
                        try:
                            return json.loads(json_block)
                        except json.JSONDecodeError:
                            continue
        
        logger.error(f"Error extracting JSON: Could not find valid JSON in the response")
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
            logger.debug(f"Cleaning JSON text: {json_text[:100]}...")
            
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
            
            # Handle unclosed quotes
            cleaned = re.sub(r'"([^"]*?)(?=[,}\]])', r'"\1"', cleaned)
            
            # Fix issues with backslashes in strings
            cleaned = re.sub(r'([^\\])\\([^"\\\\/bfnrtu])', r'\1\\\\\2', cleaned)
            
            # ADD NEW CLEANING: Fix missing commas between objects in arrays
            cleaned = re.sub(r'}\s*{', '},{', cleaned)
            
            # ADD NEW CLEANING: Fix missing commas between array elements
            cleaned = re.sub(r'"\s*"', '","', cleaned)
            
            # ADD NEW CLEANING: Fix property name quoting in more cases
            cleaned = re.sub(r'([{\[,]\s*)(\w+)(\s*:)', r'\1"\2"\3', cleaned)
            
            return cleaned
            
        except Exception as e:
            # If any cleanup fails, log it but return the original
            logger.error(f"Error during JSON cleanup: {e}")
            return json_text
    
    def _parse_list_response(self, response: str) -> List[str]:
        """
        Parse list of items from LLM response.
        
        Args:
            response: Raw response text from the LLM
            
        Returns:
            List of strings
        """
        # First try JSON array parsing
        try:
            if '[' in response and ']' in response:
                array_text = response[response.find('['):response.rfind(']')+1]
                return json.loads(array_text)
        except:
            pass
        
        # Fallback: Extract bullet points or numbered items
        items = []
        
        # Check for bullet points or numbered lists
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # Match bullet points (-, *, •) or numbered items (1., 2., etc.)
            if re.match(r'^[-*•]\s+(.+)$', line) or re.match(r'^\d+\.\s+(.+)$', line):
                # Extract the content after the bullet or number
                content = re.sub(r'^[-*•]\s+', '', line)
                content = re.sub(r'^\d+\.\s+', '', content)
                items.append(content.strip())
        
        if items:
            return items
        
        # If no structured format is found, split by newlines and filter empty lines
        return [line.strip() for line in lines if line.strip()]

    def _extract_json(self, text):
        """
        Extract JSON from the LLM response text, with improved error handling.
        
        Args:
            text: The text containing JSON response
            
        Returns:
            The parsed JSON object, or None if extraction fails
        """
        try:
            # Try to find JSON between triple backticks
            import re
            # Fix: Use proper string literals for regex pattern instead of actual backticks
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            
            if json_match:
                json_str = json_match.group(1).strip()
                # Use existing _clean_json_text function to handle escaping issues
                cleaned_json = self._clean_json_text(json_str)
                try:
                    return json.loads(cleaned_json)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in code block: {e}")
                    # Try a more aggressive cleaning approach
                    ultra_clean = re.sub(r'[^\x20-\x7E]', '', cleaned_json)  # Remove non-printable chars
                    return json.loads(ultra_clean)
            
            # If not found, try to find JSON between single backticks
            # Fix: Use proper string literals for regex pattern
            json_match = re.search(r'`([\s\S]*?)`', text)
            if json_match:
                json_str = json_match.group(1).strip()
                # Use existing _clean_json_text function
                cleaned_json = self._clean_json_text(json_str)
                try:
                    return json.loads(cleaned_json)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error in inline code: {e}")
            
            # Last resort: try to find any JSON-like structure in the text
            try:
                # Look for anything between curly braces with added validation
                brace_match = re.search(r'(\{[\s\S]*\})', text)
                if brace_match:
                    json_str = brace_match.group(1).strip()
                    # Use existing _clean_json_text function
                    cleaned_json = self._clean_json_text(json_str)
                    try:
                        return json.loads(cleaned_json)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error in brace extraction: {e}")
                        
                # Try to find an array
                array_match = re.search(r'(\[[\s\S]*\])', text)
                if array_match:
                    json_str = array_match.group(1).strip()
                    # Use existing _clean_json_text function
                    cleaned_json = self._clean_json_text(json_str)
                    try:
                        return json.loads(cleaned_json)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error in array extraction: {e}")
            except Exception as inner_error:
                logger.error(f"Inner parsing error: {inner_error}")
            
            # If all else fails, try to clean and parse the entire text
            try:
                # First, try direct parsing
                try:
                    return json.loads(text)
                except json.JSONDecodeError as e:
                    pass
                    
                # Then try with cleaned text
                cleaned_text = self._clean_json_text(text)
                try:
                    return json.loads(cleaned_text)
                except json.JSONDecodeError as e:
                    # Log specific error location
                    logger.error(f"JSON decode error in full text: {e}")
                    # Try to handle the "Extra data" error specifically
                    if "Extra data" in str(e):
                        pos = int(str(e).split("char ")[1].strip(")")) if "char " in str(e) else -1
                        if pos > 0:
                            # Just use the part before the extra data
                            return json.loads(text[:pos])
            except Exception as e:
                # If nothing works, return None and log
                logger.error(f"No valid JSON found in: {text[:100]}...")
                return None
            
        except Exception as e:
            logger.error(f"Error extracting JSON: {e}")
            # Fallback: try a more manual extraction approach
            try:
                logger.info("Attempting to extract JSON using fallback method")
                # Find JSON-like patterns - handle arrays or objects
                if '[' in text and ']' in text:
                    start_idx = text.find('[')
                    end_idx = text.rfind(']') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        # Extract just the array part
                        array_text = text[start_idx:end_idx]
                        # Clean the extracted array
                        cleaned_array = self._clean_json_text(array_text)
                        try:
                            return json.loads(cleaned_array)
                        except Exception as array_error:
                            logger.error(f"Failed to parse array: {array_error}")
                
                elif '{' in text and '}' in text:
                    start_idx = text.find('{')
                    end_idx = text.rfind('}') + 1
                    if start_idx >= 0 and end_idx > start_idx:
                        # Extract just the object part
                        obj_text = text[start_idx:end_idx]
                        # Clean the extracted object
                        cleaned_obj = self._clean_json_text(obj_text)
                        try:
                            return json.loads(cleaned_obj)
                        except Exception as obj_error:
                            logger.error(f"Failed to parse object: {obj_error}")
                
                # Return None if all attempts fail
                return None
                
            except Exception as fallback_error:
                logger.error(f"Fallback extraction also failed: {fallback_error}")
                return None