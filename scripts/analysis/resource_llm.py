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
    
    def generate_structured_response(self, prompt: str, format_type: str = "json", 
                                     temperature: float = 0.1, max_tokens: int = 512) -> Any:
        """
        Generate a structured response from the LLM, with specific parameters
        optimized for predictable JSON generation.
        
        Args:
            prompt: The prompt to send to the model
            format_type: Type of formatting to expect ("json", "list", "text")
            temperature: Temperature setting (lower for more deterministic output)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Parsed structured response (JSON object, list, or text)
        """
        # Verify model exists before attempting to use it
        try:
            model_check_response = requests.get(f"{self.base_url}/api/tags")
            if model_check_response.status_code == 200:
                models_data = model_check_response.json()
                available_models = [m.get("name") for m in models_data.get("models", [])]
                
                if self.model not in available_models:
                    logger.error(f"Model {self.model} not found! Please pull it with: ollama pull {self.model}")
                    # Fall back to whatever model is available
                    available_models = [m for m in available_models if 'instruct' in m.lower()]
                    if available_models:
                        self.model = available_models[0]
                        logger.warning(f"Falling back to available model: {self.model}")
                    else:
                        logger.error("No instruction models available. Resource processing will fail.")
                        return None
        except Exception as e:
            logger.warning(f"Error checking model availability: {e}")
            # Continue anyway, we'll catch errors during the main API call
        # Add formatting guidance and system instructions specific to Mistral
        if format_type == "json":
            # Add Mistral-specific system prompt for better JSON formatting
            prompt = f"""<s>[INST] You are an expert JSON formatter with perfect accuracy.
When asked to generate JSON, you ONLY output valid, parseable JSON without any explanation or markdown formatting.
            
{prompt.strip()}

Output ONLY the JSON with no additional text. [/INST]</s>"""
        else:
            # For non-JSON responses, still use the Mistral instruction format
            prompt = f"""<s>[INST] {prompt.strip()} [/INST]</s>"""
        
        # Prepare request for structured generation
        try:
            logger.info(f"Generating structured response with format={format_type}")
            
            # Make API call with optimized parameters and thread configuration
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
                timeout=3600.0  # Extended to 1 hour for Raspberry Pi (from 1000 seconds)
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Ollama API response time: {elapsed:.2f} seconds")
            
            if response.status_code != 200:
                logger.error(f"Error from Ollama API: {response.status_code}")
                # Try initiating a pull for the model if it's a 404 (model not found)
                if response.status_code == 404:
                    try:
                        logger.warning(f"Model {self.model} not found. Attempting to pull...")
                        pull_response = requests.post(
                            f"{self.base_url}/api/pull",
                            json={"name": self.model},
                            timeout=10.0  # Just to check if pull started
                        )
                        if pull_response.status_code == 200:
                            logger.info(f"Model {self.model} is being pulled in the background. Try again soon.")
                        else:
                            logger.error(f"Failed to pull model: {pull_response.status_code}")
                    except Exception as pull_error:
                        logger.error(f"Error pulling model: {pull_error}")
                return None
            
            result = response.json()
            if not result or 'response' not in result:
                logger.error("Invalid response format from API")
                return None
            
            # Extract the raw response text
            raw_response = result['response']
            logger.debug(f"Raw response: {raw_response[:100]}...")
            
            # Process based on expected format
            if format_type == "json":
                try:
                    return self._parse_json_response(raw_response)
                except Exception as e:
                    logger.error(f"Error parsing JSON response: {e}")
                    return None
            elif format_type == "list":
                return self._parse_list_response(raw_response)
            else:
                return raw_response
                
        except Exception as e:
            logger.error(f"Error generating structured response: {e}")
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
    
    def _parse_json_response(self, response_text: str) -> Any:
        """
        Parse JSON from LLM response with enhanced cleanup and validation.
        
        Args:
            response_text: Raw response text from the LLM
            
        Returns:
            Parsed JSON object/array
        """
        # Check for empty responses or model errors
        if not response_text or response_text.strip() == "":
            logger.error("Received empty response from LLM, returning None")
            return None
            
        # Handle API error responses that might be in JSON format
        if "error" in response_text and len(response_text) < 200:
            try:
                error_json = json.loads(response_text)
                if "error" in error_json:
                    logger.error(f"LLM API error: {error_json['error']}")
                    return None
            except:
                pass  # Not a JSON error, continue with parsing
        
        # Log the raw response for debugging (full response for better debugging)
        logger.debug(f"Parsing JSON from: {response_text}")
        
        try:
            # First attempt: direct JSON parsing
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            # Log specific JSON error to aid debugging
            logger.error(f"Error extracting JSON: {e}")
            
            # Second attempt: Extract JSON portion from response and fix common issues
            try:
                # Find the first occurrence of '[' or '{'
                start = None
                for i, char in enumerate(response_text):
                    if char in '[{':
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
                        logger.debug(f"Extracted JSON: {json_text}")
                        
                        try:
                            return json.loads(json_text)
                        except json.JSONDecodeError as e:
                            # Try cleaning up common issues
                            logger.debug(f"Error with extracted JSON: {e}")
                            cleaned_json = self._clean_json_text(json_text)
                            
                            # Attempt fix for missing comma delimiters (most common issue)
                            if "Expecting ',' delimiter" in str(e):
                                position = int(str(e).split("char ")[1].strip(")")) if "char " in str(e) else -1
                                if position > 0 and position < len(cleaned_json):
                                    # Insert a comma at the position where it's expected
                                    fixed_json = cleaned_json[:position] + "," + cleaned_json[position:]
                                    logger.info(f"Attempting to fix missing comma delimiter at position {position}")
                                    try:
                                        return json.loads(fixed_json)
                                    except:
                                        pass  # If this fix fails, continue with other methods
                                        
                            # Try regular cleaning
                            try:
                                return json.loads(cleaned_json)
                            except json.JSONDecodeError:
                                # Extra aggressive comma fixing for arrays
                                if "[" in cleaned_json and "]" in cleaned_json and "{" in cleaned_json and "}" in cleaned_json:
                                    # Fix missing commas between objects in an array
                                    aggressive_fix = re.sub(r'}\s*{', '},{', cleaned_json)
                                    try:
                                        return json.loads(aggressive_fix)
                                    except:
                                        pass  # Continue to next method
            except Exception as e:
                logger.error(f"Error extracting JSON portion: {e}")
                
            # Third attempt: Use regex to find JSON patterns
            try:
                # Look for array of objects pattern
                if '[' in response_text and ']' in response_text:
                    array_match = re.search(r'\[\s*(\{.*?\}\s*(?:,\s*\{.*?\}\s*)*)\]', response_text, re.DOTALL)
                    if array_match:
                        array_text = f"[{array_match.group(1)}]"
                        cleaned_json = self._clean_json_text(array_text)
                        # Fix missing commas between objects
                        aggressive_fix = re.sub(r'}\s*{', '},{', cleaned_json)
                        try:
                            return json.loads(aggressive_fix)
                        except:
                            # Try just the cleaned version
                            try:
                                return json.loads(cleaned_json)
                            except:
                                pass
                
                # Look for bare objects
                if '{' in response_text and '}' in response_text:
                    object_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
                    if object_match:
                        object_text = object_match.group(0)
                        cleaned_json = self._clean_json_text(object_text)
                        try:
                            return json.loads(cleaned_json)
                        except:
                            pass
            except Exception as e:
                logger.error(f"Error with regex JSON extraction: {e}")
            
            # Final fallback for list extraction
            if '[' in response_text and ']' in response_text:
                try:
                    # For arrays of strings, looser extraction
                    array_content = response_text[response_text.find('[')+1:response_text.rfind(']')]
                    # Handle both comma-separated and non-comma-separated items
                    if ',' in array_content:
                        items = [item.strip().strip('"\'') for item in array_content.split(',')]
                    else:
                        # Try to split by whitespace or newlines if no commas
                        items = [item.strip().strip('"\'') for item in re.split(r'\s+', array_content) if item.strip()]
                    
                    logger.info(f"Extracted array with {len(items)} items using fallback method")
                    return items
                except Exception as e:
                    logger.error(f"Error with fallback array extraction: {e}")
            
            # Last resort: try to extract any valid-looking tags with regex
            try:
                # Look for what appear to be tags in the response
                tag_matches = re.findall(r'[\"\']([a-z0-9-]{2,30})[\"\']', response_text.lower())
                if tag_matches and len(tag_matches) > 0:
                    logger.info(f"Extracted {len(tag_matches)} tags using regex pattern matching")
                    return tag_matches[:5]  # Limit to 5 tags
            except Exception as e:
                logger.error(f"Error with regex tag extraction: {e}")
            
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