#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for extracting structured information from content using LLMs.
This module separates extraction logic for better modularity and code maintenance.
"""

import logging
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ExtractionUtils:
    """
    Utilities for extracting structured information from content using LLMs.
    This class centralizes extraction functionality to avoid code duplication.
    """
    
    def __init__(self, resource_llm=None):
        """
        Initialize with an optional ResourceLLM instance.
        
        Args:
            resource_llm: ResourceLLM instance to use for extraction
        """
        self.resource_llm = resource_llm
        
        # If no ResourceLLM was provided, create one with default settings
        if self.resource_llm is None:
            try:
                from scripts.analysis.resource_llm import ResourceLLM
                from scripts.llm.processor_config_utils import get_adaptive_model_config, get_best_available_model
                
                # Get optimal configuration and model
                model = get_best_available_model()
                config = get_adaptive_model_config()
                
                # Create ResourceLLM instance with optimal settings
                self.resource_llm = ResourceLLM(model=model)
                self.resource_llm.model_config = config
                
                logger.info(f"Created ResourceLLM with model {model} for extraction utils")
            except Exception as e:
                logger.error(f"Error creating ResourceLLM instance: {e}")
                raise
    
    def extract_definitions(self, title: str, url: str, content: str) -> List[Dict]:
        """
        Extract definitions from resource content.
        
        Args:
            title: Resource title
            url: Resource URL
            content: Resource content text
            
        Returns:
            List of definition dictionaries with term and definition keys
        """
        logger.info(f"Extracting definitions from {title}")
        try:
            # First step: Extract key topics from the entire content
            # Use a larger content size for context
            max_content_length = 4000
            truncated_content = content[:max_content_length] if content else ""
            
            # Step 1: Identify key topics in the content
            topics_prompt = f"""<s>[INST] You are analyzing a web resource to identify key topics or terminologies.
            
            Title: {title}
            URL: {url}
            
            Content excerpt:
            {truncated_content[:2000]}
            
            Your task: Extract 3-5 key topics, concepts, or terms that appear important in this content.
            Focus on specialized terminology, technical concepts, skills, or projects mentioned.
            
            Return ONLY a JSON array of strings, like: ["term1", "term2", "term3"]
            [/INST]</s>"""
            
            topics_response = self.resource_llm.generate_structured_response(topics_prompt, format_type="json", temperature=0.2)
            
            if not topics_response or not isinstance(topics_response, list) or len(topics_response) == 0:
                logger.warning(f"No key topics identified for {title}, continuing with standard extraction")
                key_topics = []
            else:
                key_topics = [str(topic).strip() for topic in topics_response if topic]
                logger.info(f"Identified {len(key_topics)} key topics for {title}: {key_topics}")
            
            # Step 2: Now perform the main definition extraction with topics as context
            max_content_length = 4000
            truncated_content = content[:max_content_length] if content else ""
            
            # Add key topics to the prompt if available
            topics_text = ""
            if key_topics:
                topics_text = f"""
                Key topics already identified in this content:
                {', '.join(key_topics)}
                
                Please prioritize creating definitions for these topics if they appear relevant.
                """
            
            prompt = f"""
            You are analyzing a web resource to extract TECHNICAL TERMS and their definitions.
            Your task is to identify specialized terminology, technical concepts, methods, tools, or 
            established practices that appear in the content.
            
            Title: {title}
            URL: {url}
            
            {topics_text}
            
            Content excerpt:
            {truncated_content}
            
            IMPORTANT: You are extracting DEFINITIONS for a terminology database, NOT summarizing the website.
            Focus ONLY on extracting individual technical terms with their definitions.
            
            Instructions:
            1. Look for SPECIALIZED TECHNICAL TERMS in fields like:
               - Design methodologies (e.g., "Parametric Design", "User-Centered Design")
               - Technical tools or software (e.g., "Grasshopper", "Arduino")
               - Scientific concepts (e.g., "Biomimicry", "Synthetic Biology")
               - Research methods (e.g., "Ethnographic Research", "Data Visualization")
               
            2. Create definitions for technical terms ONLY - do NOT include:
               - People's names (unless defining a specific methodology named after them)
               - Project titles or general topics
               - General concepts without technical meaning (e.g., "portfolio", "about me")
               
            3. Each definition should explain what the technical term means in its field
            
            4. If the website doesn't contain enough technical terms, it's BETTER to return fewer high-quality 
               definitions or even an empty array than to create definitions for non-technical terms
            
            EXTREMELY IMPORTANT FORMATTING INSTRUCTIONS:
            1. Your response must be ONLY a valid JSON array containing term-definition pairs
            2. Each JSON object MUST have exactly these two fields with these exact names: "term" and "definition"
            3. "term" should be a short, specific phrase (1-5 words)
            4. "definition" should be a clear, concise explanation (1-3 sentences)
            5. Format response as a valid JSON array containing objects with these two fields
            6. Include no explanation, just the JSON array
            7. If no definitions found, return ONLY an empty array: []
            
            REQUIRED FORMAT EXAMPLE:
            [
              {{
                "term": "Design Fiction",
                "definition": "A design practice that combines elements of science fiction with product design to explore possible futures and their implications."
              }},
              {{
                "term": "Speculative Design",
                "definition": "An approach that uses design to provoke questions about possible futures and alternative presents."
              }}
            ]
            """
            
            # Use JSON format for definitions
            response = self.resource_llm.generate_structured_response(prompt, format_type="json", temperature=0.1)
            
            if response and isinstance(response, list):
                # Validate the structure of each definition
                validated_definitions = []
                for item in response:
                    try:
                        if isinstance(item, dict) and 'term' in item and 'definition' in item:
                            # Make sure both are strings
                            term = str(item['term']).strip()
                            definition = str(item['definition']).strip()
                            
                            if term and definition:
                                validated_definitions.append({
                                    'term': term,
                                    'definition': definition
                                })
                    except Exception as item_err:
                        logger.warning(f"Error validating definition item: {item_err}")
                        continue
                
                logger.info(f"Extracted {len(validated_definitions)} definitions for {title}")
                return validated_definitions
            else:
                # Return empty list if extraction fails
                logger.warning(f"No definitions extracted for {title}")
                return []
                
        except Exception as e:
            logger.error(f"Error extracting definitions: {e}")
            return []  # Return empty list in case of any error
    
    def identify_projects(self, title: str, url: str, content: str) -> List[Dict]:
        """
        Identify projects from resource content.
        
        Args:
            title: Resource title
            url: Resource URL
            content: Resource content text
            
        Returns:
            List of project dictionaries
        """
        logger.info(f"Identifying projects from {title}")
        
        try:
            # Use a larger content size for finding more projects
            max_content_length = 4000
            truncated_content = content[:max_content_length] if content else ""
            
            prompt = f"""<s>[INST] You are analyzing a web resource to identify ACTUAL PROJECTS or significant works
            that are presented in detail. Focus on finding tangible, completed or in-progress works.

            Title: {title}
            URL: {url}
            
            Content excerpt:
            {truncated_content}
            
            Your task: Identify up to 3 SPECIFIC, REAL PROJECTS or works that are featured in this content.
            
            ONLY INCLUDE ACTUAL PROJECTS:
            - For design portfolios: Include specific design projects with concrete outputs
            - For research sites: Include specific research studies or experiments
            - For academic sites: Include publications, courses developed, or research initiatives
            - For personal sites: Include major works, exhibitions, or substantial creations
            
            DO NOT INCLUDE:
            - Generic sections of the website (like "About Me" or "Contact")
            - Skills or capabilities (unless they are part of a specific project)
            - Vague ideas or concepts without concrete implementation
            - General topics or interests without a specific project output
            
            For each project, extract:
            - An exact project title as it appears on the site
            - A specific description of what the project actually produced or created
            - The stated goals or purpose of the project
            - Relevant fields or categories that the project belongs to
            
            If no specific projects are described in enough detail, return an empty array.
            
            EXTREMELY IMPORTANT FORMATTING INSTRUCTIONS:
            1. Your response must be a valid JSON array of objects
            2. Each object must have these exact fields:
               - "title": The exact name of the specific project (string)
               - "description": A brief description (string, 1-2 sentences)
               - "goals": The stated objectives of this project (string, if available)
               - "fields": Array of strings with 2-5 relevant field/category tags
            3. Return ONLY the JSON array with no explanation text before or after
            4. If no projects are found, return an empty array: []
            5. The JSON MUST be valid and parseable
            
            REQUIRED FORMAT:
            [
              {{
                "title": "Project Name",
                "description": "Brief description of what this project is",
                "goals": "The stated objectives of this project",
                "fields": ["field1", "field2", "field3"]
              }}
            ]
            [/INST]</s>"""
            
            # Use JSON format for projects
            response = self.resource_llm.generate_structured_response(prompt, format_type="json", temperature=0.1)
            
            if response and isinstance(response, list):
                # Validate the structure of each project
                validated_projects = []
                for item in response:
                    try:
                        if isinstance(item, dict) and 'title' in item and 'description' in item:
                            # Convert all fields to appropriate types
                            title_str = str(item['title']).strip()
                            desc_str = str(item['description']).strip()
                            goals_str = str(item.get('goals', '')).strip()
                            
                            # Ensure fields are lists of strings if present
                            fields_list = []
                            if 'fields' in item:
                                if isinstance(item['fields'], list):
                                    fields_list = [str(field).strip() for field in item['fields'] if field]
                                elif isinstance(item['fields'], str):
                                    # Try to parse string as JSON array if it looks like one
                                    if item['fields'].startswith('[') and item['fields'].endswith(']'):
                                        try:
                                            parsed_fields = json.loads(item['fields'])
                                            if isinstance(parsed_fields, list):
                                                fields_list = [str(field).strip() for field in parsed_fields if field]
                                        except:
                                            # If parsing fails, use it as a single tag
                                            fields_list = [item['fields'].strip('[]').strip('"\'').strip()]
                                    else:
                                        # Use as a single tag
                                        fields_list = [item['fields'].strip()]
                            
                            # Only add if we have the required fields with content
                            if title_str and desc_str:
                                validated_projects.append({
                                    'title': title_str,
                                    'description': desc_str,
                                    'goals': goals_str,
                                    'fields': fields_list
                                })
                    except Exception as item_err:
                        logger.warning(f"Error validating project item: {item_err}")
                        continue
                
                logger.info(f"Identified {len(validated_projects)} projects for {title}")
                return validated_projects
            else:
                # Return empty list if identification fails
                logger.warning(f"No projects identified for {title}")
                return []
                
        except Exception as e:
            logger.error(f"Error identifying projects: {e}")
            return []  # Return empty list in case of any error
    
    def extract_methods(self, title: str, url: str, content: str) -> List[Dict]:
        """
        Extract methods and procedures from resource content.
        
        Args:
            title: Resource title
            url: Resource URL
            content: Resource content text
            
        Returns:
            List of method dictionaries
        """
        # If a separate MethodLLM is available, use it
        try:
            from scripts.analysis.method_llm import MethodLLM
            method_llm = MethodLLM()
            logger.info(f"Using dedicated MethodLLM to extract methods from {title}")
            return method_llm.extract_methods(title, url, content)
        except Exception as e:
            # If MethodLLM import fails, fallback to ResourceLLM
            logger.warning(f"Error using MethodLLM, falling back to ResourceLLM: {e}")
            
            # Construct a method extraction prompt
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
            
            methods_json = self.resource_llm.generate_structured_response(prompt, format_type="json")
            
            # Validate and process the methods
            if methods_json and isinstance(methods_json, list):
                valid_methods = []
                for method in methods_json:
                    if isinstance(method, dict) and 'title' in method and 'description' in method and 'steps' in method:
                        # Process steps to ensure they're a list
                        steps_list = []
                        if isinstance(method['steps'], list):
                            steps_list = [str(step).strip() for step in method['steps'] if step]
                        else:
                            steps_list = [str(method['steps']).strip()]
                        
                        # Process tags to ensure they're a list
                        tags_list = []
                        if 'tags' in method:
                            if isinstance(method['tags'], list):
                                tags_list = [str(tag).strip().lower() for tag in method['tags'] if tag]
                            else:
                                tags_list = [str(method['tags']).strip().lower()]
                        
                        # Add the validated method
                        valid_methods.append({
                            'title': str(method['title']).strip(),
                            'description': str(method['description']).strip(),
                            'steps': steps_list,
                            'tags': tags_list
                        })
                        
                logger.info(f"Extracted {len(valid_methods)} methods from {title}")
                return valid_methods
            else:
                logger.warning(f"No methods found in {title}")
                return []
