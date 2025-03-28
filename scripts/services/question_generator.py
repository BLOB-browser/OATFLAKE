import asyncio
import logging
import json
import pandas as pd
from pathlib import Path
import uuid
from datetime import datetime
import httpx
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Status tracking
_generation_active = False
_generation_progress = 0
_generation_error = None

def get_config_path():
    """Get the path to the config file in the project directory"""
    # First try to use config.json in the project root
    local_config = Path("config.json")
    if local_config.exists():
        return local_config
    
    # If it doesn't exist, try the user's home directory as fallback
    home_config = Path.home() / '.blob' / 'config.json'
    if home_config.exists():
        return home_config
    
    # If neither exists, return the local path as default
    return local_config

def get_generation_status():
    """Get the current status of question generation"""
    return {
        "active": _generation_active,
        "progress": _generation_progress,
        "error": str(_generation_error) if _generation_error else None
    }

async def generate_questions(num_questions: int = 5) -> List[Dict[str, Any]]:
    """Generate questions based on the vector store content"""
    global _generation_active, _generation_progress, _generation_error
    _generation_active = True
    _generation_progress = 0
    _generation_error = None
    
    try:
        logger.info(f"Starting question generation... (target: {num_questions} questions)")
        
        # Get data path from config
        config_path = get_config_path()
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        data_path = Path(config.get('data_path', ''))
        
        # Get an import reference to OllamaClient here to avoid circular imports
        from fastapi import Request
        from fastapi.requests import Request as FastAPIRequest
        from scripts.llm.ollama_client import OllamaClient
        
        # Create a client
        client = OllamaClient()
        
        # Build context from vector stores
        context = ""
        
        _generation_progress = 10
        
        # Generate 5 diverse queries to get comprehensive context
        sample_queries = [
            "What is this content about?",
            "What are the key concepts in this material?",
            "What practical applications does this knowledge have?",
            "What are the relationships between major components?",
            "What are the limitations or challenges in this domain?"
        ]
        
        logger.info("Gathering context from vector stores...")
        # Get context from each query with increased document count
        for query in sample_queries:
            retrieved_context = await client.get_relevant_context(query, k=8)
            if retrieved_context:
                context += retrieved_context + "\n\n"
        
        # If we couldn't get context, try to read from CSV files
        if not context:
            logger.info("No context from vector stores, trying CSV files...")
            # Try to get content from CSV files
            for csv_file in ["definitions.csv", "materials.csv", "methods.csv"]:
                csv_path = data_path / csv_file
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        if csv_file == "materials.csv" and "content" in df.columns:
                            # Take the first 5 materials
                            for _, row in df.head(5).iterrows():
                                context += f"{row.get('title', 'Untitled')}: {row.get('content', '')}\n\n"
                        elif csv_file == "definitions.csv" and "definition" in df.columns:
                            # Take the first 10 definitions
                            for _, row in df.head(10).iterrows():
                                context += f"{row.get('term', 'Term')}: {row.get('definition', '')}\n\n"
                        elif csv_file == "methods.csv" and "description" in df.columns:
                            # Take the first 5 methods
                            for _, row in df.head(5).iterrows():
                                context += f"{row.get('name', 'Method')}: {row.get('description', '')}\n\n"
                    except Exception as e:
                        logger.error(f"Error reading {csv_file}: {e}")
        
        _generation_progress = 30
        
        if not context:
            logger.warning("No context available for question generation")
            _generation_active = False
            return []
        
        logger.info(f"Generated context with {len(context.split())} words")
        
        # Create improved prompt for generating questions with better context utilization
        prompt = f"""
Based on the following context, generate {num_questions} thought-provoking research questions that could guide scholarly inquiry into these topics.
Each question should prompt critical analysis, synthesis of ideas, or exploration of relationships between concepts.
Draw specifically from the provided context to ensure questions are highly relevant to the actual content.

CONTEXT:
{context}

INSTRUCTIONS:
1. Generate exactly {num_questions} unique research questions that directly reference concepts from the context
2. Include a diverse mix of question types:
   - Analytical questions that examine relationships between specific concepts mentioned in the context
   - Evaluative questions that assess methodologies or approaches described in the material
   - Exploratory questions that investigate implications or future directions of the key ideas
   - Comparative questions that contrast different perspectives or theories presented in the context
   - Practical questions that address real-world applications of the concepts
3. Focus on questions that would require substantive research to answer properly
4. Frame questions to encourage deep exploration rather than factual recall
5. Ensure questions are specific enough to guide focused research, not overly broad
6. IMPORTANT: Your response must be ONLY a valid JSON array with each question having a "question" field
7. Do not include any explanations or additional text outside the JSON array

JSON FORMAT EXAMPLE:
[
  {{"question": "How might the relationship between concept A and concept B influence the development of new methodologies in this field?"}},
  {{"question": "What theoretical frameworks best explain the observed patterns in this domain, and what are their limitations?"}},
  {{"question": "To what extent can the findings from this area be generalized across different contexts or populations?"}}
]

Remember to only output the JSON array and nothing else.
"""

        _generation_progress = 50
        logger.info("Generating research questions...")
        
        # Make request to Ollama API with a longer timeout
        try:
            async with httpx.AsyncClient() as http_client:
                logger.info("Sending request to Ollama API with extended timeout...")
                response = await http_client.post(
                    f"{client.base_url}/api/generate",
                    json={
                        "model": client.model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.7,
                        # Increase max_tokens for more comprehensive responses
                        "max_tokens": 1024,
                    },
                    timeout=1200.0  # Increase timeout to 20 minutes for larger context processing
                )
                
                if response.status_code != 200:
                    logger.error(f"Error from Ollama API: {response.status_code}")
                    _generation_active = False
                    _generation_error = f"API Error: {response.status_code}"
                    return []
                    
                result = response.json()
                if not result or 'response' not in result:
                    logger.error("Invalid response format from Ollama API")
                    _generation_active = False
                    _generation_error = "Invalid API response"
                    return []
                    
                generated_text = result['response']
                
        except httpx.TimeoutException as e:
            logger.error(f"Request timed out after 20 minutes: {e}")
            _generation_error = f"API timeout after 20 minutes: {str(e)}"
            _generation_active = False
            # Return empty list instead of generic questions on timeout
            return []
            
        except Exception as e:
            logger.error(f"Error making request to Ollama API: {e}")
            _generation_active = False
            _generation_error = f"API error: {str(e)}"
            return []

        _generation_progress = 80
        logger.info(f"Received {len(generated_text.split())} words from LLM")
        
        # Extract JSON array from response
        try:
            # First attempt: Use regex to extract JSON array
            import re
            
            # Try different regex patterns to extract the JSON
            patterns = [
                r'\[\s*{.*}\s*\]',  # Standard JSON array pattern
                r'\[\s*\{\s*"question":.*\}\s*\]',  # More specific for our format
                r'\[(?:\s*\{(?:"question":\s*"[^"]*")\}(?:,|\s*)?)+\]'  # Even more specific
            ]
            
            json_data = None
            json_str = None
            
            # Try each pattern until we find a match
            for pattern in patterns:
                json_match = re.search(pattern, generated_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    try:
                        json_data = json.loads(json_str)
                        logger.info(f"Successfully extracted JSON using pattern: {pattern}")
                        break
                    except json.JSONDecodeError:
                        logger.warning(f"Found matching pattern but JSON decode failed: {pattern}")
            
            # If regex extraction failed, try some manual cleanup
            if not json_data:
                logger.info("Regex extraction failed, trying manual cleanup")
                # Try to find start and end of array
                start_idx = generated_text.find('[')
                end_idx = generated_text.rfind(']') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = generated_text[start_idx:end_idx]
                    try:
                        json_data = json.loads(json_str)
                        logger.info("Successfully extracted JSON with manual indices")
                    except json.JSONDecodeError:
                        logger.warning("Manual extraction found brackets but JSON decode failed")
            
            # If all extraction attempts failed, try to parse line by line
            if not json_data:
                logger.info("All extraction attempts failed, trying line-by-line parsing")
                questions_data = []
                
                # Look for lines containing "question"
                lines = generated_text.split('\n')
                for line in lines:
                    # Look for question pattern like: "question": "What is...?"
                    match = re.search(r'"question":\s*"([^"]+)"', line)
                    if match:
                        question_text = match.group(1)
                        questions_data.append({"question": question_text})
                
                if questions_data:
                    json_data = questions_data
                    logger.info(f"Line-by-line parsing found {len(questions_data)} questions")
            
            # If we still don't have valid JSON, try to generate questions from the text
            if not json_data:
                logger.warning("JSON extraction failed completely, generating questions from text")
                questions_data = []
                
                # Split text by sentences and look for questions
                sentences = re.split(r'[.!?]', generated_text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    # Check if sentence is a question
                    if sentence.endswith('?') or ('?' in sentence and len(sentence) > 15):
                        questions_data.append({"question": sentence})
                    # Look for numbered questions like "1. What is..."
                    elif re.match(r'^\d+[\s.)]+\w+', sentence) and len(sentence) > 15:
                        # Remove number prefix and add to questions
                        clean_question = re.sub(r'^\d+[\s.)]+', '', sentence)
                        questions_data.append({"question": clean_question})
                
                if questions_data:
                    json_data = questions_data
                    logger.info(f"Text-based parsing found {len(questions_data)} questions")
            
            # If all else fails, log error and return empty list
            if not json_data:
                logger.error("Failed to extract any questions from response")
                return []
                
            # Format each question with ID
            questions = []
            for item in json_data:
                if isinstance(item, dict) and "question" in item:
                    questions.append({
                        "question_id": str(uuid.uuid4()),
                        "question_text": item["question"],
                        "created_by": "blob",  # Changed from 'auto-generator' to 'blob'
                        "created_at": datetime.now().isoformat()
                    })
            
            logger.info(f"Successfully parsed {len(questions)} questions")
            
            # If we didn't get enough questions, that's fine - return what we have
            # Limit to requested number if we got more than requested
            questions = questions[:num_questions]
            return questions
            
        except Exception as e:
            logger.error(f"Error parsing questions: {e}", exc_info=True)
            _generation_error = str(e)
            return []
        
    except Exception as e:
        logger.error(f"Error generating questions: {e}", exc_info=True)
        _generation_error = str(e)
        return []
    finally:
        _generation_active = False
        _generation_progress = 100

async def save_questions(questions: List[Dict[str, Any]]) -> bool:
    """Save generated questions to CSV"""
    try:
        if not questions:
            logger.warning("No questions to save")
            return False
            
        # Get path
        config_path = get_config_path()
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        data_path = Path(config.get('data_path', ''))
        questions_path = data_path / "questions.csv"
        
        # Create or append to CSV
        questions_df = pd.DataFrame(questions)
        
        if questions_path.exists():
            # Append to existing CSV
            existing_df = pd.read_csv(questions_path)
            # Avoid duplicates by question ID
            existing_ids = set(existing_df['question_id'])
            new_questions = [q for q in questions if q['question_id'] not in existing_ids]
            
            if new_questions:
                new_df = pd.DataFrame(new_questions)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                updated_df.to_csv(questions_path, index=False)
                logger.info(f"Added {len(new_questions)} new questions to existing file")
            else:
                logger.info("No new questions to add")
        else:
            # Create new CSV
            questions_df.to_csv(questions_path, index=False)
            logger.info(f"Created new questions file with {len(questions)} questions")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving questions: {e}")
        return False
