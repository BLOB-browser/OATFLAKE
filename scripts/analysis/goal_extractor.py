#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import asyncio
from datetime import datetime
from pathlib import Path
import httpx

logger = logging.getLogger(__name__)

class GoalExtractor:
    """
    Extracts goals from vector stores using Mistral LLM.
    Analyzes content and topic stores to identify goals and their importance.
    """
    
    def __init__(self, data_folder: str):
        """
        Initialize the goal extractor.
        
        Args:
            data_folder: Path to data directory
        """
        self.data_folder = Path(data_folder)
        
    async def extract_goals(self, ollama_client=None, max_goals: int = 50) -> Dict[str, Any]:
        """
        Extract goals from vector stores.
        
        Args:
            ollama_client: Optional OllamaClient instance
            max_goals: Maximum number of goals to extract (default 50)
            
        Returns:
            Dictionary with statistics about the goal extraction process
        """
        start_time = datetime.now()
        
        # Stats dictionary to track results
        stats = {
            "goals_extracted": 0,
            "content_samples": 0,
            "stores_analyzed": [],
            "start_time": start_time.isoformat(),
            "duration_seconds": 0
        }
        
        try:
            # Use provided client or create a new one
            if ollama_client is None:
                from scripts.llm.ollama_client import OllamaClient
                ollama_client = OllamaClient()
            
            logger.info("Starting goal extraction from vector stores")
            
            # 1. Gather content from vector stores using diversity queries
            content_samples = await self._gather_content_samples(ollama_client)
            stats["content_samples"] = len(content_samples)
            
            if not content_samples:
                logger.warning("No content samples found in vector stores")
                return {
                    "status": "warning",
                    "message": "No content samples found in vector stores",
                    "stats": stats
                }
            
            # 2. Extract goals from content samples (with configurable limit)
            # Goals are now saved incrementally during extraction, not just at the end
            goals = await self._extract_goals_with_llm(content_samples, ollama_client, max_goals)
            stats["goals_extracted"] = len(goals)
            
            if goals:
                logger.info(f"Successfully extracted and saved {len(goals)} goals")
            else:
                logger.warning("No goals were extracted from content")
            
            # Calculate duration
            stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()
            
            return {
                "status": "success",
                "message": f"Extracted {len(goals)} goals from vector stores",
                "stats": stats,
                "goals": goals
            }
            
        except Exception as e:
            logger.error(f"Error extracting goals: {e}", exc_info=True)
            
            stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()
            stats["error"] = str(e)
            
            return {
                "status": "error",
                "message": f"Error extracting goals: {str(e)}",
                "stats": stats
            }
    
    async def _gather_content_samples(self, ollama_client) -> List[Dict[str, Any]]:
        """
        Gather diverse content samples from vector stores to extract goals.
        This implementation properly queries individual stores including topic stores.
        """
        content_samples = []
        all_stores_searched = set()
        
        # Define diverse queries to get a wide range of content
        queries = [
            "What are the main goals mentioned in this knowledge base?",
            "What educational objectives are important in this domain?",
            "What learning outcomes are valued in this field?",
            "What are people trying to achieve in this area?",
            "What problems are people working to solve?",
            "What are the aspirations mentioned in these materials?",
            "What are the key targets mentioned in this content?",
            "What achievements are valued according to these resources?",
        ]
        
        # Get content from standard stores first
        standard_stores = ["content_store", "reference_store"]
        
        # First search the standard stores using the regular context method
        # This ensures we get a baseline of content with the established hybrid approach
        logger.info("Gathering content from standard stores (reference and content)")
        for query in queries[:4]:  # Use first 4 queries for standard stores to avoid redundancy
            try:
                # Use standard context retrieval which searches both stores
                context = await ollama_client.get_relevant_context(query=query, k=5)
                
                if context and len(context) > 100:
                    content_samples.append({
                        "query": query,
                        "store_type": "combined_stores",
                        "content": context
                    })
                    logger.info(f"Retrieved content for query: {query[:40]}...")
            except Exception as e:
                logger.error(f"Error querying with standard method: {e}")
        
        # Now look for topic stores
        try:
            # Check if the client has the topic_stores attribute
            if hasattr(ollama_client, 'topic_stores'):
                topic_stores = ollama_client.topic_stores
                
                # Skip checking if topic_stores is empty or None
                if not topic_stores:
                    logger.info("No topic stores available in client")
                else:
                    # Only try to access topic stores that exist and are loaded
                    available_topic_stores = []
                    for name, store in topic_stores.items():
                        if store is not None:
                            available_topic_stores.append(name)
                    
                    if available_topic_stores:
                        logger.info(f"Found {len(available_topic_stores)} loaded topic stores: {', '.join(available_topic_stores)}")
                        
                        # For each available topic store, get content
                        for store_name in available_topic_stores:
                            # Extract topic name for more targeted queries
                            topic_name = store_name.replace("topic_", "").replace("-", " ")
                            topic_queries = [
                                f"What are the learning goals related to {topic_name}?",
                                f"What skills should students develop about {topic_name}?",
                                f"What are the key educational objectives for {topic_name}?"
                            ]
                            
                            # Only try one query per topic to avoid too many warnings
                            selected_query = topic_queries[0]
                            try:
                                if hasattr(ollama_client, 'get_relevant_context_from_store'):
                                    context = await ollama_client.get_relevant_context_from_store(
                                        query=selected_query,
                                        store_name=store_name,
                                        k=3
                                    )
                                    
                                    if context and len(context) > 100:
                                        content_samples.append({
                                            "query": selected_query,
                                            "store_type": store_name,
                                            "content": context,
                                            "is_topic": True
                                        })
                                        all_stores_searched.add(store_name)
                                        logger.info(f"Retrieved content from topic store '{store_name}'")
                            except Exception as e:
                                logger.warning(f"Error querying topic store '{store_name}': {e}")
                    else:
                        logger.info("No properly loaded topic stores found")
            else:
                logger.info("Client does not support topic stores")
                
        except Exception as e:
            logger.error(f"Error processing topic stores: {e}")
        
        # Update stats
        all_stores_searched.update(standard_stores)
        logger.info(f"Gathered {len(content_samples)} content samples from {len(all_stores_searched)} stores")
        
        # If we didn't get enough samples, try a more aggressive search on content_store
        if len(content_samples) < 5:
            logger.warning(f"Only found {len(content_samples)} samples, trying a more aggressive search")
            try:
                # Use a more general query with higher k
                context = await ollama_client.get_relevant_context(
                    query="What is this knowledge base about?", 
                    k=10
                )
                
                if context and len(context) > 100:
                    content_samples.append({
                        "query": "General knowledge base content",
                        "store_type": "combined_stores",
                        "content": context
                    })
                    logger.info("Added fallback content sample")
            except Exception as e:
                logger.error(f"Error with fallback query: {e}")
        
        return content_samples
            
    async def _extract_goals_with_llm(self, content_samples: List[Dict[str, Any]], ollama_client, max_goals: int = 50) -> List[Dict[str, Any]]:
        """
        Extract goals from content samples using Mistral LLM.
        Limits to a configurable maximum number of most important goals.
        """
        goals = []
        extracted_goal_texts = set()  # To avoid duplicates
        max_goals_limit = max_goals  # Use the passed parameter instead of hardcoded value
        
        logger.info(f"🎯 GOAL EXTRACTION: Processing {len(content_samples)} content samples")
        logger.info(f"🔢 PHASE LIMIT: Will extract maximum {max_goals_limit} goals")
        
        # Process content samples in batches
        batch_size = 3  # Process this many samples at once
        
        # This loop is the key to determining completion - it processes ALL content samples
        for i in range(0, len(content_samples), batch_size):
            batch = content_samples[i:i+batch_size]
            
            # Process each sample in the batch
            batch_tasks = []
            for sample in batch:
                task = self._process_content_sample(sample, ollama_client)
                batch_tasks.append(task)
                
            # Process batch concurrently
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results and add new goals
            batch_goals = []
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error processing content sample: {result}")
                    continue
                    
                # Add non-duplicate goals
                for goal in result:
                    # Check if this is a duplicate goal text
                    if goal["goal_text"] not in extracted_goal_texts:
                        goals.append(goal)
                        batch_goals.append(goal)
                        extracted_goal_texts.add(goal["goal_text"])
            
            # Save goals after each batch is processed
            if batch_goals:
                logger.info(f"Saving batch of {len(batch_goals)} goals to CSV")
                self._save_goals_to_csv(batch_goals)
                
            # If we've collected more than max_goals_limit, keep only the most important ones
            if len(goals) > max_goals_limit:
                logger.info(f"Collected {len(goals)} goals, keeping only the top {max_goals_limit} by importance")
                # Sort goals by importance (high to low)
                goals.sort(key=lambda g: g["importance"], reverse=True)
                # Keep only the top max_goals_limit
                goals = goals[:max_goals_limit]
                # Update the extracted_goal_texts set to match the current goals
                extracted_goal_texts = {goal["goal_text"] for goal in goals}
        
        # After the for loop completes, ALL batches have been processed
        # The function returns goals, indicating the process is complete
        goals.sort(key=lambda g: g["importance"], reverse=True)
        logger.info(f"🎯 GOAL EXTRACTION COMPLETE: {len(goals)} goals extracted (limited to maximum of {max_goals_limit})")
        return goals

    async def _process_content_sample(self, sample: Dict[str, Any], ollama_client) -> List[Dict[str, Any]]:
        """
        Process a single content sample to extract goals.
        """
        try:
            # Use the GoalLLM implementation directly
            from scripts.analysis.goal_llm import GoalLLM
            
            # Create the GoalLLM instance if we don't have one
            if not hasattr(self, 'goal_llm'):
                self.goal_llm = GoalLLM()
                logger.info("Created GoalLLM instance for goal extraction")
            
            # Extract goals using the dedicated goal_llm
            try:
                goals = self.goal_llm.extract_goals(
                    content=sample['content'],
                    store_type=sample['store_type']
                )
                
                if goals:
                    logger.info(f"Extracted {len(goals)} goals from {sample['store_type']} content")
                    return goals
                else:
                    # Fallback to async method if direct extraction fails
                    logger.info("Direct goal extraction failed, trying with async method")
                    return await self._extract_goals_with_async_api(sample, ollama_client)
            except Exception as extraction_err:
                logger.error(f"Error in goal extraction: {extraction_err}")
                # Fallback to async method if direct extraction fails
                return await self._extract_goals_with_async_api(sample, ollama_client)
        
        except Exception as e:
            logger.error(f"Error processing content sample: {e}")
            return []
    
    async def _extract_goals_with_async_api(self, sample: Dict[str, Any], ollama_client) -> List[Dict[str, Any]]:
        """
        Fallback method to extract goals using async API when direct extraction fails.
        """
        try:
            # Use the async chat_completion_async method from GoalLLM
            from scripts.analysis.goal_llm import GoalLLM
            
            # Create the GoalLLM instance if we don't have one
            if not hasattr(self, 'goal_llm'):
                self.goal_llm = GoalLLM()
                logger.info("Created GoalLLM instance for async goal extraction")
            
            # Prepare prompt for goal extraction
            prompt = f"""
You are an educational goals analyst. Your task is to extract clear, actionable learning goals from the content below.

Content from {sample['store_type']}:
{sample['content']}

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
"""
            # Call the LLM to extract goals
            response = await self.goal_llm.chat_completion_async(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.2
            )
            
            # Extract the response text
            if response and "message" in response:
                response_text = response["message"]["content"]
            else:
                logger.warning("Unexpected response format from LLM")
                return []
            
            # Parse the JSON response
            # First, try to find JSON array in the response
            import re
            json_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                goals = json.loads(json_str)
            else:
                # If no clear JSON found, try parsing the whole response
                try:
                    goals = json.loads(response_text)
                except:
                    logger.warning(f"Could not parse JSON from response: {response_text[:100]}...")
                    return []
            
            # Validate and clean up each goal
            valid_goals = []
            for goal in goals:
                if isinstance(goal, dict) and "goal_text" in goal and "importance" in goal:
                    # Ensure importance is a number between 1-10
                    try:
                        importance = float(goal["importance"])
                        if importance < 1:
                            importance = 1
                        elif importance > 10:
                            importance = 10
                        goal["importance"] = importance
                    except:
                        goal["importance"] = 5  # Default if not parseable
                        
                    # Add store type as additional metadata
                    goal["store_type"] = sample["store_type"]
                    goal["extracted_at"] = datetime.now().isoformat()
                    
                    valid_goals.append(goal)
            
            return valid_goals
            
        except Exception as e:
            logger.error(f"Error in async goal extraction: {e}")
            return []
    
    def _save_goals_to_csv(self, goals: List[Dict[str, Any]]) -> bool:
        """
        Save extracted goals to a CSV file.
        Ensures a maximum of 50 goals are saved based on importance.
        
        Args:
            goals: List of goal dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        try:
            max_goals = 50  # Maximum number of goals to keep
            
            # Create a DataFrame with the goals
            df = pd.DataFrame(goals)
            
            # Ensure output directory exists
            self.data_folder.mkdir(parents=True, exist_ok=True)
            
            # Define CSV path
            csv_path = self.data_folder / "goals.csv"
            
            # If file exists, merge with existing goals
            if csv_path.exists():
                try:
                    existing_df = pd.read_csv(csv_path)
                    
                    # Check if we already have 'goal_text' column to avoid duplicates
                    if "goal_text" in existing_df.columns:
                        # Get existing goal texts
                        existing_goals = set(existing_df["goal_text"])
                        
                        # Filter out goals we already have
                        new_goals = df[~df["goal_text"].isin(existing_goals)]
                        
                        if len(new_goals) > 0:
                            # Add goal_id to new goals if it doesn't exist
                            if 'goal_id' not in new_goals.columns:
                                # Calculate the next goal_id based on existing data
                                next_id = 0
                                if 'goal_id' in existing_df.columns:
                                    # Extract numeric part of existing IDs and find max
                                    existing_ids = existing_df['goal_id'].astype(str)
                                    numeric_ids = [int(id.replace('goal_', '')) for id in existing_ids if id.startswith('goal_')]
                                    if numeric_ids:
                                        next_id = max(numeric_ids) + 1
                                
                                # Generate sequential goal IDs
                                new_goals['goal_id'] = [f'goal_{next_id + i}' for i in range(len(new_goals))]
                            
                            # Combine existing and new goals
                            combined_df = pd.concat([existing_df, new_goals], ignore_index=True)
                            
                            # If we have more than max_goals, keep only the most important ones
                            if len(combined_df) > max_goals:
                                logger.info(f"Combined goals exceeds limit of {max_goals}, keeping only the most important ones")
                                # Ensure 'importance' column is a number for proper sorting
                                combined_df['importance'] = pd.to_numeric(combined_df['importance'], errors='coerce').fillna(0)
                                # Sort by importance (high to low) and keep top max_goals
                                combined_df = combined_df.sort_values('importance', ascending=False).head(max_goals)
                                
                            combined_df.to_csv(csv_path, index=False)
                            logger.info(f"Added goals - final count: {len(combined_df)} goals (limit: {max_goals})")
                        else:
                            logger.info("No new goals to add to CSV")
                        
                    else:
                        # If existing file doesn't have proper structure, overwrite it
                        # Add goal_id to all goals
                        df['goal_id'] = [f'goal_{i}' for i in range(len(df))]
                        
                        # If we have more than max_goals, keep only the most important ones
                        if len(df) > max_goals:
                            logger.info(f"Number of goals exceeds limit of {max_goals}, keeping only the most important ones")
                            # Sort by importance (high to low) and keep top max_goals
                            df = df.sort_values('importance', ascending=False).head(max_goals)
                            
                        df.to_csv(csv_path, index=False)
                        logger.info(f"Saved {len(df)} goals to CSV (overwriting improperly formatted file)")
                        
                except Exception as e:
                    logger.error(f"Error merging with existing goals CSV: {e}")
                    # Fall back to overwriting the file
                    # Add goal_id to all goals
                    df['goal_id'] = [f'goal_{i}' for i in range(len(df))]
                    
                    # If we have more than max_goals, keep only the most important ones
                    if len(df) > max_goals:
                        logger.info(f"Number of goals exceeds limit of {max_goals}, keeping only the most important ones")
                        # Sort by importance (high to low) and keep top max_goals
                        df = df.sort_values('importance', ascending=False).head(max_goals)
                        
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Saved {len(df)} goals to CSV (overwriting after error)")
            else:
                # Save to new file
                # Add goal_id to all goals
                df['goal_id'] = [f'goal_{i}' for i in range(len(df))]
                
                # If we have more than max_goals, keep only the most important ones
                if len(df) > max_goals:
                    logger.info(f"Number of goals exceeds limit of {max_goals}, keeping only the most important ones")
                    # Sort by importance (high to low) and keep top max_goals
                    df = df.sort_values('importance', ascending=False).head(max_goals)
                    
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved {len(df)} goals to new CSV file (limit: {max_goals})")
                
            return True
                
        except Exception as e:
            logger.error(f"Error saving goals to CSV: {e}")
            return False
