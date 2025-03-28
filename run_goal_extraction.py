#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import os
from pathlib import Path
from scripts.analysis.goal_extractor import GoalExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def main():
    # Get data path from config.json
    config_path = Path("config.json")
    if not config_path.exists():
        logger.error("Config file not found. Please create a config.json file with a 'data_path' key.")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_path = config.get('data_path')
    if not data_path:
        logger.error("No data_path specified in config.json")
        return
    
    logger.info(f"Starting goal extraction with data path: {data_path}")
    logger.info("Goals will be saved incrementally as they are extracted")
    
    # Initialize the client explicitly so we can check for topic stores
    from scripts.llm.ollama_client import OllamaClient
    client = OllamaClient()
    
    # Check if any topic stores were loaded (with better error handling)
    topic_stores = getattr(client, 'topic_stores', {})
    if (topic_stores):
        # Count only properly loaded stores
        valid_stores = {}
        for name, store in topic_stores.items():
            if store is not None and hasattr(store, 'docstore'):
                doc_count = len(store.docstore._dict) if hasattr(store.docstore, '_dict') else 0
                if doc_count > 0:
                    valid_stores[name] = doc_count
        
        if valid_stores:
            logger.info(f"Found {len(valid_stores)} valid topic stores: {', '.join(valid_stores.keys())}")
            for name, count in valid_stores.items():
                # Format topic name for better readability in logs
                topic_name = name.replace('topic_', '').replace('_', ' ').title()
                logger.info(f"  - {topic_name}: {count} documents")
        else:
            logger.info("No valid topic stores found (stores exist but contain no documents)")
    else:
        logger.info("No topic stores found - extraction will only use reference and content stores")
    
    # Create extractor and run extraction with our initialized client
    extractor = GoalExtractor(data_path)
    result = await extractor.extract_goals(ollama_client=client)
    
    # Display results
    stats = result.get("stats", {})
    goals = result.get("goals", [])
    
    logger.info(f"Extraction completed: {result.get('status', 'unknown')}")
    logger.info(f"Total goals extracted: {stats.get('goals_extracted', 0)}")
    logger.info(f"Content samples analyzed: {stats.get('content_samples', 0)}")
    logger.info(f"Duration: {stats.get('duration_seconds', 0):.2f} seconds")
    logger.info(f"Goals have been saved to {data_path}/goals.csv")
    
    # Display stats by store type
    store_stats = {}
    topic_goals = 0
    
    for goal in goals:
        store = goal.get("store_type", "unknown")
        store_stats[store] = store_stats.get(store, 0) + 1
        if store.startswith("topic_"):
            topic_goals += 1
    
    logger.info("\nGoals by store type:")
    for store, count in store_stats.items():
        logger.info(f"  - {store}: {count} goals")
    
    if topic_goals > 0:
        logger.info(f"Topic stores contributed {topic_goals} goals ({(topic_goals/len(goals))*100:.1f}% of total)")
    
    if goals:
        logger.info("\nTop 5 Goals (by importance):")
        for i, goal in enumerate(goals[:5], 1):
            logger.info(f"{i}. {goal['goal_text']} (Importance: {goal['importance']:.1f})")
            logger.info(f"   Topic: {goal.get('topic', 'Unknown')}")
            logger.info(f"   Justification: {goal.get('justification', 'None provided')}")
            logger.info("")
    
    # Save results to JSON for reference
    output_path = Path(data_path) / "goal_extraction_results.json"
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
