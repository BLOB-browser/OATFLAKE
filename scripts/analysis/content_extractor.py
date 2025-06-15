#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import json
import aiohttp
import asyncio
from typing import List, Dict, Any
from pathlib import Path
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

async def extract_content_for_urls(processed_urls: List[Dict], data_folder: str) -> List[Path]:
    """
    Extract content from processed URLs and prepare it for vector store generation.
    
    Args:
        processed_urls: List of dictionaries containing URL data
        data_folder: Path to data folder for storing extracted content
        
    Returns:
        List of paths to content files
    """
    logger.info(f"Extracting content from {len(processed_urls)} processed URLs")
    
    # Create a temporary directory for content files
    from scripts.storage.temporary_storage_service import TemporaryStorageService
    temp_storage = TemporaryStorageService(data_folder)
    
    # Initialize content file path
    content_file = Path(temp_storage.get_temp_path()) / "url_content.jsonl"
    
    # Set to track URLs we've already processed
    processed_url_ids = set()
    
    # Process URLs in batches
    batch_size = 10
    total_urls = len(processed_urls)
    content_entries = []
    
    # Process in batches asynchronously
    for i in range(0, total_urls, batch_size):
        batch = processed_urls[i:i+batch_size]
        logger.info(f"Processing URL batch {i//batch_size + 1}/{(total_urls-1)//batch_size + 1}")
        
        batch_entries = await fetch_batch_content(batch)
        content_entries.extend(batch_entries)
        
        # Write batch to file to avoid keeping everything in memory
        await write_content_batch(content_entries, content_file)
        content_entries = []  # Reset after writing
    
    # Return the path to the content file
    return [content_file]

async def fetch_batch_content(batch: List[Dict]) -> List[Dict]:
    """
    Fetch content for a batch of URLs asynchronously
    
    Args:
        batch: List of URL dictionaries
        
    Returns:
        List of content entry dictionaries
    """
    async def fetch_url(url_data):
        url = url_data.get('origin_url')
        if not url:
            return None
            
        try:
            # Create a basic async HTTP client
            async with aiohttp.ClientSession() as session:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Extract text using BeautifulSoup
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script and style elements
                        for script_or_style in soup(['script', 'style']):
                            script_or_style.extract()
                            
                        # Extract text
                        text = soup.get_text(separator=' ', strip=True)
                        
                        # Limit text length
                        if len(text) > 4000:
                            text = text[:4000]
                        
                        # Create content entry
                        return {
                            "url": url,
                            "depth": url_data.get('depth', 0),
                            "origin": url_data.get('origin', ''),
                            "content": text,
                            "type": "url_content"
                        }
                    else:
                        logger.warning(f"Failed to fetch {url}: HTTP status {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    # Process all URLs in the batch concurrently
    tasks = [fetch_url(url_data) for url_data in batch]
    results = await asyncio.gather(*tasks)
    
    # Filter out None results and return valid content entries
    return [result for result in results if result]

async def write_content_batch(entries: List[Dict], file_path: Path):
    """
    Write content entries to a JSONL file
    
    Args:
        entries: List of content entry dictionaries
        file_path: Path to the output file
    """
    if not entries:
        return
        
    # Ensure the directory exists
    os.makedirs(file_path.parent, exist_ok=True)
    
    # Open in append mode to add to existing file if it exists
    mode = 'a' if file_path.exists() else 'w'
    
    with open(file_path, mode, encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    logger.info(f"Added {len(entries)} content entries to {file_path}")
