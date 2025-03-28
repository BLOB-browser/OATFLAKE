from typing import List
import httpx
import numpy as np
from langchain.embeddings.base import Embeddings
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class OllamaEmbeddings(Embeddings):
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model_name: str = "nomic-embed-text",  # Use nomic-embed-text as default
        batch_size: int = 10,  # Default batch size (will be adjusted based on system)
        dimension: int = 768,  # nomic-embed-text dimension
        timeout: float = 3600.0  # Default timeout in seconds (very high to ensure completion)
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimension = dimension
        self.timeout = timeout  # Store timeout as instance variable
        self._http_client = httpx.AsyncClient(timeout=timeout)
        logger.info(f"Initialized OllamaEmbeddings with model: {model_name}, batch_size: {batch_size}, timeout: {timeout}s")

    async def aembeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        try:
            # Filter out empty texts to avoid errors
            filtered_texts = [text for text in texts if text and text.strip()]
            if len(filtered_texts) < len(texts):
                logger.warning(f"Filtered out {len(texts) - len(filtered_texts)} empty texts")
                
            if not filtered_texts:
                logger.error("No valid texts to embed after filtering")
                # Return a dummy embedding to avoid crashes
                return [[0.0] * self.dimension]
                
            # Continue with filtered texts
            texts = filtered_texts
            embeddings = []
            total_texts = len(texts)
            logger.info(f"Generating embeddings for {total_texts} texts using {self.model_name}")
            
            # Check if we need more detailed progress logging
            high_volume = total_texts > 500
            log_interval = max(self.batch_size * 5, 100) if high_volume else self.batch_size
            
            # Use a standard effective batch size for consistency
            effective_batch_size = 30 if high_volume else self.batch_size
            logger.info(f"Using batch size {effective_batch_size} for embeddings ({total_texts} texts)")
            
            # Process in batches
            for i in range(0, len(texts), effective_batch_size):
                # Start batch timing
                batch_start_time = datetime.now()
                
                batch = texts[i:i + effective_batch_size]
                batch_embeddings = []
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    tasks = []
                    for text in batch:
                        tasks.append(
                            client.post(
                                f"{self.base_url}/api/embeddings",
                                json={
                                    "model": self.model_name,
                                    "prompt": text,  # No need for formatting with nomic-embed-text
                                }
                            )
                        )
                    responses = await asyncio.gather(*tasks)
                    
                    for response in responses:
                        if response.status_code == 200:
                            result = response.json()
                            if 'embedding' in result:
                                batch_embeddings.append(result['embedding'])
                            else:
                                logger.error(f"Missing embedding in response: {result}")
                                batch_embeddings.append([0.0] * self.dimension)
                        else:
                            logger.error(f"Embedding failed: {response.text}")
                            batch_embeddings.append([0.0] * self.dimension)
                            
                embeddings.extend(batch_embeddings)
                # Calculate batch processing time for performance metrics
                batch_end_time = datetime.now()
                batch_duration = (batch_end_time - batch_start_time).total_seconds()
                batch_text_len = sum(len(t) for t in batch)
                
                # Log progress less frequently for large batches to reduce log spam
                if i % log_interval == 0 or i + effective_batch_size >= total_texts:
                    time_per_item = batch_duration / len(batch) if len(batch) > 0 else 0
                    chars_per_second = batch_text_len / batch_duration if batch_duration > 0 else 0
                    
                    # Estimate remaining time
                    remaining_items = total_texts - (i + len(batch))
                    estimated_remaining_seconds = remaining_items * time_per_item
                    
                    # Format as hours:minutes:seconds
                    hours, remainder = divmod(estimated_remaining_seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    eta = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                    
                    logger.info(f"Processed batch of {len(batch)} texts ({i + len(batch)}/{total_texts}) - {((i + len(batch))/total_texts*100):.1f}% complete")
                    logger.info(f"Batch stats: {time_per_item:.2f}s/item, {chars_per_second:.1f} chars/sec, ETA: {eta}")
                
            logger.info(f"Successfully generated {len(embeddings)} embeddings (dimension: {len(embeddings[0]) if embeddings else self.dimension})")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            return [[0.0] * self.dimension] * len(texts)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding for multiple documents"""
        # Filter out empty texts to avoid errors
        filtered_texts = [text for text in texts if text and text.strip()]
        if len(filtered_texts) < len(texts):
            logger.warning(f"Filtered out {len(texts) - len(filtered_texts)} empty texts")
            
        if not filtered_texts:
            logger.error("No valid texts to embed after filtering")
            # Return a dummy embedding to avoid crashes
            return [[0.0] * self.dimension]
            
        # Handle running event loop better
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # We're in an event loop, so we need a different approach
            # This requires Python 3.11+ for asyncio.Runner
            logger.warning("Running embed_documents inside an existing event loop - creating new one")
            with asyncio.Runner() as runner:
                return runner.run(self.aembeddings(filtered_texts))
        except RuntimeError:
            # No running event loop, we can use the traditional approach
            return asyncio.run(self.aembeddings(filtered_texts))

    def embed_query(self, text: str) -> List[float]:
        """Synchronous embedding for a single query"""
        # Handle running event loop better
        try:
            # Check if we're already in an event loop
            loop = asyncio.get_running_loop()
            # We're in an event loop, so we need a different approach
            logger.warning("Running embed_query inside an existing event loop - creating new one")
            with asyncio.Runner() as runner:
                results = runner.run(self.aembeddings([text]))
                return results[0]
        except RuntimeError:
            # No running event loop, we can use the traditional approach
            results = asyncio.run(self.aembeddings([text]))
            return results[0]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._http_client.aclose()
