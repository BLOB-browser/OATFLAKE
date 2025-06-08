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
            total_texts = len(texts)
            embeddings = []
            
            # Check if we need more detailed progress logging
            high_volume = total_texts > 500
            log_interval = max(self.batch_size * 5, 100) if high_volume else self.batch_size
            
            # Use a standard effective batch size for consistency
            effective_batch_size = 30 if high_volume else self.batch_size
            
            # Track metrics for this batch
            cache_hits = 0
            cache_misses = 0
            saved_time = 0  # estimated time saved in seconds
            
            # Process in batches
            for i in range(0, len(texts), effective_batch_size):
                # Start batch timing
                batch_start_time = datetime.now()
                
                batch = texts[i:i + effective_batch_size]
                batch_embeddings = []
                texts_to_embed = []
                cache_indices = []
                
                # Check cache for each text in the batch
                for j, text in enumerate(batch):
                    # Try to get embedding from cache
                    cached_embedding = self.cache.get(text)
                    if cached_embedding:
                        batch_embeddings.append(cached_embedding)
                        cache_hits += 1
                        # Add None as a placeholder to maintain alignment with batch indices
                        texts_to_embed.append(None)
                    else:
                        # Need to generate embedding
                        batch_embeddings.append(None)  # Placeholder
                        texts_to_embed.append(text)
                        cache_indices.append(j)
                        cache_misses += 1
                
                # Generate embeddings for texts not found in cache
                if texts_to_embed and any(text is not None for text in texts_to_embed):
                    texts_to_actually_embed = [text for text in texts_to_embed if text is not None]
                    logger.debug(f"Generating {len(texts_to_actually_embed)} embeddings not found in cache")
                    
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        tasks = []
                        for text in texts_to_actually_embed:
                            tasks.append(
                                client.post(
                                    f"{self.base_url}/api/embeddings",
                                    json={
                                        "model": self.model_name,
                                        "prompt": text,
                                    }
                                )
                            )
                        responses = await asyncio.gather(*tasks)
                        
                        # Process generated embeddings
                        generated_embeddings = []
                        for response in responses:
                            if response.status_code == 200:
                                result = response.json()
                                if 'embedding' in result:
                                    generated_embeddings.append(result['embedding'])
                                else:
                                    logger.error(f"Missing embedding in response: {result}")
                                    generated_embeddings.append([0.0] * self.dimension)
                            else:
                                logger.error(f"Embedding failed: {response.text}")
                                generated_embeddings.append([0.0] * self.dimension)
                          # Update cache with new embeddings
                        non_none_texts = [text for text in texts_to_actually_embed if text is not None]
                        for idx, (text, embedding) in enumerate(zip(non_none_texts, generated_embeddings)):
                            self.cache.set(text, embedding)
                        
                        # Merge generated embeddings with cached ones
                        generated_idx = 0
                        for j, embedding in enumerate(batch_embeddings):
                            if embedding is None:
                                batch_embeddings[j] = generated_embeddings[generated_idx]
                                generated_idx += 1
                    
                embeddings.extend(batch_embeddings)
                
                # Calculate batch processing time for performance metrics
                batch_end_time = datetime.now()
                batch_duration = (batch_end_time - batch_start_time).total_seconds()
                batch_text_len = sum(len(t) for t in batch if t)
                
                # Estimate time saved through caching for this batch
                if cache_hits > 0:
                    # Assume each embedding would have taken the same time as this batch per item
                    avg_time_per_miss = batch_duration / max(1, cache_misses) if cache_misses > 0 else 0.1  # seconds
                    time_saved_this_batch = cache_hits * avg_time_per_miss
                    saved_time += time_saved_this_batch
                
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
                    logger.info(f"Cache performance: {cache_hits} hits, {cache_misses} misses, est. {saved_time:.2f}s saved")
                
            # Log overall stats
            cache_stats = self.cache.stats()
            logger.info(f"Embedding generation complete - Cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses, {cache_stats['size']}/{cache_stats['max_size']} entries")
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
