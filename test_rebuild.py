#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import logging
import sys
import time
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

async def test_rebuild_indexes():
    """Test all vector store rebuild approaches to ensure consistency."""
    try:
        # Get data path from config.json
        config_path = Path("config.json")
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                data_path = Path(config.get('data_path', ''))
        else:
            logger.error("Config file not found")
            return False

        logger.info(f"Using data path: {data_path}")
        
        # Measure execution time
        start_time = time.time()
        
        # STEP 1: Check initial state to compare later
        vector_path = data_path / "vector_stores" / "default"
        initial_stores = []
        
        if vector_path.exists():
            # Count all store directories
            for store_dir in vector_path.glob("*"):
                if store_dir.is_dir():
                    initial_stores.append(store_dir.name)
            
            logger.info(f"Initial state: Found {len(initial_stores)} stores: {initial_stores}")
        
        # STEP 2: Test JSONL-based approach (same as in knowledge.py)
        logger.info("TESTING JSONL-BASED VECTOR GENERATION")
        logger.info("=====================================")
        
        jsonl_result = None
        try:
            # Import required components
            from scripts.analysis.vector_generator import VectorGenerator
            from scripts.storage.content_storage_service import ContentStorageService
            
            # Find JSONL content files in same locations as knowledge.py
            content_storage = ContentStorageService(str(data_path))
            content_paths = list(content_storage.temp_storage_path.glob("*.jsonl"))
            
            # Check temp directory
            temp_dir = data_path / "temp"
            if temp_dir.exists():
                content_paths.extend(list(temp_dir.glob("*.jsonl")))
                
            # Check content directory
            content_dir = data_path / "content"
            if content_dir.exists():
                content_paths.extend(list(content_dir.glob("*.jsonl")))
            
            if content_paths:
                logger.info(f"Found {len(content_paths)} JSONL files for processing")
                for path in content_paths:
                    logger.info(f"  - {path}")
                
                # Use VectorGenerator to process JSONL files
                vector_generator = VectorGenerator(str(data_path))
                jsonl_result = await vector_generator.generate_vector_stores(content_paths)
                
                logger.info(f"JSONL-based approach results:")
                logger.info(f"  - Status: {jsonl_result.get('status')}")
                logger.info(f"  - Documents processed: {jsonl_result.get('documents_processed', 0)}")
                logger.info(f"  - Stores created: {jsonl_result.get('stores_created', [])}")
                logger.info(f"  - Topic stores: {jsonl_result.get('topic_stores_created', [])}")
            else:
                logger.info("No JSONL files found for processing")
        except Exception as e:
            logger.error(f"Error in JSONL-based approach: {e}")
        
        # STEP 3: Test CSV-based approach (using ProcessingManager)
        logger.info("TESTING CSV-BASED VECTOR REBUILD")
        logger.info("===============================")
        
        csv_result = None
        try:
            # Import ProcessingManager
            from scripts.data.processing_manager import ProcessingManager
            processing_manager = ProcessingManager(data_path)
            
            # Rebuild all indexes from CSV files
            logger.info("Starting index rebuild using ProcessingManager...")
            csv_result = await processing_manager.rebuild_all_indexes()
            
            # Verify CSV inputs were available
            csv_files = ["definitions.csv", "methods.csv", "projects.csv", "resources.csv", "materials.csv"]
            missing_files = []
            for csv_file in csv_files:
                if not (data_path / csv_file).exists():
                    missing_files.append(csv_file)
            
            if missing_files:
                logger.warning(f"Some source CSV files were not found: {missing_files}")
            else:
                logger.info("✅ All source CSV files were found and should have been processed")
            
            # Check results of CSV-based rebuild
            if csv_result["status"] == "success":
                # Check if required stores are present
                required_stores = ["reference_store", "content_store"]
                rebuilt_stores = csv_result.get('stores_rebuilt', [])
                topic_stores = [store for store in rebuilt_stores if store.startswith("topic_")]
                
                logger.info("✅ Successfully rebuilt indexes using CSV-based approach")
                logger.info(f"Rebuilt {len(rebuilt_stores)} stores ({len(topic_stores)} topic stores)")
                
                # Check for mandatory stores
                missing_required = [store for store in required_stores if store not in rebuilt_stores]
                if missing_required:
                    logger.warning(f"❌ Some required stores were not rebuilt: {missing_required}")
                else:
                    logger.info("✅ All required stores (reference_store, content_store) were rebuilt")
                
                # Check topic stores
                if topic_stores:
                    logger.info(f"✅ Created {len(topic_stores)} topic stores: {topic_stores[:5]}")
                    if len(topic_stores) > 5:
                        logger.info(f"   ...and {len(topic_stores) - 5} more")
                else:
                    logger.warning("⚠️ No topic stores were created during rebuild")
                
                # Show individual store document counts
                logger.info("Document counts by store:")
                for store_name, doc_count in csv_result.get('document_counts', {}).items():
                    logger.info(f"  - {store_name}: {doc_count} documents")
            else:
                logger.error(f"❌ Failed to rebuild indexes: {csv_result.get('message')}")
        except Exception as e:
            logger.error(f"Error in CSV-based approach: {e}")
            csv_result = {"status": "error", "error": str(e)}
        
        # STEP 4: Compare results from both approaches
        logger.info("COMPARING BOTH APPROACHES")
        logger.info("========================")
        
        # Check if both approaches were successful
        jsonl_success = jsonl_result and jsonl_result.get("status") == "success"
        csv_success = csv_result and csv_result.get("status") == "success"
        
        if jsonl_success and csv_success:
            # Compare stores created
            jsonl_stores = set(jsonl_result.get("stores_created", []))
            csv_stores = set(csv_result.get("stores_rebuilt", []))
            
            # Stores in both approaches
            common_stores = jsonl_stores.intersection(csv_stores)
            only_jsonl = jsonl_stores - csv_stores
            only_csv = csv_stores - jsonl_stores
            
            logger.info(f"Stores in both approaches: {len(common_stores)}")
            logger.info(f"Stores only in JSONL approach: {len(only_jsonl)}")
            if only_jsonl:
                logger.info(f"  - {only_jsonl}")
            logger.info(f"Stores only in CSV approach: {len(only_csv)}")
            if only_csv:
                logger.info(f"  - {only_csv}")
            
            # Compare topic stores
            jsonl_topics = set(jsonl_result.get("topic_stores_created", []))
            csv_topics = set([s for s in csv_result.get("stores_rebuilt", []) if s.startswith("topic_")])
            
            common_topics = jsonl_topics.intersection(csv_topics)
            only_jsonl_topics = jsonl_topics - csv_topics
            only_csv_topics = csv_topics - jsonl_topics
            
            logger.info(f"Topic stores in both approaches: {len(common_topics)}")
            logger.info(f"Topic stores only in JSONL approach: {len(only_jsonl_topics)}")
            if only_jsonl_topics:
                logger.info(f"  - {only_jsonl_topics}")
            logger.info(f"Topic stores only in CSV approach: {len(only_csv_topics)}")
            if only_csv_topics:
                logger.info(f"  - {only_csv_topics}")
        elif jsonl_success:
            logger.info("✅ JSONL-based approach succeeded, but CSV-based approach failed")
        elif csv_success:
            logger.info("✅ CSV-based approach succeeded, but JSONL-based approach failed")
        else:
            logger.warning("⚠️ Both approaches failed")
        
        # STEP 5: Verify final state of the vector stores
        logger.info("VERIFYING FINAL VECTOR STORE STATE")
        logger.info("=================================")
        
        # Get final state
        final_stores = []
        if vector_path.exists():
            for store_dir in vector_path.glob("*"):
                if store_dir.is_dir():
                    final_stores.append(store_dir.name)
        
        # Compare with initial state
        new_stores = set(final_stores) - set(initial_stores)
        removed_stores = set(initial_stores) - set(final_stores)
        
        logger.info(f"Final state: {len(final_stores)} stores")
        logger.info(f"New stores created: {len(new_stores)}")
        if new_stores:
            logger.info(f"  - {new_stores}")
        logger.info(f"Stores removed: {len(removed_stores)}")
        if removed_stores:
            logger.info(f"  - {removed_stores}")
        
        # STEP 6: Verify integrity of final stores
        logger.info("VERIFYING STORE INTEGRITY")
        logger.info("========================")
        
        valid_stores = 0
        stores_with_stats = 0
        
        for store_name in final_stores:
            store_path = vector_path / store_name
            store_valid = (store_path / "index.faiss").exists() and (store_path / "documents.json").exists()
            
            if store_valid:
                valid_stores += 1
                
                # Check for embedding stats
                stats_path = store_path / "embedding_stats.json"
                if stats_path.exists():
                    try:
                        with open(stats_path, 'r') as f:
                            stats = json.load(f)
                            if "embedding_count" in stats and "chunk_count" in stats:
                                stores_with_stats += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Error reading stats for {store_name}: {e}")
            else:
                logger.warning(f"⚠️ Store {store_name} is missing required files")
        
        logger.info(f"✅ {valid_stores}/{len(final_stores)} stores have valid file structure")
        logger.info(f"✅ {stores_with_stats}/{len(final_stores)} stores have valid embedding stats")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        logger.info(f"⏱️ Total execution time: {execution_time:.2f} seconds")
        
        # Determine overall success
        overall_success = (jsonl_success or csv_success) and valid_stores > 0
        if overall_success:
            logger.info("🎉 Test completed successfully")
        else:
            logger.error("💥 Test failed")
            
        return overall_success
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    # Run the test
    if asyncio.run(test_rebuild_indexes()):
        sys.exit(0)
    else:
        sys.exit(1)