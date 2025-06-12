#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import logging
from pathlib import Path

# Add the project directory to the path so we can import from scripts
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def debug_level_status():
    """Debug the current level status and URL filtering"""
    try:
        from scripts.analysis.url_storage import URLStorageManager
        from scripts.analysis.level_processor import LevelBasedProcessor
        from utils.config import get_data_path
        import json
        
        # Get data path
        data_path = get_data_path()
        print(f"Data path: {data_path}")
        
        # Load config to check current process level
        config_path = Path(__file__).parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            current_level = config.get('current_process_level', 2)
            max_depth = config.get('crawl_config', {}).get('max_depth', 4)
            print(f"Current process level from config: {current_level}")
            print(f"Max depth: {max_depth}")
        else:
            print("Config file not found")
            return
        
        # Initialize URL storage 
        processed_urls_file = os.path.join(data_path, "processed_urls.csv")
        url_storage = URLStorageManager(processed_urls_file)
        
        # Initialize level processor
        level_processor = LevelBasedProcessor(data_path)
        
        print("\n=== URL STORAGE STATUS ===")
        print(f"URL storage file: {processed_urls_file}")
        print(f"URL storage discovery mode: {getattr(url_storage, 'discovery_mode', 'Not set')}")
        print(f"URL storage rediscovery mode: {getattr(url_storage, 'rediscovery_mode', 'Not set')}")
        
        print("\n=== LEVEL STATUS (from LevelBasedProcessor) ===")
        level_status = level_processor.get_level_status()
        for level, status in level_status.items():
            print(f"Level {level}: {status['pending']} pending URLs, Complete: {status['is_complete']}")
        
        print("\n=== DETAILED URL COUNTS BY LEVEL (Discovery Mode) ===")
        url_storage.set_discovery_mode(True)
        for level in range(1, 6):
            urls = url_storage.get_pending_urls(depth=level)
            print(f"Level {level} (discovery mode): {len(urls)} URLs")
            if urls and level <= 3:  # Show first few URLs for levels 1-3
                for i, url_info in enumerate(urls[:3]):
                    if isinstance(url_info, dict):
                        url = url_info.get('url', url_info)
                    else:
                        url = url_info
                    print(f"  {i+1}. {url}")
                if len(urls) > 3:
                    print(f"  ... and {len(urls) - 3} more")
        
        print("\n=== DETAILED URL COUNTS BY LEVEL (Analysis Mode) ===")
        url_storage.set_discovery_mode(False)
        for level in range(1, 6):
            urls = url_storage.get_pending_urls(depth=level)
            print(f"Level {level} (analysis mode): {len(urls)} URLs")
            if urls and level <= 3:  # Show first few URLs for levels 1-3
                for i, url_info in enumerate(urls[:3]):
                    if isinstance(url_info, dict):
                        url = url_info.get('url', url_info)
                    else:
                        url = url_info
                    print(f"  {i+1}. {url}")
                if len(urls) > 3:
                    print(f"  ... and {len(urls) - 3} more")
        
        print(f"\n=== RECOMMENDATION ===")
        if current_level == 2:
            # Check if level 2 has pending URLs in analysis mode
            url_storage.set_discovery_mode(False)
            level_2_urls = url_storage.get_pending_urls(depth=2)
            
            if level_2_urls:
                print(f"✅ Level 2 has {len(level_2_urls)} pending URLs ready for processing")
                print("   Next step: Process these URLs")
            else:
                print("❌ Level 2 has no pending URLs in analysis mode")
                # Check discovery mode
                url_storage.set_discovery_mode(True)
                level_2_discovery_urls = url_storage.get_pending_urls(depth=2)
                if level_2_discovery_urls:
                    print(f"   But level 2 has {len(level_2_discovery_urls)} URLs in discovery mode")
                    print("   Issue: URL filtering is preventing analysis mode from seeing URLs")
                    print("   Solution: Fix URL storage filtering logic")
                else:
                    print("   Level 2 complete - should advance to level 3")
                    print(f"   Next step: Update config.json current_process_level from {current_level} to {current_level + 1}")
        
    except Exception as e:
        logger.error(f"Error debugging level status: {e}", exc_info=True)

if __name__ == "__main__":
    debug_level_status()
