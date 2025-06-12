#!/usr/bin/env python3
"""
Debug script to check URL cache distribution and filtering
"""

import sys
import os
import csv
from collections import defaultdict

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.analysis.url_storage import URLStorageManager
from utils.config import get_data_path

def analyze_url_distribution():
    """Analyze the distribution of URLs by level and processed status"""
    
    data_folder = get_data_path()
    processed_urls_file = os.path.join(data_folder, "processed_urls.csv")
    pending_urls_file = os.path.join(data_folder, "pending_urls.csv")
    
    print(f"🔍 Analyzing URL distribution")
    print(f"📁 Data folder: {data_folder}")
    print(f"📄 Processed URLs file: {processed_urls_file}")
    print(f"📄 Pending URLs file: {pending_urls_file}")
    print("=" * 60)
    
    # Initialize URL storage manager
    url_storage = URLStorageManager(processed_urls_file)
    
    # Count URLs by level in the raw files
    print("\n📊 RAW FILE ANALYSIS:")
    print("-" * 30)
    
    # Analyze pending URLs file
    pending_by_level = defaultdict(int)
    total_pending_raw = 0
    
    if os.path.exists(pending_urls_file):
        with open(pending_urls_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            print(f"Pending URLs header: {header}")
            
            for row in reader:
                if row and len(row) >= 3:
                    url = row[1] if len(row) > 1 else row[0]  # Handle both old and new schema
                    depth_col = 2 if len(row) > 2 else 1
                    depth = int(row[depth_col]) if row[depth_col].isdigit() else 0
                    
                    pending_by_level[depth] += 1
                    total_pending_raw += 1
        
        print(f"📄 Total pending URLs in file: {total_pending_raw:,}")
        for level in sorted(pending_by_level.keys()):
            print(f"   Level {level}: {pending_by_level[level]:,} URLs")
    
    # Analyze processed URLs file  
    processed_by_level = defaultdict(int)
    total_processed_raw = 0
    
    if os.path.exists(processed_urls_file):
        with open(processed_urls_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            print(f"Processed URLs header: {header}")
            
            for row in reader:
                if row and len(row) >= 2:
                    depth = int(row[1]) if row[1].isdigit() else 0
                    processed_by_level[depth] += 1
                    total_processed_raw += 1
        
        print(f"📄 Total processed URLs in file: {total_processed_raw:,}")
        for level in sorted(processed_by_level.keys()):
            print(f"   Level {level}: {processed_by_level[level]:,} URLs")
    
    print("\n🧠 CACHE ANALYSIS:")
    print("-" * 30)
    
    # Test in discovery mode
    print("\n🔍 DISCOVERY MODE:")
    url_storage.set_discovery_mode(True)
    url_storage.load_pending_urls_cache()
    
    discovery_cache_total = len(url_storage._pending_urls_cache)
    discovery_by_level = defaultdict(int)
    
    for url, metadata in url_storage._pending_urls_cache.items():
        depth = metadata.get("depth", 0)
        discovery_by_level[depth] += 1
    
    print(f"   Cache total: {discovery_cache_total:,}")
    for level in sorted(discovery_by_level.keys()):
        print(f"   Level {level}: {discovery_by_level[level]:,} URLs")
    
    # Test in analysis mode
    print("\n📊 ANALYSIS MODE:")
    url_storage.set_discovery_mode(False)
    url_storage.load_pending_urls_cache()
    
    analysis_cache_total = len(url_storage._pending_urls_cache)
    analysis_by_level = defaultdict(int)
    
    for url, metadata in url_storage._pending_urls_cache.items():
        depth = metadata.get("depth", 0)
        analysis_by_level[depth] += 1
    
    print(f"   Cache total: {analysis_cache_total:,}")
    for level in sorted(analysis_by_level.keys()):
        print(f"   Level {level}: {analysis_by_level[level]:,} URLs")
    
    # Test get_pending_urls for specific levels
    print("\n🎯 GET_PENDING_URLS TESTS:")
    print("-" * 30)
    
    for level in [1, 2, 3, 4, 5]:
        pending_urls = url_storage.get_pending_urls(depth=level)
        print(f"   Level {level}: {len(pending_urls)} URLs returned by get_pending_urls()")
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete!")

if __name__ == "__main__":
    analyze_url_distribution()
