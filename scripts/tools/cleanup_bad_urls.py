#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
URL Cleanup Tool - Production Utility

This tool identifies and removes problematic URLs from pending_urls.csv that are:
- Invalid (malformed URLs)
- Inaccessible (404, 403, connection errors)
- Blocked by robots.txt or other access restrictions

Usage:
    python scripts/tools/cleanup_bad_urls.py [--dry-run] [--level=3] [--max-test=50]
    
Options:
    --dry-run     : Show what would be removed without actually removing
    --level=N     : Only check URLs at specific level (default: all levels)
    --max-test=N  : Maximum URLs to test per level (default: 50)
"""

import sys
import os
import csv
import asyncio
import aiohttp
import argparse
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict, Tuple
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class URLCleanupTool:
    """Tool to clean up problematic URLs from pending_urls.csv"""
    
    def __init__(self, data_folder: str):
        self.data_folder = data_folder
        self.pending_urls_file = os.path.join(data_folder, "pending_urls.csv")
        self.backup_file = os.path.join(data_folder, "pending_urls_backup.csv")
        
    async def test_url_accessibility(self, url: str, timeout: int = 10) -> Tuple[bool, str]:
        """
        Test if a URL is accessible.
        
        Returns:
            Tuple of (is_accessible, reason)
        """
        # First check if URL is valid
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False, "invalid_url"
            if parsed.scheme not in ['http', 'https']:
                return False, "invalid_scheme"
            if not url.strip() or url == "https://" or url == "http://":
                return False, "empty_url"
        except Exception:
            return False, "malformed_url"
            
        # Test HTTP accessibility
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.head(url, allow_redirects=True) as response:
                    if response.status == 200:
                        return True, "accessible"
                    elif response.status == 404:
                        return False, "http_404"
                    elif response.status == 403:
                        return False, "http_403"
                    elif response.status == 401:
                        return False, "http_401"
                    elif response.status >= 500:
                        return False, f"http_{response.status}"
                    else:
                        return False, f"http_{response.status}"
                        
        except aiohttp.ClientError as e:
            return False, "connection_error"
        except asyncio.TimeoutError:
            return False, "timeout"
        except Exception as e:
            return False, f"unknown_error"
    
    async def test_urls_batch(self, urls_data: List[Dict], max_concurrent: int = 10) -> Dict[str, Tuple[bool, str]]:
        """Test multiple URLs concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def test_single(url_data: Dict) -> Tuple[str, bool, str]:
            url = url_data.get('origin_url', '')
            async with semaphore:
                accessible, reason = await self.test_url_accessibility(url)
                return url, accessible, reason
        
        print(f"🧪 Testing {len(urls_data)} URLs accessibility...")
        
        tasks = [test_single(url_data) for url_data in urls_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        url_status = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                url = urls_data[i].get('origin_url', 'unknown')
                url_status[url] = (False, "test_error")
                print(f"❌ Error testing {url}: {result}")
            else:
                url, accessible, reason = result
                url_status[url] = (accessible, reason)
                status = "✅ ACCESSIBLE" if accessible else "❌ FAILED"
                short_url = url[:80] + "..." if len(url) > 80 else url
                print(f"[{i+1}/{len(urls_data)}] {status} - {short_url} ({reason})")
        
        return url_status
    
    def load_pending_urls(self, level_filter: int = None) -> List[Dict]:
        """Load pending URLs, optionally filtered by level."""
        if not os.path.exists(self.pending_urls_file):
            print(f"❌ Pending URLs file not found: {self.pending_urls_file}")
            return []
        
        try:
            df = pd.read_csv(self.pending_urls_file)
            print(f"📋 Loaded {len(df)} total pending URLs")
            
            if level_filter is not None:
                df = df[df['depth'] == level_filter]
                print(f"📋 Filtered to {len(df)} URLs at level {level_filter}")
            
            return df.to_dict('records')
            
        except Exception as e:
            print(f"❌ Error loading pending URLs: {e}")
            return []
    
    def backup_pending_urls(self) -> bool:
        """Create a backup of the current pending URLs file."""
        try:
            if os.path.exists(self.pending_urls_file):
                import shutil
                shutil.copy2(self.pending_urls_file, self.backup_file)
                print(f"💾 Backup created: {self.backup_file}")
                return True
            return False
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return False
    
    def remove_bad_urls(self, bad_urls: List[str], dry_run: bool = True) -> bool:
        """Remove bad URLs from pending_urls.csv."""
        if not bad_urls:
            print("✅ No bad URLs to remove")
            return True
            
        try:
            df = pd.read_csv(self.pending_urls_file)
            original_count = len(df)
            
            # Remove bad URLs
            df = df[~df['origin_url'].isin(bad_urls)]
            new_count = len(df)
            removed_count = original_count - new_count
            
            if dry_run:
                print(f"🔍 DRY RUN: Would remove {removed_count} bad URLs")
                print(f"🔍 Would keep {new_count} URLs")
                return True
            else:
                # Create backup first
                if not self.backup_pending_urls():
                    print("❌ Could not create backup - aborting removal")
                    return False
                
                # Save cleaned file
                df.to_csv(self.pending_urls_file, index=False)
                print(f"✅ Removed {removed_count} bad URLs")
                print(f"✅ Kept {new_count} working URLs")
                return True
                
        except Exception as e:
            print(f"❌ Error removing bad URLs: {e}")
            return False
    
    async def cleanup_urls(self, level_filter: int = None, max_test: int = 50, dry_run: bool = True) -> Dict:
        """Main cleanup function."""
        print("🧹 URL CLEANUP TOOL")
        print("=" * 60)
        
        # Load URLs
        urls_data = self.load_pending_urls(level_filter)
        if not urls_data:
            return {"status": "error", "message": "No URLs to process"}
        
        # Limit testing if requested
        if len(urls_data) > max_test:
            print(f"⚠️ Limiting test to first {max_test} URLs (use --max-test to change)")
            urls_data = urls_data[:max_test]
        
        # Test accessibility
        url_status = await self.test_urls_batch(urls_data)
        
        # Categorize results
        working_urls = []
        bad_urls = []
        failure_reasons = {}
        
        for url, (accessible, reason) in url_status.items():
            if accessible:
                working_urls.append(url)
            else:
                bad_urls.append(url)
                if reason not in failure_reasons:
                    failure_reasons[reason] = 0
                failure_reasons[reason] += 1
        
        # Print summary
        print(f"\n📊 CLEANUP RESULTS:")
        print(f"   ✅ Working URLs: {len(working_urls)}")
        print(f"   ❌ Bad URLs: {len(bad_urls)}")
        
        if failure_reasons:
            print(f"\n💥 FAILURE REASONS:")
            for reason, count in sorted(failure_reasons.items()):
                print(f"   {reason}: {count} URLs")
        
        # Remove bad URLs
        if bad_urls:
            success = self.remove_bad_urls(bad_urls, dry_run)
            if not success:
                return {"status": "error", "message": "Failed to remove bad URLs"}
        
        return {
            "status": "success",
            "working_urls": len(working_urls),
            "bad_urls": len(bad_urls),
            "failure_reasons": failure_reasons,
            "dry_run": dry_run
        }

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Clean up problematic URLs from pending_urls.csv")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without actually removing")
    parser.add_argument("--level", type=int, help="Only check URLs at specific level")
    parser.add_argument("--max-test", type=int, default=50, help="Maximum URLs to test")
    parser.add_argument("--execute", action="store_true", help="Actually remove bad URLs (default is dry-run)")
    
    args = parser.parse_args()
    
    # Import config
    from utils.config import get_data_path
    data_folder = get_data_path()
    
    # Create cleanup tool
    cleanup_tool = URLCleanupTool(data_folder)
    
    # Run cleanup
    dry_run = not args.execute
    if dry_run:
        print("🔍 Running in DRY-RUN mode (use --execute to actually remove URLs)")
    else:
        print("💥 EXECUTION mode - will actually remove bad URLs!")
    
    result = await cleanup_tool.cleanup_urls(
        level_filter=args.level,
        max_test=args.max_test,
        dry_run=dry_run
    )
    
    print(f"\n🏁 Cleanup completed with status: {result.get('status')}")
    
    if result.get('status') == 'success' and dry_run and result.get('bad_urls', 0) > 0:
        print("\n💡 To actually remove the bad URLs, run:")
        level_arg = f" --level={args.level}" if args.level else ""
        max_test_arg = f" --max-test={args.max_test}" if args.max_test != 50 else ""
        print(f"   python scripts/tools/cleanup_bad_urls.py --execute{level_arg}{max_test_arg}")

if __name__ == "__main__":
    asyncio.run(main())
