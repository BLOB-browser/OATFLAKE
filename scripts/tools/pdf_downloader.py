#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF Downloader - Downloads PDFs from pending URLs to the materials folder for processing.
Saves PDFs to data_folder/materials/ for later processing by the existing PDF processor.
"""

import os
import sys
import json
import requests
import argparse
from pathlib import Path
from urllib.parse import urlparse
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import PDF utilities
from scripts.tools.pdf_utils import is_pdf_url, download_pdf_to_materials

def extract_pdfs_from_pending_urls(data_folder):
    """Extract PDF URLs from pending_urls.csv."""
    from scripts.analysis.url_storage import URLStorageManager
    
    processed_urls_file = os.path.join(data_folder, "processed_urls.csv")
    url_storage = URLStorageManager(processed_urls_file)
    
    pdf_urls = []
    
    # Check all levels for PDF URLs
    for level in range(1, 6):
        pending_urls = url_storage.get_pending_urls(depth=level)
        if pending_urls:
            for url_data in pending_urls:
                url = url_data.get('url', '')
                if is_pdf_url(url):
                    pdf_urls.append({
                        'url': url,
                        'origin': url_data.get('origin', ''),
                        'depth': url_data.get('depth', level),
                        'resource_id': url_data.get('resource_id', '')
                    })
    
    return pdf_urls

def main():
    parser = argparse.ArgumentParser(description='Download PDFs from pending URLs')
    parser.add_argument('--level', type=int, help='Specific level to process')
    parser.add_argument('--max-pdfs', type=int, default=10, help='Maximum number of PDFs to download')
    parser.add_argument('--download-dir', help='Directory to save PDFs (default: data_folder/materials)')
    parser.add_argument('--list-only', action='store_true', help='Just list PDF URLs without downloading')
    parser.add_argument('--timeout', type=int, default=30, help='Download timeout in seconds')
    
    args = parser.parse_args()
    
    try:
        from utils.config import get_data_path
        data_folder = get_data_path()
        
        # Set download directory
        if args.download_dir:
            download_dir = args.download_dir
        else:
            download_dir = os.path.join(data_folder, 'materials')
        
        os.makedirs(download_dir, exist_ok=True)
        
        print("🔍 PDF DOWNLOADER")
        print("=" * 50)
        print(f"📁 Data folder: {data_folder}")
        print(f"📄 Download directory: {download_dir}")
        
        # Extract PDF URLs
        print("\n🔍 Extracting PDF URLs from pending URLs...")
        pdf_urls = extract_pdfs_from_pending_urls(data_folder)
        
        if not pdf_urls:
            print("✅ No PDF URLs found in pending URLs")
            return
        
        # Filter by level if specified
        if args.level:
            pdf_urls = [pdf for pdf in pdf_urls if pdf.get('depth') == args.level]
            print(f"📊 Found {len(pdf_urls)} PDF URLs at level {args.level}")
        else:
            print(f"📊 Found {len(pdf_urls)} PDF URLs across all levels")
        
        # Group by level for display
        pdfs_by_level = {}
        for pdf in pdf_urls:
            level = pdf.get('depth', 'unknown')
            if level not in pdfs_by_level:
                pdfs_by_level[level] = []
            pdfs_by_level[level].append(pdf)
        
        print("\n📋 PDF URLs by level:")
        for level in sorted(pdfs_by_level.keys()):
            print(f"   Level {level}: {len(pdfs_by_level[level])} PDFs")
        
        if args.list_only:
            print("\n📄 PDF URLs found:")
            for i, pdf in enumerate(pdf_urls[:20]):  # Show first 20
                print(f"   {i+1}. {pdf['url']}")
                print(f"      Origin: {pdf.get('origin', 'N/A')}")
                print(f"      Level: {pdf.get('depth', 'N/A')}")
                print()
            
            if len(pdf_urls) > 20:
                print(f"   ... and {len(pdf_urls) - 20} more")
            return
        
        # Download PDFs
        pdfs_to_download = pdf_urls[:args.max_pdfs]
        print(f"\n⬇️  Downloading {len(pdfs_to_download)} PDFs...")
        
        downloaded = []
        failed = []
        
        for i, pdf_data in enumerate(pdfs_to_download):
            url = pdf_data['url']
            print(f"\n[{i+1}/{len(pdfs_to_download)}] Processing: {url}")
            
            result = download_pdf_to_materials(
                url=url, 
                data_folder=data_folder,
                resource_id=pdf_data.get('resource_id', ''),
                logging_resource_id=f"{pdf_data.get('resource_id', 'unknown')}::{url}"
            )
            
            if result['success']:
                downloaded.append(result)
                # Add metadata to result
                result.update(pdf_data)
            else:
                failed.append(result)
        
        # Summary
        print(f"\n📊 DOWNLOAD SUMMARY")
        print("=" * 50)
        print(f"✅ Successfully downloaded: {len(downloaded)} PDFs")
        print(f"❌ Failed downloads: {len(failed)} PDFs")
        
        if downloaded:
            total_size = sum(pdf['size'] for pdf in downloaded)
            print(f"📁 Total size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
            print(f"📄 Files saved to: {download_dir}")
        
        if failed:
            print(f"\n❌ Failed URLs:")
            for fail in failed:
                print(f"   {fail['url']}: {fail['error']}")
        
        # Save download log
        log_file = os.path.join(download_dir, 'download_log.json')
        log_data = {
            'timestamp': time.time(),
            'downloaded': downloaded,
            'failed': failed,
            'summary': {
                'total_found': len(pdf_urls),
                'attempted': len(pdfs_to_download),
                'successful': len(downloaded),
                'failed': len(failed)
            }
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📝 Download log saved to: {log_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
