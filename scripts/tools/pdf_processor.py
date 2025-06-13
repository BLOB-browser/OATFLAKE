#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF Processing Utility for Knowledge Orchestration

This script handles PDF URLs that were saved during regular URL processing.
PDFs require special handling and can't be processed by the regular web scraper.
"""

import os
import json
import requests
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.config import get_data_path

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handle PDF processing for the knowledge orchestration system."""
    
    def __init__(self, data_folder: str = None):
        """Initialize PDF processor."""
        self.data_folder = data_folder or get_data_path()
        self.pdf_queue_file = os.path.join(self.data_folder, 'pdfs_to_process.json')
        self.pdf_storage_dir = os.path.join(self.data_folder, 'pdfs')
        
        # Create PDF storage directory
        os.makedirs(self.pdf_storage_dir, exist_ok=True)
    
    def load_pdf_queue(self) -> List[Dict]:
        """Load the PDF processing queue."""
        if not os.path.exists(self.pdf_queue_file):
            return []
        
        try:
            with open(self.pdf_queue_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading PDF queue: {e}")
            return []
    
    def save_pdf_queue(self, pdf_queue: List[Dict]) -> None:
        """Save the PDF processing queue."""
        try:
            with open(self.pdf_queue_file, 'w') as f:
                json.dump(pdf_queue, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving PDF queue: {e}")
    
    def download_pdf(self, url: str, filename: str = None) -> str:
        """Download a PDF file."""
        if not filename:
            # Generate filename from URL
            filename = url.split('/')[-1]
            if not filename.endswith('.pdf'):
                filename += '.pdf'
        
        # Clean filename for filesystem
        import re
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        pdf_path = os.path.join(self.pdf_storage_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(pdf_path):
            logger.info(f"PDF already exists: {filename}")
            return pdf_path
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(pdf_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded PDF: {filename} ({len(response.content)} bytes)")
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error downloading PDF {url}: {e}")
            return None
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using PyPDF2 or similar."""
        try:
            # Try to import PyPDF2
            try:
                import PyPDF2
            except ImportError:
                logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
                print("⚠️  PyPDF2 not installed. To enable PDF text extraction, run:")
                print("   pip install PyPDF2")
                return ""
            
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip()
                
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_pdfs(self, max_pdfs: int = 5, download_only: bool = False) -> Dict[str, Any]:
        """Process PDFs from the queue."""
        pdf_queue = self.load_pdf_queue()
        
        if not pdf_queue:
            return {
                "status": "no_pdfs",
                "message": "No PDFs in processing queue"
            }
        
        processed = 0
        downloaded = 0
        errors = []
        
        print(f"📄 Processing {min(max_pdfs, len(pdf_queue))} PDFs from queue of {len(pdf_queue)}")
        
        for i, pdf_data in enumerate(pdf_queue[:max_pdfs]):
            url = pdf_data.get('url', '')
            print(f"\n[{i+1}/{min(max_pdfs, len(pdf_queue))}] Processing: {url[:60]}...")
            
            try:
                # Download PDF
                pdf_path = self.download_pdf(url)
                if pdf_path:
                    downloaded += 1
                    print(f"✅ Downloaded: {os.path.basename(pdf_path)}")
                    
                    if not download_only:
                        # Extract text
                        text = self.extract_pdf_text(pdf_path)
                        if text:
                            print(f"📝 Extracted {len(text)} characters of text")
                            
                            # TODO: Process with UniversalAnalysisLLM
                            # This would need to be implemented similar to regular URL processing
                            
                        else:
                            print("⚠️ No text extracted from PDF")
                    
                    processed += 1
                else:
                    errors.append(f"Failed to download: {url}")
                    
            except Exception as e:
                error_msg = f"Error processing {url}: {e}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        return {
            "status": "success",
            "processed": processed,
            "downloaded": downloaded,
            "errors": errors,
            "total_in_queue": len(pdf_queue)
        }
    
    def list_pdfs(self) -> Dict[str, Any]:
        """List PDFs in the processing queue."""
        pdf_queue = self.load_pdf_queue()
        
        print(f"📄 PDF Processing Queue ({len(pdf_queue)} PDFs)")
        print("=" * 60)
        
        if not pdf_queue:
            print("No PDFs in queue")
            return {"count": 0, "pdfs": []}
        
        for i, pdf_data in enumerate(pdf_queue[:10]):  # Show first 10
            url = pdf_data.get('url', '')
            origin = pdf_data.get('origin', '')
            depth = pdf_data.get('depth', 'unknown')
            
            print(f"{i+1:2d}. {url}")
            print(f"    Origin: {origin}")
            print(f"    Depth: {depth}")
            print()
        
        if len(pdf_queue) > 10:
            print(f"... and {len(pdf_queue) - 10} more PDFs")
        
        return {"count": len(pdf_queue), "pdfs": pdf_queue}
    
    def clear_processed_pdfs(self) -> int:
        """Remove successfully downloaded PDFs from the queue."""
        pdf_queue = self.load_pdf_queue()
        original_count = len(pdf_queue)
        
        remaining_pdfs = []
        
        for pdf_data in pdf_queue:
            url = pdf_data.get('url', '')
            # Generate expected filename
            filename = url.split('/')[-1]
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            import re
            filename = re.sub(r'[^\w\-_\.]', '_', filename)
            pdf_path = os.path.join(self.pdf_storage_dir, filename)
            
            # Keep in queue if not downloaded yet
            if not os.path.exists(pdf_path):
                remaining_pdfs.append(pdf_data)
        
        self.save_pdf_queue(remaining_pdfs)
        removed_count = original_count - len(remaining_pdfs)
        
        print(f"🧹 Removed {removed_count} processed PDFs from queue")
        print(f"📄 {len(remaining_pdfs)} PDFs remaining in queue")
        
        return removed_count

def main():
    """Command line interface for PDF processing."""
    parser = argparse.ArgumentParser(description="Process PDF URLs from knowledge orchestration")
    parser.add_argument('--action', choices=['list', 'process', 'download', 'clear'], 
                       default='list', help='Action to perform')
    parser.add_argument('--max-pdfs', type=int, default=5, 
                       help='Maximum number of PDFs to process')
    parser.add_argument('--data-folder', help='Data folder path (uses config if not specified)')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = PDFProcessor(args.data_folder)
    
    if args.action == 'list':
        processor.list_pdfs()
    elif args.action == 'process':
        result = processor.process_pdfs(args.max_pdfs, download_only=False)
        print(f"\n📊 Processing Results:")
        print(f"   Processed: {result['processed']}")
        print(f"   Downloaded: {result['downloaded']}")
        print(f"   Errors: {len(result['errors'])}")
    elif args.action == 'download':
        result = processor.process_pdfs(args.max_pdfs, download_only=True)
        print(f"\n📊 Download Results:")
        print(f"   Downloaded: {result['downloaded']}")
        print(f"   Errors: {len(result['errors'])}")
    elif args.action == 'clear':
        processor.clear_processed_pdfs()

if __name__ == "__main__":
    main()
