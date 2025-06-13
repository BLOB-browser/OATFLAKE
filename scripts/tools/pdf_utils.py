#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF Utilities - Core PDF detection and download functions for use across the system.
"""

import os
import requests
import time
import logging
from urllib.parse import urlparse
from typing import Dict, Any

logger = logging.getLogger(__name__)

def is_pdf_url(url: str) -> bool:
    """
    Check if URL points to a PDF file.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL is a PDF, False otherwise
    """
    # Simple filename check
    if url.lower().endswith('.pdf'):
        return True
    
    # Advanced check via HEAD request
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        content_type = response.headers.get('content-type', '').lower()
        return 'pdf' in content_type
    except Exception as e:
        # If HEAD request fails, assume it's not a PDF to avoid blocking
        logger.debug(f"HEAD request failed for {url}: {e}")
        return False

def download_pdf_to_materials(url: str, data_folder: str, resource_id: str = "", logging_resource_id: str = "") -> Dict[str, Any]:
    """
    Download a PDF from URL to the materials folder.
    
    Args:
        url: URL of the PDF to download
        data_folder: Base data folder path
        resource_id: Resource ID for the PDF (used in filename)
        logging_resource_id: ID for logging purposes
        
    Returns:
        Dictionary with download result
    """
    try:
        # Create materials folder if it doesn't exist
        materials_folder = os.path.join(data_folder, 'materials')
        os.makedirs(materials_folder, exist_ok=True)
        
        logger.info(f"[Resource: {logging_resource_id}] Downloading PDF: {url}")
        
        # Generate filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # If no filename or doesn't end with .pdf, generate one
        if not filename or not filename.lower().endswith('.pdf'):
            if resource_id:
                filename = f"{resource_id}_{int(time.time())}.pdf"
            else:
                filename = f"document_{int(time.time())}.pdf"
        
        # Ensure unique filename
        filepath = os.path.join(materials_folder, filename)
        counter = 1
        base_name = filename[:-4]  # Remove .pdf
        while os.path.exists(filepath):
            if resource_id:
                filename = f"{resource_id}_{base_name}_{counter}.pdf"
            else:
                filename = f"{base_name}_{counter}.pdf"
            filepath = os.path.join(materials_folder, filename)
            counter += 1
        
        # Download the PDF
        start_time = time.time()
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Save the PDF
        total_size = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Verify file was saved
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            logger.info(f"[Resource: {logging_resource_id}] PDF downloaded successfully!")
            logger.info(f"[Resource: {logging_resource_id}]   📁 Saved to: {filepath}")
            logger.info(f"[Resource: {logging_resource_id}]   📊 Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            logger.info(f"[Resource: {logging_resource_id}]   ⏱️  Download time: {duration:.2f} seconds")
            
            return {
                "success": True,
                "filepath": filepath,
                "filename": filename,
                "size": file_size,
                "duration": duration,
                "url": url
            }
        else:
            return {
                "success": False,
                "error": "File was not saved properly",
                "url": url
            }
            
    except Exception as e:
        logger.error(f"[Resource: {logging_resource_id}] PDF download failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "url": url
        }


