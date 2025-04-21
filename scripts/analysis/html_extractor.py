#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import List, Dict
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

def extract_text(html_content: str, size_limit: int = 2000) -> str:
    """
    Extract readable text content from HTML with enhanced methods to get meaningful content.
    
    Args:
        html_content: Raw HTML content
        size_limit: Size limit for the extracted text
        
    Returns:
        Extracted text content
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script, style elements and hidden content
        for element in soup(["script", "style", "noscript", "iframe", "head"]):
            element.extract()
            
        # Look for main content sections and give them priority
        content_sections = []
        
        # Look for all headings as they typically introduce important content
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in headings:
            heading_text = heading.get_text().strip()
            if heading_text:
                # Get the next siblings (content that follows the heading)
                content = []
                for sibling in heading.find_next_siblings():
                    # Stop if we hit another heading
                    if sibling.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        break
                    # Add the text from this element
                    sibling_text = sibling.get_text().strip()
                    if sibling_text:
                        content.append(sibling_text)
                
                # Add this section with its heading
                if content:
                    section = f"{heading_text}:\n" + "\n".join(content)
                    content_sections.append(section)
        
        # Look for typical content elements
        for element_type in ['main', 'article', 'section', 'div.content', 'div.main']:
            elements = soup.select(element_type) if '.' in element_type else soup.find_all(element_type)
            for element in elements:
                element_text = element.get_text().strip()
                if len(element_text) > 100:  # Only keep substantial content blocks
                    content_sections.append(element_text)
        
        # If we found content sections, use them with size limit
        if content_sections:
            focused_text = "\n\n".join(content_sections)
            total_chars = len(focused_text)
            truncated_text = focused_text[:size_limit]
            logger.info(f"Extracted {total_chars} chars, truncated to {len(truncated_text)} chars from {len(content_sections)} sections")
            return truncated_text
        
        # Fall back to regular text extraction if structured approach didn't work
        # Get text from the whole page
        text = soup.get_text(separator=' ')
        
        # Break into lines and remove leading/trailing whitespace
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Apply size limit
        truncated_text = text[:size_limit]
        logger.info(f"Using fallback extraction: {len(text)} chars truncated to {len(truncated_text)} chars")
        return truncated_text
    except Exception as e:
        logger.error(f"Error extracting text content: {e}")
        return ""

def extract_page_texts(pages_dict: Dict[str, str], size_limit: int = 2000) -> Dict[str, str]:
    """
    Extract text from multiple HTML pages
    
    Args:
        pages_dict: Dictionary of HTML content keyed by page name
        size_limit: Size limit for each extracted page text
        
    Returns:
        Dictionary of extracted text content keyed by page name
    """
    texts = {}
    for page_name, html_content in pages_dict.items():
        # Skip error entries
        if page_name == "error":
            continue
            
        extracted_text = extract_text(html_content, size_limit)
        if extracted_text:
            texts[page_name] = extracted_text
            logger.info(f"Extracted {len(extracted_text)} chars from page '{page_name}'")
        else:
            logger.warning(f"No content could be extracted from page '{page_name}'")
            
    return texts
