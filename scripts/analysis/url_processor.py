#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import requests
from typing import Tuple, Dict, Any, List, Set
from bs4 import BeautifulSoup
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

def clean_url(url: str) -> str:
    """
    Clean and normalize a URL.
    
    Args:
        url: The URL to clean
        
    Returns:
        Cleaned URL
    """
    url = url.strip()
    
    # Add http:// if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
        
    return url

def get_base_domain_and_url(url: str) -> Tuple[str, str, str]:
    """
    Extract base domain and path from a URL.
    
    Args:
        url: The URL to process
        
    Returns:
        Tuple of base_domain, base_path, and base_url
    """
    parsed_url = urlparse(url)
    base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
    base_path = parsed_url.path.rsplit('/', 1)[0] if '/' in parsed_url.path else ''
    base_url = f"{base_domain}{base_path}"
    
    return base_domain, base_path, base_url

def resolve_absolute_url(href: str, base_domain: str, current_url: str = None) -> Tuple[bool, str]:
    """
    Convert a relative URL to absolute URL.
    
    Args:
        href: The link href
        base_domain: The base domain for the website
        current_url: The current page URL (used for relative links)
        
    Returns:
        Tuple of (is_valid, absolute_url)
    """
    parsed_base = urlparse(base_domain)
    
    # Skip unwanted links
    if (href.startswith('#') or any(domain in href for domain in 
                                 ['twitter.com', 'facebook.com', 'linkedin.com', 
                                  'instagram.com', 'mailto:'])):
        return False, ""
        
    # Convert to absolute URL if needed
    if href.startswith('/'):
        absolute_url = f"{base_domain}{href}"
    elif not href.startswith(('http://', 'https://')):
        # For relative URLs, use the current URL as base if provided
        if current_url:
            current_base = current_url.rsplit('/', 1)[0] if '/' in current_url else current_url
            absolute_url = f"{current_base}/{href}"
        else:
            # Fallback to base_domain if no current URL provided
            absolute_url = f"{base_domain}/{href}"
    else:
        # Skip links to other domains
        if parsed_base.netloc not in href:
            return False, ""
        absolute_url = href
    
    return True, absolute_url

def extract_navigation_links(soup: BeautifulSoup) -> List[Dict]:
    """
    Extract links from navigation elements.
    
    Args:
        soup: BeautifulSoup object of the HTML
        
    Returns:
        List of link objects
    """
    nav_links = []
    
    # Find links in the navigation menu - these are typically important sections
    for nav_element in soup.select('nav, header, .navigation, .navbar, .menu, .nav, ul.nav, ul.menu'):
        for link in nav_element.find_all('a', href=True):
            nav_links.append(link)
    
    # If no nav links found, look for any links with important text
    if not nav_links:
        important_terms = ['about', 'profile', 'bio', 'project', 'work', 'portfolio', 
                          'contact', 'info', 'research', 'cv', 'resume', 'publication']
        
        for link in soup.find_all('a', href=True):
            link_text = link.get_text().lower().strip()
            if any(term in link_text for term in important_terms):
                nav_links.append(link)
                
    return nav_links

def extract_project_links(soup: BeautifulSoup, project_keywords: List[str] = None) -> List[Dict]:
    """
    Extract links that are likely to be project-related.
    
    Args:
        soup: BeautifulSoup object of the HTML
        project_keywords: List of keywords to look for in link text
        
    Returns:
        List of link objects
    """
    if project_keywords is None:
        project_keywords = ['project', 'work', 'case', 'study', 'portfolio', 'article', 
                           'publication', 'research', 'detail', 'post']
    
    project_links = []
    
    # Look in likely project containers
    for container in soup.select('.projects, .works, .portfolio, .case-studies, article, .research, .publications'):
        for link in container.find_all('a', href=True):
            project_links.append(link)
    
    # Also look for links with project-related text if not enough found
    if not project_links:
        for link in soup.find_all('a', href=True):
            link_text = link.get_text().lower().strip()
            if any(keyword in link_text for keyword in project_keywords):
                project_links.append(link)
                
    return project_links

def extract_detail_links(soup: BeautifulSoup, detail_keywords: List[str] = None) -> List[Dict]:
    """
    Extract links that are likely to contain detailed content.
    
    Args:
        soup: BeautifulSoup object of the HTML
        detail_keywords: List of keywords to look for in link text
        
    Returns:
        List of link objects
    """
    if detail_keywords is None:
        detail_keywords = ['detail', 'more', 'read', 'view', 'full', 'case', 'story', 
                          'about', 'learn', 'article', 'page', 'post']
    
    detail_links = []
    
    # Look in likely content containers
    for container in soup.select('main, article, section, .content, .projects, .details, .post'):
        for link in container.find_all('a', href=True):
            detail_links.append(link)
    
    # Also look for links with detail-related text if not enough found
    if len(detail_links) < 5:
        for link in soup.find_all('a', href=True):
            link_text = link.get_text().lower().strip()
            if any(keyword in link_text for keyword in detail_keywords):
                detail_links.append(link)
                
    return detail_links

def process_links(links: List, base_domain: str, current_url: str, 
                  visited_urls: Set[str], additional_urls: List[str] = None) -> List[str]:
    """
    Process a list of links and extract valid, unique URLs.
    
    Args:
        links: List of BeautifulSoup link objects
        base_domain: The base domain for the website
        current_url: The current page URL (for relative links)
        visited_urls: Set of already visited URLs
        additional_urls: List of already collected URLs to check against
        
    Returns:
        List of new URLs
    """
    new_urls = []
    additional_urls = additional_urls or []
    
    for link in links:
        href = link['href']
        
        # Resolve to absolute URL
        is_valid, absolute_url = resolve_absolute_url(href, base_domain, current_url)
        if not is_valid:
            continue
        
        # Skip if already visited or in any queue
        if (absolute_url not in visited_urls and 
            absolute_url not in additional_urls and 
            absolute_url not in new_urls):
            new_urls.append(absolute_url)
    
    return new_urls
