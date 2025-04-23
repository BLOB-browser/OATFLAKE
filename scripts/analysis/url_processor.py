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
    Enhanced for better handling of documentation sites and educational resources.
    
    Args:
        href: The link href
        base_domain: The base domain for the website
        current_url: The current page URL (used for relative links)
        
    Returns:
        Tuple of (is_valid, absolute_url)
    """
    parsed_base = urlparse(base_domain)
    
    # Skip fragment-only links, email, phone, and social media
    if (href.startswith('#') or href.startswith('mailto:') or href.startswith('tel:') or 
        href.startswith('javascript:') or href.startswith('data:')):
        return False, ""
    
    # Skip common social media platforms
    social_media_domains = [
        'twitter.com', 'facebook.com', 'linkedin.com', 'instagram.com', 
        'youtube.com', 'reddit.com', 'tiktok.com', 'pinterest.com', 'tumblr.com'
    ]
    
    if any(domain in href for domain in social_media_domains):
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
        # For absolute URLs, we'll keep links to some important external domains that are
        # commonly used in educational content and documentation
        allowed_domains = [
            # Common educational domains
            '.edu', '.ac.uk', '.ac.', '.edu.', '.school', '.uni-', '.university',
            # Documentation, code and academic resources
            'github.com', 'gitlab.com', 'bitbucket.org', 'docs.', 'documentation.',
            'arxiv.org', 'scholar.google.com', 'research.', 'academia.edu',
            'researchgate.net', 'springer.com', 'ieee.org', 'acm.org',
            # Most common general domains
            parsed_base.netloc
        ]
        
        # Check if this is an allowed external domain
        if any(domain in href for domain in allowed_domains):
            absolute_url = href
        else:
            # Skip links to other external domains
            return False, ""
            
    # Skip common file types that are not relevant for content analysis
    skip_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',  # Images
                      '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm',  # Media
                      '.zip', '.rar', '.tar', '.gz', '.7z',  # Archives
                      '.exe', '.dmg', '.apk', '.msi']  # Executables
    
    if any(absolute_url.lower().endswith(ext) for ext in skip_extensions):
        return False, ""
    
    return True, absolute_url

def extract_navigation_links(soup: BeautifulSoup) -> List[Dict]:
    """
    Extract links from navigation elements.
    Enhanced to better handle documentation sites, course websites, and student portfolios.
    
    Args:
        soup: BeautifulSoup object of the HTML
        
    Returns:
        List of link objects
    """
    nav_links = []
    
    # Find links in conventional navigation elements (expanded list)
    for nav_element in soup.select('nav, header, .navigation, .navbar, .menu, .nav, ul.nav, ul.menu, .sidebar, .toc, .table-of-contents, .site-nav, .main-nav, .top-nav, .primary-nav, .secondary-nav, #navigation, #main-nav, #menu, .tabs, .breadcrumb, .breadcrumbs, .subnav, .submenu, .dropdown, .dropdown-menu, aside, .sidebar, #sidebar'):
        for link in nav_element.find_all('a', href=True):
            nav_links.append(link)
    
    # Look for documentation-specific navigation elements
    for doc_nav in soup.select('.documentation-nav, .doc-nav, .docs-sidebar, .docs-nav, .docs-menu, .api-nav, .toc, .contents, .index, #toc, #contents, #index'):
        for link in doc_nav.find_all('a', href=True):
            nav_links.append(link)
    
    # Find links in list items within the first 1/4 of the page - often navigation
    body_content = soup.find('body')
    if body_content:
        # Approximate the top part of the page by using the string representation length
        body_str = str(body_content)
        top_section_end = len(body_str) // 4
        top_html = body_str[:top_section_end]
        top_soup = BeautifulSoup(top_html, 'html.parser')
        
        # Find all lists in the top section
        for list_element in top_soup.find_all(['ul', 'ol']):
            # Skip if it's already part of a known navigation element
            if list_element.parent and list_element.parent.name in ['nav', 'header']:
                continue
                
            for link in list_element.find_all('a', href=True):
                nav_links.append(link)
    
    # If we still don't have many navigation links, look for any links with important text
    if len(nav_links) < 5:
        important_terms = [
            # General navigation terms
            'about', 'profile', 'bio', 'project', 'work', 'portfolio', 'contact', 'info', 
            'research', 'cv', 'resume', 'publication', 'home', 'main', 'index', 'overview',
            # Documentation terms
            'guide', 'tutorial', 'doc', 'documentation', 'api', 'reference', 'manual',
            'handbook', 'start', 'getting-started', 'quickstart', 'quick-start',
            # Academic terms
            'syllabus', 'schedule', 'assignment', 'course', 'class', 'lecture', 'lab',
            'resource', 'material', 'reading', 'download', 'slides', 'notes'
        ]
        
        # Gather all links in the document
        all_links = soup.find_all('a', href=True)
        
        # First pass: look for links with navigation-specific text
        for link in all_links:
            link_text = link.get_text().lower().strip()
            if any(term in link_text for term in important_terms):
                if link not in nav_links:
                    nav_links.append(link)
        
        # Second pass: if still not enough, look for top-level directory links
        if len(nav_links) < 10:
            for link in all_links:
                href = link.get('href', '')
                # Count path segments as indicator of depth
                path_segments = [p for p in href.split('/') if p]
                # If it's a root-level link (0 or 1 path segment), it's likely navigation
                if len(path_segments) <= 1 and not href.startswith(('#', 'mailto:', 'tel:')):
                    if link not in nav_links:
                        nav_links.append(link)
                        
    return nav_links

def extract_project_links(soup: BeautifulSoup, project_keywords: List[str] = None) -> List[Dict]:
    """
    Extract links that are likely to be project-related.
    Enhanced to handle student websites and documentation sites.
    
    Args:
        soup: BeautifulSoup object of the HTML
        project_keywords: List of keywords to look for in link text
        
    Returns:
        List of link objects
    """
    if project_keywords is None:
        project_keywords = [
            'project', 'work', 'case', 'study', 'portfolio', 'article', 'publication', 
            'research', 'detail', 'post', 'assignment', 'homework', 'lab', 'exercise',
            'experiment', 'gallery', 'showcase', 'collection', 'demo', 'thesis', 
            'dissertation', 'capstone', 'final', 'class', 'course', 'semester'
        ]
    
    project_links = []
    
    # Look in likely project containers (expanded list)
    for container in soup.select('.projects, .works, .portfolio, .case-studies, article, .research, .publications, .assignments, .course-work, .gallery, .showcase, .examples, .demos, ul.projects, ol.projects, div.projects, .collection, .items, .entries, .cards, .grid, .list, .repo, .repository'):
        for link in container.find_all('a', href=True):
            project_links.append(link)
    
    # Look for headings that might introduce project lists
    project_heading_keywords = ['projects', 'assignments', 'portfolio', 'work', 'publications']
    for heading in soup.find_all(['h1', 'h2', 'h3', 'h4']):
        heading_text = heading.get_text().lower().strip()
        if any(keyword in heading_text for keyword in project_heading_keywords):
            # Find the next unordered list or div that might contain projects
            next_element = heading.find_next_sibling()
            while next_element and not (next_element.name in ['ul', 'ol', 'div'] or next_element.name in ['h1', 'h2', 'h3', 'h4']):
                next_element = next_element.find_next_sibling()
                
            if next_element and next_element.name in ['ul', 'ol', 'div']:
                for link in next_element.find_all('a', href=True):
                    project_links.append(link)
    
    # Look for links with project-related text in href or link text
    all_links = soup.find_all('a', href=True)
    for link in all_links:
        link_text = link.get_text().lower().strip()
        href = link.get('href', '').lower()
        
        # Check both link text and URL
        if (any(keyword in link_text for keyword in project_keywords) or 
            any(keyword in href for keyword in project_keywords)):
            if link not in project_links:
                project_links.append(link)
    
    # If very few links found through targeted methods, look for projects in patterns
    if len(project_links) < 5:
        # Look for links that might be projects based on URL patterns
        for link in all_links:
            href = link.get('href', '')
            # Check for numeric patterns that often indicate project listings
            if (('/project/' in href or '/projects/' in href or 
                 '/work/' in href or '/assignment/' in href) or
                (href.endswith('.html') or href.endswith('.htm') or 
                 href.endswith('.php'))):
                if link not in project_links:
                    project_links.append(link)
    
    # If still not enough links found, get links from the main content area
    if len(project_links) < 5:
        # Try to find the main content section
        main_content = soup.find('main') or soup.find(id='content') or soup.find(class_='content')
        if main_content:
            for link in main_content.find_all('a', href=True):
                if link not in project_links:
                    project_links.append(link)
                
    return project_links

def extract_detail_links(soup: BeautifulSoup, detail_keywords: List[str] = None) -> List[Dict]:
    """
    Extract links that are likely to contain detailed content.
    More comprehensive approach to find links throughout the document,
    suitable for documentation sites and student projects.
    
    Args:
        soup: BeautifulSoup object of the HTML
        detail_keywords: List of keywords to look for in link text
        
    Returns:
        List of link objects
    """
    if detail_keywords is None:
        detail_keywords = [
            'detail', 'more', 'read', 'view', 'full', 'case', 'story', 'about', 'learn', 
            'article', 'page', 'post', 'doc', 'documentation', 'tutorial', 'guide', 'example',
            'sample', 'reference', 'api', 'manual', 'chapter', 'section', 'topic', 'content',
            'assignment', 'project', 'paper', 'report', 'thesis', 'journal', 'publication'
        ]
    
    detail_links = []
    
    # First, try to find links in conventional content containers
    for container in soup.select('main, article, section, .content, .projects, .details, .post, div.entry-content, .documentation, .container, .wrapper, #content, .page-content'):
        for link in container.find_all('a', href=True):
            detail_links.append(link)
    
    # Find all links in paragraph tags - often used in documentation
    for p_tag in soup.find_all('p'):
        for link in p_tag.find_all('a', href=True):
            detail_links.append(link)
    
    # Find all links in list items - common in docs and student sites
    for li_tag in soup.find_all('li'):
        for link in li_tag.find_all('a', href=True):
            detail_links.append(link)
            
    # Find links in documentation-specific elements like code blocks
    for code_container in soup.select('pre, code, .code-block, .example'):
        for link in code_container.find_all('a', href=True):
            detail_links.append(link)
    
    # Process links with specific keywords in their text or href
    all_links = soup.find_all('a', href=True)
    keyword_links = []
    
    for link in all_links:
        link_text = link.get_text().lower().strip()
        href = link.get('href', '').lower()
        
        # Check if any keyword is in link text or href
        if (any(keyword in link_text for keyword in detail_keywords) or
            any(keyword in href for keyword in detail_keywords)):
            keyword_links.append(link)
    
    # Look for documentation file extensions in hrefs
    doc_extensions = ['.html', '.htm', '.md', '.pdf', '.doc', '.docx', '.txt', '.tex', '.rst']
    for link in all_links:
        href = link.get('href', '').lower()
        if any(href.endswith(ext) for ext in doc_extensions):
            keyword_links.append(link)
    
    # Add any links that weren't already captured
    for link in keyword_links:
        if link not in detail_links:
            detail_links.append(link)
    
    # If we still don't have many links, just get all links on the page
    # This ensures we don't miss anything in unconventional page structures
    if len(detail_links) < 10:
        logger.info("Few links found with specific criteria - adding all page links")
        for link in all_links:
            if link not in detail_links:
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
