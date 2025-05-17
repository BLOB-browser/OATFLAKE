#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import requests
from typing import Tuple, Dict, Any, List, Set
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlunparse

logger = logging.getLogger(__name__)

def clean_url(url: str) -> str:
    """
    Clean and normalize a URL.
    
    Args:
        url: The URL to clean
        
    Returns:
        Cleaned URL
    """
    if not url:
        return url
        
    url = url.strip()
    
    # Remove trailing slashes
    while url.endswith('/') and len(url) > 1:
        url = url[:-1]
    
    # Fix common URL issues
    
    # Remove embedded file:/// protocol if present
    if "file:///" in url:
        url = url.replace("file:///", "")
    
    # Handle URLs that mix protocols
    if url.count("://") > 1:
        parts = url.split("://", 1)
        second_part = parts[1]
        if "://" in second_part:
            # Keep only the part after the second ://
            url = parts[0] + "://" + second_part.split("://", 1)[1]
    
    # Add http:// if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Remove query parameters for certain file types
    ext_to_clean = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.zip', '.rar']
    for ext in ext_to_clean:
        if ext in url and '?' in url:
            url = url.split('?')[0]
            break
    
    # Handle anchor tags
    if '#' in url and any(ext in url for ext in ['.html', '.htm', '.php', '.aspx']):
        url = url.split('#')[0]
        
    # Additional normalization to avoid duplicates
    url = normalize_url(url)
        
    return url

def normalize_url(url: str) -> str:
    """
    Normalize a URL by removing common tracking parameters, fragments, and
    standardizing paths to avoid duplicate content.
    
    Args:
        url: The URL to normalize
        
    Returns:
        Normalized URL
    """
    if not url:
        return url
        
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Strip fragments (anchors)
    fragment = ''
    
    # Keep the path, but normalize trailing slashes
    path = parsed_url.path
    if path.endswith('/') and len(path) > 1:
        path = path[:-1]
        
    # Remove common tracking and session parameters
    if parsed_url.query:
        # Parse the query parameters
        from urllib.parse import parse_qs, urlencode
        query_params = parse_qs(parsed_url.query)
        
        # List of parameters to remove
        params_to_remove = [
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content', 
            'gclid', 'fbclid', '_ga', 'ref', 'source', 'WT.mc_id', 'sid',
            'share', '_hsenc', '_hsmi', 'yclid', 'mkt_tok',
            'redirect', 'return', 'return_to', 'auth', 'token'
        ]
        
        # Remove parameters
        filtered_params = {k: v for k, v in query_params.items() if k not in params_to_remove}
        
        # Reconstruct query
        query = urlencode(filtered_params, doseq=True) if filtered_params else ''
    else:
        query = ''
        
    # Reconstruct the URL
    normalized_url = urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        path,
        parsed_url.params,
        query,
        fragment
    ))
    
    # Remove standard ports (http:80, https:443)
    if parsed_url.port == 80 and parsed_url.scheme == 'http':
        normalized_url = normalized_url.replace(':80', '')
    elif parsed_url.port == 443 and parsed_url.scheme == 'https':
        normalized_url = normalized_url.replace(':443', '')
        
    return normalized_url

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
        logger.debug(f"Skipping special URL format: {href[:30]}...")
        return False, ""
    
    # Skip common social media platforms
    social_media_domains = [
        'twitter.com', 'facebook.com', 'linkedin.com', 'instagram.com', 
        'youtube.com', 'reddit.com', 'tiktok.com', 'pinterest.com', 'tumblr.com'
    ]
    
    if any(domain in href for domain in social_media_domains):
        logger.debug(f"Skipping social media URL: {href[:50]}...")
        return False, ""
        
    # Skip platform-specific pages that are typically not useful (login, auth, settings, etc.)
    skip_patterns = [
        # GitHub and Git platforms
        '/login', '/signup', '/join', '/settings', '/notifications', 
        '/issues/new', '/pull/new', '/marketplace', '/pricing', 
        '/features', '/customer-stories', '/security', '/site/terms',
        '/site/privacy', '/contact', '/about', '/careers',
        'github.com/login', 'github.com/join', 'github.com/features',
        # Common auth & utility paths
        'sign-in', 'log-in', 'register', 'password-reset',
        # Notion and documentation platforms
        'notion.so/login', 'notion.so/signup', 'notion.so/pricing',
        '/api-docs', '/graphql', '/api-reference', '/robots.txt',
        # Analytics, tracking, and utility
        '/favicon.ico', '/sitemap.xml', '/rss', '/feed',
        '/xmlrpc.php', '/wp-login.php', '/wp-admin',
        # Admin-specific paths
        '/admin/', '/dashboard/', '/wp-admin/', '/cpanel/'
    ]
    
    # Check for skip patterns
    for pattern in skip_patterns:
        if pattern.lower() in href.lower():
            logger.debug(f"Skipping platform utility URL with pattern '{pattern}': {href[:50]}...")
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
        skipped_ext = next((ext for ext in skip_extensions if absolute_url.lower().endswith(ext)), None)
        logger.debug(f"Skipping file with extension {skipped_ext}: {absolute_url[:50]}...")
        return False, ""
    
    # Additional GitHub-specific pattern filtering
    # These are GitHub repository pages that often don't contain useful content
    if 'github.com/' in absolute_url.lower():
        # First, check if this is a content page we want to keep
        github_keep_patterns = [
            # README and documentation files
            '/blob/main/README.md', '/blob/master/README.md',
            '/blob/main/docs/', '/blob/master/docs/',
            '/blob/main/documentation/', '/blob/master/documentation/',
            '/wiki/Home', '/wiki/Documentation',
            # Actual content files
            '/blob/main/', '/blob/master/', 
            '/tree/main/', '/tree/master/',
            # Specific issues or PRs (these might have useful content)
            '/issues/[0-9]+$', '/pull/[0-9]+$'
        ]
        
        # Use regex to check keep patterns
        import re
        for pattern in github_keep_patterns:
            if '[0-9]+' in pattern:
                if re.search(pattern, absolute_url):
                    # This is a content URL we want to keep
                    logger.debug(f"Keeping GitHub content URL: {absolute_url[:50]}...")
                    break
            elif pattern in absolute_url:
                # This is a content URL we want to keep
                logger.debug(f"Keeping GitHub content URL: {absolute_url[:50]}...")
                break
        else:
            # No keep pattern matched, now check skip patterns
            github_skip_patterns = [
                '/commit/', '/commits/', '/network/', '/stargazers',
                '/blame/', '/watchers/', '/graphs/', '/settings/',
                '/branches/', '/releases/', '/tags/', '/milestones/',
                '/compare/', '/fork/', '/forks/', '/wiki/',
                '/actions/', '/activity/', '/deployments/',
                '/labels/', '/discussions/', '/pulse/',
                '/community/', '/actions/runs/', '/projects/',
                '/actions/workflows/', '/traffic/', '/security/',
                '/subscription', '/install/', '/events/', '/members/',
                '/people/', '/topics/', '/followers/', '/following/',
                '/raw/'
            ]
            
            for pattern in github_skip_patterns:
                if pattern in absolute_url:
                    logger.debug(f"Skipping GitHub utility URL with pattern '{pattern}': {absolute_url[:50]}...")
                    return False, ""
                    
            # Additional regex-based filtering for GitHub
            import re
            
            # Skip issue and pull request listings, but allow specific issues/PRs that might contain content
            if re.search(r'github\.com/[^/]+/[^/]+/(issues|pulls)$', absolute_url):
                logger.debug(f"Skipping GitHub issue/PR listing page: {absolute_url}")
                return False, ""
                
            # Skip repository listing pages
            if re.search(r'github\.com/[^/]+/[^/]+\?tab=repositories$', absolute_url):
                logger.debug(f"Skipping GitHub repository listing page: {absolute_url}")
                return False, ""
                
            # Skip stargazer pages with pagination
            if re.search(r'github\.com/[^/]+\?tab=(stars|repositories|projects|packages|followers|following)', absolute_url):
                logger.debug(f"Skipping GitHub profile tab: {absolute_url}")
                return False, ""
                
            # Skip user profile pages that don't contain specific content
            if re.search(r'github\.com/[^/]+$', absolute_url) and not re.search(r'github\.com/[^/]+/[^/]+', absolute_url):
                logger.debug(f"Skipping GitHub user profile page: {absolute_url}")
                return False, ""
    
    # Notion-specific filtering
    if 'notion.so' in absolute_url.lower():
        notion_skip_patterns = [
            '/login', '/signup', '/onboarding', '/request-invite', 
            '/changelog', '/pricing', '/enterprise', '/teams', '/personal',
            '/product', '/customers', '/remote', '/guides', '/help',
            '/security', '/terms', '/privacy',
            # Notion system controls and features
            '?openExternalBrowser=1', '?nav=', '?pvs=', '&pvs=', 
            '&openExternalBrowser=', '?p=', '&p='
        ]
        
        for pattern in notion_skip_patterns:
            if pattern in absolute_url:
                logger.debug(f"Skipping Notion utility page with pattern '{pattern}': {absolute_url[:50]}...")
                return False, ""
                
    # Common documentation platform patterns to skip
    doc_platform_skip_patterns = {
        'readthedocs.io': ['/search/', '/builds/', '/downloads/', '/versions/', '/edit/'],
        'docs.google.com': ['settings', 'u/0', 'document/create'],
        'confluence': ['/plugins/', '/spaces/', '/admin/', '/setup/', '/display/'],
        'gitbook.io': ['/settings/', '/account/', '/billing/'],
        'docusaurus': ['/docs/next/', '/blog/tags/']
    }
    
    for platform, patterns in doc_platform_skip_patterns.items():
        if platform in absolute_url.lower():
            for pattern in patterns:
                if pattern in absolute_url:
                    logger.debug(f"Skipping documentation platform ({platform}) utility page: {absolute_url[:50]}...")
                    return False, ""
    
    # Skip URLs with common tracking or non-content parameters
    parsed_url = urlparse(absolute_url)
    if parsed_url.query:
        skip_params = [
            'utm_', 'ref=', 'source=', 'medium=', 'campaign=',
            'content=', 'term=', 'gclid=', 'fbclid=', 'share=',
            'token=', 'auth=', 'session=', 'login=', '_ga=',
            'redirect=', 'return_to=', 'lang=', 'locale=',
            'page=', 'sort=', 'order=', 'filter=', 'view=',
            'theme=', 'v=', 'sid=', '_hsenc=', '_hsmi='
        ]
        
        # Check for query parameters that suggest the URL is not content-rich
        from urllib.parse import parse_qs
        query_params = parse_qs(parsed_url.query)
        
        # If there are too many query parameters (suggesting filters, sorting, etc.) 
        # or if they contain tracking/session parameters, skip the URL
        if len(query_params) > 3 or any(param.startswith(prefix) for param in query_params for prefix in skip_params):
            logger.debug(f"Skipping URL with tracking/utility query parameters: {absolute_url[:50]}...")
            return False, ""
            
    # Apply URL quality scoring to filter out low-quality URLs
    # This uses heuristics to determine if a URL likely contains valuable content
    # Higher scores (closer to 1.0) indicate more valuable content
    quality_score = score_url_quality(absolute_url)
    if quality_score < 0.3:  # Threshold can be adjusted based on observations
        logger.debug(f"Skipping low-quality URL (score={quality_score:.2f}): {absolute_url[:50]}...")
        return False, ""
        
    # The quality score is already calculated above, no need to calculate it twice
    # Skip URLs with low quality scores - this redundant check was removed
    
    return True, absolute_url

def score_url_quality(url: str) -> float:
    """
    Assigns a quality score to a URL based on heuristics,
    higher scores suggest more valuable content.
    
    Args:
        url: The URL to score
        
    Returns:
        Score between 0.0 and 1.0 (higher is better)
    """
    if not url:
        return 0.0
        
    score = 0.5  # Default mid-point score
    
    # Parse the URL
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()
    domain = parsed_url.netloc.lower()
    
    # Domain-based scoring
    if any(edu_domain in domain for edu_domain in ['.edu', '.ac.uk', '.ac.']):
        score += 0.2  # Educational domains likely have good content
    
    if 'github.com' in domain and '/blob/' in path:
        # GitHub file content
        if any(ext in path for ext in ['.md', '.rst', '.txt', '.ipynb']):
            score += 0.25  # Markdown and documentation files
        elif any(ext in path for ext in ['.py', '.js', '.java', '.cpp', '.h', '.c']):
            score += 0.15  # Source code files
        elif 'readme' in path.lower():
            score += 0.25  # README files
    
    # Path-based scoring
    if '/docs/' in path or '/documentation/' in path:
        score += 0.2  # Documentation paths
    elif '/blog/' in path:
        score += 0.15  # Blog content
    elif '/tutorials/' in path or '/guide/' in path:
        score += 0.2  # Tutorials and guides
    elif '/examples/' in path:
        score += 0.15  # Example code
        
    # File extensions that suggest good content
    if path.endswith(('.html', '.htm')):
        score += 0.05
    elif path.endswith(('.md', '.rst', '.txt', '.pdf')):
        score += 0.15
    elif path.endswith(('.ipynb')):
        score += 0.2  # Jupyter notebooks are often content-rich
        
    # Penalize certain patterns
    if 'login' in path or 'signin' in path or 'signup' in path:
        score -= 0.3
    if '/admin/' in path or '/settings/' in path:
        score -= 0.3
        
    # Length of path can be an indicator of specific content
    # (but not too long, as that might be a generated path)
    path_segments = [p for p in path.split('/') if p]
    if 2 <= len(path_segments) <= 5:
        score += 0.05
    elif len(path_segments) > 5:
        score -= 0.1
        
    # Ensure score is in [0.0, 1.0] range
    return max(0.0, min(1.0, score))

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
