#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import requests
from typing import Tuple, Dict, Any
from bs4 import BeautifulSoup
import urllib3
import csv
import os

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

class ContentFetcher:
    """Responsible for fetching and extracting content from websites"""
    
    def __init__(self, timeout: int = 90):
        self.timeout = timeout  # Increased from 30 to 90 seconds for Raspberry Pi
        # Add size limits matching ResourceLLM
        self.size_limits = {
            'description': 4000,    # For description generation
            'tags': 2500,          # For tag generation
            'definitions': 4000,    # For definition extraction
            'projects': 4000,      # For project identification
            'default': 2000        # Default limit
        }
        # Add SSL verification flag - useful for Raspberry Pi which might have issues
        self.verify_ssl = True
        
        # Add platform-specific User-Agents with updated Chrome version
        self.user_agents = {
            'windows': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'mac': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'raspberry': 'Mozilla/5.0 (Linux; Android 13; Raspberry Pi 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'
        }
        
        # Detect platform and set appropriate User-Agent and settings
        import platform
        self.system = platform.system().lower()
        
        # Standardized timeout and settings for all platforms
        self.timeout = 90  # Standard 90 second timeout for all platforms
        self.verify_ssl = False  # Disable SSL verification for compatibility
        
        # Set User-Agent based on platform but keep same behavior
        if 'darwin' in self.system:
            self.user_agent = self.user_agents['mac']
        elif 'linux' in self.system:
            self.user_agent = self.user_agents['raspberry']
        else:
            self.user_agent = self.user_agents['windows']
            
        logger.info(f"ContentFetcher initialized for {self.system} platform")
        logger.info(f"ContentFetcher using User-Agent for: {self.user_agent[:30]}...")
        logger.info(f"ContentFetcher initialized with size limits: {self.size_limits}")
        
        # SSL verification settings
        self.verify_ssl = False  # Disable SSL verification
        logger.info("SSL verification disabled for content fetching")
        
        # Initialize processed URLs storage
        self.processed_urls_file = "processed_urls.csv"
        if not os.path.exists(self.processed_urls_file):
            with open(self.processed_urls_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["url"])

    def _load_processed_urls(self):
        """Load processed URLs from the CSV file."""
        processed_urls = set()
        if os.path.exists(self.processed_urls_file):
            with open(self.processed_urls_file, mode='r') as file:
                reader = csv.reader(file)
                next(reader, None)  # Skip header
                for row in reader:
                    processed_urls.add(row[0])
        return processed_urls

    def _save_processed_url(self, url):
        """Save a processed URL to the CSV file."""
        with open(self.processed_urls_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([url])

    def get_main_page_with_links(self, url: str, max_depth: int = 2, go_deeper: bool = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Retrieves the main page content and identifies links to additional pages.
        Can go multiple layers deep to find more content-rich pages.
        
        Args:
            url: The URL to fetch
            max_depth: Maximum depth level to crawl (1=just main page links, 2=two levels, 3=three levels)
            go_deeper: (Deprecated) If True, will look for links two levels deep (use max_depth instead)
            
        Returns:
            Tuple of (success_flag, dict with main_html and additional_urls)
        """
        # Handle backward compatibility with go_deeper parameter
        if go_deeper is not None:
            logger.warning("Parameter 'go_deeper' is deprecated, use 'max_depth' instead")
            if go_deeper is False:
                max_depth = 1  # Only get main page links
            else:
                max_depth = 2  # Go two levels deep (original behavior)
        
        try:
            # Clean URL, handling trailing slashes and common issues
            url = url.strip()
            
            # Add http:// if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            # Store the base domain for building absolute URLs
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
            base_path = parsed_url.path.rsplit('/', 1)[0] if '/' in parsed_url.path else ''
            base_url = f"{base_domain}{base_path}"
            
            # Set headers with platform-specific User-Agent
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0'
            }
            
            # Get the main page
            main_response = requests.get(
                url, 
                headers=headers, 
                timeout=self.timeout,
                verify=self.verify_ssl,
                allow_redirects=True
            )
            main_response.raise_for_status()
            main_html = main_response.text
            
            # Parse the page to find important links
            soup = BeautifulSoup(main_html, 'html.parser')
            
            # Track visited URLs to avoid duplicates
            visited_urls = {url}
            additional_urls = []
            
            # Find links in the navigation menu - these are typically important sections
            nav_links = []
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
            
            # Process links, converting to absolute URLs
            for link in nav_links:
                href = link['href']
                # Skip social media and external links 
                if (href.startswith('#') or 'twitter.com' in href or 'facebook.com' in href or
                    'linkedin.com' in href or 'instagram.com' in href or 'mailto:' in href):
                    continue
                    
                # Convert to absolute URL if needed
                if href.startswith('/'):
                    absolute_url = f"{base_domain}{href}"
                elif not href.startswith(('http://', 'https://')):
                    absolute_url = f"{base_url}/{href}"
                else:
                    # Skip links to other domains
                    if parsed_url.netloc not in href:
                        continue
                    absolute_url = href
                
                # Skip if already visited or in the queue
                if absolute_url not in visited_urls and absolute_url not in additional_urls:
                    additional_urls.append(absolute_url)
            
            # If we're going deeper, fetch the level 1 pages to find level 2 links and possibly level 3
            level2_urls = []
            level3_urls = []
            
            if max_depth >= 2 and additional_urls:
                logger.info(f"Going deeper: examining up to 100 first-level pages")
                pages_to_check = additional_urls[:100]  # Check up to 100 pages at level 1
                
                for first_level_url in pages_to_check:
                    try:
                        logger.info(f"Checking for deeper links in {first_level_url}")
                        
                        # Fetch the first-level page
                        response = requests.get(
                            first_level_url,
                            headers=headers,
                            timeout=60,  # Increased from 10 to 60 seconds for Raspberry Pi
                            verify=self.verify_ssl,
                            allow_redirects=True
                        )
                        
                        if response.status_code != 200:
                            continue
                            
                        # Parse the HTML for links
                        first_level_soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find project-specific links that might contain detailed content
                        project_keywords = ['project', 'work', 'case', 'study', 'portfolio', 'article', 
                                           'publication', 'research', 'detail', 'post']
                        
                        # Find links with these keywords or in specific content sections
                        project_links = []
                        
                        # Look in likely project containers
                        for container in first_level_soup.select('.projects, .works, .portfolio, .case-studies, article, .research, .publications'):
                            for link in container.find_all('a', href=True):
                                project_links.append(link)
                        
                        # Also look for links with project-related text
                        if not project_links:
                            for link in first_level_soup.find_all('a', href=True):
                                link_text = link.get_text().lower().strip()
                                if any(keyword in link_text for keyword in project_keywords):
                                    project_links.append(link)
                        
                        # Process the found links
                        for link in project_links:
                            href = link['href']
                            
                            # Skip unwanted links
                            if (href.startswith('#') or any(domain in href for domain in 
                                                           ['twitter.com', 'facebook.com', 'linkedin.com', 
                                                            'instagram.com', 'mailto:'])):
                                continue
                                
                            # Convert to absolute URL if needed
                            if href.startswith('/'):
                                absolute_url = f"{base_domain}{href}"
                            elif not href.startswith(('http://', 'https://')):
                                # For relative URLs, we need to be more careful here
                                # If we're already on a first-level page, build from that URL
                                first_level_base = first_level_url.rsplit('/', 1)[0] if '/' in first_level_url else first_level_url
                                absolute_url = f"{first_level_base}/{href}"
                            else:
                                # Skip links to other domains
                                if parsed_url.netloc not in href:
                                    continue
                                absolute_url = href
                            
                            # Skip if already visited or in any queue
                            if (absolute_url not in visited_urls and 
                                absolute_url not in additional_urls and 
                                absolute_url not in level2_urls):
                                level2_urls.append(absolute_url)
                        
                        logger.info(f"Found {len(project_links)} potential project links on {first_level_url}")
                        
                    except Exception as e:
                        logger.warning(f"Error processing deeper links for {first_level_url}: {e}")
            
            # Process level 3 if requested
            if max_depth >= 3 and level2_urls:
                logger.info(f"Going to level 3: examining up to 50 second-level pages")
                pages_to_check_l2 = level2_urls[:50]  # Check up to 50 pages at level 2
                
                for second_level_url in pages_to_check_l2:
                    try:
                        logger.info(f"Checking for level 3 links in {second_level_url}")
                        
                        # Fetch the second-level page
                        response = requests.get(
                            second_level_url,
                            headers=headers,
                            timeout=60,
                            verify=self.verify_ssl,
                            allow_redirects=True
                        )
                        
                        if response.status_code != 200:
                            continue
                            
                        # Parse the HTML for links
                        second_level_soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Find detailed content links that might contain rich information
                        detail_keywords = ['detail', 'more', 'read', 'view', 'full', 'case', 'story', 
                                          'about', 'learn', 'article', 'page', 'post']
                        
                        # Find links with these keywords or in specific content sections
                        detail_links = []
                        
                        # Look in likely content containers
                        for container in second_level_soup.select('main, article, section, .content, .projects, .details, .post'):
                            for link in container.find_all('a', href=True):
                                detail_links.append(link)
                        
                        # Also look for links with detail-related text
                        if len(detail_links) < 5:  # If not enough links found in containers
                            for link in second_level_soup.find_all('a', href=True):
                                link_text = link.get_text().lower().strip()
                                if any(keyword in link_text for keyword in detail_keywords):
                                    detail_links.append(link)
                        
                        # Process the found links
                        for link in detail_links:
                            href = link['href']
                            
                            # Skip unwanted links
                            if (href.startswith('#') or any(domain in href for domain in 
                                                          ['twitter.com', 'facebook.com', 'linkedin.com', 
                                                           'instagram.com', 'mailto:'])):
                                continue
                                
                            # Convert to absolute URL if needed
                            if href.startswith('/'):
                                absolute_url = f"{base_domain}{href}"
                            elif not href.startswith(('http://', 'https://')):
                                # For relative URLs, build from the second-level URL's base
                                second_level_base = second_level_url.rsplit('/', 1)[0] if '/' in second_level_url else second_level_url
                                absolute_url = f"{second_level_base}/{href}"
                            else:
                                # Skip links to other domains
                                if parsed_url.netloc not in href:
                                    continue
                                absolute_url = href
                            
                            # Skip if already visited or in any queue
                            if (absolute_url not in visited_urls and 
                                absolute_url not in additional_urls and 
                                absolute_url not in level2_urls and
                                absolute_url not in level3_urls):
                                level3_urls.append(absolute_url)
                        
                        # Mark this URL as visited to avoid re-processing
                        visited_urls.add(second_level_url)
                        
                        logger.info(f"Found {len(detail_links)} potential level 3 links on {second_level_url}")
                        
                    except Exception as e:
                        logger.warning(f"Error processing level 3 links for {second_level_url}: {e}")
            
            # Combine all URLs, prioritizing higher levels
            if level3_urls:
                logger.info(f"Found {len(level3_urls)} third-level (deepest) URLs")
                combined_urls = additional_urls + level2_urls + level3_urls
            elif level2_urls:
                logger.info(f"Found {len(level2_urls)} second-level (deeper) URLs")
                combined_urls = additional_urls + level2_urls
            else:
                combined_urls = additional_urls
            
            # Return the main page HTML and the list of all additional URLs
            return True, {
                "main_html": main_html,
                "additional_urls": combined_urls,
                "base_url": base_url,
                "base_domain": base_domain,
                "headers": headers,
                "visited_urls": visited_urls
            }
            
        except requests.RequestException as e:
            logger.error(f"Error fetching main page {url}: {e}")
            return False, {"error": str(e)}

    def fetch_additional_page(self, url: str, headers: Dict = None, timeout: int = 60) -> Tuple[bool, str]:  # Increased timeout from 10 to 60 seconds for Raspberry Pi
        """
        Fetch a single additional page.
        
        Args:
            url: The URL to fetch
            headers: Request headers
            timeout: Request timeout
            
        Returns:
            Tuple of (success_flag, html_content)
        """
        try:
            if headers is None:
                headers = {
                    'User-Agent': self.user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                }
            
            response = requests.get(
                url,
                headers=headers,
                timeout=timeout,
                verify=self.verify_ssl,
                allow_redirects=True
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully fetched content from {url}")
                return True, response.text
            else:
                logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                return False, ""
                
        except Exception as e:
            logger.warning(f"Error fetching page {url}: {e}")
            return False, ""
    
    def fetch_content(self, url: str, max_additional_pages: int = 250, max_depth: int = 3) -> Tuple[bool, Dict[str, str]]:
        """
        Safely retrieves website content with error handling.
        Fetches main page and related pages, storing each separately.
        
        Args:
            url: The URL to fetch
            max_additional_pages: Maximum number of additional pages to fetch (increased to 250)
            max_depth: Maximum depth level to crawl (default=3 for three levels deep)
            
        Returns:
            Tuple of (success_flag, dict_of_pages)
            dict_of_pages format: {'main': main_html, 'url1': page1_html, ...}
        """
        try:
            # Load processed URLs
            processed_urls = self._load_processed_urls()

            if url in processed_urls:
                logger.info(f"Skipping already processed URL: {url}")
                return True, {"main": "", "error": "URL already processed"}

            # Get the main page and extract links to other pages with specified depth
            success, main_data = self.get_main_page_with_links(url, max_depth=max_depth)

            if not success or "error" in main_data:
                return False, {"error": main_data.get("error", "Unknown error fetching main page")}

            # Initialize the pages dictionary with the main page
            pages_dict = {'main': main_data["main_html"]}

            # Get the list of additional URLs
            additional_urls = main_data["additional_urls"]
            headers = main_data["headers"]
            visited_urls = main_data["visited_urls"]

            # Limit number of additional pages if needed
            initial_count = len(additional_urls)
            if initial_count > max_additional_pages:
                logger.info(f"Limiting from {initial_count} to {max_additional_pages} additional pages")
                additional_urls = additional_urls[:max_additional_pages]
            if additional_urls:
                logger.info(f"Found {len(additional_urls)} additional pages to fetch from {url}")

            # Fetch additional pages one by one
            for idx, additional_url in enumerate(additional_urls):
                if additional_url in processed_urls:
                    logger.info(f"Skipping already processed subpage URL: {additional_url}")
                    continue

                logger.info(f"Fetching additional page {idx+1}/{len(additional_urls)}: {additional_url}")
                success, html_content = self.fetch_additional_page(additional_url, headers)

                if success and html_content:
                    # Use page name as key (simplified URL)
                    section_name = additional_url.rsplit('/', 1)[-1] or f'page{idx+1}'
                    # Store each page separately
                    pages_dict[section_name] = html_content
                    visited_urls.add(additional_url)

                    # Save the processed subpage URL
                    self._save_processed_url(additional_url)

            # Save the main URL as processed
            self._save_processed_url(url)

            # Return all pages as a dictionary
            return True, pages_dict

        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return False, {"error": str(e)}
    
    def extract_text(self, html_content: str, limit_size: int = None) -> str:
        """
        Extracts readable text content from HTML with enhanced methods to get meaningful content.
        
        Args:
            html_content: Raw HTML content
            limit_size: Optional size limit for the extracted text
            
        Returns:
            Extracted text content
        """
        try:
            # Use the specified limit or the default
            size_limit = limit_size or self.size_limits['default']
            
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
            
    def extract_page_texts(self, pages_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Extract text from multiple HTML pages
        
        Args:
            pages_dict: Dictionary of HTML content keyed by page name
            
        Returns:
            Dictionary of extracted text content keyed by page name
        """
        texts = {}
        for page_name, html_content in pages_dict.items():
            # Skip error entries
            if page_name == "error":
                continue
                
            # For individual pages processing, use a consistent size limit to ensure equal treatment
            # Each page gets the same amount of content for fair analysis
            size_limit = 2000
            
            extracted_text = self.extract_text(html_content, size_limit)
            if extracted_text:
                texts[page_name] = extracted_text
                logger.info(f"Extracted {len(extracted_text)} chars from page '{page_name}'")
            else:
                logger.warning(f"No content could be extracted from page '{page_name}'")
                
        return texts