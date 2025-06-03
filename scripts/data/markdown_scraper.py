import re
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import httpx
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import sys
import traceback
from langchain_community.document_loaders import TextLoader

logger = logging.getLogger(__name__)

class MarkdownScraper:
    def __init__(self, data_path: Path, output_path: Optional[Path] = None):
        """
        Initialize the markdown scraper.
        
        Args:
            data_path: Directory containing markdown files to process
            output_path: Directory to output CSV files (defaults to data_path)
        """
        self.data_path = Path(data_path)
        self.output_path = output_path or self.data_path
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers - keep only resources
        self.resources = []
        self.urls_to_scrape = set()
        
        # Regex patterns for extracting data
        self.name_pattern = re.compile(r'- \*\*(.*?)\*\*')
        self.website_pattern = re.compile(r'\[(.*?)\]\((https?://.*?)\)')
        self.tag_pattern = re.compile(r'#(\w+(-\w+)*)')
        
        # HTTP client with reasonable timeout
        self.http_timeout = 10.0
    
    def process_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single markdown file and extract URLs."""
        logger.info(f"Processing file: {file_path}")
        logger.info(f"File exists: {file_path.exists()}")
        stats = {"resources": 0}
        
        try:
            # For simplicity and better debugging, use TextLoader directly
            with open(file_path, 'r') as f:
                content = f.read()
                logger.info(f"Raw file content:\n{content[:500]}...")
                
            # Create simple document with the content
            docs = [{
                "page_content": content,
                "metadata": {"source": str(file_path)}
            }]
            
            for doc in docs:
                # Get content depending on format
                if isinstance(doc, dict):
                    content = doc["page_content"]
                else:
                    content = doc.page_content
                
                # Extract resources (URLs) only
                extracted_data = self._extract_data_from_content(content, file_path)
                
                # Update stats
                num_resources = len(extracted_data.get("resources", []))
                stats["resources"] += num_resources
                
                # Log what was found in this file
                logger.info(f"Extracted from {file_path.name}: {num_resources} resources")
                
                # If resources were found, log details for debugging
                if num_resources > 0:
                    for i, resource in enumerate(extracted_data.get("resources", [])):
                        logger.info(f"  Resource {i+1}: '{resource.get('title')}' - {resource.get('url')}")
                
                # Add to global collections - only resources now
                self.resources.extend(extracted_data.get("resources", []))
                
            return {
                "status": "success",
                "file_path": str(file_path),
                "stats": stats
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {
                "status": "error",
                "file_path": str(file_path),
                "error": str(e)
            }
    
    def _extract_data_from_content(self, content: str, source_file: Path) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract URLs from markdown content.
        
        Returns a dictionary with lists of resources.
        
        Supports both:
        1. Regular markdown with [title](url) format links
        2. YAML frontmatter with website URLs
        """
        results = {"resources": []}
        
        # Split content into lines for processing
        lines = content.split('\n')
        
        # Check for YAML frontmatter format with websites
        in_frontmatter = False
        website_field_found = False
        
        # Process YAML frontmatter if present
        for i, line in enumerate(lines):
            # Detect YAML frontmatter start/end
            if line.strip() == '---' and i == 0:
                in_frontmatter = True
                continue
            elif line.strip() == '---' and in_frontmatter:
                in_frontmatter = False
                continue
                
            if in_frontmatter and 'website:' in line:
                website_field_found = True
                
            # Extract websites from YAML or structured content
            if 'website:' in line:
                website_field_found = True
                parts = line.split('website:', 1)
                if len(parts) == 2:
                    url = parts[1].strip()
                    # Get name from previous lines if possible
                    title = "Website"
                    
                    # For student list format, get the name from the corresponding line
                    # Format is typically "    Name:\n        website: http://..."
                    current_indentation = len(line) - len(line.lstrip())
                    
                    # Find the immediate parent line (with less indentation)
                    for prev_idx in range(i-1, max(0, i-5), -1):
                        prev_line = lines[prev_idx]
                        prev_indentation = len(prev_line) - len(prev_line.lstrip())
                        
                        # If this line has less indentation and ends with colon, it's likely the name
                        if prev_indentation < current_indentation and ':' in prev_line:
                            potential_title = prev_line.strip().rstrip(':').strip()
                            if potential_title and len(potential_title) > 0:
                                title = potential_title
                                break
                            
                    if url:
                        logger.info(f"Found URL in content: {title} - {url}")
                        self.urls_to_scrape.add((title, url))
                        
                        resource = {
                            "title": title,
                            "url": url,
                            "description": f"Website for {title}",
                            "type": "website",
                            "category": "imported",
                            "tags": "",
                            "created_at": datetime.now().isoformat()
                        }
                        results["resources"].append(resource)
        
        # If we didn't find websites in YAML, try regular markdown link format
        if not website_field_found:
            # Check for structured list with names and websites
            name_line_pattern = re.compile(r'- \*\*(.*?)\*\*')
            current_name = None
            
            # Collect URLs from all lines
            for i, line in enumerate(lines):
                # First check if this is a name line
                name_match = name_line_pattern.search(line)
                if name_match:
                    current_name = name_match.group(1).strip()
                    logger.info(f"Found name: {current_name}")
                    continue
                
                # Extract websites - these are what we'll actually scrape
                website_matches = self.website_pattern.findall(line)
                for title, url in website_matches:
                    # If this line is right after a name line, use that name as the title
                    if current_name and i > 0 and name_line_pattern.search(lines[i-1]):
                        resource_id = current_name
                    else:
                        resource_id = title
                    
                    logger.info(f"Found URL to scrape: {resource_id} - {url}")
                    
                    # Add to URLs to scrape later
                    self.urls_to_scrape.add((resource_id, url))
                    
                    # Extract tags if any
                    tags = self.tag_pattern.findall(line)
                    logger.info(f"Tags for {resource_id}: {tags}")
                    
                    # Extract just the tag names
                    tag_names = [tag[0] for tag in tags]
                    
                    resource = {
                        "title": resource_id,
                        "url": url,
                        "description": line,
                        "type": "website",
                        "category": "imported",
                        "tags": ",".join(tag_names),  # Store as comma-separated values for consistency
                        "created_at": datetime.now().isoformat()
                    }
                    results["resources"].append(resource)
        
        return results
        
    async def scrape_url(self, title: str, url: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Basic URL scraping to get title and description for resources.
        Only extracts minimal information needed for the resource entry.
        No longer attempts to extract definitions, projects, or methods.
        """
        results = {"resources": []}
        logger.info(f"Basic scraping of URL for metadata: {url}")
        
        try:
            # Use httpx for the request
            async with httpx.AsyncClient(timeout=self.http_timeout) as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                
                # Parse the HTML with BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Get page title if not provided
                page_title = title
                if not page_title and soup.title:
                    page_title = soup.title.text.strip()
                if not page_title:
                    page_title = urlparse(url).netloc
                
                # Try to get a better description
                description = f"Content from {url}"
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    description = meta_desc.get('content')
                elif soup.find('p'):
                    # Get first paragraph as fallback description
                    first_p = soup.find('p')
                    description = first_p.text.strip()[:200] + '...'
                
                # Create resource entry
                resource = {
                    "title": page_title,
                    "url": url,
                    "description": description,
                    "type": "website",
                    "category": "scraped",
                    "tags": "",  # Empty string for consistency with comma-separated format
                    "created_at": datetime.now().isoformat()
                }
                results["resources"].append(resource)
                
                return results
                
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.error(f"Error scraping {url}: {e}")
            return results
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {e}")
            return results
    
    async def process_directory(self) -> Dict[str, Any]:
        """Process all markdown files in the data directory and scrape URLs found."""
        logger.info(f"Processing markdown files in {self.data_path}")
        
        # Find all markdown files
        markdown_files = list(self.data_path.glob("**/*.md"))
        if not markdown_files:
            logger.warning(f"No markdown files found in {self.data_path}")
            return {"status": "warning", "message": "No markdown files found"}
        
        # First pass: Process files and collect URLs
        results = []
        for file_path in markdown_files:
            result = self.process_file(file_path)
            results.append(result)
            
        # Print summary of URLs found
        if self.urls_to_scrape:
            logger.info(f"Found URLs to scrape:")
            for title, url in self.urls_to_scrape:
                logger.info(f"  - {title}: {url}")
        else:
            logger.warning("No URLs found to scrape")
        
        # Second pass: Scrape all collected URLs
        if self.urls_to_scrape:
            logger.info(f"Found {len(self.urls_to_scrape)} URLs to scrape")
            scrape_tasks = []
            
            # Create tasks for parallel scraping (with a reasonable limit)
            for title, url in self.urls_to_scrape:
                scrape_tasks.append(self.scrape_url(title, url))
            
            # Execute scraping tasks with concurrency control (max 5 concurrent requests)
            # Process in batches to avoid overwhelming the system
            BATCH_SIZE = 5
            for i in range(0, len(scrape_tasks), BATCH_SIZE):
                batch = scrape_tasks[i:i+BATCH_SIZE]
                scrape_results = await asyncio.gather(*batch)
                
                # Process each result and add to our collections
                for result in scrape_results:
                    self.resources.extend(result.get("resources", []))
        
        # Save all extracted data to CSV files
        self._save_to_csv()
        
        return {
            "status": "success",
            "files_processed": len(results),
            "urls_scraped": len(self.urls_to_scrape),
            "results": results,
            "data_extracted": {
                "resources": len(self.resources)
            }
        }
    
    def _save_to_csv(self) -> None:
        """Save extracted resources to CSV file."""
        logger.info(f"Saving extracted data to CSV files in {self.output_path}")
        
        # Save resources
        if self.resources:
            logger.info(f"Saving {len(self.resources)} resources to resources.csv")
            self._save_dataframe(
                pd.DataFrame(self.resources),
                "resources.csv"
            )
        else:
            logger.info("No resources to save")
    
    def _save_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """Save dataframe to CSV, merging with existing data if available."""
        file_path = self.output_path / filename
        
        try:
            if file_path.exists():
                # Read existing data
                existing_df = pd.read_csv(file_path)
                
                # Add processed flag column if needed
                if "processed" not in existing_df.columns:
                    existing_df["processed"] = False
                    logger.info(f"Added 'processed' column to {filename}")
                
                # Merge without duplicates
                if "url" in df.columns:
                    if "processed" not in df.columns:
                        df["processed"] = False
                    merged_df = pd.concat([existing_df, df]).drop_duplicates(subset=["url"])
                else:
                    if "processed" not in df.columns:
                        df["processed"] = False
                    merged_df = pd.concat([existing_df, df])
                
                merged_df.to_csv(file_path, index=False)
                logger.info(f"Updated {file_path} with {len(df)} new records")
            else:
                # Create new file - add processed flag
                if "processed" not in df.columns:
                    df["processed"] = False
                df.to_csv(file_path, index=False)
                logger.info(f"Created {file_path} with {len(df)} records")
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")

# Standalone execution
async def main_async():
    import argparse
    parser = argparse.ArgumentParser(description="Scrape markdown files and websites to extract structured data")
    parser.add_argument("--input", "-i", required=True, help="Directory containing markdown files")
    parser.add_argument("--output", "-o", help="Directory to output CSV files")
    parser.add_argument("--no-scrape", action="store_true", help="Skip web scraping, only process markdown files")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the scraper
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    scraper = MarkdownScraper(input_path, output_path)
    
    if args.no_scrape:
        # Modified version that doesn't do web scraping
        markdown_files = list(input_path.glob("**/*.md"))
        results = []
        for file_path in markdown_files:
            result = scraper.process_file(file_path)
            results.append(result)
            
        scraper._save_to_csv()
        
        result_summary = {
            "status": "success",
            "files_processed": len(results),
            "urls_scraped": 0,
            "data_extracted": {
                "resources": len(scraper.resources)
            }
        }
    else:
        # Run full process including web scraping
        result_summary = await scraper.process_directory()
    
    # Print summary
    print(f"Processing complete!")
    print(f"Files processed: {result_summary['files_processed']}")
    if not args.no_scrape:
        print(f"URLs scraped: {result_summary['urls_scraped']}")
    print(f"Data extracted:")
    for key, value in result_summary['data_extracted'].items():
        print(f"  - {key}: {value}")

def main():
    # Create async event loop and run the async main
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        # Handle deprecation warning by creating a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async main function
        loop.run_until_complete(main_async())
    except Exception as e:
        print(f"Error running markdown scraper: {e}")
        traceback.print_exc()
    finally:
        if loop and not loop.is_closed():
            loop.close()

if __name__ == "__main__":
    main()