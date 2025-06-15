from pathlib import Path
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from .markdown_scraper import MarkdownScraper

logger = logging.getLogger(__name__)

class MarkdownProcessor:
    def __init__(self, data_path: Path, group_id: str = "default"):
        """
        Initialize the markdown processor.
        
        Args:
            data_path: Base path for data
            group_id: Group ID for vector store organization
        """
        self.data_path = Path(data_path)
        self.group_id = group_id
        self.markdown_path = self.data_path / "markdown"
        self.markdown_path.mkdir(parents=True, exist_ok=True)
        
    async def process_markdown_files(self, skip_scraping: bool = False) -> Dict[str, Any]:
        """
        Process markdown files and extract resources to CSV.
        
        Args:
            skip_scraping: If True, only extract links without web scraping
        """
        try:
            logger.info(f"Starting markdown processing from {self.markdown_path}")
            
            # Initialize the markdown scraper
            logger.info(f"Initializing markdown scraper with input path: {self.markdown_path}, output path: {self.data_path}")
            scraper = MarkdownScraper(self.markdown_path, self.data_path)
            
            # Process markdown files
            if skip_scraping:
                # Manual non-scraping implementation
                markdown_files = list(self.markdown_path.glob("**/*.md"))
                logger.info(f"Found {len(markdown_files)} markdown files: {[f.name for f in markdown_files]}")
                
                # Check for existing processed status with file modification tracking
                processed_files = {}  # Change from set to dict to track modification times
                md_status_path = self.data_path / "markdown_processed.json"
                if md_status_path.exists():
                    try:
                        with open(md_status_path, 'r') as f:
                            processed_status = json.load(f)
                            processed_files = processed_status.get("processed_files", {})
                            logger.info(f"Found {len(processed_files)} previously processed markdown files with timestamps")
                    except Exception as e:
                        logger.error(f"Error reading markdown processed status: {e}")
                
                # Process only new or modified files
                file_results = []
                new_processed_files = {}
                for file_path in markdown_files:
                    file_path_str = str(file_path)
                    file_mod_time = file_path.stat().st_mtime
                    
                    # Skip if already processed and not modified since last processing
                    if file_path_str in processed_files:
                        last_processed_time = processed_files[file_path_str]
                        if file_mod_time <= last_processed_time:
                            logger.info(f"Skipping already processed markdown file (not modified): {file_path}")
                            continue
                        else:
                            logger.info(f"Re-processing modified markdown file: {file_path}")
                    
                    logger.info(f"Processing markdown file: {file_path}")
                    
                    # Track resources count before processing to detect duplications
                    resources_before = len(scraper.resources)
                    
                    result = scraper.process_file(file_path)
                    file_results.append(result)
                    
                    # Store with modification time to prevent future duplications
                    new_processed_files[file_path_str] = file_mod_time
                    
                    resources_after = len(scraper.resources)
                    new_resources = resources_after - resources_before
                    logger.info(f"Processed file {file_path}: extracted {new_resources} new resources (total: {resources_after})")
                
                # Save processed status with file modification times
                all_processed_files = {**processed_files, **new_processed_files}
                try:
                    with open(md_status_path, 'w') as f:
                        json.dump({
                            "processed_files": all_processed_files, 
                            "updated_at": datetime.now().isoformat(),
                            "processing_note": "File modification times tracked to prevent URL extraction duplications"
                        }, f, indent=2)
                    logger.info(f"Updated markdown processing status: {len(new_processed_files)} new/modified files, {len(all_processed_files)} total tracked")
                except Exception as e:
                    logger.error(f"Error saving markdown processed status: {e}")
                
                # Check for potential duplications in extracted resources
                unique_urls = set()
                duplicate_count = 0
                for resource in scraper.resources:
                    url = resource.get('origin_url', '')
                    if url in unique_urls:
                        duplicate_count += 1
                    else:
                        unique_urls.add(url)
                
                if duplicate_count > 0:
                    logger.warning(f"⚠️ Detected {duplicate_count} potential duplicate URLs in extracted resources")
                    logger.warning("Consider cleaning up the resources.csv file to remove duplicates")
                else:
                    logger.info(f"✅ No duplicate URLs detected in {len(scraper.resources)} extracted resources")
                
                # Save the extracted resources to CSV with deduplication
                logger.info(f"Saving extracted resources to CSV: {len(scraper.resources)} resources")
                scraper._save_to_csv()
                
                logger.info(f"Markdown processing complete - resources saved to CSV with state tracking")
                
                results = {
                    "status": "success",
                    "files_processed": len(file_results),
                    "new_files_processed": len(new_processed_files),
                    "skipped_files": len(markdown_files) - len(new_processed_files),
                    "urls_scraped": 0,
                    "duplicate_urls_detected": duplicate_count,
                    "unique_resources_extracted": len(unique_urls),
                    "data_extracted": {
                        "resources": len(scraper.resources),
                    },
                    "state_tracking": {
                        "enabled": True,
                        "files_tracked": len(all_processed_files),
                        "modification_times_tracked": True
                    }
                }
            else:
                # Full process with web scraping
                results = await scraper.process_directory()
                
                # Save the extracted resources to CSV
                logger.info(f"Saving extracted resources to CSV after web scraping")
                scraper._save_to_csv()
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing markdown files: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
