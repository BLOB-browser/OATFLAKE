from pathlib import Path
import logging
import json
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime
from .data_processor import DataProcessor
from .markdown_scraper import MarkdownScraper
from scripts.services.data_analyser import DataAnalyser

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
        
    async def process_markdown_files(self, skip_scraping: bool = False, analyze_resources: bool = True) -> Dict[str, Any]:
        """
        Process markdown files and update the knowledge base.
        
        Args:
            skip_scraping: If True, only extract links without web scraping
            analyze_resources: If True, analyze extracted resources with LLM
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
                
                # Check for existing processed status
                processed_files = set()
                md_status_path = self.data_path / "markdown_processed.json"
                if md_status_path.exists():
                    try:
                        with open(md_status_path, 'r') as f:
                            processed_status = json.load(f)
                            processed_files = set(processed_status.get("processed_files", []))
                            logger.info(f"Found {len(processed_files)} previously processed markdown files")
                    except Exception as e:
                        logger.error(f"Error reading markdown processed status: {e}")
                
                # Process only new or modified files
                file_results = []
                new_processed_files = []
                for file_path in markdown_files:
                    file_path_str = str(file_path)
                    
                    # Skip if already processed (unless forcing update)
                    if file_path_str in processed_files:
                        logger.info(f"Skipping already processed markdown file: {file_path}")
                        continue
                    
                    logger.info(f"Processing markdown file: {file_path}")
                    result = scraper.process_file(file_path)
                    file_results.append(result)
                    new_processed_files.append(file_path_str)
                    logger.info(f"Processed file {file_path}: extracted {len(scraper.resources)} resources")
                
                # Save processed status
                all_processed_files = list(processed_files) + new_processed_files
                try:
                    with open(md_status_path, 'w') as f:
                        json.dump({"processed_files": all_processed_files, "updated_at": datetime.now().isoformat()}, f)
                    logger.info(f"Updated markdown processing status: {len(new_processed_files)} new files, {len(all_processed_files)} total")
                except Exception as e:
                    logger.error(f"Error saving markdown processed status: {e}")
                
                # First save the extracted resources to CSV
                logger.info(f"Saving extracted resources to CSV: {len(scraper.resources)} resources")
                scraper._save_to_csv()
                
                # For the markdown processor, we just save the resources without analyzing
                # This avoids duplicating the analysis work and allows for separate analysis command
                logger.info(f"Resource analysis during markdown processing is now handled separately")
                logger.info(f"Use the /api/data/stats/knowledge/analyze-resources endpoint to analyze resources")
                
                # Initialize analysis metrics (since we're not analyzing in this path)
                resources_analyzed = 0
                
                results = {
                    "status": "success",
                    "files_processed": len(file_results),
                    "new_files_processed": len(new_processed_files),
                    "skipped_files": len(markdown_files) - len(new_processed_files),
                    "urls_scraped": 0,
                    "data_extracted": {
                        "resources": len(scraper.resources),
                    },
                    "analysis": {
                        "resources_analyzed": resources_analyzed,
                    }
                }
            else:
                # Full process with web scraping
                results = await scraper.process_directory()
                
                # Analyze resources after web scraping
                if analyze_resources:
                    resources_path = self.data_path / "resources.csv"
                    if resources_path.exists():
                        try:
                            logger.info(f"Analyzing resources after web scraping")
                            analyzer = DataAnalyser()
                            updated_resources, new_projects = analyzer.analyze_resources(
                                csv_path=str(resources_path)
                            )
                            
                            # Save the analyzed resources and projects
                            analyzer.save_updated_resources(updated_resources)
                            analyzer.save_projects_csv(new_projects)
                            
                            results["analysis"] = {
                                "resources_analyzed": len(updated_resources),
                                "projects_identified": len(new_projects),
                                "definitions_extracted": len(analyzer._get_definitions_from_resources(updated_resources))
                            }
                        except Exception as analyze_error:
                            logger.error(f"Error analyzing resources after scraping: {analyze_error}")
            
            # In the unified processing flow, we don't want to trigger a separate vector update
            # because the calling function (stats.py) will handle that immediately after
            # This prevents duplicate processing
            
            # Simply note that vector processing is deferred to the calling function
            results["note"] = "Vector processing deferred to calling function (knowledge/process endpoint)"
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing markdown files: {e}")
            return {
                "status": "error",
                "error": str(e)
            }