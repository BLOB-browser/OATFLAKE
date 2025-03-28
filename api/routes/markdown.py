from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from pathlib import Path
import shutil
import json
import logging
from datetime import datetime
from scripts.data.markdown_processor import MarkdownProcessor
from utils.config import BACKEND_CONFIG

router = APIRouter(prefix="/api/data/markdown", tags=["markdown"])
logger = logging.getLogger(__name__)

@router.post("/process")
async def process_markdown(skip_scraping: bool = False):
    """
    Process all markdown files in the markdown directory.
    
    This will extract structured data from the markdown files and save it to the appropriate CSV files.
    
    Args:
        skip_scraping: If True, only extract links without web scraping (faster but less comprehensive)
    """
    try:
        # Get data path from config
        data_path = Path(BACKEND_CONFIG['data_path'])
        group_id = 'default'  # Default group ID
        
        # Initialize markdown processor
        processor = MarkdownProcessor(data_path, group_id)
        
        # Process markdown files
        results = await processor.process_markdown_files(skip_scraping=skip_scraping)
        
        return {
            "status": "success",
            "message": "Markdown files processed successfully",
            "data": results
        }
        
    except Exception as e:
        logger.error(f"Error processing markdown files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_markdown_file(
    file: UploadFile = File(...),
    title: str = Form(None),
    description: str = Form(None),
    tags: str = Form(None),
    group_id: str = Form("default"),
    process_now: bool = Form(False),
    skip_scraping: bool = Form(False),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a markdown file to be processed.
    
    Args:
        file: The markdown file to upload
        title: Optional title for the file
        description: Optional description of the content
        tags: Optional comma-separated list of tags
        group_id: Group ID for organization (defaults to "default")
        process_now: Whether to process the file immediately
        skip_scraping: Whether to skip web scraping during processing
        background_tasks: For background processing
    
    The file will be saved to the markdown directory and can be processed later.
    """
    try:
        # Get data path from config
        data_path = Path(BACKEND_CONFIG['data_path'])
        
        # Create markdown path without group-specific subdirectory
        markdown_path = data_path / "markdown"
        markdown_path.mkdir(parents=True, exist_ok=True)
        
        # Validate file is a markdown file
        if not file.filename.lower().endswith(('.md', '.markdown')):
            raise HTTPException(
                status_code=400, 
                detail="Only markdown files (.md, .markdown) are allowed"
            )
        
        # Create a more descriptive filename if title is provided
        if title:
            # Convert title to slug-like filename
            import re
            safe_filename = re.sub(r'[^\w\s-]', '', title.lower())
            safe_filename = re.sub(r'[\s]+', '-', safe_filename)
            new_filename = f"{safe_filename}.md"
            file_path = markdown_path / new_filename
        else:
            file_path = markdown_path / file.filename
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Save metadata alongside the file if provided
        if title or description or tags:
            metadata = {
                "title": title or file.filename,
                "description": description or "",
                "tags": tags.split(",") if tags else [],
                "uploaded_at": datetime.now().isoformat(),
                "file_path": str(file_path)
            }
            
            metadata_path = file_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Process now if requested, or schedule in background
        if process_now:
            processor = MarkdownProcessor(data_path, group_id)
            if background_tasks:
                background_tasks.add_task(processor.process_markdown_files, skip_scraping=skip_scraping)
            else:
                try:
                    # Process immediately (may take time)
                    results = await processor.process_markdown_files(skip_scraping=skip_scraping)
                    
                    # Check if we had partial success
                    if results.get("status") == "partial_success":
                        return {
                            "status": "partial_success",
                            "message": f"File {file.filename} uploaded and basic processing completed, but there was an error with vector store updates",
                            "file_path": str(file_path),
                            "processing_results": results
                        }
                    
                    return {
                        "status": "success",
                        "message": f"File {file.filename} uploaded and processed successfully",
                        "file_path": str(file_path),
                        "processing_results": results
                    }
                except Exception as process_error:
                    logger.error(f"Error during processing: {process_error}")
                    # File was uploaded successfully but processing failed
                    return {
                        "status": "partial_success",
                        "message": f"File {file.filename} uploaded but processing failed: {str(process_error)}",
                        "file_path": str(file_path),
                        "error": str(process_error)
                    }
        
        return {
            "status": "success",
            "message": f"File {file.filename} uploaded successfully",
            "file_path": str(file_path)
        }
        
    except Exception as e:
        logger.error(f"Error uploading markdown file: {e}")
        # Return a JSON error response instead of raising an exception
        # This helps the frontend handle errors better
        return {
            "status": "error",
            "message": f"Error uploading markdown: {str(e)}",
            "error": str(e)
        }