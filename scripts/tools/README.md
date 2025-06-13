# Scripts Tools

This directory contains production utilities for maintaining and managing OATFLAKE.

## URL Management Tools

### `cleanup_bad_urls.py`
Identifies and removes problematic URLs from `pending_urls.csv` that are:
- Invalid (malformed URLs like "https://")
- Inaccessible (404, 403, connection errors)
- Blocked by robots.txt or other access restrictions

**Usage:**
```bash
# Check what would be removed (dry run)
python scripts/tools/cleanup_bad_urls.py --level=3 --dry-run

# Actually remove bad URLs
python scripts/tools/cleanup_bad_urls.py --level=3 --execute

# Check all levels, test up to 100 URLs
python scripts/tools/cleanup_bad_urls.py --max-test=100 --execute
```

**Options:**
- `--dry-run`: Show what would be removed without actually removing (default)
- `--execute`: Actually remove bad URLs
- `--level=N`: Only check URLs at specific level
- `--max-test=N`: Maximum URLs to test per level (default: 50)

**Safety Features:**
- Always creates a backup before making changes
- Detailed reporting of what was found and removed
- Concurrent URL testing for speed
- Comprehensive error handling

### `pdf_downloader.py`
Downloads PDF files from pending URLs for processing by the existing PDF processor.

**Usage:**
```bash
# List PDF URLs without downloading
python scripts/tools/pdf_downloader.py --list-only

# Download PDFs from specific level
python scripts/tools/pdf_downloader.py --level=3 --max-pdfs=5

# Download all PDFs (up to limit)
python scripts/tools/pdf_downloader.py --max-pdfs=20

# Custom download directory
python scripts/tools/pdf_downloader.py --download-dir=/path/to/pdfs --max-pdfs=10
```

**Options:**
- `--list-only`: Show PDF URLs without downloading
- `--level=N`: Process only URLs at specific level
- `--max-pdfs=N`: Maximum PDFs to download (default: 10)
- `--timeout=N`: Download timeout in seconds (default: 30)

**Features:**
- Detects PDF URLs by file extension and content-type
- Downloads to `data_folder/materials/` for later processing
- Creates unique filenames to avoid conflicts
- Generates download log with results

### `pdf_utils.py`
Core PDF detection and download utilities used by other components.

**Key Functions:**
- `is_pdf_url(url)`: Detects if URL points to a PDF
- `download_pdf_to_materials(url, data_folder, resource_id, logging_resource_id)`: Downloads PDF to materials folder
- `extract_pdfs_from_pending_urls(data_folder, level)`: Finds PDF URLs in pending URLs

**Integration:**
Used by `SingleResourceProcessorUniversal` to automatically handle PDFs in the analysis flow:
- PDFs are detected during URL processing
- Downloaded to materials folder instead of being analyzed by LLM
- URL marked as processed after successful download
- Regular web content continues through normal analysis pipeline

**Benefits:**
- Prevents LLM from trying to analyze binary PDF content
- Centralizes PDF handling logic
- Maintains clean separation between analysis and download logic
- Provides consistent error handling and logging

## Integration Patterns

### Knowledge Orchestrator Integration
The tools are designed to integrate with the main Knowledge Orchestrator workflow:

1. **URL Cleanup**: Run `cleanup_bad_urls.py` before processing to remove problematic URLs
2. **PDF Handling**: `pdf_utils.py` automatically handles PDFs during `process_specific_url()` calls
3. **Bulk Processing**: Use `pdf_downloader.py` for batch processing of discovered PDFs
4. **Materials Management**: All PDFs go to `data_folder/materials/` for later analysis by existing PDF processors

### Error Handling
All tools include comprehensive error handling:
- Timeouts for network operations
- Retry logic for failed operations
- Detailed logging and reporting
- Graceful degradation on failures
