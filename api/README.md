# BLOB API Documentation

This directory contains the FastAPI routes and endpoints for the BLOB RAG platform. The API provides access to all platform functionality including authentication, data processing, question answering, and system management.

## API Structure

The API is organized into logical modules:

- `main.py`: Entry point and router registration
- `routes/`: Directory containing all route handlers grouped by functionality

## Available Endpoints

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/login` | POST | User login with email/password |
| `/api/auth/connect` | POST | Connect to a specific group |
| `/api/auth/status` | GET | Get current authentication status |
| `/api/auth/logout` | POST | Log out current user |

### Data Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/scan` | POST | Scan and process all data |
| `/api/storage/set` | POST | Set data storage path |
| `/api/data/markdown/process` | POST | Process all markdown files |
| `/api/data/markdown/upload` | POST | Upload a markdown file |

### Question Answering

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/questions` | GET | List all questions with answers |
| `/api/questions/ask` | POST | Ask a question against the knowledge base |
| `/api/questions/answers` | POST | Create a new answer |
| `/api/questions/generate` | POST | Trigger question generation |
| `/api/questions/generation/status` | GET | Get question generation status |

### Resource Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/resources` | GET | List all resources |
| `/api/resources/{resource_id}` | GET | Get a specific resource |
| `/api/methods` | GET | List all methods |
| `/api/definitions` | GET | List all definitions |
| `/api/projects` | GET | List all projects |

### Bucket Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/bucket/{bucket_type}` | POST | Create/update a bucket |
| `/api/bucket/info` | GET | Get bucket information |
| `/api/bucket/{bucket_type}/list` | GET | List buckets of a specific type |
| `/api/bucket/{bucket_type}/{name}` | DELETE | Delete a bucket |
| `/api/bucket/{bucket_type}/{name}/update` | POST | Update bucket metadata |
| `/api/search` | GET | Search across buckets |

### System Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Get system status |
| `/health` | GET | Health check endpoint |
| `/api/check-update` | GET | Check for system updates |
| `/api/system-settings` | GET/POST | Get/update system settings |
| `/api/training/schedule` | GET/POST | Get/update training schedule |

### LLM Interaction

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/web` | POST | Send prompts directly to the LLM |
| `/api/references` | POST | Get relevant references for a query |

## Example Requests

### Asking a Question

```bash
curl -X POST "http://localhost:8999/api/questions/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is BLOB?", "project_id": "default"}'
```

### Processing Markdown Content

```bash
curl -X POST "http://localhost:8999/api/data/markdown/process"
```

### Uploading a Markdown File

```bash
curl -X POST "http://localhost:8999/api/data/markdown/upload" \
  -F "file=@/path/to/document.md" \
  -F "title=Document Title" \
  -F "tags=tag1,tag2"
```

### Getting System Status

```bash
curl "http://localhost:8999/api/status"
```

## Authentication

Most endpoints require authentication. To authenticate, first call the login endpoint to receive a token, then include it in subsequent requests as a cookie or Authorization header.

## Error Handling

All API endpoints return standard error responses with appropriate HTTP status codes:

- 400: Bad Request - Missing parameters or invalid input
- 401: Unauthorized - Authentication required
- 403: Forbidden - Insufficient permissions
- 404: Not Found - Resource not found
- 500: Internal Server Error - Server-side error

## Response Format

Most endpoints return JSON responses with the following structure:

```json
{
  "status": "success",  // or "error"
  "data": {},           // Response data or null on error
  "message": ""         // Error message or null on success
}
```