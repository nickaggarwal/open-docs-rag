# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI-based RAG (Retrieval Augmented Generation) documentation assistant that crawls documentation websites, processes and embeds content, and uses LLMs to answer questions. The system supports both Azure OpenAI and standard OpenAI APIs with seamless switching via environment variables.

## Core Architecture

The application follows a modular architecture with these main components:

- **FastAPI Application** (`app/main.py`): Main server with streaming SSE endpoints, authentication middleware, and background job management
- **Vector Store** (`app/faiss_vector_store.py`): FAISS-based similarity search with OpenAI embeddings
- **LLM Interface** (`app/llm_interface.py`): Flexible interface supporting both Azure OpenAI and OpenAI APIs with streaming responses
- **Web Crawler** (`app/crawler.py`): Async crawler with incremental crawling and memory persistence
- **Document Processor** (`app/document_processor.py`): Text chunking with timeout protection
- **Database Layer** (`app/database.py`): SQLite database for Q&A history storage
- **Widget Component** (`widget/`): TypeScript widget for embedding RAG functionality into websites

## Development Commands

### Python Application
```bash
# ALWAYS USE build_and_run.sh to start the application (handles Docker setup properly)
sh build_and_run.sh                 # Primary method to build and run the application

# Alternative methods (only if Docker is unavailable)
python -m app.main                   # Start FastAPI server on localhost:8000 (requires venv activation)
source venv/bin/activate && python -m app.main  # With virtual environment

# Testing
python test_rag_app.py               # Run comprehensive integration tests
python test_api_endpoints.py         # Test API endpoints
python real_world_test.py            # Real-world testing script
pytest                               # Run pytest test suite (configured in pytest.ini)

# Docker management (if needed manually)
docker-compose up -d                 # Start containerized application
docker-compose down                  # Stop containers

# Dependencies (only if not using Docker)
pip install -r requirements.txt      # Install Python dependencies in virtual environment
```

### Widget Development
```bash
cd widget/
npm install                          # Install dependencies
npm run dev                          # Development server with hot reload
npm run build                        # Production build (creates dist/ for NPM)
npm run build:bundle                 # Webpack bundle build
npm run build:npm                    # Rollup NPM package build
npm run lint                         # ESLint code analysis
npm run type-check                   # TypeScript type checking
npm run clean                        # Clean dist directory
```

## Environment Configuration

The application uses a flexible configuration system supporting both Azure OpenAI and standard OpenAI:

### Key Environment Variables
- `USE_AZURE_OPENAI`: Set to "true" for Azure OpenAI, "false" for standard OpenAI
- `AUTH_TOKEN`: Optional token for endpoint authentication (if not set, auth is disabled)
- `DATABASE_URL`: SQLite database path (defaults to `sqlite:///./data/rag_database.db`)

### API Configuration
When `USE_AZURE_OPENAI=true`:
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT`, `AZURE_EMBEDDING_DEPLOYMENT`

When `USE_AZURE_OPENAI=false`:
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL`

## Key Features

### Streaming SSE Responses
The `/question` endpoint uses Server-Sent Events for real-time streaming responses. The LLM interface supports streaming for both Azure OpenAI and OpenAI APIs.

### Incremental Crawling
The crawler includes memory persistence (`data/crawl_memory.json`) and incremental updates. When `incremental=true`, only changed pages are re-processed and old documents from the domain are replaced.

### Authentication System
Optional token-based authentication protects `/crawl` and `/question` endpoints. If `AUTH_TOKEN` is not set, all endpoints work without authentication.

### Background Job Processing
Crawl operations run as background jobs with status tracking via `/job-status/{job_id}` endpoint.

## Data Storage

- **Vector Index**: FAISS index stored in `data/faiss_index/` with JSON metadata files
- **Database**: SQLite database at `data/rag_database.db` for Q&A history
- **Crawl Memory**: JSON file tracking crawled pages and checksums for incremental updates

## API Integration Patterns

The codebase demonstrates dual API support patterns:
- Dynamic client initialization based on environment flags
- Unified interface classes that abstract API differences
- Environment-driven configuration switching

## Testing Strategy

- `test_rag_app.py`: Comprehensive integration test covering crawling, processing, and Q&A
- `test_api_endpoints.py`: API endpoint validation
- `real_world_test.py`: Real-world scenario testing
- Widget has separate TypeScript testing setup with Jest/ESLint