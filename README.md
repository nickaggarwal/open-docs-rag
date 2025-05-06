# Azure-Powered RAG Documentation Assistant

A FastAPI application that implements Retrieval Augmented Generation (RAG) for documentation sites using Azure OpenAI for embeddings and LLM inference. The application crawls documentation websites, processes and embeds the content, and uses LLMs to answer questions based on the retrieved documents.

## Features

- **Web Crawler**: Asynchronously crawls documentation websites with configurable concurrency
- **Document Processing**: Chunks documents into appropriate sizes for embedding with timeout protection
- **Vector Storage Options**: 
  - **FAISS**: High-performance similarity search with efficient serialization
- **Azure OpenAI Integration**: Generates embeddings and answers using Azure's OpenAI service
- **Q&A History**: Stores all questions and answers in a database
- **API Endpoints**: FastAPI-based REST API with async/await for concurrent processing

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`
- Azure OpenAI API key and endpoint

## Installation

### Local Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory with your Azure OpenAI settings:
   ```
   # Azure OpenAI Configuration
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_API_VERSION=2024-02-01
   
   # Azure OpenAI Deployments
   AZURE_OPENAI_DEPLOYMENT=your_model_deployment_name
   AZURE_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
   
   # Database configuration
   DATABASE_URL=sqlite:///./data/rag_database.db
   ```

### Docker Deployment

The application can be easily deployed using Docker and Docker Compose:

1. Make sure Docker and Docker Compose are installed on your system
2. Create a `.env` file as described above
3. Build and start the container:
   ```bash
   docker-compose up -d
   ```
4. The application will be available at http://localhost:8000

To stop the container:
```bash
docker-compose down
```

To view logs:
```bash
docker-compose logs -f
```

The `data` directory is mounted as a volume, so your vector indices and database will persist between container restarts.

## Usage

Run the test script to verify functionality:
```bash
python test_rag_app.py
```

To test the API endpoints:
```bash
python test_api_endpoints.py
```

Start the application locally:
```bash
python -m app.main
```

This will start the FastAPI server on `http://localhost:8000`.

## Architecture

- **Vector Store**: FAISS-based vector storage for efficient similarity search
- **Concurrent Web Crawler**: Async crawler for efficient document retrieval
- **Document Processor**: Intelligent text chunking with timeout protection
- **LLM Interface**: Direct integration with Azure OpenAI API
- **Database**: SQLite database for storing question-answer history
- **Error Handling**: Improved error detection and recovery

## Azure OpenAI Integration

The system uses Azure OpenAI for two key components:
1. **Text Embeddings**: Converting document chunks to vector embeddings
2. **LLM Inference**: Generating coherent answers based on retrieved context

The application has been optimized to work with Azure OpenAI models including newer versions like `gpt-4o-mini`.

## API Endpoints

- **GET /** - Welcome page with endpoint descriptions
- **POST /crawl** - Crawl a documentation site
  ```json
  {
    "url": "https://example.com/docs",
    "max_pages": 100
  }
  ```
- **GET /job-status/{job_id}** - Check status of a crawl job
- **POST /question** - Ask a question and get an answer
  ```json
  {
    "question": "How do I use this API?",
    "num_results": 5
  }
  ```
- **POST /add-qa** - Add a custom Q&A pair
  ```json
  {
    "question": "What is RAG?",
    "answer": "Retrieval Augmented Generation is a technique...",
    "sources": ["https://example.com/rag-docs"]
  }
  ```
- **GET /history** - View Q&A history

## Testing

The test suite includes:
- Connection testing to Azure OpenAI
- Web crawling with timeout protection
- Document processing and chunking
- Vector store operations (add, search, save, load)
- End-to-end question answering

Run tests with:
```bash
python test_rag_app.py
