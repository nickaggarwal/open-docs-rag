import os
import logging
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio
import uvicorn
from dotenv import load_dotenv
from .middleware.json_error_handler import JSONErrorHandlerMiddleware
from .middleware.rate_limiter import RateLimiterMiddleware
import uuid
import urllib.parse

from app.faiss_vector_store import FAISSVectorStore
from app.crawler import crawl_website
from app.document_processor import DocumentProcessor
from app.llm_interface import LLMInterface
from app.database import Database

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Documentation Assistant",
    description="A RAG-based API for crawling documentation sites and answering questions using Azure OpenAI or OpenAI APIs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add JSON error handler middleware
app.add_middleware(
    JSONErrorHandlerMiddleware,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Add rate limiter middleware
app.add_middleware(RateLimiterMiddleware)

# Create data directory if it doesn't exist
os.makedirs("./data", exist_ok=True)
os.makedirs("./data/faiss_index", exist_ok=True)

# Determine which API to use
use_azure = os.getenv("USE_AZURE_OPENAI", "true").lower() == "true"
if use_azure:
    logger.info("Using Azure OpenAI API for language model")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    llm_interface = LLMInterface(deployment_name=deployment_name)
else:
    logger.info("Using direct OpenAI API for language model")
    model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    llm_interface = LLMInterface(model_name=model_name)

# Initialize components
vector_store = FAISSVectorStore(index_path="./data/faiss_index/index")
document_processor = DocumentProcessor()
database = Database()

# Authentication dependency
async def verify_token(authorization: str = Header(None)):
    """
    Verify the authorization token if AUTH_TOKEN environment variable is set.
    If AUTH_TOKEN is not set, authentication is skipped.
    """
    required_token = os.getenv("AUTH_TOKEN")
    
    # If no token is configured, skip authentication
    if not required_token:
        return True
    
    # Check if authorization header is provided
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is required"
        )
    
    # Extract token from "Bearer <token>" format
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization header must be in format 'Bearer <token>'"
        )
    
    token = authorization[7:]  # Remove "Bearer " prefix
    
    # Verify token
    if token != required_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )
    
    return True

# Pydantic models for request/response validation
class CrawlRequest(BaseModel):
    url: str
    max_pages: int = Field(default=100, ge=1, le=1000)
    incremental: bool = Field(default=True)

class CrawlResponse(BaseModel):
    job_id: str
    message: str
    status: str

class QuestionRequest(BaseModel):
    question: str
    num_results: int = Field(default=5, ge=1, le=10)

class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]
    question_id: Optional[int] = None

class QAPairRequest(BaseModel):
    question: str
    answer: str
    sources: Optional[List[str]] = None

class QAPairResponse(BaseModel):
    question_id: int
    answer_id: int
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    message: str
    documents_processed: Optional[int] = None

# Store active jobs
active_jobs = {}

@app.get("/")
async def root():
    # Get information about which API is being used
    api_type = "Azure OpenAI" if os.getenv("USE_AZURE_OPENAI", "true").lower() == "true" else "OpenAI"
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT") if api_type == "Azure OpenAI" else os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    return {
        "message": f"RAG Documentation Assistant API using {api_type}",
        "model": model,
        "endpoints": [
            "/crawl - Crawl a documentation site",
            "/question - Ask a question (streaming SSE response)",
            "/add-qa - Add a Q&A pair",
            "/job-status/{job_id} - Check job status",
            "/history - View Q&A history"
        ]
    }

@app.post("/crawl")
async def start_crawl(req: CrawlRequest, _: bool = Depends(verify_token)) -> Dict[str, Any]:
    """
    Start a crawl job for a documentation site
    
    Args:
        req: Crawl request with URL and max pages
        
    Returns:
        Job information with ID
    """
    try:
        url = req.url
        max_pages = req.max_pages
        incremental = req.incremental if hasattr(req, 'incremental') else True
        
        # Generate a job ID
        job_id = str(uuid.uuid4())
        
        # Add to active jobs
        active_jobs[job_id] = {
            "status": "running",
            "message": "Starting crawl job",
            "documents_processed": 0
        }
        
        # Launch crawl task in background
        asyncio.create_task(process_crawl_job(job_id, url, max_pages, incremental))
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": "Crawl job started"
        }
    except Exception as e:
        logger.error(f"Error starting crawl: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to start crawl: {str(e)}"
        }

async def process_crawl_job(job_id: str, url: str, max_pages: int, incremental: bool = True):
    """
    Process a crawl job in the background
    
    Args:
        job_id: Job ID
        url: URL to crawl
        max_pages: Maximum pages to crawl
        incremental: Whether to use incremental crawling
    """
    try:
        # Crawl the website - now returns both documents and whether changes were detected
        logger.info(f"Starting crawl of {url} with max_pages={max_pages}, incremental={incremental}")
        documents, has_changes = await crawl_website(url, max_pages, incremental)
        
        active_jobs[job_id]["message"] = f"Crawled {len(documents)} documents, changes detected: {has_changes}"
        
        if not documents:
            active_jobs[job_id]["status"] = "completed"
            active_jobs[job_id]["message"] = "No documents found or no changes detected"
            return
            
        # If incremental crawling is enabled and changes were detected, we need to remove old documents 
        # from this URL domain before adding new ones
        if incremental and has_changes:
            # Parse the domain from the URL to remove documents from that domain
            parsed_url = urllib.parse.urlparse(url)
            domain = parsed_url.netloc
            
            # Remove existing documents with the same domain
            active_jobs[job_id]["message"] = f"Removing old documents from {domain} before adding new ones"
            removed_count = await vector_store.remove_documents_by_domain(domain)
            logger.info(f"Removed {removed_count} old documents from domain {domain}")
        
        # Process documents into chunks
        processed_docs = await document_processor.process_documents(documents)
        
        active_jobs[job_id]["message"] = f"Processed into {len(processed_docs)} chunks, indexing..."
        active_jobs[job_id]["documents_processed"] = len(processed_docs)
        
        # Add to vector store
        await vector_store.add_documents(processed_docs)
        
        # ChromaDB automatically persists data, so no need for explicit save
        
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["message"] = f"Indexed {len(processed_docs)} document chunks successfully"
        
    except Exception as e:
        logger.error(f"Error in crawl job {job_id}: {str(e)}")
        active_jobs[job_id]["status"] = "error"
        active_jobs[job_id]["message"] = f"Error: {str(e)}"

@app.get("/job-status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Check the status of a background job
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        **active_jobs[job_id]
    }

@app.post("/question")
async def ask_question(request: QuestionRequest, _: bool = Depends(verify_token)):
    """
    Ask a question and get a streaming answer using Server-Sent Events (SSE)
    """
    async def generate_sse_stream():
        try:
            # Search for relevant documents
            relevant_docs = await vector_store.search(request.question, k=request.num_results)
            
            if not relevant_docs:
                # Send empty response for no documents
                yield f"data: {json.dumps({'type': 'sources', 'content': []})}\n\n"
                no_info_msg = "I don't have enough information to answer this question. Try crawling more documentation or asking a different question."
                yield f"data: {json.dumps({'type': 'answer', 'content': no_info_msg})}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Generate streaming answer using LLM
            full_answer = ""
            sources = []
            async for chunk in llm_interface.generate_answer_stream(request.question, relevant_docs):
                if chunk.startswith("data: [DONE]"):
                    # Store Q&A in database after streaming is complete
                    if full_answer and sources:
                        answer_data = {
                            "answer": full_answer,
                            "sources": sources
                        }
                        try:
                            await database.store_qa(request.question, answer_data)
                        except Exception as db_error:
                            logger.error(f"Error storing Q&A in database: {str(db_error)}")
                yield chunk
                
                # Parse chunk to extract data for database storage
                if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
                    try:
                        chunk_data = chunk[6:].strip()  # Remove "data: " prefix
                        if chunk_data and chunk_data != "[DONE]":
                            parsed = json.loads(chunk_data)
                            if parsed.get("type") == "answer_chunk":
                                full_answer += parsed.get("content", "")
                            elif parsed.get("type") == "sources":
                                sources = parsed.get("content", [])
                            elif parsed.get("type") == "answer":
                                full_answer = parsed.get("content", "")
                    except json.JSONDecodeError:
                        pass  # Ignore parsing errors for now
                        
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            error_msg = f"Error processing question: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache", 
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@app.post("/question_full", response_model=QuestionResponse)
async def ask_question_full(request: QuestionRequest, _: bool = Depends(verify_token)):
    """
    Ask a question and get a complete static answer (non-streaming)
    """
    try:
        logger.info(f"Processing question: {request.question}")
        logger.info(f"Requesting {request.num_results} results")
        
        # Debug: Check vector store state
        logger.info(f"Vector store has {len(vector_store.documents) if vector_store.documents else 0} documents")
        logger.info(f"Vector store index exists: {vector_store.index is not None}")
        
        # Search for relevant documents
        relevant_docs = await vector_store.search(request.question, k=request.num_results)
        
        logger.info(f"Found {len(relevant_docs)} relevant documents")
        if relevant_docs:
            for i, doc in enumerate(relevant_docs):
                logger.info(f"Document {i+1}: score={doc.get('score', 'N/A')}, metadata={doc.get('metadata', {})}")
        
        if not relevant_docs:
            no_info_msg = "I don't have enough information to answer this question. Try crawling more documentation or asking a different question."
            logger.warning("No relevant documents found for the question")
            return QuestionResponse(
                answer=no_info_msg,
                sources=[]
            )
        
        # Generate static answer using LLM
        logger.info("Generating answer using LLM interface")
        result = await llm_interface.generate_answer(request.question, relevant_docs)
        
        logger.info(f"Generated answer with {len(result['sources'])} sources")
        
        # Store Q&A in database
        if result["answer"] and result["sources"]:
            try:
                question_id = await database.store_qa(request.question, result)
                return QuestionResponse(
                    answer=result["answer"],
                    sources=result["sources"],
                    question_id=question_id
                )
            except Exception as db_error:
                logger.error(f"Error storing Q&A in database: {str(db_error)}")
                # Return response even if database storage fails
                return QuestionResponse(
                    answer=result["answer"],
                    sources=result["sources"]
                )
        
        return QuestionResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
        
    except Exception as e:
        logger.error(f"Error in static response: {str(e)}")
        error_msg = f"Error processing question: {str(e)}"
        return QuestionResponse(
            answer=error_msg,
            sources=[]
        )


@app.post("/add-qa", response_model=QAPairResponse)
async def add_qa_pair(request: QAPairRequest):
    """
    Add a Q&A pair to improve the system
    """
    try:
        # Store in database
        stored_data = await database.add_qa_pair(
            request.question, 
            request.answer,
            request.sources
        )
        
        # Also add to vector store for retrieval
        doc = {
            "text": f"Question: {request.question}\nAnswer: {request.answer}",
            "metadata": {
                "question_id": stored_data["question_id"],
                "source": "manual_entry"
            }
        }
        await vector_store.add_document(doc)
        
        return {
            "question_id": stored_data["question_id"],
            "answer_id": stored_data["answer_id"],
            "message": "Q&A pair added successfully"
        }
    except Exception as e:
        logger.error(f"Error adding QA pair: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding QA pair: {str(e)}")

@app.get("/history")
async def get_history(limit: int = Query(20, ge=1, le=100)):
    """
    Get recent Q&A history
    """
    history = await database.get_qa_history(limit=limit)
    return {"history": history}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
