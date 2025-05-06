import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
import uvicorn
from dotenv import load_dotenv
from .middleware.json_error_handler import JSONErrorHandlerMiddleware

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
    description="A RAG-based API for crawling documentation sites and answering questions using Azure OpenAI",
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

# Create data directory if it doesn't exist
os.makedirs("./data", exist_ok=True)
os.makedirs("./data/faiss_index", exist_ok=True)

# Initialize components
vector_store = FAISSVectorStore(index_path="./data/faiss_index/index")
document_processor = DocumentProcessor()
llm_interface = LLMInterface()
database = Database()

# Pydantic models for request/response validation
class CrawlRequest(BaseModel):
    url: str
    max_pages: int = Field(default=100, ge=1, le=1000)

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
    return {
        "message": "RAG Documentation Assistant API with Azure OpenAI",
        "endpoints": [
            "/crawl - Crawl a documentation site",
            "/question - Ask a question",
            "/add-qa - Add a Q&A pair",
            "/job-status/{job_id} - Check job status",
            "/history - View Q&A history"
        ]
    }

async def crawl_and_index(job_id: str, url: str, max_pages: int):
    """Background task to crawl a website and index its content"""
    try:
        active_jobs[job_id] = {
            "status": "running",
            "message": "Starting crawl job",
            "documents_processed": 0
        }
        
        # Crawl the website
        logger.info(f"Starting crawl of {url} with max_pages={max_pages}")
        documents = await crawl_website(url, max_pages)
        
        active_jobs[job_id]["message"] = f"Crawled {len(documents)} documents, processing..."
        
        # Process documents into chunks
        processed_docs = await document_processor.process_documents(documents)
        
        active_jobs[job_id]["message"] = f"Processed into {len(processed_docs)} chunks, indexing..."
        active_jobs[job_id]["documents_processed"] = len(processed_docs)
        
        # Add to vector store
        await vector_store.add_documents(processed_docs)
        
        # ChromaDB automatically persists data, so no need for explicit save
        
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["message"] = f"Indexed {len(processed_docs)} document chunks successfully"
        
        logger.info(f"Completed crawl job {job_id}")
        
    except Exception as e:
        logger.error(f"Error in crawl job {job_id}: {str(e)}")
        active_jobs[job_id] = {
            "status": "error",
            "message": f"Error: {str(e)}",
            "documents_processed": active_jobs[job_id].get("documents_processed", 0)
        }

@app.post("/crawl", response_model=CrawlResponse)
async def crawl_site(request: CrawlRequest, background_tasks: BackgroundTasks):
    """
    Crawl a documentation site and index its content for RAG
    """
    # Generate a job ID
    job_id = f"crawl_{len(active_jobs) + 1}"
    
    # Start the background task
    background_tasks.add_task(crawl_and_index, job_id, request.url, request.max_pages)
    
    return {
        "job_id": job_id,
        "message": f"Started crawling {request.url}",
        "status": "running"
    }

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

@app.post("/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question and get an answer based on indexed documentation
    """
    try:
        # Search for relevant documents
        relevant_docs = await vector_store.search(request.question, k=request.num_results)
        
        if not relevant_docs:
            answer_data = {
                "answer": "I don't have enough information to answer this question. Try crawling more documentation or asking a different question.",
                "sources": []
            }
        else:
            # Generate answer using LLM
            answer_data = await llm_interface.generate_answer(request.question, relevant_docs)
        
        # Store Q&A in database
        stored_data = await database.store_qa(request.question, answer_data)
        
        return {
            "answer": answer_data["answer"],
            "sources": answer_data["sources"],
            "question_id": stored_data["question_id"]
        }
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

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
