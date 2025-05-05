#!/usr/bin/env python3
"""
Real-world test case for the RAG Documentation Assistant

This script tests a complete end-to-end workflow by:
1. Crawling a specific product documentation
2. Asking domain-specific questions
3. Comparing answers to expected responses
4. Simulating user feedback
"""

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
import json
import time
from pprint import pprint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import app components
from app.faiss_vector_store import FAISSVectorStore
from app.crawler import crawl_website
from app.document_processor import DocumentProcessor
from app.llm_interface import LLMInterface
from app.database import Database

# Load environment variables
load_dotenv()

# Test configuration for FastAPI documentation
FASTAPI_DOCS_URL = "https://fastapi.tiangolo.com/tutorial/first-steps/"
MAX_PAGES = 10

# Domain-specific questions about FastAPI
DOMAIN_QUESTIONS = [
    "How do I create a FastAPI application?",
    "What are path parameters in FastAPI?",
    "How do I handle form data in FastAPI?",
    "What is dependency injection in FastAPI?",
    "How do I create a background task in FastAPI?"
]

# Expected keywords in answers (for basic validation)
EXPECTED_KEYWORDS = {
    "How do I create a FastAPI application?": ["import", "fastapi", "app", "FastAPI()", "uvicorn"],
    "What are path parameters in FastAPI?": ["path", "parameter", "{}"],
    "How do I handle form data in FastAPI?": ["Form", "form-data", "post"],
    "What is dependency injection in FastAPI?": ["Depends", "dependency", "function"],
    "How do I create a background task in FastAPI?": ["BackgroundTasks", "add_task", "async"]
}

async def setup_test_environment():
    """Set up the test environment with FastAPI documentation"""
    logger.info("Setting up test environment...")
    
    try:
        # Initialize components
        vector_store = FAISSVectorStore(index_path="./data/real_world_test_db/index")
        document_processor = DocumentProcessor(chunk_size=600, chunk_overlap=100)
        llm_interface = LLMInterface()
        
        # Crawl FastAPI documentation
        logger.info(f"Crawling FastAPI documentation at {FASTAPI_DOCS_URL}...")
        documents = await crawl_website(FASTAPI_DOCS_URL, max_pages=MAX_PAGES)
        logger.info(f"Crawled {len(documents)} documents")
        
        # Process documents
        processed_docs = await document_processor.process_documents(documents)
        logger.info(f"Processed into {len(processed_docs)} document chunks")
        
        # Store in vector database
        await vector_store.add_documents(processed_docs)
        logger.info(f"Indexed {len(processed_docs)} document chunks")
        
        return {
            "vector_store": vector_store,
            "llm_interface": llm_interface,
            "db": Database(),
            "document_count": len(processed_docs)
        }
    except Exception as e:
        logger.error(f"Error setting up test environment: {str(e)}")
        raise

def validate_answer(question, answer):
    """Basic validation of answers by checking for expected keywords"""
    if question not in EXPECTED_KEYWORDS:
        return True  # No validation criteria
        
    keywords = EXPECTED_KEYWORDS[question]
    score = sum(1 for keyword in keywords if keyword.lower() in answer.lower())
    percentage = (score / len(keywords)) * 100
    
    logger.info(f"Answer validation score: {percentage:.1f}% ({score}/{len(keywords)} keywords found)")
    return percentage >= 50  # At least 50% of keywords should be present

async def test_qa_workflow(env):
    """Test the complete Q&A workflow"""
    vector_store = env["vector_store"]
    llm_interface = env["llm_interface"]
    db = env["db"]
    
    results = []
    
    for question in DOMAIN_QUESTIONS:
        logger.info(f"\n--- Testing question: {question} ---")
        
        # Search for relevant documents
        relevant_docs = await vector_store.search(question, k=5)
        logger.info(f"Found {len(relevant_docs)} relevant documents")
        
        if not relevant_docs:
            logger.warning("No relevant documents found!")
            results.append({
                "question": question,
                "answer": "No relevant documents found",
                "valid": False,
                "stored_id": None
            })
            continue
        
        # Generate answer
        start_time = time.time()
        answer_data = await llm_interface.generate_answer(question, relevant_docs)
        elapsed_time = time.time() - start_time
        
        answer = answer_data.get("answer", "")
        sources = answer_data.get("sources", [])
        
        logger.info(f"Answer generated in {elapsed_time:.2f} seconds")
        logger.info(f"Answer: {answer[:200]}..." if len(answer) > 200 else f"Answer: {answer}")
        logger.info(f"Sources: {sources}")
        
        # Validate answer
        valid = validate_answer(question, answer)
        
        # Store Q&A in database
        stored = await db.store_qa(question, answer_data)
        logger.info(f"Stored Q&A with ID: {stored.get('question_id')}")
        
        # Store results
        results.append({
            "question": question,
            "answer": answer,
            "sources": sources,
            "valid": valid,
            "stored_id": stored.get('question_id'),
            "time_seconds": elapsed_time
        })
        
        # Simulate thinking time between questions
        await asyncio.sleep(1)
    
    return results

async def simulate_user_feedback(env, results):
    """Simulate user feedback by adding custom answers for low-quality responses"""
    db = env["db"]
    
    for result in results:
        if not result["valid"]:
            logger.info(f"\n--- Adding custom answer for: {result['question']} ---")
            
            # Create a better answer for the questions that didn't pass validation
            if result["question"] == "How do I create a FastAPI application?":
                custom_answer = """
                To create a FastAPI application:
                
                1. Install FastAPI and uvicorn:
                   ```
                   pip install fastapi uvicorn
                   ```
                
                2. Create a Python file (e.g., main.py) with:
                   ```python
                   from fastapi import FastAPI
                   
                   app = FastAPI()
                   
                   @app.get("/")
                   def read_root():
                       return {"Hello": "World"}
                   ```
                
                3. Run the server:
                   ```
                   uvicorn main:app --reload
                   ```
                
                This creates a basic FastAPI application with a single endpoint.
                """
            else:
                # Generic improvement for other failed answers
                custom_answer = f"Here's an improved answer for the question: {result['question']}"
            
            # Add to database
            custom_qa = await db.add_qa_pair(
                result["question"],
                custom_answer,
                ["https://fastapi.tiangolo.com/tutorial/"]
            )
            
            logger.info(f"Added custom Q&A with ID: {custom_qa.get('question_id')}")

async def run_real_world_test():
    """Run the real-world test scenario"""
    logger.info("Starting real-world test scenario for RAG application")
    
    # Set up test environment
    env = await setup_test_environment()
    logger.info(f"Test environment ready with {env['document_count']} indexed documents")
    
    # Test Q&A workflow
    results = await test_qa_workflow(env)
    
    # Calculate statistics
    valid_count = sum(1 for r in results if r["valid"])
    avg_time = sum(r["time_seconds"] for r in results) / len(results)
    
    logger.info("\n--- Test Results Summary ---")
    logger.info(f"Total questions: {len(results)}")
    logger.info(f"Valid answers: {valid_count}/{len(results)} ({valid_count/len(results)*100:.1f}%)")
    logger.info(f"Average response time: {avg_time:.2f} seconds")
    
    # Simulate user feedback for improvement
    await simulate_user_feedback(env, results)
    
    logger.info("\nReal-world test completed!")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./data/real_world_test_db", exist_ok=True)
    
    # Run test
    asyncio.run(run_real_world_test())
