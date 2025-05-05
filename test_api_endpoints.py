#!/usr/bin/env python3
"""
Test script for RAG API endpoints
This script tests the FastAPI endpoints of the RAG application.
It uses the requests library to send HTTP requests to a running instance of the API.
"""

import requests
import json
import time
import logging
from pprint import pprint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API configuration
BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Test the root endpoint"""
    logger.info("Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    
    if response.status_code == 200:
        logger.info("Root endpoint test successful")
        pprint(response.json())
        return True
    else:
        logger.error(f"Root endpoint test failed: {response.status_code}")
        logger.error(response.text)
        return False

def test_crawl_endpoint():
    """Test the crawl endpoint"""
    logger.info("Testing crawl endpoint...")
    
    data = {
        "url": "https://docs.python.org/3/tutorial/introduction.html",
        "max_pages": 3  # Small number for testing
    }
    
    response = requests.post(f"{BASE_URL}/crawl", json=data)
    
    if response.status_code == 200:
        result = response.json()
        logger.info("Crawl endpoint test successful")
        logger.info(f"Job ID: {result.get('job_id')}")
        logger.info(f"Status: {result.get('status')}")
        logger.info(f"Message: {result.get('message')}")
        return result.get('job_id')
    else:
        logger.error(f"Crawl endpoint test failed: {response.status_code}")
        logger.error(response.text)
        return None

def test_job_status(job_id):
    """Test the job status endpoint"""
    logger.info(f"Testing job status endpoint for job {job_id}...")
    
    # Poll job status until completed or error
    max_polls = 20
    poll_count = 0
    
    while poll_count < max_polls:
        response = requests.get(f"{BASE_URL}/job-status/{job_id}")
        
        if response.status_code == 200:
            result = response.json()
            status = result.get('status')
            logger.info(f"Job status: {status}")
            logger.info(f"Message: {result.get('message')}")
            
            if status == "completed" or status == "error":
                return status == "completed"
            
            # Wait before polling again
            time.sleep(2)
            poll_count += 1
        else:
            logger.error(f"Job status endpoint test failed: {response.status_code}")
            logger.error(response.text)
            return False
    
    logger.error("Job status polling timed out")
    return False

def test_question_endpoint():
    """Test the question endpoint"""
    logger.info("Testing question endpoint...")
    
    data = {
        "question": "What are lists in Python?",
        "num_results": 3
    }
    
    response = requests.post(f"{BASE_URL}/question", json=data)
    
    if response.status_code == 200:
        result = response.json()
        logger.info("Question endpoint test successful")
        logger.info(f"Answer: {result.get('answer')}")
        logger.info(f"Sources: {result.get('sources')}")
        logger.info(f"Question ID: {result.get('question_id')}")
        return True
    else:
        logger.error(f"Question endpoint test failed: {response.status_code}")
        logger.error(response.text)
        return False

def test_add_qa_endpoint():
    """Test the add-qa endpoint"""
    logger.info("Testing add-qa endpoint...")
    
    data = {
        "question": "What is the best way to learn Python?",
        "answer": "The best way to learn Python is through practice and working on real projects.",
        "sources": ["https://example.com/python-learning"]
    }
    
    response = requests.post(f"{BASE_URL}/add-qa", json=data)
    
    if response.status_code == 200:
        result = response.json()
        logger.info("Add-qa endpoint test successful")
        logger.info(f"Question ID: {result.get('question_id')}")
        logger.info(f"Answer ID: {result.get('answer_id')}")
        logger.info(f"Message: {result.get('message')}")
        return True
    else:
        logger.error(f"Add-qa endpoint test failed: {response.status_code}")
        logger.error(response.text)
        return False

def test_history_endpoint():
    """Test the history endpoint"""
    logger.info("Testing history endpoint...")
    
    response = requests.get(f"{BASE_URL}/history?limit=10")
    
    if response.status_code == 200:
        result = response.json()
        history = result.get('history', [])
        logger.info("History endpoint test successful")
        logger.info(f"Retrieved {len(history)} history items")
        
        if history:
            logger.info("Sample history item:")
            pprint(history[0])
        
        return True
    else:
        logger.error(f"History endpoint test failed: {response.status_code}")
        logger.error(response.text)
        return False

def run_api_tests():
    """Run all API tests"""
    logger.info("Starting RAG API Test Suite")
    logger.info("Make sure the API server is running at " + BASE_URL)
    
    # Test root endpoint
    if not test_root_endpoint():
        logger.error("Root endpoint test failed. Is the server running?")
        return
    
    # Test crawl endpoint
    job_id = test_crawl_endpoint()
    if not job_id:
        logger.error("Crawl endpoint test failed.")
        return
    
    # Test job status endpoint
    if not test_job_status(job_id):
        logger.warning("Job status endpoint test failed or job failed to complete.")
        # Continue with other tests anyway
    
    # Test question endpoint
    test_question_endpoint()
    
    # Test add-qa endpoint
    test_add_qa_endpoint()
    
    # Test history endpoint
    test_history_endpoint()
    
    logger.info("API test suite completed!")

if __name__ == "__main__":
    run_api_tests()
