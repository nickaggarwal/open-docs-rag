#!/usr/bin/env python3
"""
Test script for RAG Documentation Assistant application
This script tests the core functionality of the RAG application:
1. Connecting to Azure OpenAI
2. Crawling a documentation site
3. Processing and indexing content
4. Asking questions and generating answers
5. Adding custom Q&A pairs
"""

import pytest

# Skip these tests by default when run under pytest
pytest.skip(
    "Integration tests require external services", allow_module_level=True
)

import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from pprint import pprint

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add app directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import app components
from app.faiss_vector_store import FAISSVectorStore
from app.crawler import crawl_website, WebCrawler
from app.document_processor import DocumentProcessor
from app.llm_interface import LLMInterface
from app.database import Database

# Load environment variables
load_dotenv()

# Test configuration
TEST_URL = "https://docs.python.org/3/tutorial/introduction.html"  # Small Python docs page
MAX_PAGES = 3  # Limit crawling to a very small number of pages for testing
TEST_QUESTIONS = [
    "What are lists in Python?",
    "How do I use string methods?",
]
CUSTOM_QA = {
    "question": "What version of Python is this documentation for?",
    "answer": "This documentation is for Python 3.",
    "sources": ["https://docs.python.org/3/tutorial/introduction.html"]
}

# Test documents to ensure consistent testing without crawling
TEST_DOCUMENTS = [
    {
        "text": "Python lists are mutable sequences, typically used to store collections of homogeneous items. Lists may be constructed in several ways: Using square brackets to denote an empty list: []; Using square brackets and elements: [a, b, c]; Using a list comprehension: [x for x in iterable]; Using the type constructor: list() or list(iterable).",
        "metadata": {
            "url": "https://docs.python.org/3/tutorial/datastructures.html",
            "title": "Python Lists"
        }
    },
    {
        "text": "String methods are built-in methods that can be called on string objects to perform various transformations and operations. Some common string methods include: str.upper(), str.lower(), str.strip(), str.replace(old, new), str.split(sep), and many more. Strings are immutable, so these methods always return a new string and never modify the original.",
        "metadata": {
            "url": "https://docs.python.org/3/library/stdtypes.html#string-methods",
            "title": "String Methods"
        }
    },
    {
        "text": "Dictionaries are indexed by keys, which can be any immutable type; strings and numbers can always be keys. Tuples can be used as keys if they contain only strings, numbers, or tuples; if a tuple contains any mutable object either directly or indirectly, it cannot be used as a key. You can't use lists as keys, since lists can be modified in place using index assignments, slice assignments, or methods like append() and extend().",
        "metadata": {
            "url": "https://docs.python.org/3/tutorial/datastructures.html#dictionaries",
            "title": "Python Dictionaries"
        }
    }
]

async def test_azure_openai_connection():
    """Test connecting to Azure OpenAI"""
    logger.info("Testing Azure OpenAI connection...")
    
    # Check if Azure OpenAI credentials are set
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    
    if not endpoint or not api_key:
        logger.error("Azure OpenAI credentials not set in .env file")
        return False
    
    try:
        # Create a simple test document
        test_doc = {
            "text": "2+2=4", 
            "metadata": {"url": "test", "title": "Test Document"}
        }
        
        # Initialize LLM interface
        llm = LLMInterface()
        
        # Test with a simple question - using the direct prompt to avoid template errors
        answer_data = await llm.generate_answer("What is 2+2?", [test_doc])
        
        if answer_data and "answer" in answer_data:
            logger.info("Successfully connected to Azure OpenAI")
            logger.info(f"Test response: {answer_data['answer']}")
            return True
        else:
            logger.error("Failed to get proper response from Azure OpenAI")
            return False
    except Exception as e:
        logger.error(f"Error connecting to Azure OpenAI: {str(e)}")
        return False

async def test_crawl_and_process():
    """Test crawling website and processing documents"""
    logger.info(f"Crawling {TEST_URL} (limited to {MAX_PAGES} pages)...")
    
    try:
        # Create a crawler with more restrictive settings for testing
        crawler = WebCrawler(
            base_url=TEST_URL,
            max_pages=MAX_PAGES,
            concurrency=2,  # Lower concurrency
            timeout=10,      # Shorter timeout
            overall_timeout=60  # Only wait 1 minute max
        )
        
        # Crawl directly instead of using the utility function
        documents = await crawler.crawl()
        logger.info(f"Crawled {len(documents)} documents")
        
        if not documents:
            logger.warning("No documents crawled, using test documents instead")
            documents = TEST_DOCUMENTS
            
        # Process documents
        processor = DocumentProcessor()
        processed_docs = await processor.process_documents(documents)
        logger.info(f"Processed into {len(processed_docs)} document chunks")
        
        return processed_docs
    except Exception as e:
        logger.error(f"Error during crawling/processing: {str(e)}")
        logger.info("Using test documents as fallback")
        # Use test documents as fallback
        processor = DocumentProcessor()
        processed_docs = await processor.process_documents(TEST_DOCUMENTS)
        logger.info(f"Processed test documents into {len(processed_docs)} chunks")
        return processed_docs

async def test_vector_store(documents):
    """Test adding documents to vector store and searching"""
    logger.info("Testing vector store...")
    
    try:
        # Initialize FAISS vector store with a test-specific directory
        vector_store = FAISSVectorStore(index_path="./data/test_faiss_index")
        
        # Add documents
        doc_ids = await vector_store.add_documents(documents)
        logger.info(f"Added {len(doc_ids)} documents to vector store")
        
        # Test search
        test_query = "What are Python lists?"
        results = await vector_store.search(test_query, k=2)
        
        logger.info(f"Search results for '{test_query}':")
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1} (score: {doc.get('score', 0):.4f}):")
            logger.info(f"  Metadata: {doc.get('metadata', {})}")
            text_snippet = doc.get('text', '')[:100] + ('...' if len(doc.get('text', '')) > 100 else '')
            logger.info(f"  Text snippet: {text_snippet}")
        
        return vector_store
    except Exception as e:
        logger.error(f"Error testing vector store: {str(e)}")
        return None

async def test_qa(vector_store, llm_interface):
    """Test question answering"""
    logger.info("Testing question answering...")
    
    try:
        # Test with the first question
        question = TEST_QUESTIONS[0]
        logger.info(f"\nQuestion: {question}")
        
        # Get documents to use
        if not vector_store:
            logger.warning("Vector store not available, using test documents directly")
            # Convert test documents to simple dictionaries with just text and metadata
            relevant_docs = [
                {
                    "text": doc["text"],
                    "metadata": doc["metadata"]
                }
                for doc in TEST_DOCUMENTS
            ]
        else:
            # Search for relevant documents
            try:
                relevant_docs = await vector_store.search(question, k=2)
                if not relevant_docs:
                    logger.warning("No relevant documents found, using test documents")
                    # Convert test documents to simple dictionaries
                    relevant_docs = [
                        {
                            "text": doc["text"],
                            "metadata": doc["metadata"]
                        }
                        for doc in TEST_DOCUMENTS
                    ]
            except Exception as e:
                logger.error(f"Search failed: {str(e)}, using test documents instead")
                # Convert test documents to simple dictionaries
                relevant_docs = [
                    {
                        "text": doc["text"],
                        "metadata": doc["metadata"]
                    }
                    for doc in TEST_DOCUMENTS
                ]
        
        # Log the document structure we're using
        logger.info(f"Using {len(relevant_docs)} documents for generating answer")
        for i, doc in enumerate(relevant_docs):
            logger.info(f"Document {i+1} type: {type(doc)}")
            
        # Generate answer with simplified document structure
        answer_data = await llm_interface.generate_answer(question, relevant_docs)
        
        logger.info(f"Answer: {answer_data.get('answer', 'No answer generated')}")
        logger.info(f"Sources: {answer_data.get('sources', [])}")
    except Exception as e:
        logger.error(f"Error in QA test: {str(e)}")
        logger.error(f"Error type: {type(e)}")

async def test_database():
    """Test database operations"""
    logger.info("Testing database operations...")
    
    try:
        # Initialize database
        db = Database()
        
        # Test storing Q&A
        stored = await db.store_qa(
            "What is Python?", 
            {"answer": "Python is a programming language.", "sources": ["test"]}
        )
        
        logger.info(f"Stored Q&A with ID: {stored.get('question_id')}")
        
        # Test custom Q&A pair
        custom = await db.add_qa_pair(
            CUSTOM_QA["question"],
            CUSTOM_QA["answer"],
            CUSTOM_QA["sources"]
        )
        
        logger.info(f"Added custom Q&A with ID: {custom.get('question_id')}")
        
        # Test retrieving history
        history = await db.get_qa_history(limit=5)
        logger.info(f"Retrieved {len(history)} Q&A pairs from history")
        
        return True
    except Exception as e:
        logger.error(f"Error testing database: {str(e)}")
        return False

async def run_tests():
    """Run all tests sequentially"""
    logger.info("Starting RAG Application Test Suite")
    
    # Test Azure OpenAI connection
    if not await test_azure_openai_connection():
        logger.error("Azure OpenAI connection test failed. Aborting remaining tests.")
        return
    
    # Test crawling and processing
    processed_docs = await test_crawl_and_process()
    if not processed_docs:
        logger.error("Document processing test failed. Aborting remaining tests.")
        return
    
    # Test vector store
    vector_store = await test_vector_store(processed_docs)
    if not vector_store:
        logger.error("Vector store test failed. Aborting remaining tests.")
        return
    
    # Initialize LLM interface for QA testing
    llm_interface = LLMInterface()
    
    # Test QA
    await test_qa(vector_store, llm_interface)
    
    # Test database
    await test_database()
    
    logger.info("Test suite completed!")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./data/test_faiss_index", exist_ok=True)
    
    # Run tests
    asyncio.run(run_tests())
