#!/usr/bin/env python3
"""
Test script for the enhanced RAG system with MD-aware processing
"""

import asyncio
import sys
import os
import json

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from app.document_processor import DocumentProcessor
from app.faiss_vector_store import FAISSVectorStore  
from app.llm_interface import LLMInterface

async def test_enhanced_processing():
    """Test the enhanced document processing capabilities"""
    
    print("🔄 Testing Enhanced RAG System")
    print("=" * 50)
    
    # Initialize components
    doc_processor = DocumentProcessor()
    
    # Test document classification
    print("\n1. Testing Document Classification:")
    test_url = "https://docs.inferless.com/concepts/configuring-the-input-output-schema"
    test_content = """
# Configuring Input and Output Schema

This document explains how to configure input and output schemas in Inferless.

## Setting Up Input Schema

To set up your input schema, you need to define the structure of data your model expects.

```python
INPUT_SCHEMA = {
    "prompt": {
        "datatype": "STRING",
        "required": True,
        "shape": [-1],
        "example": ["Hello world"]
    }
}
```

## Defining Output Schema

The output schema defines what your model returns.

```python
OUTPUT_SCHEMA = {
    "generated_text": {
        "datatype": "STRING", 
        "shape": [-1]
    }
}
```

## Code Example

Here's a complete app.py example:

```python
import inferless
from inferless import RequestObjects, ResponseObjects

app = inferless.Cls(gpu="T4")

class InferlessPythonModel:
    @app.load
    def initialize(self):
        # Load your model here
        pass
        
    @app.infer  
    def infer(self, inputs: RequestObjects) -> ResponseObjects:
        # Process inputs and return outputs
        prompt = inputs["prompt"]
        result = f"Processed: {prompt}"
        return ResponseObjects(generated_text=result)
```
"""
    
    # Test document classification
    classification = doc_processor.classify_document_type(test_url, test_content)
    print(f"   URL: {test_url}")
    print(f"   Document Type: {classification['document_type']}")
    print(f"   Priority Level: {classification['priority_level']}")
    print(f"   Has Code: {classification['has_code_examples']}")
    print(f"   Topic Hierarchy: {classification['topic_hierarchy']}")
    
    # Test MD structure parsing
    print("\n2. Testing Markdown Structure Parsing:")
    md_structure = doc_processor.parse_markdown_structure(test_content)
    print(f"   Title: {md_structure['title']}")
    print(f"   Number of Sections: {len(md_structure['sections'])}")
    print(f"   Number of Code Blocks: {len(md_structure['code_blocks'])}")
    
    for i, section in enumerate(md_structure['sections'][:3]):
        print(f"   Section {i+1}: {section['title']} (Level {section['level']})")
    
    # Test enhanced chunking
    print("\n3. Testing Enhanced Chunking:")
    metadata = {
        "url": test_url,
        "title": "Configuring Input Output Schema",
        "source": "web_crawler"
    }
    
    chunks = doc_processor.create_enhanced_chunks(test_content, metadata)
    print(f"   Generated {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks[:3]):
        chunk_meta = chunk['metadata']
        print(f"   Chunk {i+1}: {chunk_meta.get('chunk_type', 'unknown')} "
              f"(Priority: {chunk_meta.get('priority', 'N/A')})")
    
    # Test LLM interface enhancements
    print("\n4. Testing LLM Interface Enhancements:")
    llm = LLMInterface()
    
    # Test query classification
    test_questions = [
        "How to accept Input and return Output in app.py?",
        "What is Inferless?", 
        "What are the API endpoints?",
        "Show me configuration examples"
    ]
    
    for question in test_questions:
        intent = llm._classify_query_intent(question)
        print(f"   '{question}' -> {intent}")
    
    # Test semantic relevance scoring
    print("\n5. Testing Semantic Relevance Scoring:")
    test_doc = {
        "text": test_content,
        "metadata": {**metadata, **classification},
        "score": 0.75
    }
    
    for question in test_questions[:2]:
        relevance = llm._calculate_semantic_relevance(question, test_doc)
        priority = llm._calculate_priority_score(test_doc, llm._classify_query_intent(question))
        print(f"   '{question}':")
        print(f"     Semantic: {relevance:.3f}, Priority: {priority:.3f}")
    
    print("\n✅ Enhanced RAG System Test Complete!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_enhanced_processing())