#!/usr/bin/env python3
"""
Test script for enhanced FAISS vector store with code block support

This script demonstrates:
1. Code block detection and language identification
2. Enhanced document processing with context
3. Code-aware search functionality
4. Different query types and result ranking
"""

import asyncio
import os
import logging
from typing import List, Dict, Any
import sys

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.faiss_vector_store import FAISSVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample documents with code blocks for testing
SAMPLE_DOCUMENTS = [
    {
        "text": """
# Python Authentication Guide

User authentication is crucial for web applications. Here's how to implement it:

```python
def authenticate_user(username: str, password: str) -> bool:
    # Hash the password and compare with stored hash
    password_hash = hash_password(password)
    user = get_user_from_database(username)
    
    if user and user.password_hash == password_hash:
        return True
    return False

def hash_password(password: str) -> str:
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()
```

This authentication system provides basic security for user login.
        """,
        "metadata": {
            "url": "https://docs.example.com/auth/python",
            "title": "Python Authentication Guide",
            "source": "test"
        }
    },
    {
        "text": """
# JavaScript API Integration

When working with REST APIs in JavaScript, you need to handle authentication properly:

```javascript
async function authenticateUser(username, password) {
    const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, password })
    });
    
    if (response.ok) {
        const data = await response.json();
        localStorage.setItem('token', data.token);
        return true;
    }
    return false;
}

// Usage example
const loginForm = document.getElementById('login-form');
loginForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const username = e.target.username.value;
    const password = e.target.password.value;
    
    if (await authenticateUser(username, password)) {
        window.location.href = '/dashboard';
    } else {
        alert('Login failed!');
    }
});
```

This code demonstrates modern JavaScript authentication with async/await.
        """,
        "metadata": {
            "url": "https://docs.example.com/auth/javascript",
            "title": "JavaScript API Authentication",
            "source": "test"
        }
    },
    {
        "text": """
# Database Configuration

Setting up a secure database connection is essential for any application.

## Connection Parameters

The following parameters are required:
- Host: Database server address
- Port: Usually 5432 for PostgreSQL
- Database name: Your application database
- Username and password: Authentication credentials

## SQL Example

```sql
-- Create a user table for authentication
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert a sample user
INSERT INTO users (username, password_hash, email) 
VALUES ('admin', 'hashed_password_here', 'admin@example.com');

-- Query to find user for authentication
SELECT id, username, password_hash 
FROM users 
WHERE username = $1;
```

Always use parameterized queries to prevent SQL injection attacks.
        """,
        "metadata": {
            "url": "https://docs.example.com/database/setup",
            "title": "Database Configuration Guide",
            "source": "test"
        }
    },
    {
        "text": """
# Security Best Practices

When implementing authentication systems, follow these security principles:

## Password Security
- Always hash passwords using a strong algorithm like bcrypt
- Use salt to prevent rainbow table attacks
- Implement password complexity requirements
- Consider two-factor authentication

## Session Management
- Use secure, httpOnly cookies for session tokens
- Implement proper session expiration
- Invalidate sessions on logout
- Protect against CSRF attacks

## General Security
- Always use HTTPS in production
- Validate and sanitize all user inputs
- Implement rate limiting for login attempts
- Log authentication events for monitoring

These practices help protect your application from common security vulnerabilities.
        """,
        "metadata": {
            "url": "https://docs.example.com/security/best-practices",
            "title": "Security Best Practices",
            "source": "test"
        }
    },
    {
        "text": """
# Docker Deployment Guide

Deploy your authentication service using Docker for better scalability:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

This Dockerfile creates a secure, minimal container for your Python authentication service.

## Docker Compose

```yaml
version: '3.8'
services:
  auth-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/authdb
    depends_on:
      - db
  
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=authdb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Use docker-compose for multi-service deployments.
        """,
        "metadata": {
            "url": "https://docs.example.com/deployment/docker",
            "title": "Docker Deployment Guide",
            "source": "test"
        }
    }
]

async def test_enhanced_vector_store():
    """Test the enhanced FAISS vector store functionality"""
    
    print("🚀 Testing Enhanced FAISS Vector Store")
    print("=" * 50)
    
    # Initialize vector store
    vector_store = FAISSVectorStore(index_path="./data/test_enhanced_index")
    
    # Clear any existing data
    vector_store.clear()
    
    print("\n📝 Adding sample documents with code blocks...")
    
    # Add documents
    doc_ids = await vector_store.add_documents(SAMPLE_DOCUMENTS)
    print(f"✅ Added {len(doc_ids)} main documents")
    print(f"📊 Total indexed documents: {len(vector_store.documents)}")
    
    # Analyze what was created
    code_blocks = sum(1 for doc in vector_store.documents 
                     if doc["metadata"].get("content_type") == "code_block")
    mixed_docs = sum(1 for doc in vector_store.documents 
                    if doc["metadata"].get("content_type") == "mixed")
    
    print(f"📋 Mixed content documents: {mixed_docs}")
    print(f"💻 Code block documents: {code_blocks}")
    
    # Test different search scenarios
    test_queries = [
        {
            "query": "python authentication function",
            "description": "Code-focused query (auto-detected)",
            "expected_focus": True
        },
        {
            "query": "security best practices authentication",
            "description": "Text-focused query (auto-detected)",
            "expected_focus": False
        },
        {
            "query": "javascript async login",
            "description": "Code query with language",
            "code_focused": True
        },
        {
            "query": "docker deployment configuration",
            "description": "Mixed query",
            "code_focused": None
        },
        {
            "query": "SQL user table create",
            "description": "Database code query",
            "code_focused": True
        }
    ]
    
    print("\n🔍 Testing search functionality:")
    print("-" * 40)
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        description = test_case["description"]
        code_focused = test_case.get("code_focused")
        
        print(f"\n{i}. {description}")
        print(f"   Query: '{query}'")
        
        if code_focused is not None:
            results = await vector_store.search(query, k=3, code_focused=code_focused)
            print(f"   Mode: {'Code-focused' if code_focused else 'Text-focused'} (explicit)")
        else:
            results = await vector_store.search(query, k=3)
            # Detect what mode was used
            is_code_query = vector_store._is_code_query(query)
            print(f"   Mode: {'Code-focused' if is_code_query else 'Text-focused'} (auto-detected)")
        
        print(f"   Results: {len(results)}")
        
        for j, result in enumerate(results):
            metadata = result["metadata"]
            content_type = metadata.get("content_type", "unknown")
            language = metadata.get("programming_language", "N/A")
            score = result["score"]
            
            print(f"     {j+1}. Score: {score:.3f} | Type: {content_type} | Lang: {language}")
            print(f"        Title: {metadata.get('title', 'No title')}")
            
            # Show snippet of text
            text_snippet = result["text"][:100].replace('\n', ' ')
            print(f"        Text: {text_snippet}...")
            
            if content_type == "code_block":
                raw_code = metadata.get("raw_code", "")
                if raw_code:
                    code_snippet = raw_code[:60].replace('\n', '\\n')
                    print(f"        Code: {code_snippet}...")
    
    print("\n📊 Document Analysis:")
    print("-" * 30)
    
    # Analyze document distribution
    language_counts = {}
    type_counts = {}
    
    for doc in vector_store.documents:
        metadata = doc["metadata"]
        content_type = metadata.get("content_type", "unknown")
        language = metadata.get("programming_language", "N/A")
        
        type_counts[content_type] = type_counts.get(content_type, 0) + 1
        if language != "N/A":
            language_counts[language] = language_counts.get(language, 0) + 1
    
    print("Content Types:")
    for content_type, count in type_counts.items():
        print(f"  {content_type}: {count}")
    
    print("\nProgramming Languages:")
    for language, count in language_counts.items():
        print(f"  {language}: {count}")
    
    print("\n🎯 Code Detection Test:")
    print("-" * 25)
    
    test_code_queries = [
        "python function definition",
        "how to create user table",
        "security best practices",
        "javascript async function",
        "def authenticate_user",
        "docker container setup"
    ]
    
    for query in test_code_queries:
        is_code = vector_store._is_code_query(query)
        print(f"  '{query}' -> {'Code-focused' if is_code else 'Text-focused'}")
    
    print("\n✅ Enhanced FAISS Vector Store test completed!")
    print("\n📈 Key Improvements Demonstrated:")
    print("  • Automatic code block detection and language identification")
    print("  • Context-aware embeddings for code blocks")
    print("  • Intelligent query type detection")
    print("  • Enhanced scoring for code vs text content")
    print("  • Rich metadata for programming languages and context")
    print("  • Smart result grouping to avoid duplication")

async def main():
    """Run the test"""
    await test_enhanced_vector_store()

if __name__ == "__main__":
    asyncio.run(main())