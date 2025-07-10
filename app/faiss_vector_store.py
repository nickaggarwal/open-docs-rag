from typing import List, Dict, Any, Optional, Tuple
import os
import logging
import numpy as np
import faiss
from dotenv import load_dotenv
import openai
import uuid
import time
import asyncio
import pickle
import json
import re
from bs4 import BeautifulSoup

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, index_path="./data/faiss_index", dimension=1536):
        """
        Initialize a vector store with FAISS for similarity search with enhanced code block support
        
        Args:
            index_path: Base path to save/load the FAISS index
            dimension: Dimension of embeddings (1536 for OpenAI's ada-002)
        """
        # Force reload environment variables
        load_dotenv(override=True)
        
        # Determine whether to use Azure OpenAI or direct OpenAI API
        self.use_azure = os.getenv("USE_AZURE_OPENAI", "true").lower() == "true"
        
        if self.use_azure:
            logger.info("Using Azure OpenAI API for embeddings")
            # Load Azure OpenAI configuration
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
            self.azure_embedding_deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
            
            if not self.azure_endpoint or not self.azure_api_key:
                logger.warning("Azure OpenAI credentials not found. Vector operations will fail.")
                
            # Configure OpenAI client for Azure
            self.client = openai.OpenAI(
                api_key=self.azure_api_key,
                base_url=f"{self.azure_endpoint}/openai/deployments/{self.azure_embedding_deployment}",
                default_query={"api-version": self.azure_api_version},
            )
            # Store model name for embedding
            self.embedding_model = self.azure_embedding_deployment
        else:
            logger.info("Using direct OpenAI API for embeddings")
            # Load OpenAI configuration
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
            
            if not self.openai_api_key:
                logger.warning("OpenAI API key not found. Vector operations will fail.")
            
            # Create standard OpenAI client
            self.client = openai.OpenAI(
                api_key=self.openai_api_key,
            )
            # Store model name for embedding
            self.embedding_model = self.openai_embedding_model
        
        self.dimension = dimension
        self.index_path = index_path
        
        # Initialize default empty collections in case loading fails
        self.documents = []
        self.id_map = {}
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Try to load existing index and documents if they exist
        self._load_index()
        
        # Initialize code patterns for language detection
        self._init_code_patterns()
    
    def _init_code_patterns(self):
        """Initialize patterns for code block detection and language identification"""
        self.code_patterns = {
            'python': [
                r'\bdef\s+\w+\s*\(',
                r'\bclass\s+\w+\s*\(',
                r'\bimport\s+\w+',
                r'\bfrom\s+\w+\s+import',
                r'__name__\s*==\s*["\']__main__["\']',
                r'\bprint\s*\(',
                r'\bif\s+__name__',
            ],
            'javascript': [
                r'\bfunction\s+\w+\s*\(',
                r'\bconst\s+\w+\s*=',
                r'\blet\s+\w+\s*=',
                r'\bvar\s+\w+\s*=',
                r'\bconsole\.log\s*\(',
                r'=>',
                r'\brequire\s*\(',
                r'\bmodule\.exports',
            ],
            'typescript': [
                r'\binterface\s+\w+',
                r'\btype\s+\w+\s*=',
                r':\s*\w+\[\]',
                r':\s*string\b',
                r':\s*number\b',
                r':\s*boolean\b',
                r'\bimport\s+.*from\s+["\']',
            ],
            'java': [
                r'\bpublic\s+class\s+\w+',
                r'\bprivate\s+\w+\s+\w+',
                r'\bpublic\s+static\s+void\s+main',
                r'\bSystem\.out\.println',
                r'\bpackage\s+\w+',
                r'\bimport\s+java\.',
            ],
            'cpp': [
                r'#include\s*<\w+>',
                r'\bint\s+main\s*\(',
                r'\bstd::\w+',
                r'\busing\s+namespace\s+std',
                r'\bcout\s*<<',
                r'\bcin\s*>>',
            ],
            'bash': [
                r'#!/bin/bash',
                r'#!/bin/sh',
                r'\$\w+',
                r'\becho\s+',
                r'\bif\s*\[\s*',
                r'\bfor\s+\w+\s+in\s+',
                r'\bchmod\s+',
                r'\bmkdir\s+',
            ],
            'sql': [
                r'\bSELECT\s+',
                r'\bFROM\s+\w+',
                r'\bWHERE\s+',
                r'\bINSERT\s+INTO',
                r'\bUPDATE\s+\w+\s+SET',
                r'\bCREATE\s+TABLE',
                r'\bDROP\s+TABLE',
            ],
            'yaml': [
                r'^\s*\w+:\s*$',
                r'^\s*-\s+\w+',
                r'version:\s*["\']?\d+',
                r'apiVersion:',
                r'kind:',
                r'metadata:',
            ],
            'json': [
                r'^\s*{',
                r'^\s*\[',
                r'"\w+":\s*"',
                r'"\w+":\s*\d+',
                r'"\w+":\s*true|false',
            ],
            'dockerfile': [
                r'^FROM\s+\w+',
                r'^RUN\s+',
                r'^COPY\s+',
                r'^ADD\s+',
                r'^WORKDIR\s+',
                r'^EXPOSE\s+',
                r'^CMD\s+',
                r'^ENTRYPOINT\s+',
            ],
            'html': [
                r'<html',
                r'<head>',
                r'<body>',
                r'<div\s+',
                r'<span\s+',
                r'<p>',
                r'<!DOCTYPE\s+html',
            ],
            'css': [
                r'\.\w+\s*{',
                r'#\w+\s*{',
                r'\w+:\s*\w+;',
                r'@media\s+',
                r'@import\s+',
                r'color:\s*#\w+',
            ],
        }
    
    def _detect_programming_language(self, code_text: str) -> str:
        """
        Detect the programming language of a code block
        
        Args:
            code_text: The code text to analyze
            
        Returns:
            Detected language or 'unknown' if not detected
        """
        if not code_text.strip():
            return 'unknown'
        
        language_scores = {}
        
        for language, patterns in self.code_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, code_text, re.MULTILINE | re.IGNORECASE))
                score += matches
            
            if score > 0:
                language_scores[language] = score
        
        if language_scores:
            # Return language with highest score
            return max(language_scores.items(), key=lambda x: x[1])[0]
        
        return 'unknown'
    
    def _extract_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract code blocks from text with improved detection
        
        Args:
            text: Text to extract code blocks from
            
        Returns:
            List of code block dictionaries with metadata
        """
        code_blocks = []
        
        # Pattern 1: Markdown code blocks with language specification
        markdown_pattern = r'```(\w+)?\n(.*?)\n```'
        for match in re.finditer(markdown_pattern, text, re.DOTALL):
            language = match.group(1) or 'unknown'
            code_content = match.group(2).strip()
            
            if code_content:
                # If no language specified, try to detect it
                if language == 'unknown':
                    language = self._detect_programming_language(code_content)
                
                code_blocks.append({
                    'content': code_content,
                    'language': language,
                    'type': 'markdown_code_block',
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
        
        # Pattern 2: HTML pre/code tags
        soup = BeautifulSoup(text, 'html.parser')
        for pre_tag in soup.find_all('pre'):
            code_content = pre_tag.get_text().strip()
            if code_content:
                # Check for language class
                language = 'unknown'
                if pre_tag.get('class'):
                    for cls in pre_tag.get('class'):
                        if cls.startswith('language-'):
                            language = cls.replace('language-', '')
                            break
                        elif cls.startswith('lang-'):
                            language = cls.replace('lang-', '')
                            break
                
                if language == 'unknown':
                    language = self._detect_programming_language(code_content)
                
                code_blocks.append({
                    'content': code_content,
                    'language': language,
                    'type': 'html_pre_block',
                    'start_pos': 0,  # Approximate for HTML
                    'end_pos': len(code_content)
                })
        
        # Pattern 3: Indented code blocks (4+ spaces or tabs)
        indented_pattern = r'^((?:    |\t).+(?:\n(?:    |\t).*)*)$'
        for match in re.finditer(indented_pattern, text, re.MULTILINE):
            code_content = match.group(1).strip()
            if code_content and len(code_content.split('\n')) >= 2:  # At least 2 lines
                # Remove indentation
                lines = code_content.split('\n')
                min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
                cleaned_lines = [line[min_indent:] for line in lines]
                cleaned_content = '\n'.join(cleaned_lines)
                
                language = self._detect_programming_language(cleaned_content)
                
                code_blocks.append({
                    'content': cleaned_content,
                    'language': language,
                    'type': 'indented_code_block',
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
        
        return code_blocks
    
    def _enhance_document_with_code_blocks(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enhance document by extracting and processing code blocks separately
        
        Args:
            document: Original document with text and metadata
            
        Returns:
            List of enhanced document chunks including separate code block documents
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(text)
        
        enhanced_docs = []
        
        # Create main document (with code blocks marked but not removed)
        main_doc = {
            "text": text,
            "metadata": {
                **metadata,
                "content_type": "mixed",
                "has_code_blocks": len(code_blocks) > 0,
                "code_block_count": len(code_blocks)
            }
        }
        enhanced_docs.append(main_doc)
        
        # Create separate documents for each code block
        for i, code_block in enumerate(code_blocks):
            # Create context by extracting surrounding text
            context_before = ""
            context_after = ""
            
            if code_block['start_pos'] > 0:
                # Get up to 200 characters before the code block
                start_context = max(0, code_block['start_pos'] - 200)
                context_before = text[start_context:code_block['start_pos']].strip()
                if context_before:
                    # Try to get complete sentences
                    sentences = context_before.split('.')
                    if len(sentences) > 1:
                        context_before = '.'.join(sentences[-2:]).strip()
            
            if code_block['end_pos'] < len(text):
                # Get up to 200 characters after the code block
                end_context = min(len(text), code_block['end_pos'] + 200)
                context_after = text[code_block['end_pos']:end_context].strip()
                if context_after:
                    # Try to get complete sentences
                    sentences = context_after.split('.')
                    if len(sentences) > 1:
                        context_after = '.'.join(sentences[:2]).strip()
            
            # Create enhanced text for code block embedding
            enhanced_code_text = f"""
Context before: {context_before}

Code ({code_block['language']}):
{code_block['content']}

Context after: {context_after}
            """.strip()
            
            code_doc = {
                "text": enhanced_code_text,
                "metadata": {
                    **metadata,
                    "content_type": "code_block",
                    "programming_language": code_block['language'],
                    "code_block_type": code_block['type'],
                    "code_block_index": i,
                    "raw_code": code_block['content'],
                    "context_before": context_before,
                    "context_after": context_after,
                    "is_code_block": True
                }
            }
            enhanced_docs.append(code_doc)
        
        return enhanced_docs
    
    def _load_index(self):
        """Load FAISS index and document data if exists"""
        index_file = f"{self.index_path}.index"
        docs_file = f"{self.index_path}.documents.json"
        id_map_file = f"{self.index_path}.id_map.json"
        
        # Fall back to old format files if new ones don't exist
        old_docs_file = f"{self.index_path}.documents"
        old_id_map_file = f"{self.index_path}.id_map"
        
        try:
            # Check if the index file exists
            if os.path.exists(index_file):
                # Load FAISS index
                self.index = faiss.read_index(index_file)
                
                # Load documents - prefer JSON over pickle
                if os.path.exists(docs_file):
                    with open(docs_file, 'r') as f:
                        self.documents = json.load(f)
                elif os.path.exists(old_docs_file):
                    with open(old_docs_file, 'rb') as f:
                        self.documents = pickle.load(f)
                else:
                    self.documents = []
                    logger.warning(f"Documents file not found at {docs_file} or {old_docs_file}")
                
                # Load ID map - prefer JSON over pickle
                if os.path.exists(id_map_file):
                    with open(id_map_file, 'r') as f:
                        string_id_map = json.load(f)
                        self.id_map = {k: int(v) for k, v in string_id_map.items()}
                elif os.path.exists(old_id_map_file):
                    with open(old_id_map_file, 'rb') as f:
                        self.id_map = pickle.load(f)
                else:
                    self.id_map = {}
                    logger.warning(f"ID map file not found at {id_map_file} or {old_id_map_file}")
                
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            else:
                # Create new index
                self.index = faiss.IndexFlatL2(self.dimension)
                self.documents = []
                self.id_map = {}
                logger.info("Created new FAISS index")
        except Exception as e:
            # Create new index if loading fails
            logger.error(f"Error loading index: {str(e)}")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []
            self.id_map = {}
            logger.info("Created new FAISS index due to load error")
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using OpenAI's embedding model"""
        try:
            # Handle empty text
            if not text or text.isspace():
                logger.warning("Empty text provided for embedding. Using zeros.")
                return np.zeros(self.dimension, dtype=np.float32)
                
            # Estimate token count (rough estimate - 1 token ~= 4 chars)
            est_tokens = len(text) // 4
            
            # Check if text exceeds token limit
            if est_tokens > 8000:  # OpenAI embedding models have 8192 token limit
                logger.warning(f"Text exceeds embedding model token limit (~{est_tokens} tokens). Splitting into multiple parts.")
                
                # Calculate how many parts we need
                num_parts = (est_tokens // 8000) + 1
                chars_per_part = len(text) // num_parts
                
                # Split text into multiple parts
                embeddings = []
                for i in range(num_parts):
                    start_idx = i * chars_per_part
                    end_idx = min((i + 1) * chars_per_part, len(text))
                    part_text = text[start_idx:end_idx]
                    
                    # Skip empty parts
                    if not part_text or part_text.isspace():
                        continue
                        
                    # Get embedding for this part
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=part_text
                    )
                    
                    part_embedding = np.array(response.data[0].embedding, dtype=np.float32)
                    embeddings.append(part_embedding)
                
                # Combine embeddings by averaging them
                if embeddings:
                    combined_embedding = np.mean(embeddings, axis=0)
                    # Normalize the combined embedding
                    norm = np.linalg.norm(combined_embedding)
                    if norm > 0:
                        combined_embedding = combined_embedding / norm
                    return combined_embedding
                else:
                    # Fallback if something went wrong
                    return np.zeros(self.dimension, dtype=np.float32)
            
            # Handle normal case (within token limit)
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero embedding as fallback
            return np.zeros(self.dimension, dtype=np.float32)
            
    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if not texts:
            return np.array([])
            
        # Use a cache to avoid re-embedding identical texts
        embedding_cache = {}
        all_embeddings = []
        
        # Reusable function to estimate tokens
        def estimate_tokens(text):
            return len(text) // 4
        
        # Use retry mechanism for embedding generation
        max_retries = 3
        batch_size = 10  # Reduce batch size for stability
        
        # Calculate total batches for logging
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # For tracking unique texts in batch
            unique_texts = []
            unique_indices = []
            
            # Find unique texts to minimize API calls
            for j, text in enumerate(batch_texts):
                # Skip empty texts
                if not text or text.isspace():
                    logger.warning(f"Empty text at index {i+j}. Using zero embedding.")
                    all_embeddings.append(np.zeros(self.dimension, dtype=np.float32))
                    continue
                    
                # Check token count
                est_tokens = estimate_tokens(text)
                
                # For very long texts, handle them individually
                if est_tokens > 8000:
                    logger.warning(f"Text too long for embedding: ~{est_tokens} tokens. Will process individually.")
                    
                    # Process long text separately
                    emb = await self.generate_embedding(text)
                    all_embeddings.append(emb)
                    continue
                
                if text in embedding_cache:
                    # Use cached embedding
                    all_embeddings.append(embedding_cache[text])
                else:
                    # Mark for embedding
                    unique_texts.append(text)
                    unique_indices.append(j)
            
            # If we have unique texts to embed
            if unique_texts:
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        # Call embedding API only for unique texts
                        logger.info(f"Batch {i//batch_size + 1}/{total_batches}: Embedding {len(unique_texts)} unique texts")
                        
                        # Log request to OpenAI Embedding API
                        api_type = "Azure OpenAI" if self.use_azure else "OpenAI"
                        logger.info(f"{api_type} Embedding request: model={self.embedding_model}, input length={len(unique_texts)}")
                        
                        response = self.client.embeddings.create(
                            model=self.embedding_model,
                            input=unique_texts
                        )
                        
                        # Log response details
                        try:
                            logger.info(f"{api_type} Embedding response: model={response.model}, usage={response.usage}")
                        except Exception as e:
                            logger.info(f"{api_type} Embedding response received but couldn't log all details: {str(e)}")
                        
                        # Extract embeddings from response
                        batch_embeddings = [np.array(data.embedding, dtype=np.float32) 
                                          for data in response.data]
                        
                        # Update cache and store embeddings
                        for text, emb in zip(unique_texts, batch_embeddings):
                            embedding_cache[text] = emb
                            
                        # Add embeddings in original order
                        for j, text in enumerate(batch_texts):
                            if j in unique_indices:
                                # Get index in unique_texts
                                unique_idx = unique_indices.index(j)
                                all_embeddings.append(batch_embeddings[unique_idx])
                        
                        # Break retry loop on success
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        error_msg = str(e)
                        
                        # Handle rate limits with exponential backoff
                        if "rate limit" in error_msg.lower():
                            wait_time = 2 ** retry_count  # Exponential backoff
                            logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry...")
                            await asyncio.sleep(wait_time)
                        elif "invalid" in error_msg.lower() and "input" in error_msg.lower():
                            # Handle invalid input errors with individual embedding
                            logger.warning("Invalid input detected. Attempting individual text embedding...")
                            
                            # Try to embed each text individually
                            for j, single_text in enumerate(unique_texts):
                                try:
                                    # Check if text is too long
                                    if estimate_tokens(single_text) <= 8000:
                                        logger.info(f"Individual embedding for text {j+1}/{len(unique_texts)}")
                                        emb = await self.generate_embedding(single_text)
                                        embedding_cache[single_text] = emb
                                        
                                    else:
                                        # Single text is too long, use zero embedding
                                        logger.error("Single text exceeds token limit. Using zero embedding.")
                                        embedding_cache[single_text] = np.zeros(self.dimension, dtype=np.float32)
                                        
                                except Exception as inner_e:
                                    logger.error(f"Error embedding individual text: {str(inner_e)}")
                                    # Add zero embedding as fallback
                                    embedding_cache[single_text] = np.zeros(self.dimension, dtype=np.float32)
                            
                            # Add embeddings in original order
                            for j, text in enumerate(batch_texts):
                                if j in unique_indices:
                                    if text in embedding_cache:
                                        all_embeddings.append(embedding_cache[text])
                                    else:
                                        all_embeddings.append(np.zeros(self.dimension, dtype=np.float32))
                            
                            # Break retry loop since we've handled individual texts
                            break
                        else:
                            # Other errors
                            logger.error(f"Error generating embeddings (attempt {retry_count}/{max_retries}): {error_msg}")
                            
                            if retry_count == max_retries:
                                logger.error("Max retries reached. Using zero embeddings.")
                                # Add zero embeddings for this batch as fallback
                                for _ in range(len(batch_texts)):
                                    if len(all_embeddings) < i + _:  # Only add if not already added
                                        all_embeddings.append(np.zeros(self.dimension, dtype=np.float32))
                            
                            # Wait before retrying
                            await asyncio.sleep(retry_count * 2)
        
        # Ensure we have the right number of embeddings
        assert len(all_embeddings) <= len(texts), f"Generated {len(all_embeddings)} embeddings for {len(texts)} texts"
        
        # Fill any missing embeddings with zeros (shouldn't happen, but just in case)
        while len(all_embeddings) < len(texts):
            all_embeddings.append(np.zeros(self.dimension, dtype=np.float32))
            
        logger.info(f"Completed embedding generation for {len(texts)} texts")
        return np.array(all_embeddings, dtype=np.float32)

    async def add_document(self, document: Dict[str, Any]) -> str:
        """
        Add a document to the vector store with enhanced code block processing
        
        Args:
            document: Document with text to embed and metadata
            
        Returns:
            ID of the added document
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        
        # Generate a unique ID if not provided
        if "id" in document:
            doc_id = document["id"]
        else:
            base_id = metadata.get("url", "")
            if not base_id:
                # Generate a unique ID
                doc_id = str(uuid.uuid4())
            else:
                # Use URL as base ID with a UUID suffix to ensure uniqueness
                doc_id = f"{base_id}_{uuid.uuid4().hex[:8]}"
        
        try:
            # Enhance document with code block processing
            enhanced_docs = self._enhance_document_with_code_blocks(document)
            
            added_ids = []
            for i, enhanced_doc in enumerate(enhanced_docs):
                # Create unique ID for each enhanced document
                if i == 0:
                    # Main document keeps the original ID
                    enhanced_doc_id = doc_id
                else:
                    # Code block documents get suffixed IDs
                    enhanced_doc_id = f"{doc_id}_code_block_{i-1}"
                
                # Generate embedding
                embedding = await self.generate_embedding(enhanced_doc["text"])
                
                # Log embedding request details
                content_type = enhanced_doc["metadata"].get("content_type", "mixed")
                language = enhanced_doc["metadata"].get("programming_language", "N/A")
                logger.info(f"Generated embedding for document ID: {enhanced_doc_id}, "
                           f"type: {content_type}, language: {language}, "
                           f"text length: {len(enhanced_doc['text'])} characters")
                
                # Add to index
                faiss_id = len(self.documents)
                self.index.add(np.array([embedding], dtype=np.float32))
                
                # Store document with metadata
                self.documents.append({
                    "text": enhanced_doc["text"],
                    "metadata": enhanced_doc["metadata"],
                    "id": enhanced_doc_id
                })
                
                # Map external ID to index
                self.id_map[enhanced_doc_id] = faiss_id
                added_ids.append(enhanced_doc_id)
            
            # Save periodically (every 100 documents)
            if len(self.documents) % 100 == 0:
                self.save(self.index_path)
                
            logger.info(f"Added document with {len(enhanced_docs)} chunks (main + {len(enhanced_docs)-1} code blocks)")
            return doc_id  # Return the main document ID
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple documents to the vector store with enhanced code block processing
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of IDs for added documents (main document IDs only)
        """
        if not documents:
            return []
            
        # Initialize documents list if it doesn't exist
        if not hasattr(self, 'documents'):
            self.documents = []
            
        # Initialize id_map if it doesn't exist
        if not hasattr(self, 'id_map'):
            self.id_map = {}
        
        try:
            # Process all documents with code block enhancement
            all_enhanced_docs = []
            main_doc_ids = []
            
            for doc in documents:
                metadata = doc.get("metadata", {})
                if "id" in doc:
                    doc_id = doc["id"]
                else:
                    base_id = metadata.get("url", "")
                    if not base_id:
                        doc_id = str(uuid.uuid4())
                    else:
                        doc_id = f"{base_id}_{uuid.uuid4().hex[:8]}"
                
                main_doc_ids.append(doc_id)
                
                # Enhance document with code block processing
                enhanced_docs = self._enhance_document_with_code_blocks(doc)
                
                # Add enhanced document IDs
                for i, enhanced_doc in enumerate(enhanced_docs):
                    if i == 0:
                        # Main document keeps the original ID
                        enhanced_doc_id = doc_id
                    else:
                        # Code block documents get suffixed IDs
                        enhanced_doc_id = f"{doc_id}_code_block_{i-1}"
                    
                    enhanced_doc["id"] = enhanced_doc_id
                    all_enhanced_docs.append(enhanced_doc)
            
            # Generate embeddings for all enhanced documents
            texts = [doc["text"] for doc in all_enhanced_docs]
            embeddings = await self.generate_embeddings(texts)
            
            # Add to index
            start_idx = len(self.documents)
            self.index.add(embeddings)
            
            # Store enhanced documents with metadata
            for i, enhanced_doc in enumerate(all_enhanced_docs):
                # Map external ID to index
                faiss_id = start_idx + i
                self.id_map[enhanced_doc["id"]] = faiss_id
                
                # Store document
                self.documents.append({
                    "text": enhanced_doc["text"],
                    "metadata": enhanced_doc["metadata"],
                    "id": enhanced_doc["id"]
                })
            
            # Save the index
            self.save(self.index_path)
            
            total_code_blocks = len(all_enhanced_docs) - len(documents)
            logger.info(f"Added {len(documents)} documents with {total_code_blocks} code blocks to FAISS")
            logger.info(f"Total enhanced documents: {len(all_enhanced_docs)}")
            
            return main_doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    async def search(self, query: str, k: int = 5, code_focused: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents with enhanced code block support
        
        Args:
            query: Query string
            k: Number of results to return
            code_focused: If True, prioritize code blocks; if False, prioritize text; if None, auto-detect
            
        Returns:
            List of document dictionaries with similarity scores
        """
        # Load index if not already loaded
        self._load_index()
        
        if not self.index:
            logger.warning("No index loaded")
            return []
            
        if not self.documents:
            logger.warning("No documents in the index")
            return []
            
        try:
            # Auto-detect if query is code-focused
            if code_focused is None:
                code_focused = self._is_code_query(query)
            
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Search the index - retrieve more results than needed for re-ranking
            retrieval_k = min(k * 4, len(self.documents))  # Retrieve 4x more for re-ranking
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32), k=retrieval_k
            )
            
            # First pass results
            initial_results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1:  # Valid result
                    doc = self.documents[idx].copy()
                    # Convert distance to similarity score (lower distance = higher similarity)
                    embedding_score = 1.0 / (1.0 + dist)
                    doc["embedding_score"] = float(embedding_score)
                    initial_results.append(doc)
            
            # If we have no results, return empty list
            if not initial_results:
                return []
                
            # Re-rank results with enhanced scoring
            weighted_results = []
            for doc in initial_results:
                # Start with the embedding score
                final_score = doc["embedding_score"]
                
                # Extract text and metadata
                text = doc["text"]
                metadata = doc.get("metadata", {})
                content_type = metadata.get("content_type", "mixed")
                
                # Code-focused search enhancements
                if code_focused:
                    # Boost code blocks significantly
                    if content_type == "code_block":
                        final_score += 0.4
                        
                        # Additional boost for matching programming language
                        prog_lang = metadata.get("programming_language", "")
                        if prog_lang and prog_lang.lower() in query.lower():
                            final_score += 0.2
                    
                    # Slight boost for documents with code blocks
                    elif metadata.get("has_code_blocks", False):
                        final_score += 0.1
                        
                else:
                    # Text-focused search: slight penalty for code blocks
                    if content_type == "code_block":
                        final_score -= 0.1
                    # Boost for mixed content
                    elif content_type == "mixed":
                        final_score += 0.05
                
                # Enhanced keyword matching
                query_terms = [term.lower() for term in query.split() if len(term) > 2]
                text_lower = text.lower()
                
                # Count exact matches
                exact_matches = sum(1 for term in query_terms if term in text_lower)
                if exact_matches > 0:
                    match_boost = min(0.3, exact_matches * 0.05)
                    final_score += match_boost
                
                # Special handling for code blocks
                if content_type == "code_block":
                    raw_code = metadata.get("raw_code", "")
                    if raw_code:
                        # Boost for function/class name matches in code
                        code_matches = sum(1 for term in query_terms if term in raw_code.lower())
                        if code_matches > 0:
                            final_score += min(0.25, code_matches * 0.08)
                
                # Weight for document structure
                heading = metadata.get("heading", "")
                if heading:
                    # Check if any query terms appear in the heading
                    heading_lower = heading.lower()
                    heading_match_count = sum(1 for term in query_terms if term in heading_lower)
                    # Apply heading boost
                    heading_boost = min(0.25, heading_match_count * 0.08)
                    final_score += heading_boost
                
                # Level boost for headings with higher hierarchy
                level = metadata.get("level", 0)
                if level > 0:
                    # Level 1 (H1) gets highest boost, decreasing for H2, H3, etc.
                    level_boost = max(0, 0.1 - ((level - 1) * 0.03))
                    final_score += level_boost
                
                # Context relevance for code blocks
                if content_type == "code_block":
                    context_before = metadata.get("context_before", "")
                    context_after = metadata.get("context_after", "")
                    context_text = f"{context_before} {context_after}".lower()
                    
                    context_matches = sum(1 for term in query_terms if term in context_text)
                    if context_matches > 0:
                        final_score += min(0.15, context_matches * 0.05)
                
                # Remove the embedding_score from the final document
                doc.pop("embedding_score", None)
                # Update with final weighted score
                doc["score"] = min(1.0, final_score)  # Cap at 1.0
                weighted_results.append(doc)
            
            # Group results by main document ID to avoid too many code blocks from same source
            grouped_results = {}
            for doc in weighted_results:
                doc_id = doc["id"]
                main_id = doc_id.split("_code_block_")[0]  # Get main document ID
                
                if main_id not in grouped_results:
                    grouped_results[main_id] = []
                grouped_results[main_id].append(doc)
            
            # Select best result from each group and flatten
            final_results = []
            for main_id, docs in grouped_results.items():
                # Sort docs by score and take the best one
                docs.sort(key=lambda x: x["score"], reverse=True)
                
                # If code-focused, prefer code blocks; otherwise prefer mixed content
                if code_focused:
                    # Prefer code blocks, but ensure we have the best scoring one
                    final_results.append(docs[0])
                else:
                    # Prefer mixed content, fall back to code blocks
                    mixed_docs = [d for d in docs if d["metadata"].get("content_type") == "mixed"]
                    if mixed_docs:
                        final_results.append(mixed_docs[0])
                    else:
                        final_results.append(docs[0])
            
            # Sort by final score and take top k
            final_results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"Search completed: query='{query}', code_focused={code_focused}, "
                       f"initial_results={len(initial_results)}, final_results={len(final_results[:k])}")
            
            return final_results[:k]
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            import traceback
            logger.error(f"Complete search error: {traceback.format_exc()}")
            return []
    
    def _is_code_query(self, query: str) -> bool:
        """
        Determine if a query is focused on code/programming concepts
        
        Args:
            query: Search query
            
        Returns:
            True if query appears to be code-focused
        """
        code_keywords = [
            'function', 'method', 'class', 'variable', 'import', 'export',
            'def', 'return', 'if', 'else', 'for', 'while', 'try', 'catch',
            'code', 'example', 'snippet', 'implementation', 'syntax',
            'python', 'javascript', 'typescript', 'java', 'cpp', 'c++',
            'bash', 'sql', 'html', 'css', 'yaml', 'json', 'dockerfile',
            'api', 'endpoint', 'parameter', 'argument', 'callback',
            'async', 'await', 'promise', 'lambda', 'arrow function'
        ]
        
        query_lower = query.lower()
        
        # Check for programming language mentions
        for language in self.code_patterns.keys():
            if language in query_lower:
                return True
        
        # Check for code-related keywords
        code_keyword_count = sum(1 for keyword in code_keywords if keyword in query_lower)
        
        # If more than 20% of words are code-related, consider it code-focused
        total_words = len(query.split())
        if total_words > 0 and (code_keyword_count / total_words) > 0.2:
            return True
        
        # Check for code-like patterns
        if re.search(r'[(){}\[\]<>]', query) or '.' in query or '_' in query:
            return True
        
        return False
    
    async def remove_documents_by_domain(self, domain: str) -> int:
        """
        Remove all documents from a specific domain from the vector store
        
        Args:
            domain: Domain name (e.g., 'example.com')
            
        Returns:
            Number of documents removed
        """
        # Load index if not already loaded
        self._load_index()
        
        if not self.index:
            logger.warning("No index loaded to remove documents from")
            return 0
            
        # Load the document metadata
        self.load(self.index_path)
        
        # Find document IDs to remove
        ids_to_remove = []
        
        for doc in self.documents:
            url = doc.get("metadata", {}).get("url", "")
            
            # Check if the URL contains this domain
            if domain in url:
                ids_to_remove.append(doc["id"])
                
        if not ids_to_remove:
            logger.info(f"No documents found with domain '{domain}'")
            return 0
            
        logger.info(f"Removing {len(ids_to_remove)} documents from domain '{domain}'")
        
        # Generate list of IDs to keep
        all_ids = list(self.id_map.keys())
        keep_ids = [doc_id for doc_id in all_ids if doc_id not in ids_to_remove]
        
        # Create new index and transfer data
        new_index = faiss.IndexFlatL2(self.dimension)
        
        if keep_ids:
            # Build a list of vectors to retain
            keep_vectors = []
            new_documents = []
            new_id_map = {}
            
            # Transfer vectors and metadata for documents we're keeping
            for doc_id in keep_ids:
                vector_idx = self.id_map[doc_id]
                vector = self.index.reconstruct(vector_idx)
                keep_vectors.append(vector)
                new_documents.append(self.documents[vector_idx])
                new_id_map[doc_id] = len(new_documents) - 1
                
            # Add vectors to new index
            keep_vectors = np.array(keep_vectors).astype('float32')
            new_index.add(keep_vectors)
            
            # Update index and metadata
            self.index = new_index
            self.documents = new_documents
            self.id_map = new_id_map
        else:
            # If no documents remain, just reset everything
            self.index = new_index
            self.documents = []
            self.id_map = {}
        
        # Save the updated index and metadata
        self.save(self.index_path)
        
        return len(ids_to_remove)
    
    def save(self, path: str) -> None:
        """
        Save the index and documents to disk
        
        Args:
            path: Path to save the index to
        """
        try:
            # Create extremely simplified documents for saving
            # This avoids deep recursion issues with pickle
            simplified_documents = []
            for doc in self.documents:
                # Only keep the bare minimum: text and ID
                simplified_doc = {
                    "text": doc.get("text", ""),
                    "id": doc.get("id", "")
                }
                
                # Create a flat metadata dictionary with only string values
                metadata = doc.get("metadata", {})
                flat_metadata = {}
                
                # Convert all metadata items to simple strings to avoid recursion
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        flat_metadata[key] = value
                    else:
                        # Convert complex objects to strings
                        try:
                            flat_metadata[key] = str(value)
                        except:
                            # If conversion fails, skip this metadata item
                            logger.warning(f"Skipping metadata item {key} due to serialization issues")
                
                simplified_doc["metadata"] = flat_metadata
                simplified_documents.append(simplified_doc)
            
            # Save FAISS index directly without pickling
            faiss.write_index(self.index, f"{path}.index")
            
            # Use more reliable JSON for documents instead of pickle
            import json
            with open(f"{path}.documents.json", "w") as f:
                json.dump(simplified_documents, f)
                
            # Simplify the ID map to just key-value pairs (string keys)
            string_id_map = {str(k): int(v) for k, v in self.id_map.items()}
            with open(f"{path}.id_map.json", "w") as f:
                json.dump(string_id_map, f)
                
            logger.info(f"Saved index with {len(self.documents)} documents to {path}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            # Don't re-raise to avoid crashing the application
            
    def load(self, path: str) -> bool:
        """
        Load the index and documents from disk
        
        Args:
            path: Base path to load files from (without extension)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Check for both pickle and JSON files for backward compatibility
            document_file = f"{path}.documents.json" if os.path.exists(f"{path}.documents.json") else f"{path}.documents"
            id_map_file = f"{path}.id_map.json" if os.path.exists(f"{path}.id_map.json") else f"{path}.id_map"
            
            if os.path.exists(document_file):
                # Try loading from JSON first (preferred)
                if document_file.endswith('.json'):
                    import json
                    with open(document_file, "r") as f:
                        self.documents = json.load(f)
                    logger.info(f"Loaded {len(self.documents)} documents from JSON")
                else:
                    # Fall back to pickle if JSON not available
                    with open(document_file, 'rb') as f:
                        self.documents = pickle.load(f)
                    logger.info(f"Loaded {len(self.documents)} documents from pickle")
            else:
                logger.warning(f"Document file not found at {document_file}")
                return False
            
            # Load ID map similarly
            if os.path.exists(id_map_file):
                if id_map_file.endswith('.json'):
                    import json
                    with open(id_map_file, "r") as f:
                        string_id_map = json.load(f)
                        # Convert string keys back to original type if needed
                        self.id_map = {k: int(v) for k, v in string_id_map.items()}
                else:
                    with open(id_map_file, 'rb') as f:
                        self.id_map = pickle.load(f)
            else:
                logger.warning(f"ID map file not found at {id_map_file}")
                return False
            
            logger.info(f"Successfully loaded FAISS index with {len(self.documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            # Initialize with empty collections
            self.documents = []
            self.id_map = {}
            return False
    
    def clear(self) -> None:
        """Clear the index and all documents"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.id_map = {}
        logger.info("Cleared all documents from index")
