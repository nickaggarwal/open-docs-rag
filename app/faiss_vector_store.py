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

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    def __init__(self, index_path="./data/faiss_index", dimension=1536):
        """
        Initialize a vector store with FAISS for similarity search
        
        Args:
            index_path: Base path to save/load the FAISS index
            dimension: Dimension of embeddings (1536 for OpenAI's ada-002)
        """
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
        
        self.dimension = dimension
        self.index_path = index_path
        
        # Initialize default empty collections in case loading fails
        self.documents = []
        self.id_map = {}
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Initialize or load existing index and documents
        if os.path.exists(f"{index_path}.index"):
            logger.info(f"Loading existing index from {index_path}")
            try:
                self.index = faiss.read_index(f"{index_path}.index")
                if not self.load(index_path):
                    # If loading fails, create a new index
                    logger.warning("Failed to load existing data, creating new index")
                    self.index = faiss.IndexFlatL2(dimension)
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}")
                # Create new index if loading fails
                self.index = faiss.IndexFlatL2(dimension)
        else:
            logger.info(f"Creating new FAISS index with dimension {dimension}")
            self.index = faiss.IndexFlatL2(dimension)  # L2 distance
    
    def estimate_tokens(self, text):
        """Estimate number of tokens in a text string"""
        # OpenAI uses ~4 chars per token on average
        return len(text) // 3  # Conservative estimate (assuming 3 chars per token)
        
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using Azure OpenAI"""
        if not text.strip():
            return np.zeros(self.dimension, dtype=np.float32)
            
        # Estimate token count
        est_tokens = self.estimate_tokens(text)
        
        # If text might exceed token limit, truncate it
        if est_tokens > 8000:  # Azure OpenAI embedding models have 8192 token limit
            logger.warning(f"Text exceeds embedding model token limit (~{est_tokens} tokens). Truncating to ~8000 tokens.")
            # Find a good truncation point - preferably at paragraph break
            chars_to_keep = 8000 * 3  # ~8000 tokens
            
            # Try to find a paragraph break near the limit
            last_paragraph = text[:chars_to_keep].rfind('\n\n')
            if last_paragraph > chars_to_keep * 0.8:  # If we can keep at least 80% of the text
                truncated_text = text[:last_paragraph]
            else:
                # Otherwise just truncate at character limit
                truncated_text = text[:chars_to_keep]
                
            # Add truncation note
            truncated_text += "\n\n[... Content truncated due to length ...]"
            text = truncated_text
            
        try:
            response = self.client.embeddings.create(
                model=self.azure_embedding_deployment,
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
            return np.empty((0, self.dimension), dtype=np.float32)
            
        # Use a cache to avoid re-embedding identical texts
        embedding_cache = {}
        all_embeddings = []
        
        # Check for texts that are too long and truncate them
        processed_texts = []
        for text in texts:
            # Estimate token count
            est_tokens = self.estimate_tokens(text)
            
            # The embedding model has a limit of 8192 tokens
            if est_tokens > 8000:
                logger.warning(f"Text too long for embedding: ~{est_tokens} tokens. Truncating.")
                # Find a good truncation point - preferably at paragraph break
                chars_to_keep = 8000 * 3  # ~8000 tokens
                
                # Try to find a paragraph break near the limit
                last_paragraph = text[:chars_to_keep].rfind('\n\n')
                if last_paragraph > chars_to_keep * 0.8:  # If we can keep at least 80% of the text
                    truncated_text = text[:last_paragraph]
                else:
                    # Otherwise just truncate at character limit
                    truncated_text = text[:chars_to_keep]
                    
                # Add truncation note
                truncated_text += "\n\n[... Content truncated due to length ...]"
                processed_texts.append(truncated_text)
            else:
                processed_texts.append(text)
        
        # Replace original texts with processed ones
        texts = processed_texts
            
        # Reduce batch size to prevent errors
        batch_size = 16  # Further reduced from 20 to avoid rate limits and token limits
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"Generating embeddings for {len(texts)} texts in {total_batches} batches")
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            unique_texts = []
            indices = []
            
            # De-duplicate texts within the batch
            for j, text in enumerate(batch):
                if text in embedding_cache:
                    # Use cached embedding
                    all_embeddings.append(embedding_cache[text])
                else:
                    unique_texts.append(text)
                    indices.append(j)
            
            if unique_texts:
                retry_count = 0
                max_retries = 5
                while retry_count < max_retries:
                    try:
                        # Call Azure OpenAI embedding API only for unique texts
                        logger.info(f"Batch {i//batch_size + 1}/{total_batches}: Embedding {len(unique_texts)} unique texts")
                        
                        # Log request to Azure OpenAI Embedding API
                        logger.info(f"Azure Embedding request: model={self.azure_embedding_deployment}, input length={len(unique_texts)}")
                        
                        response = self.client.embeddings.create(
                            model=self.azure_embedding_deployment,
                            input=unique_texts,
                        )
                        
                        try:
                            logger.info(f"Azure Embedding response: model={response.model}, usage={response.usage}")
                        except Exception as e:
                            logger.info(f"Azure Embedding response received but couldn't log all details: {str(e)}")
                        
                        # Extract embeddings from response
                        batch_embeddings = [np.array(data.embedding, dtype=np.float32) 
                                            for data in response.data]
                        
                        # Add to cache and results
                        for text, emb in zip(unique_texts, batch_embeddings):
                            embedding_cache[text] = emb
                            all_embeddings.append(emb)
                        
                        # Success, break out of retry loop
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        error_msg = str(e)
                        
                        # Check if it's a rate limit error (HTTP 429)
                        if "429" in error_msg:
                            # Exponential backoff
                            wait_time = 2 ** retry_count
                            logger.warning(f"Rate limit hit. Retrying in {wait_time} seconds...")
                            await asyncio.sleep(wait_time)
                        # Check if token limit error 
                        elif "maximum context length" in error_msg or "token" in error_msg.lower():
                            logger.error(f"Token limit error: {error_msg}")
                            # Try to process texts individually
                            if len(unique_texts) > 1:
                                logger.info("Trying to process texts individually...")
                                for single_text in unique_texts:
                                    try:
                                        # Truncate if still too long
                                        est_tokens = self.estimate_tokens(single_text)
                                        if est_tokens > 8000:
                                            logger.warning(f"Text too long: ~{est_tokens} tokens. Truncating further.")
                                            single_text = single_text[:24000]  # Roughly 8000 tokens
                                            
                                        emb = await self.generate_embedding(single_text)
                                        embedding_cache[single_text] = emb
                                        all_embeddings.append(emb)
                                        # Small delay to avoid rate limits
                                        await asyncio.sleep(0.5)
                                    except Exception as inner_e:
                                        logger.error(f"Error embedding individual text: {str(inner_e)}")
                                        # Add zero embedding as fallback
                                        all_embeddings.append(np.zeros(self.dimension, dtype=np.float32))
                                # Consider this batch processed
                                break
                            else:
                                # Single text is too long, use zero embedding
                                logger.error("Single text exceeds token limit. Using zero embedding.")
                                for _ in unique_texts:
                                    all_embeddings.append(np.zeros(self.dimension, dtype=np.float32))
                                break
                        else:
                            # For other errors, retry with backoff
                            logger.error(f"Error generating embeddings (attempt {retry_count}/{max_retries}): {error_msg}")
                            if retry_count >= max_retries:
                                logger.error("Max retries reached. Using zero embeddings.")
                                # Add zero embeddings for this batch as fallback
                                for _ in unique_texts:
                                    all_embeddings.append(np.zeros(self.dimension, dtype=np.float32))
                            else:
                                # Wait before retrying (with exponential backoff)
                                wait_time = 2 ** retry_count
                                await asyncio.sleep(wait_time)
                
                # Don't overwhelm the API - longer delay between batches
                if i + batch_size < len(texts):
                    await asyncio.sleep(1)  # Increased from 0.2 to avoid rate limits
        
        logger.info(f"Completed embedding generation for {len(texts)} texts")
        return np.array(all_embeddings, dtype=np.float32)
    
    async def add_document(self, document: Dict[str, Any]) -> str:
        """
        Add a document to the vector store
        
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
            # Generate embedding
            embedding = await self.generate_embedding(text)
            
            # Log embedding request details
            logger.info(f"Generated embedding for document ID: {doc_id}, text length: {len(text)} characters")
            
            # Add to index
            faiss_id = len(self.documents)
            self.index.add(np.array([embedding], dtype=np.float32))
            
            # Store document with metadata
            self.documents.append({
                "text": text,
                "metadata": metadata,
                "id": doc_id
            })
            
            # Map external ID to index
            self.id_map[doc_id] = faiss_id
            
            # Save periodically (every 100 documents)
            if len(self.documents) % 100 == 0:
                self.save(self.index_path)
                
            logger.info(f"Added document with ID {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise
    
    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple documents to the vector store
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of IDs for added documents
        """
        if not documents:
            return []
            
        # Initialize documents list if it doesn't exist
        if not hasattr(self, 'documents'):
            self.documents = []
            
        # Initialize id_map if it doesn't exist
        if not hasattr(self, 'id_map'):
            self.id_map = {}
            
        # Generate unique IDs for all documents
        doc_ids = []
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
            doc_ids.append(doc_id)
        
        try:
            # Generate embeddings for all documents
            texts = [doc.get("text", "") for doc in documents]
            embeddings = await self.generate_embeddings(texts)
            
            # Add to index
            start_idx = len(self.documents)
            self.index.add(embeddings)
            
            # Store documents with metadata
            for i, (doc, doc_id) in enumerate(zip(documents, doc_ids)):
                # Map external ID to index
                faiss_id = start_idx + i
                self.id_map[doc_id] = faiss_id
                
                # Store document
                self.documents.append({
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                    "id": doc_id
                })
            
            # Save the index
            self.save(self.index_path)
            
            logger.info(f"Added {len(documents)} documents to FAISS")
            return doc_ids
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of documents with similarity scores
        """
        if not self.documents:
            logger.warning("No documents in the index")
            return []
            
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Search the index
            k = min(k, len(self.documents))
            distances, indices = self.index.search(
                np.array([query_embedding], dtype=np.float32), k=k
            )
            
            # Format results
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx != -1:  # Valid result
                    doc = self.documents[idx].copy()
                    # Convert distance to similarity score (lower distance = higher similarity)
                    score = 1.0 / (1.0 + dist)
                    doc["score"] = float(score)
                    results.append(doc)
            
            return results
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            return []
    
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
                    with open(document_file, "rb") as f:
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
                    with open(id_map_file, "rb") as f:
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
