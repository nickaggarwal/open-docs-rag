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
    
    def _load_index(self):
        """Load FAISS index and document data if exists"""
        index_file = f"{self.index_path}/faiss.index"
        docs_file = f"{self.index_path}/documents.pkl"
        
        try:
            if os.path.exists(index_file) and os.path.exists(docs_file):
                # Load FAISS index
                self.index = faiss.read_index(index_file)
                
                # Load documents and ID map
                with open(docs_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', [])
                    self.id_map = data.get('id_map', {})
                
                logger.info(f"Loaded existing index with {len(self.documents)} documents")
            else:
                # Create new index
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            # Create new index as fallback
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info("Created new FAISS index due to loading error")
    
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
