from typing import List, Dict, Any, Optional, Tuple
import os
import logging
import numpy as np
import pickle
import faiss
from dotenv import load_dotenv
import openai
import uuid
import time
import asyncio

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
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using Azure OpenAI"""
        try:
            # Call Azure OpenAI embedding API
            response = self.client.embeddings.create(
                model=self.azure_embedding_deployment,
                input=[text],
            )
            
            # Extract embedding from response
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
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
        
        # Increase batch size but stay within API limits
        batch_size = 100  # Increased from 20
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
                try:
                    # Call Azure OpenAI embedding API only for unique texts
                    logger.info(f"Batch {i//batch_size + 1}/{total_batches}: Embedding {len(unique_texts)} unique texts")
                    response = self.client.embeddings.create(
                        model=self.azure_embedding_deployment,
                        input=unique_texts,
                    )
                    
                    # Extract embeddings from response
                    batch_embeddings = [np.array(data.embedding, dtype=np.float32) 
                                        for data in response.data]
                    
                    # Add to cache and results
                    for text, emb in zip(unique_texts, batch_embeddings):
                        embedding_cache[text] = emb
                        all_embeddings.append(emb)
                    
                    # Don't overwhelm the API - longer delay between larger batches
                    if i + batch_size < len(texts):
                        await asyncio.sleep(0.2)  # Reduced from 0.5
                        
                except Exception as e:
                    logger.error(f"Error generating embeddings for batch: {str(e)}")
                    # Add zero embeddings for this batch as fallback
                    zero_embedding = np.zeros(self.dimension, dtype=np.float32)
                    for _ in unique_texts:
                        all_embeddings.append(zero_embedding)
        
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
            path: Base path to save files (without extension)
        """
        try:
            # Save FAISS index directly - no pickling
            faiss.write_index(self.index, f"{path}.index")
            
            # Create a clean version of documents for pickling
            clean_documents = []
            for doc in self.documents:
                # Create a simplified copy without potential circular references
                clean_doc = {
                    "text": doc.get("text", ""),
                    "metadata": {k: v for k, v in doc.get("metadata", {}).items()},
                    "id": doc.get("id", "")
                }
                clean_documents.append(clean_doc)
                
            # Save documents as pickle
            with open(f"{path}.documents", "wb") as f:
                pickle.dump(clean_documents, f, protocol=4)  # Use a stable protocol
                
            # Save ID mapping as JSON instead of pickle for better compatibility
            with open(f"{path}.id_map.json", "w") as f:
                import json
                # Convert keys to strings since JSON requires string keys
                json_id_map = {str(k): v for k, v in self.id_map.items()}
                json.dump(json_id_map, f)
                
            logger.info(f"Saved index with {len(self.documents)} documents to {path}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            
    def load(self, path: str) -> bool:
        """
        Load the index and documents from disk
        
        Args:
            path: Base path to load files from (without extension)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load FAISS index
            if os.path.exists(f"{path}.index"):
                self.index = faiss.read_index(f"{path}.index")
            else:
                logger.warning(f"No index file found at {path}.index")
                return False
                
            # Load documents
            if os.path.exists(f"{path}.documents"):
                with open(f"{path}.documents", "rb") as f:
                    self.documents = pickle.load(f)
            else:
                logger.warning(f"No documents file found at {path}.documents")
                return False
                
            # Load ID mapping - try the new JSON format first, then fall back to pickle
            if os.path.exists(f"{path}.id_map.json"):
                import json
                with open(f"{path}.id_map.json", "r") as f:
                    json_id_map = json.load(f)
                    # Convert keys back from strings
                    self.id_map = {k: v for k, v in json_id_map.items()}
            elif os.path.exists(f"{path}.id_map"):
                with open(f"{path}.id_map", "rb") as f:
                    self.id_map = pickle.load(f)
            else:
                logger.warning(f"No ID mapping file found")
                self.id_map = {}
                
            logger.info(f"Loaded index with {len(self.documents)} documents from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def clear(self) -> None:
        """Clear the index and all documents"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.id_map = {}
        logger.info("Cleared all documents from index")
