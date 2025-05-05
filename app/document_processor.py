from typing import List, Dict, Any, Optional
import asyncio
import re
import logging
import time
from concurrent.futures import TimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, max_process_time: int = 120):
        """
        Initialize the document processor with configuration for chunking
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between chunks
            max_process_time: Maximum time in seconds to spend processing a single document
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_process_time = max_process_time  # 2 minutes per document
        self.max_chunk_time = 5  # 5 seconds max for chunking algorithm
        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, max_process_time={max_process_time}s")

    async def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of documents: split into chunks for embedding
        
        Args:
            documents: List of documents with text and metadata
            
        Returns:
            List of processed document chunks with text and metadata
        """
        processed_documents = []
        
        logger.info(f"Processing {len(documents)} documents...")
        start_time = time.time()
        
        for i, doc in enumerate(documents):
            try:
                # Set a timeout for processing each document
                doc_start_time = time.time()
                
                # Create a task with timeout for processing
                try:
                    result = await asyncio.wait_for(
                        self._process_document(doc),
                        timeout=self.max_process_time
                    )
                    processed_documents.extend(result)
                    
                    # Log progress
                    doc_time = time.time() - doc_start_time
                    logger.info(f"Processed document {i+1}/{len(documents)} in {doc_time:.2f}s - " 
                               f"Created {len(result)} chunks")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Document {i+1}/{len(documents)} processing timed out after {self.max_process_time}s. "
                                  f"Using emergency simplified chunking.")
                    
                    # Use emergency simplified chunking with tight timeout
                    try:
                        text = doc.get("text", "")
                        metadata = doc.get("metadata", {})
                        
                        # Just split by paragraphs and limit chunk size
                        simple_chunks = text.split('\n\n')
                        simple_chunks = [chunk for chunk in simple_chunks if chunk.strip()]
                        
                        # Make sure chunks aren't too large
                        final_chunks = []
                        for chunk in simple_chunks:
                            if len(chunk) > self.chunk_size * 2:  # Very large paragraph
                                # Split by sentences or just force split
                                sentences = chunk.split('. ')
                                current_chunk = ""
                                for sentence in sentences:
                                    if len(current_chunk) + len(sentence) > self.chunk_size:
                                        if current_chunk:
                                            final_chunks.append(current_chunk)
                                        current_chunk = sentence
                                    else:
                                        current_chunk += sentence + ". "
                                if current_chunk:
                                    final_chunks.append(current_chunk)
                            else:
                                final_chunks.append(chunk)
                        
                        # Create processed chunks with metadata
                        emergency_chunks = []
                        for j, chunk_text in enumerate(final_chunks):
                            if not chunk_text.strip():
                                continue
                                
                            chunk_metadata = metadata.copy()
                            chunk_metadata["chunk_id"] = j
                            chunk_metadata["emergency_chunking"] = True
                            
                            emergency_chunks.append({
                                "text": chunk_text,
                                "metadata": chunk_metadata
                            })
                            
                        processed_documents.extend(emergency_chunks)
                        logger.info(f"Emergency processed document {i+1}/{len(documents)} - "
                                   f"Created {len(emergency_chunks)} chunks with simplified method")
                    except Exception as e:
                        logger.error(f"Emergency chunking also failed for document {i+1}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error processing document {i+1}: {str(e)}")
                continue
                
        total_time = time.time() - start_time
        logger.info(f"Processed {len(documents)} documents into {len(processed_documents)} chunks in {total_time:.2f}s")
        
        return processed_documents
    
    async def _process_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single document: split into chunks with overlap
        
        Args:
            document: Document with text and metadata
            
        Returns:
            List of document chunks with text and metadata
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        
        if not text.strip():
            logger.warning(f"Empty document text for {metadata.get('url', 'unknown')}")
            return []
            
        # Choose chunking strategy based on text size
        chunks = []
        try:
            # Use asyncio to apply a timeout to the chunking process
            try:
                if len(text) > 100000:  # Very large document
                    logger.info(f"Using simplified chunking for large document ({len(text)} chars)")
                    chunks = await asyncio.wait_for(
                        self._simple_chunk_text_async(text),
                        timeout=self.max_chunk_time
                    )
                else:
                    # For regular sized documents, use the more sophisticated chunking
                    chunks = await asyncio.wait_for(
                        self._chunk_text_async(text),
                        timeout=self.max_chunk_time
                    )
            except asyncio.TimeoutError:
                logger.warning(f"Chunking timed out after {self.max_chunk_time}s, falling back to simple chunking")
                chunks = await self._simple_chunk_text_async(text)
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            # Last resort emergency chunking
            chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            
        # Add metadata to each chunk
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
                
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            
            processed_chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            
        return processed_chunks
    
    async def _chunk_text_async(self, text: str) -> List[str]:
        """Async wrapper for _chunk_text"""
        return self._chunk_text(text)
        
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using natural boundaries (paragraphs, sentences)
        with a target size and overlap
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        # First try to split by double newlines (paragraphs)
        if '\n\n' in text:
            # Split on paragraphs
            paragraphs = text.split('\n\n')
            return self._merge_splits(paragraphs)
            
        # If no paragraphs, try to split by single newlines
        if '\n' in text:
            # Split on newlines
            lines = text.split('\n')
            return self._merge_splits(lines)
            
        # If no newlines, try to split by sentences using heuristics
        # (avoid using regex for better performance)
        sentences = []
        current_sentence = ""
        for char in text:
            current_sentence += char
            if char in ['.', '!', '?'] and len(current_sentence) > 10:
                sentences.append(current_sentence)
                current_sentence = ""
        
        if current_sentence:
            sentences.append(current_sentence)
            
        if sentences:
            return self._merge_splits(sentences)
        
        # Last resort: just split by chunk size
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
        
    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Merge splits into chunks with target size and overlap
        
        Args:
            splits: List of text splits (paragraphs, sentences, etc.)
            
        Returns:
            List of text chunks
        """
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # If adding this split would exceed chunk size, save current chunk and start a new one
            if len(current_chunk) + len(split) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If a single split is larger than chunk size, we need to handle it separately
                if len(split) > self.chunk_size:
                    # For very large splits, break them down further
                    for i in range(0, len(split), self.chunk_size - self.chunk_overlap):
                        chunk = split[i:i + self.chunk_size]
                        if chunk:
                            chunks.append(chunk)
                    current_chunk = chunks[-1][-self.chunk_overlap:] if chunks else ""
                else:
                    current_chunk = split
            else:
                # Add to current chunk with a space if needed
                if current_chunk and not current_chunk.endswith(" ") and not split.startswith(" "):
                    current_chunk += " "
                current_chunk += split
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
        
    async def _simple_chunk_text_async(self, text: str) -> List[str]:
        """Async wrapper for _simple_chunk_text"""
        return self._simple_chunk_text(text)
    
    def _simple_chunk_text(self, text: str) -> List[str]:
        """
        Simple and fast chunking method for large texts that would cause the regex to hang
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        
        # First try a simple paragraph split
        paragraphs = text.split('\n\n')
        
        # If we don't have many paragraphs, split by newline
        if len(paragraphs) < 3:
            paragraphs = text.split('\n')
            
        # If we still don't have many splits, just force chunk by size
        if len(paragraphs) < 3:
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
            
        current_chunk = ""
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # If adding this paragraph would exceed chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    
                # Start overlap with last part of previous chunk
                if self.chunk_overlap > 0 and len(current_chunk) > self.chunk_overlap:
                    # Find a good break point for overlap
                    overlap_start = len(current_chunk) - self.chunk_overlap
                    # Try to find a space to break at
                    space_pos = current_chunk.rfind(' ', overlap_start)
                    if space_pos != -1 and space_pos > overlap_start:
                        current_chunk = current_chunk[space_pos+1:]
                    else:
                        current_chunk = current_chunk[-self.chunk_overlap:]
                else:
                    current_chunk = ""
                    
                # If a single paragraph is larger than chunk size, split it
                if len(para) > self.chunk_size:
                    # For very large paragraphs, break them at sentence boundaries if possible
                    sentences = para.split('. ')
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) > self.chunk_size:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            temp_chunk = sentence
                        else:
                            if temp_chunk and not temp_chunk.endswith(' '):
                                temp_chunk += ' '
                            temp_chunk += sentence
                            if not sentence.endswith('.'):
                                temp_chunk += '.'
                    
                    if temp_chunk:
                        chunks.append(temp_chunk)
                        
                    # Start with empty chunk since we've handled this paragraph
                    current_chunk = ""
                else:
                    # Add paragraph to the new chunk
                    current_chunk = para
            else:
                # Add paragraph to current chunk
                if current_chunk and not current_chunk.endswith('\n'):
                    current_chunk += '\n\n'
                current_chunk += para
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
