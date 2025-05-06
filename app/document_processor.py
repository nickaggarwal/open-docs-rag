from typing import List, Dict, Any, Optional
import asyncio
import re
import logging
import time
from concurrent.futures import TimeoutError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 400, max_process_time: int = 120):
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
        Simplified chunking that preserves code blocks and important formatting
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        # First, check if we need to split at all
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try to identify code blocks (marked with triple backticks)
        code_block_pattern = r'```(?:\w+)?\n[\s\S]*?```'
        code_blocks = re.findall(code_block_pattern, text)
        code_block_placeholders = {}
        
        # Replace code blocks with placeholders
        for i, block in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            code_block_placeholders[placeholder] = block
            text = text.replace(block, placeholder)
        
        # Split by paragraphs first
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If paragraph contains a code block placeholder, treat it specially
            if any(placeholder in paragraph for placeholder in code_block_placeholders):
                # If adding this paragraph would make the chunk too large, save current chunk first
                if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # If paragraph itself is too large, it goes into its own chunk
                if len(paragraph) > self.chunk_size:
                    # If we have a current chunk, add it first
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    
                    # Add the large paragraph as its own chunk
                    chunks.append(paragraph)
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
            else:
                # Regular paragraph handling
                # If adding this paragraph would exceed chunk size
                if len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                    # If current chunk is not empty, add it to chunks
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # Start new chunk with this paragraph
                    if len(paragraph) > self.chunk_size:
                        # The paragraph itself is too big, split by sentences
                        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                        current_chunk = ""
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                                if current_chunk:
                                    chunks.append(current_chunk)
                                    current_chunk = sentence
                                else:
                                    # Even a single sentence is too big
                                    chunks.append(sentence[:self.chunk_size])
                                    remainder = sentence[self.chunk_size:]
                                    while remainder:
                                        chunks.append(remainder[:self.chunk_size])
                                        remainder = remainder[self.chunk_size:]
                            else:
                                if current_chunk:
                                    current_chunk += " " + sentence
                                else:
                                    current_chunk = sentence
                    else:
                        current_chunk = paragraph
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Restore code blocks
        final_chunks = []
        for chunk in chunks:
            for placeholder, code_block in code_block_placeholders.items():
                chunk = chunk.replace(placeholder, code_block)
            final_chunks.append(chunk)
        
        # Ensure we have some overlap between chunks
        if len(final_chunks) > 1 and self.chunk_overlap > 0:
            overlapped_chunks = []
            for i, chunk in enumerate(final_chunks):
                if i > 0:
                    # Get overlap from previous chunk
                    prev_chunk = final_chunks[i-1]
                    if len(prev_chunk) > self.chunk_overlap:
                        overlap = prev_chunk[-self.chunk_overlap:]
                        # Add overlap at the beginning of current chunk
                        chunk = overlap + "\n\n" + chunk
                overlapped_chunks.append(chunk)
            final_chunks = overlapped_chunks
            
        return final_chunks
