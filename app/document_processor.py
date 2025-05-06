from typing import List, Dict, Any, Optional, Tuple
import asyncio
import re
import logging
import time
from concurrent.futures import TimeoutError
import tiktoken
from bs4 import BeautifulSoup
import markdown
import html2text
from .pattern_manager import PatternManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, 
                 chunk_size: int = 600,  # Default to smaller chunks for better precision
                 chunk_overlap: int = 50,  # Smaller overlap for more distinct chunks
                 max_process_time: int = 120,
                 coarse_chunk_size: int = 1000,  # Size for top-level heading chunks
                 fine_chunk_size: int = 400):    # Size for detailed chunks within sections
        """
        Initialize the document processor with configuration for hierarchical chunking
        
        Args:
            chunk_size: Default size for text chunks
            chunk_overlap: Overlap between chunks
            max_process_time: Maximum time in seconds to spend processing a single document
            coarse_chunk_size: Size for top-level heading chunks
            fine_chunk_size: Size for detailed chunks within sections
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_process_time = max_process_time
        self.max_chunk_time = 5
        self.coarse_chunk_size = coarse_chunk_size
        self.fine_chunk_size = fine_chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's tokenizer
        self.pattern_manager = PatternManager()
        self.initial_analysis_done = False
        
        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, "
                   f"chunk_overlap={chunk_overlap}, coarse_chunk_size={coarse_chunk_size}, "
                   f"fine_chunk_size={fine_chunk_size}")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))

    def analyze_initial_documents(self, documents: List[str]) -> None:
        """
        Analyze initial documents to discover common patterns
        
        Args:
            documents: List of document texts to analyze
        """
        for doc in documents:
            self.pattern_manager.analyze_document(doc)
        self.pattern_manager.update_patterns()
        self.initial_analysis_done = True
        logger.info("Initial document analysis complete")

    def _preprocess_document(self, text: str) -> str:
        """
        Preprocess document text to remove boilerplate and navigation content
        
        Args:
            text: Raw document text
            
        Returns:
            Preprocessed text
        """
        if not self.initial_analysis_done:
            # If initial analysis not done, use basic patterns
            text = self._basic_cleanup(text)
        else:
            # Use discovered patterns
            patterns = self.pattern_manager.get_patterns()
            
            # First pass: Remove headers and navigation
            for pattern in patterns["header_patterns"]:
                text = re.sub(pattern, '', text, flags=re.MULTILINE)
                
            # Second pass: Remove common elements
            for pattern in patterns["element_patterns"]:
                text = re.sub(pattern, '', text, flags=re.MULTILINE)
                
            # Third pass: Remove cleanup patterns
            for pattern in patterns["cleanup_patterns"]:
                text = re.sub(pattern, '', text, flags=re.MULTILINE)
                
            # Final cleanup
            text = self._final_cleanup(text)
            
        return text
        
    def _basic_cleanup(self, text: str) -> str:
        """
        Basic cleanup using default patterns
        
        Args:
            text: Raw document text
            
        Returns:
            Cleaned text
        """
        # First pass: Remove headers and navigation
        text = re.sub(r'Source:.*?\)\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'^.*?Inferless\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'Search\.\.\.\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'Deploy now\nDeploy now\n', '', text, flags=re.MULTILINE)
        
        # Second pass: Remove common elements
        text = re.sub(r'(?:Tutorials|Changelog|Blog)\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'(?:Cli import|Handling Input / Output|Bring custom packages)\n', '', text, flags=re.MULTILINE)
        
        # Third pass: Remove cleanup patterns
        text = re.sub(r'^.*?(?:Search\.\.\.|Deploy now|Tutorials|Changelog|Blog|Hugging face|Git \(Custom Code\)|Docker|AWS PrivateLink|Model Endpoint|Debugging your Model|File Structure Requirements|Input / Output Schema|My Volumes|My Secrets)\n', '', text, flags=re.MULTILINE)
        
        return self._final_cleanup(text)
        
    def _final_cleanup(self, text: str) -> str:
        """
        Final cleanup steps
        
        Args:
            text: Partially cleaned text
            
        Returns:
            Fully cleaned text
        """
        # Remove multiple newlines and spaces
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove any remaining HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove any remaining markdown elements
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'#{1,6}\s*', '', text)
        
        return text.strip()

    def _extract_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract document structure with headings and content
        
        Args:
            text: Preprocessed document text
            
        Returns:
            List of sections with headings and content
        """
        sections = []
        current_section = {"heading": "", "level": 0, "content": []}
        
        for line in text.split('\n'):
            # Check for headings
            heading_match = re.match(r'^(#{1,4})\s+(.+)$', line)
            if heading_match:
                # Save previous section if it has content
                if current_section["content"]:
                    sections.append(current_section)
                
                # Start new section
                level = len(heading_match.group(1))
                heading = heading_match.group(2)
                current_section = {
                    "heading": heading,
                    "level": level,
                    "content": []
                }
            else:
                current_section["content"].append(line)
        
        # Add last section
        if current_section["content"]:
            sections.append(current_section)
            
        return sections

    def _create_hierarchical_chunks(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create hierarchical chunks from document sections
        
        Args:
            sections: List of document sections with headings and content
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        for section in sections:
            # Create coarse chunk for section
            section_text = '\n'.join(section["content"])
            section_tokens = self._count_tokens(section_text)
            
            if section_tokens <= self.coarse_chunk_size:
                # Section fits in one chunk
                chunks.append({
                    "text": section_text,
                    "metadata": {
                        "heading": section["heading"],
                        "level": section["level"],
                        "chunk_type": "coarse"
                    }
                })
            else:
                # Split section into fine chunks
                fine_chunks = self._create_fine_chunks(section_text)
                for i, chunk in enumerate(fine_chunks):
                    chunks.append({
                        "text": chunk,
                        "metadata": {
                            "heading": section["heading"],
                            "level": section["level"],
                            "chunk_type": "fine",
                            "chunk_index": i
                        }
                    })
        
        return chunks

    def _create_fine_chunks(self, text: str) -> List[str]:
        """
        Create fine-grained chunks from text
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if self._count_tokens(current_chunk + paragraph) > self.fine_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    async def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of documents with improved chunking
        
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
                doc_start_time = time.time()
                
                try:
                    result = await asyncio.wait_for(
                        self._process_document(doc),
                        timeout=self.max_process_time
                    )
                    processed_documents.extend(result)
                    
                    doc_time = time.time() - doc_start_time
                    logger.info(f"Processed document {i+1}/{len(documents)} in {doc_time:.2f}s - "
                               f"Created {len(result)} chunks")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Document {i+1}/{len(documents)} processing timed out. "
                                  f"Using emergency chunking.")
                    result = await self._emergency_chunking(doc)
                    processed_documents.extend(result)
                    
            except Exception as e:
                logger.error(f"Error processing document {i+1}: {str(e)}")
                continue
                
        total_time = time.time() - start_time
        logger.info(f"Processed {len(documents)} documents into {len(processed_documents)} chunks "
                   f"in {total_time:.2f}s")
        
        return processed_documents

    async def _process_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single document with improved chunking
        
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
            
        try:
            # Preprocess document
            preprocessed_text = self._preprocess_document(text)
            
            # Extract structure
            sections = self._extract_structure(preprocessed_text)
            
            # Create hierarchical chunks
            chunks = self._create_hierarchical_chunks(sections)
            
            # Add metadata to chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update(chunk["metadata"])
                chunk_metadata["chunk_id"] = i
                
                processed_chunks.append({
                    "text": chunk["text"],
                    "metadata": chunk_metadata
                })
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}")
            return await self._emergency_chunking(document)

    async def _emergency_chunking(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Emergency fallback chunking method
        
        Args:
            document: Document with text and metadata
            
        Returns:
            List of document chunks with text and metadata
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        
        # Simple paragraph-based chunking
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if self._count_tokens(current_chunk + paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add metadata to chunks
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["emergency_chunking"] = True
            
            processed_chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return processed_chunks
