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
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, 
                 chunk_size: int = 600,  # Default to smaller chunks for better precision
                 chunk_overlap: int = 50,  # Smaller overlap for more distinct chunks
                 max_process_time: int = 120,
                 coarse_chunk_size: int = 1000,  # Size for top-level heading chunks
                 fine_chunk_size: int = 400,    # Size for detailed chunks within sections
                 concept_chunk_size: int = 2000):  # Larger chunks for concept documents
        """
        Initialize the document processor with configuration for hierarchical chunking
        
        Args:
            chunk_size: Default size for text chunks
            chunk_overlap: Overlap between chunks
            max_process_time: Maximum time in seconds to spend processing a single document
            coarse_chunk_size: Size for top-level heading chunks
            fine_chunk_size: Size for detailed chunks within sections
            concept_chunk_size: Size for concept document chunks (larger to preserve context)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_process_time = max_process_time
        self.max_chunk_time = 5
        self.coarse_chunk_size = coarse_chunk_size
        self.fine_chunk_size = fine_chunk_size
        self.concept_chunk_size = concept_chunk_size  # New parameter for concept documents
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's tokenizer
        self.pattern_manager = PatternManager()
        self.initial_analysis_done = False
        
        logger.info(f"DocumentProcessor initialized with chunk_size={chunk_size}, "
                   f"chunk_overlap={chunk_overlap}, coarse_chunk_size={coarse_chunk_size}, "
                   f"fine_chunk_size={fine_chunk_size}, concept_chunk_size={concept_chunk_size}")

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

    def _create_hierarchical_chunks(self, sections: List[Dict[str, Any]], recommended_chunk_size: int, doc_classification: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create hierarchical chunks from document sections
        
        Args:
            sections: List of document sections with headings and content
            recommended_chunk_size: The recommended chunk size for the document
            doc_classification: The document classification metadata
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        for section in sections:
            # Create coarse chunk for section
            section_text = '\n'.join(section["content"])
            section_tokens = self._count_tokens(section_text)
            
            if section_tokens <= recommended_chunk_size:
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
                fine_chunks = self._create_fine_chunks(section_text, recommended_chunk_size)
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

    def _create_fine_chunks(self, text: str, chunk_size: int = None) -> List[str]:
        """
        Create fine-grained chunks from text
        
        Args:
            text: Text to split into chunks
            chunk_size: Maximum size for chunks (defaults to fine_chunk_size)
            
        Returns:
            List of text chunks
        """
        if chunk_size is None:
            chunk_size = self.fine_chunk_size
            
        # Split by paragraphs first
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size
            if self._count_tokens(current_chunk + paragraph) > chunk_size:
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
            # Classify the document to determine optimal chunking strategy
            url = metadata.get("url", "")
            doc_classification = self.classify_document_type(url, text)
            
            # Get recommended chunk size from classification
            recommended_chunk_size = doc_classification.get("recommended_chunk_size", self.fine_chunk_size)
            doc_type = doc_classification.get("document_type", "unknown")
            
            logger.info(f"Processing {doc_type} document with recommended chunk size: {recommended_chunk_size}")
            
            # Preprocess document
            preprocessed_text = self._preprocess_document(text)
            
            # Extract structure
            sections = self._extract_structure(preprocessed_text)
            
            # Create hierarchical chunks with document-specific parameters
            chunks = self._create_hierarchical_chunks(sections, recommended_chunk_size, doc_classification)
            
            # Add metadata to chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update(chunk["metadata"])
                chunk_metadata.update(doc_classification)  # Add classification info to chunk metadata
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

    def classify_document_type(self, url: str, content: str) -> Dict[str, Any]:
        """
        Classify document type and priority based on URL structure and content
        
        Args:
            url: Document URL
            content: Document content
            
        Returns:
            Classification metadata
        """
        parsed_url = urllib.parse.urlparse(url)
        path_parts = [p for p in parsed_url.path.split('/') if p]
        
        # Document type classification
        doc_type = "unknown"
        priority = 3  # Default medium priority
        complexity = "intermediate"
        
        if 'concepts' in path_parts:
            doc_type = "concept"
            priority = 1 if 'overview' in url or 'configuring' in url else 2
            complexity = "beginner" if 'overview' in url else "intermediate"
        elif 'how-to-guides' in path_parts or 'tutorial' in path_parts:
            doc_type = "tutorial"
            priority = 2
            complexity = "intermediate"
        elif 'api-reference' in path_parts or 'api' in path_parts:
            doc_type = "api"
            priority = 1  # High priority for API queries
            complexity = "advanced"
        elif 'quickstart' in path_parts or 'getting-started' in path_parts:
            doc_type = "quickstart"
            priority = 1  # High priority for beginners
            complexity = "beginner"
        elif 'cookbook' in path_parts or 'examples' in path_parts:
            doc_type = "cookbook"
            priority = 2
            complexity = "intermediate"
        
        # Determine recommended chunk size based on document type
        if doc_type == "concept":
            recommended_chunk_size = self.concept_chunk_size  # Use larger chunks for concepts
            # Special case for schema/configuration concepts - use even larger chunks
            if 'configuring' in url or 'schema' in url or 'input-output' in url:
                recommended_chunk_size = min(3000, self.concept_chunk_size * 1.5)
        else:
            recommended_chunk_size = self.fine_chunk_size  # Use standard fine chunks for others
        
        # Content analysis
        has_code = bool(re.search(r'```|`[^`]+`', content))
        has_config = bool(re.search(r'config|setup|install|deploy', content.lower()))
        
        # Topic hierarchy extraction
        topic_hierarchy = []
        if path_parts:
            topic_hierarchy = [part.replace('-', ' ') for part in path_parts[1:]]  # Skip domain
        
        return {
            "document_type": doc_type,
            "priority_level": priority,
            "complexity": complexity,
            "recommended_chunk_size": recommended_chunk_size,
            "has_code": has_code,
            "has_config": has_config,
            "topic_hierarchy": topic_hierarchy
        }

    def parse_markdown_structure(self, content: str) -> Dict[str, Any]:
        """
        Parse markdown structure to extract headings and sections
        
        Args:
            content: Markdown content
            
        Returns:
            Structured representation of the document
        """
        lines = content.split('\n')
        structure = {
            "title": "",
            "sections": [],
            "code_blocks": [],
            "has_toc": False
        }
        
        current_section = None
        current_code_block = None
        in_code_block = False
        
        for i, line in enumerate(lines):
            # Extract title (first H1)
            if line.startswith('# ') and not structure["title"]:
                structure["title"] = line[2:].strip()
                continue
            
            # Extract headings
            heading_match = re.match(r'^(#+)\s+(.+)$', line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                section = {
                    "level": level,
                    "title": title,
                    "start_line": i,
                    "content": "",
                    "subsections": [],
                    "has_code": False
                }
                
                # Close previous section
                if current_section:
                    current_section["end_line"] = i - 1
                    structure["sections"].append(current_section)
                
                current_section = section
                continue
            
            # Track code blocks
            if line.strip().startswith('```'):
                if not in_code_block:
                    # Starting code block
                    language = line.strip()[3:].strip()
                    current_code_block = {
                        "language": language,
                        "start_line": i,
                        "content": ""
                    }
                    in_code_block = True
                    if current_section:
                        current_section["has_code"] = True
                else:
                    # Ending code block
                    if current_code_block:
                        current_code_block["end_line"] = i
                        structure["code_blocks"].append(current_code_block)
                    in_code_block = False
                    current_code_block = None
                continue
            
            # Add content to current section
            if current_section:
                current_section["content"] += line + '\n'
            
            # Add content to current code block
            if in_code_block and current_code_block:
                current_code_block["content"] += line + '\n'
        
        # Close final section
        if current_section:
            current_section["end_line"] = len(lines) - 1
            structure["sections"].append(current_section)
        
        return structure

    def create_enhanced_chunks(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create enhanced chunks for MD documents with better context
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            List of enhanced chunks
        """
        url = metadata.get("url", "")
        
        # Classify document
        doc_classification = self.classify_document_type(url, content)
        
        # Parse structure
        md_structure = self.parse_markdown_structure(content)
        
        chunks = []
        
        # 1. Document Summary Chunk
        summary_content = f"Document: {md_structure.get('title', 'Unknown')}\n"
        if md_structure.get('sections'):
            summary_content += "Sections:\n"
            for section in md_structure['sections'][:5]:  # First 5 sections
                summary_content += f"- {section['title']}\n"
        
        summary_tokens = self._count_tokens(summary_content)
        if summary_tokens <= 300:
            chunks.append({
                "text": summary_content,
                "metadata": {
                    **metadata,
                    **doc_classification,
                    "chunk_type": "document_summary",
                    "priority": doc_classification["priority_level"],
                    "md_structure": md_structure
                }
            })
        
        # 2. Section-Based Chunks
        for section in md_structure.get('sections', []):
            section_content = section.get('content', '').strip()
            if not section_content:
                continue
            
            section_tokens = self._count_tokens(section_content)
            
            # Create section header with context
            section_header = f"# {section['title']}\n"
            full_section_content = section_header + section_content
            
            if section_tokens <= self.coarse_chunk_size:
                # Section fits in one chunk
                chunks.append({
                    "text": full_section_content,
                    "metadata": {
                        **metadata,
                        **doc_classification,
                        "chunk_type": "section",
                        "heading": section['title'],
                        "level": section['level'],
                        "has_code": section.get('has_code', False),
                        "priority": doc_classification["priority_level"]
                    }
                })
            else:
                # Split into subsection chunks
                subsection_chunks = self._create_fine_chunks(section_content)
                for i, chunk in enumerate(subsection_chunks):
                    chunk_with_header = f"## {section['title']} (Part {i+1})\n{chunk}"
                    chunks.append({
                        "text": chunk_with_header,
                        "metadata": {
                            **metadata,
                            **doc_classification,
                            "chunk_type": "subsection",
                            "heading": section['title'],
                            "level": section['level'],
                            "chunk_index": i,
                            "has_code": section.get('has_code', False),
                            "priority": doc_classification["priority_level"]
                        }
                    })
        
        # 3. Dedicated Code Example Chunks
        for code_block in md_structure.get('code_blocks', []):
            if not code_block.get('content', '').strip():
                continue
            
            # Get surrounding context
            lines = content.split('\n')
            start = max(0, code_block['start_line'] - 3)
            end = min(len(lines), code_block['end_line'] + 3)
            
            context_lines = lines[start:code_block['start_line']]
            code_lines = lines[code_block['start_line']:code_block['end_line'] + 1]
            after_lines = lines[code_block['end_line'] + 1:end]
            
            code_with_context = '\n'.join(context_lines + code_lines + after_lines)
            
            if self._count_tokens(code_with_context) <= 800:
                chunks.append({
                    "text": code_with_context,
                    "metadata": {
                        **metadata,
                        **doc_classification,
                        "chunk_type": "code_example",
                        "language": code_block.get('language', 'unknown'),
                        "priority": doc_classification["priority_level"] + 0.5  # Slightly lower than section
                    }
                })
        
        return chunks
