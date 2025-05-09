import os
import json
import hashlib
import logging
from typing import Dict, Set, List, Any, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrawlMemory:
    """
    Maintains a memory of previously crawled pages to enable intelligent
    incremental crawling based on content changes.
    """
    
    def __init__(self, memory_file_path: str = "./data/crawl_memory.json"):
        """
        Initialize crawl memory
        
        Args:
            memory_file_path: Path to store memory data
        """
        self.memory_file_path = memory_file_path
        self.url_hashes: Dict[str, str] = {}  # URL -> content hash
        self.url_timestamps: Dict[str, float] = {}  # URL -> timestamp
        self.load_memory()
        
    def load_memory(self) -> None:
        """Load crawl memory from disk if it exists"""
        try:
            if os.path.exists(self.memory_file_path):
                with open(self.memory_file_path, 'r') as f:
                    data = json.load(f)
                    self.url_hashes = data.get('url_hashes', {})
                    self.url_timestamps = data.get('url_timestamps', {})
                logger.info(f"Loaded crawl memory with {len(self.url_hashes)} URLs")
            else:
                logger.info("No existing crawl memory found, starting fresh")
                # Ensure the directory exists
                os.makedirs(os.path.dirname(self.memory_file_path), exist_ok=True)
        except Exception as e:
            logger.error(f"Error loading crawl memory: {str(e)}")
            self.url_hashes = {}
            self.url_timestamps = {}
    
    def save_memory(self) -> None:
        """Save crawl memory to disk"""
        try:
            with open(self.memory_file_path, 'w') as f:
                json.dump({
                    'url_hashes': self.url_hashes,
                    'url_timestamps': self.url_timestamps
                }, f)
            logger.info(f"Saved crawl memory with {len(self.url_hashes)} URLs")
        except Exception as e:
            logger.error(f"Error saving crawl memory: {str(e)}")
    
    def compute_content_hash(self, text: str) -> str:
        """
        Compute a hash of the content to detect changes
        
        Args:
            text: Document text content
            
        Returns:
            Content hash as string
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def has_content_changed(self, url: str, content: str) -> bool:
        """
        Check if content has changed since last crawl
        
        Args:
            url: URL of the page
            content: Current text content
            
        Returns:
            True if content has changed or page is new, False otherwise
        """
        content_hash = self.compute_content_hash(content)
        
        if url not in self.url_hashes:
            # New page
            return True
        
        # Check if content hash has changed
        return self.url_hashes[url] != content_hash
    
    def update_memory(self, url: str, content: str) -> None:
        """
        Update memory with new content
        
        Args:
            url: URL of the page
            content: Text content
        """
        content_hash = self.compute_content_hash(content)
        self.url_hashes[url] = content_hash
        self.url_timestamps[url] = time.time()
    
    def get_modified_urls(self, urls: List[str], contents: List[str]) -> Dict[str, bool]:
        """
        Check which URLs have modified content
        
        Args:
            urls: List of URLs to check
            contents: List of content corresponding to URLs
            
        Returns:
            Dictionary mapping URL to boolean (True if modified, False if unchanged)
        """
        result = {}
        for url, content in zip(urls, contents):
            result[url] = self.has_content_changed(url, content)
        return result
    
    def get_known_urls(self) -> Set[str]:
        """
        Get set of all known URLs
        
        Returns:
            Set of URLs that have been crawled before
        """
        return set(self.url_hashes.keys())
    
    def get_last_crawl_time(self, url: str) -> Optional[float]:
        """
        Get the timestamp when URL was last crawled
        
        Args:
            url: URL to check
            
        Returns:
            Timestamp or None if URL not in memory
        """
        return self.url_timestamps.get(url)
    
    def is_recently_crawled(self, url: str, max_age_seconds: int = 86400) -> bool:
        """
        Check if URL was crawled recently
        
        Args:
            url: URL to check
            max_age_seconds: Maximum age to consider recent (default: 24 hours)
            
        Returns:
            True if URL was crawled within the time window
        """
        last_crawl = self.get_last_crawl_time(url)
        if last_crawl is None:
            return False
            
        return time.time() - last_crawl < max_age_seconds
