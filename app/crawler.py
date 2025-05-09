import asyncio
import logging
import time
from typing import List, Dict, Any, Set, Optional, Tuple
import httpx
from bs4 import BeautifulSoup
import urllib.parse
from .crawl_memory import CrawlMemory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebCrawler:
    def __init__(self, base_url: str, max_pages: int = 100, concurrency: int = 5, 
                 timeout: int = 30, overall_timeout: int = 300, 
                 crawl_memory: Optional[CrawlMemory] = None,
                 incremental: bool = True):
        """
        Initialize a web crawler for documentation sites
        
        Args:
            base_url: Base URL of the documentation site
            max_pages: Maximum number of pages to crawl
            concurrency: Maximum number of concurrent requests
            timeout: Timeout for individual HTTP requests in seconds
            overall_timeout: Overall timeout for the entire crawl operation
            crawl_memory: Optional CrawlMemory instance for incremental crawling
            incremental: Whether to enable incremental crawling
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.timeout = timeout
        self.overall_timeout = overall_timeout
        self.visited_urls: Set[str] = set()
        self.queue: List[str] = []
        self.semaphore = asyncio.Semaphore(concurrency)
        self.crawl_memory = crawl_memory or CrawlMemory()
        self.incremental = incremental
        self.modified_urls: Set[str] = set()  # Track which URLs have been modified
        self.unchanged_urls: Set[str] = set()  # Track which URLs are unchanged
        
    def normalize_url(self, url: str) -> str:
        """Normalize URL to avoid duplicates"""
        parsed = urllib.parse.urlparse(url)
        # Remove fragments
        return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, 
                                      parsed.params, parsed.query, ""))
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled"""
        parsed_base = urllib.parse.urlparse(self.base_url)
        parsed_url = urllib.parse.urlparse(url)
        
        # Check if URL is within the same domain
        if parsed_url.netloc != parsed_base.netloc:
            return False
        
        # Skip common file types that are not useful for documentation
        skip_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.svg', '.css', '.js', '.ico', '.woff', '.woff2', '.ttf']
        if any(parsed_url.path.endswith(ext) for ext in skip_extensions):
            return False
            
        return True
    
    async def fetch_page(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and parse a single page"""
        async with self.semaphore:
            try:
                # Use a timeout to prevent hanging on slow responses
                async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                    logger.info(f"Fetching {url}")
                    response = await client.get(url)
                    if response.status_code != 200:
                        logger.warning(f"Failed to fetch {url}: {response.status_code}")
                        return None
                    
                    # Check content type before parsing
                    content_type = response.headers.get('content-type', '')
                    if not content_type.startswith('text/html'):
                        logger.warning(f"Skipping non-HTML content at {url}: {content_type}")
                        return None
                    
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()
                    
                    # Extract main content areas
                    main_content = soup.find_all(['main', 'article', 'div', 'section'], class_=['content', 'documentation', 'main', 'article', 'docs-content'])
                    content_html = ""
                    
                    # If specific content areas found, use them
                    if main_content:
                        for content in main_content:
                            content_html += str(content)
                        content_soup = BeautifulSoup(content_html, 'html.parser')
                        text = content_soup.get_text(separator='\n')
                    else:
                        # If no specific content areas found, use the whole page
                        text = soup.get_text(separator='\n')
                    
                    # Preserve code blocks
                    code_blocks = []
                    for code in soup.find_all(['pre', 'code']):
                        code_text = code.get_text()
                        if len(code_text.strip()) > 0:
                            code_blocks.append(f"\n```\n{code_text}\n```\n")
                    
                    # Extract title
                    title = soup.title.string if soup.title else url
                    
                    # Extract all links
                    links = []
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            parsed_base = urllib.parse.urlparse(url)
                            href = urllib.parse.urlunparse((parsed_base.scheme, parsed_base.netloc, href, '', '', ''))
                        elif not href.startswith(('http://', 'https://')):
                            # Handle other relative URLs (like '../path' or 'page.html')
                            href = urllib.parse.urljoin(url, href)
                        
                        # Normalize and validate URL
                        href = self.normalize_url(href)
                        if self.is_valid_url(href):
                            links.append(href)
                    
                    # Check for content change if incremental crawling is enabled
                    content_changed = True
                    if self.incremental:
                        content_changed = self.crawl_memory.has_content_changed(url, text)
                        if content_changed:
                            # Update memory with new content
                            self.crawl_memory.update_memory(url, text)
                            self.modified_urls.add(url)
                            logger.info(f"Content changed for {url}")
                        else:
                            self.unchanged_urls.add(url)
                            logger.info(f"Content unchanged for {url}")
                    
                    return {
                        "url": url,
                        "title": title,
                        "text": text,
                        "links": links,
                        "content_changed": content_changed,
                        "code_blocks": code_blocks
                    }
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)}")
                return None
    
    async def crawl(self) -> List[Dict[str, Any]]:
        """
        Crawl the website and extract text content
        
        Returns:
            List of documents with text content and metadata
        """
        self.queue = [self.base_url]
        self.visited_urls = set()
        documents = []
        modified_documents = []  # Track documents that have changed
        
        # Set an overall timeout for the crawl process
        start_time = time.time()
        stalled_count = 0  # Track consecutive empty results
        
        while self.queue and len(self.visited_urls) < self.max_pages:
            # Check if overall timeout has been reached
            if time.time() - start_time > self.overall_timeout:
                logger.warning(f"Crawl operation timed out after {self.overall_timeout} seconds")
                break
                
            # Limit queue size to prevent memory issues
            if len(self.queue) > self.max_pages * 5:
                logger.warning(f"Queue size limit reached ({len(self.queue)} URLs). Truncating.")
                self.queue = self.queue[:self.max_pages * 5]
            
            # Process multiple URLs concurrently
            batch_size = min(self.concurrency, len(self.queue))
            batch_urls = self.queue[:batch_size]
            self.queue = self.queue[batch_size:]
            
            # Mark as visited
            for url in batch_urls:
                self.visited_urls.add(url)
            
            # Fetch pages concurrently
            tasks = [self.fetch_page(url) for url in batch_urls]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=False)
                
                # Check if we're getting results or just errors
                valid_results = [r for r in results if r is not None]
                if not valid_results:
                    stalled_count += 1
                    if stalled_count >= 3:  # If we get 3 consecutive batches with no results
                        logger.warning("Crawl stalled - 3 consecutive batches with no valid results")
                        break
                else:
                    stalled_count = 0  # Reset stalled count if we got valid results
                
                for result in valid_results:
                    # For incremental crawling, only add documents that have changed
                    if not self.incremental or result.get("content_changed", True):
                        document = {
                            "text": result["text"],
                            "metadata": {
                                "url": result["url"],
                                "title": result["title"],
                                "source": "web_crawler"
                            }
                        }
                        documents.append(document)
                        modified_documents.append(document)
                    else:
                        # Still add to all documents but track that this one is unchanged
                        document = {
                            "text": result["text"],
                            "metadata": {
                                "url": result["url"],
                                "title": result["title"],
                                "source": "web_crawler"
                            }
                        }
                        documents.append(document)
                    
                    # Add new URLs to queue with priority for shorter paths
                    # (typically better for documentation sites)
                    for url in sorted(result["links"], 
                                      key=lambda u: len(urllib.parse.urlparse(u).path.split('/'))):
                        if url not in self.visited_urls and url not in self.queue:
                            self.queue.append(url)
            
            except Exception as e:
                logger.error(f"Error during batch crawl: {str(e)}")
                # Continue with the next batch rather than failing the entire crawl
            
            logger.info(f"Crawled {len(self.visited_urls)} pages, queue size: {len(self.queue)}, " +
                       f"documents: {len(documents)}, modified: {len(modified_documents)}")
        
        # Save the updated crawl memory
        if self.incremental:
            self.crawl_memory.save_memory()
            logger.info(f"Crawl completed: {len(documents)} total documents, " +
                       f"{len(modified_documents)} modified documents, " +
                       f"{len(self.modified_urls)} modified URLs, " +
                       f"{len(self.unchanged_urls)} unchanged URLs")
        
        if not documents:
            logger.warning("Crawl completed but no documents were found!")
        
        # If incremental, return only modified documents
        if self.incremental:
            return modified_documents
        
        return documents

async def crawl_website(url: str, max_pages: int = 100, incremental: bool = True) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Utility function to crawl a website
    
    Args:
        url: URL to start crawling from
        max_pages: Maximum number of pages to crawl
        incremental: Whether to use incremental crawling (skip unchanged content)
        
    Returns:
        Tuple of (documents, has_changes)
        - documents: List of documents with text and metadata
        - has_changes: Boolean indicating if any content has changed since last crawl
    """
    # Initialize crawler with crawl memory for incremental crawling
    crawl_memory = CrawlMemory()
    crawler = WebCrawler(url, max_pages=max_pages, crawl_memory=crawl_memory, incremental=incremental)
    
    logger.info(f"Starting crawl of {url} with max_pages={max_pages}, incremental={incremental}")
    try:
        documents = await crawler.crawl()
        has_changes = len(documents) > 0 if incremental else True
        
        return documents, has_changes
    except Exception as e:
        logger.error(f"Unexpected error during crawl: {str(e)}")
        return [], False  # Return empty list and no changes
