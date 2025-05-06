import asyncio
import logging
import time
from typing import List, Dict, Any, Set, Optional
import httpx
from bs4 import BeautifulSoup
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebCrawler:
    def __init__(self, base_url: str, max_pages: int = 100, concurrency: int = 5, 
                 timeout: int = 30, overall_timeout: int = 300):
        """
        Initialize a web crawler for documentation sites
        
        Args:
            base_url: Base URL of the documentation site
            max_pages: Maximum number of pages to crawl
            concurrency: Maximum number of concurrent requests
            timeout: Timeout for individual HTTP requests in seconds
            overall_timeout: Overall timeout for the entire crawl operation
        """
        self.base_url = base_url
        self.max_pages = max_pages
        self.concurrency = concurrency
        self.timeout = timeout
        self.overall_timeout = overall_timeout
        self.visited_urls: Set[str] = set()
        self.queue: List[str] = []
        self.semaphore = asyncio.Semaphore(concurrency)
        
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
                    
                    # Append code blocks to the text
                    if code_blocks:
                        text += "\n\nCode Examples:\n" + "\n".join(code_blocks)
                    
                    # Normalize whitespace while preserving line breaks
                    text = '\n'.join([' '.join(line.split()) for line in text.split('\n') if line.strip()])
                    
                    # Skip pages with too little content
                    if len(text) < 50:
                        logger.warning(f"Skipping page with insufficient content: {url}")
                        return None
                    
                    # Extract title
                    title = soup.title.string if soup.title else url
                    
                    # Find new links - limit to a reasonable number to prevent queue explosion
                    new_urls = []
                    link_count = 0
                    for link in soup.find_all('a', href=True):
                        if link_count >= 100:  # Limit links per page
                            break
                            
                        href = link['href']
                        # Handle relative URLs
                        full_url = urllib.parse.urljoin(url, href)
                        normalized_url = self.normalize_url(full_url)
                        
                        if (normalized_url not in self.visited_urls and 
                            self.is_valid_url(normalized_url) and
                            normalized_url not in new_urls):  # Avoid duplicates
                            new_urls.append(normalized_url)
                            link_count += 1
                    
                    return {
                        "url": url,
                        "title": title,
                        "text": text,
                        "links": new_urls
                    }
            except httpx.TimeoutException:
                logger.warning(f"Timeout fetching {url}")
                return None
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)}")
                return None
    
    async def crawl(self) -> List[Dict[str, Any]]:
        """
        Crawl the documentation site
        
        Returns:
            List of documents with text content and metadata
        """
        self.queue = [self.base_url]
        self.visited_urls = set()
        documents = []
        
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
                    documents.append({
                        "text": result["text"],
                        "metadata": {
                            "url": result["url"],
                            "title": result["title"],
                            "source": "web_crawler"
                        }
                    })
                    
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
                       f"documents: {len(documents)}")
        
        if not documents:
            logger.warning("Crawl completed but no documents were found!")
            
        return documents

async def crawl_website(url: str, max_pages: int = 100) -> List[Dict[str, Any]]:
    """
    Utility function to crawl a website
    
    Args:
        url: URL to start crawling from
        max_pages: Maximum number of pages to crawl
        
    Returns:
        List of documents with text and metadata
    """
    crawler = WebCrawler(url, max_pages=max_pages)
    logger.info(f"Starting crawl of {url} with max_pages={max_pages}")
    try:
        return await crawler.crawl()
    except Exception as e:
        logger.error(f"Unexpected error during crawl: {str(e)}")
        return []  # Return empty list instead of crashing
