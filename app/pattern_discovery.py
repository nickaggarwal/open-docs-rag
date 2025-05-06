import logging
from typing import List
from .document_processor import DocumentProcessor
from .crawler import Crawler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def discover_patterns(start_urls: List[str], max_pages: int = 2) -> DocumentProcessor:
    """
    Discover patterns from initial pages
    
    Args:
        start_urls: List of URLs to start crawling from
        max_pages: Maximum number of pages to analyze
        
    Returns:
        DocumentProcessor with discovered patterns
    """
    # Initialize crawler and processor
    crawler = Crawler()
    processor = DocumentProcessor()
    
    # Crawl initial pages
    logger.info(f"Starting pattern discovery from {len(start_urls)} URLs")
    documents = []
    
    for url in start_urls:
        try:
            # Crawl the page
            result = crawler.crawl_page(url)
            if result and result.get("text"):
                documents.append(result["text"])
                logger.info(f"Successfully crawled {url}")
                
            if len(documents) >= max_pages:
                break
                
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")
            
    # Analyze documents to discover patterns
    if documents:
        logger.info(f"Analyzing {len(documents)} documents for patterns")
        processor.analyze_initial_documents(documents)
        logger.info("Pattern discovery complete")
    else:
        logger.warning("No documents found for pattern discovery")
        
    return processor

if __name__ == "__main__":
    # Example usage
    start_urls = [
        "https://docs.inferless.com/getting-started/introduction",
        "https://docs.inferless.com/concepts/overview"
    ]
    
    processor = discover_patterns(start_urls)
    
    # Print discovered patterns
    patterns = processor.pattern_manager.get_patterns()
    for category, pattern_list in patterns.items():
        print(f"\n{category}:")
        for pattern in pattern_list:
            print(f"  - {pattern}") 