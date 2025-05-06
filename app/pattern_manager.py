from typing import List, Set, Dict
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class PatternManager:
    def __init__(self):
        self.header_patterns: Set[str] = set()
        self.element_patterns: Set[str] = set()
        self.cleanup_patterns: Set[str] = set()
        self.pattern_counter: Counter = Counter()
        self.min_pattern_frequency = 2  # Minimum times a pattern must appear to be considered common
        
    def analyze_document(self, text: str) -> None:
        """
        Analyze a document to discover common patterns
        
        Args:
            text: Document text to analyze
        """
        # Find potential header patterns
        header_matches = re.finditer(r'^.*?(?:Inferless|Documentation|Navigation|Getting Started|Concepts|Integrations|API Reference|Model Import)\n', 
                                   text, re.MULTILINE | re.IGNORECASE)
        for match in header_matches:
            pattern = re.escape(match.group(0))
            self.pattern_counter[pattern] += 1
            
        # Find potential element patterns
        element_matches = re.finditer(r'^(?:Tutorials|Changelog|Blog|Introduction|Overview|Deploy|Cli|Handling|Working|Automatic|Dynamic|Hugging|Git|Docker|Cloud|AWS|Model|Version|File|Input|Output|My)\n',
                                    text, re.MULTILINE | re.IGNORECASE)
        for match in element_matches:
            pattern = re.escape(match.group(0))
            self.pattern_counter[pattern] += 1
            
        # Find potential cleanup patterns
        cleanup_matches = re.finditer(r'^.*?(?:Search\.\.\.|Deploy now|Tutorials|Changelog|Blog|Hugging face|Git \(Custom Code\)|Docker|AWS PrivateLink|Model Endpoint|Debugging your Model|File Structure Requirements|Input / Output Schema|My Volumes|My Secrets)\n',
                                    text, re.MULTILINE | re.IGNORECASE)
        for match in cleanup_matches:
            pattern = re.escape(match.group(0))
            self.pattern_counter[pattern] += 1
            
    def update_patterns(self) -> None:
        """
        Update the pattern sets based on frequency analysis
        """
        # Update header patterns
        for pattern, count in self.pattern_counter.items():
            if count >= self.min_pattern_frequency:
                if re.search(r'(?:Inferless|Documentation|Navigation|Getting Started|Concepts|Integrations|API Reference|Model Import)', pattern, re.IGNORECASE):
                    self.header_patterns.add(pattern)
                elif re.search(r'(?:Tutorials|Changelog|Blog|Introduction|Overview|Deploy|Cli|Handling|Working|Automatic|Dynamic|Hugging|Git|Docker|Cloud|AWS|Model|Version|File|Input|Output|My)', pattern, re.IGNORECASE):
                    self.element_patterns.add(pattern)
                else:
                    self.cleanup_patterns.add(pattern)
                    
    def get_patterns(self) -> Dict[str, List[str]]:
        """
        Get all discovered patterns
        
        Returns:
            Dictionary of pattern lists by category
        """
        return {
            "header_patterns": list(self.header_patterns),
            "element_patterns": list(self.element_patterns),
            "cleanup_patterns": list(self.cleanup_patterns)
        }
        
    def clear_patterns(self) -> None:
        """
        Clear all discovered patterns
        """
        self.header_patterns.clear()
        self.element_patterns.clear()
        self.cleanup_patterns.clear()
        self.pattern_counter.clear() 