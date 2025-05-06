import json
import logging
from typing import Dict, Any, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class JSONRepair:
    def __init__(self, openai_api_key: str):
        """
        Initialize JSON repair handler
        
        Args:
            openai_api_key: OpenAI API key
        """
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def repair_json(self, invalid_json: str, error_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Repair invalid JSON using OpenAI
        
        Args:
            invalid_json: The invalid JSON string
            error_context: Error context including location and message
            
        Returns:
            Repaired JSON as dictionary or None if repair failed
        """
        try:
            # Construct prompt for OpenAI
            prompt = self._construct_prompt(invalid_json, error_context)
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Using GPT-4 for better accuracy
                messages=[
                    {"role": "system", "content": "You are a JSON repair expert. Your task is to fix invalid JSON by identifying and correcting syntax errors, invalid control characters, and other issues."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Low temperature for more deterministic output
            )
            
            # Extract repaired JSON from response
            repaired_json_str = response.choices[0].message.content.strip()
            
            # Try to parse the repaired JSON
            try:
                repaired_json = json.loads(repaired_json_str)
                logger.info("Successfully repaired JSON")
                return repaired_json
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse repaired JSON: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error repairing JSON: {str(e)}")
            return None
            
    def _construct_prompt(self, invalid_json: str, error_context: Dict[str, Any]) -> str:
        """
        Construct prompt for OpenAI
        
        Args:
            invalid_json: The invalid JSON string
            error_context: Error context including location and message
            
        Returns:
            Formatted prompt string
        """
        error_type = error_context.get("type", "unknown")
        error_loc = error_context.get("loc", [])
        error_msg = error_context.get("msg", "unknown error")
        error_ctx = error_context.get("ctx", {})
        
        prompt = f"""
        Please repair the following invalid JSON. The error details are:
        - Error Type: {error_type}
        - Error Location: {error_loc}
        - Error Message: {error_msg}
        - Error Context: {error_ctx}
        
        Invalid JSON:
        {invalid_json}
        
        Please provide only the repaired JSON without any explanation or additional text.
        """
        
        return prompt
        
    def analyze_json_error(self, error: json.JSONDecodeError) -> Dict[str, Any]:
        """
        Analyze JSON decode error and extract relevant information
        
        Args:
            error: JSONDecodeError instance
            
        Returns:
            Dictionary containing error analysis
        """
        return {
            "type": "json_invalid",
            "loc": [error.pos],
            "msg": str(error),
            "ctx": {
                "error": error.__class__.__name__,
                "line": error.lineno,
                "column": error.colno
            }
        } 