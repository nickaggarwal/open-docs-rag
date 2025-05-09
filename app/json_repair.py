import json
import logging
import os
from typing import Dict, Any, Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class JSONRepair:
    def __init__(self, openai_api_key: str = None):
        """
        Initialize JSON repair handler with support for both Azure OpenAI and direct OpenAI APIs
        
        Args:
            openai_api_key: OpenAI API key (optional, will use environment variables if not provided)
        """
        # Force reload environment variables
        load_dotenv(override=True)
        
        # Determine whether to use Azure OpenAI or direct OpenAI API
        self.use_azure = os.getenv("USE_AZURE_OPENAI", "true").lower() == "true"
        
        if self.use_azure:
            logger.info("Using Azure OpenAI API for JSON repair")
            # Load Azure OpenAI configuration
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
            self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
            
            # Create OpenAI client configured for Azure
            self.client = openai.OpenAI(
                api_key=self.azure_api_key,
                base_url=f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}",
                default_query={"api-version": self.azure_api_version},
            )
            # Store the model/deployment name
            self.model = self.azure_deployment
        else:
            logger.info("Using direct OpenAI API for JSON repair")
            # Load or use provided OpenAI API key
            self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            
            if not self.openai_api_key:
                logger.warning("OpenAI API key not found. JSON repair will fail.")
            
            # Create standard OpenAI client
            self.client = openai.OpenAI(
                api_key=self.openai_api_key,
            )
            # Store the model name
            self.model = self.openai_model
        
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
            
            # Prepare messages for ChatCompletion
            messages = [
                {"role": "system", "content": "You are a JSON repair expert. Your task is to fix invalid JSON by identifying and correcting syntax errors, invalid control characters, and other issues."},
                {"role": "user", "content": prompt}
            ]
            
            # Log which API we're using
            api_type = "Azure OpenAI" if self.use_azure else "OpenAI"
            logger.info(f"Calling {api_type} API for JSON repair")
            
            # Create parameters dict with shared parameters
            completion_params = {
                "model": self.model,
                "messages": messages,
            }
            
            # Add API-specific parameters
            if self.use_azure:
                # Add Azure-specific parameter
                completion_params["max_completion_tokens"] = 2000
                # Azure doesn't support custom temperature for this model - don't set it
            else:
                # Add OpenAI-specific parameter
                completion_params["max_tokens"] = 2000
                completion_params["temperature"] = 0.1  # Low temperature for more deterministic output
            
            # Call OpenAI API using the client with the right parameters
            response = self.client.chat.completions.create(**completion_params)
            
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