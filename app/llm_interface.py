import os
import logging
import time
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import openai

# Force reload environment variables
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self, deployment_name=None):
        """
        Initialize the LLM interface for generating answers using Azure OpenAI directly
        
        Args:
            deployment_name: Azure OpenAI deployment name (optional)
        """
        # Force reload environment variables
        load_dotenv(override=True)
        
        # Load Azure OpenAI configuration
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", deployment_name)

        logger.info(f"Azure OpenAI API version: {self.azure_api_version}")
        logger.info(f"Azure OpenAI deployment: {self.azure_deployment}")
        
        logger.info(f"Initializing Azure OpenAI with endpoint: {self.azure_endpoint}, " 
                   f"deployment: {self.azure_deployment}, API version: {self.azure_api_version}")
        
        if not self.azure_endpoint or not self.azure_api_key:
            logger.warning("Azure OpenAI credentials not found. LLM operations will fail.")
        
        # Create OpenAI client configured for Azure
        self.client = openai.OpenAI(
            api_key=self.azure_api_key,
            base_url=f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}",
            default_query={"api-version": self.azure_api_version},
        )
        
        # Default QA prompt template
        self.qa_prompt_template = """You are a helpful AI assistant. Use the following context to answer the question.
            
Context:
{context}

Question:
{question}

Answer the question based on the context provided. If the answer is not contained within the context, say "I don't have enough information to answer this question" and suggest related topics the user might want to ask about instead.
"""
    
    async def generate_answer(self, question: str, documents: List[Any]) -> Dict[str, Any]:
        """
        Generate an answer to a question using retrieved documents
        
        Args:
            question: User's question
            documents: Retrieved documents from vector store
            
        Returns:
            Answer with sources
        """
        if not documents:
            return {
                "answer": "I don't have enough information to answer this question.",
                "sources": []
            }
        
        # Extract text and sources from documents
        context_parts = []
        sources = []
        
        for i, doc in enumerate(documents):
            # Handle different document formats
            if isinstance(doc, dict):
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
            else:
                # Try to extract attributes directly if not a dict
                text = getattr(doc, "text", str(doc)) if hasattr(doc, "text") else str(doc)
                metadata = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
            
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            
            # Extract source info
            source = None
            title = f"Document {i+1}"
            
            if isinstance(metadata, dict):
                source = metadata.get("url")
                title = metadata.get("title", source or title)
            elif hasattr(metadata, "url"):
                source = metadata.url
                title = getattr(metadata, "title", source or title)
            
            # Add to context
            context_parts.append(f"Document {i+1} (Source: {title}):\n{text}\n")
            
            # Add unique sources
            if source and source not in sources:
                sources.append(source)
        
        context = "\n".join(context_parts)
        
        # Format the prompt
        prompt = self.qa_prompt_template.format(
            context=context,
            question=question
        )
        
        # Generate answer with Azure OpenAI API directly
        try:
            logger.info(f"Generating answer for question: {question}")
            
            # Log API request details
            logger.info(f"Using deployment: {self.azure_deployment}, API version: {self.azure_api_version}")
            
            # Make API call to Azure OpenAI
            response = self.client.chat.completions.create(
                model=self.azure_deployment,  # This is ignored for Azure but required
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=800  # Only parameter supported by the model
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            import traceback
            logger.error(f"Complete error: {traceback.format_exc()}")
            
            # Try to provide a more helpful error message
            error_msg = str(e)
            if "api version" in error_msg.lower():
                suggestion = "Try updating AZURE_OPENAI_API_VERSION in your .env file to a newer version like 2024-02-01 or 2024-12-01-preview."
                logger.error(f"API Version issue detected. {suggestion}")
                return {
                    "answer": f"I encountered an API version compatibility error. {suggestion}",
                    "sources": []
                }
            
            return {
                "answer": "I encountered an error while generating the answer.",
                "sources": []
            }
