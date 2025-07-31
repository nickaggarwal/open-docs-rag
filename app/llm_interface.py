import os
import logging
import time
import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from dotenv import load_dotenv
import openai

# Force reload environment variables
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMInterface:
    def __init__(self, deployment_name=None, model_name=None):
        """
        Initialize the LLM interface for generating answers using OpenAI APIs
        
        Args:
            deployment_name: Azure OpenAI deployment name (optional)
            model_name: OpenAI model name for direct OpenAI API (optional)
        """
        # Force reload environment variables
        load_dotenv(override=True)
        
        # Determine whether to use Azure OpenAI or direct OpenAI API
        self.use_azure = os.getenv("USE_AZURE_OPENAI", "true").lower() == "true"
        
        if self.use_azure:
            logger.info("Using Azure OpenAI API")
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
            
            # Create async OpenAI client configured for Azure
            self.client = openai.AsyncOpenAI(
                api_key=self.azure_api_key,
                base_url=f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}",
                default_query={"api-version": self.azure_api_version},
            )
            # Store the model/deployment name
            self.model = self.azure_deployment
        else:
            logger.info("Using direct OpenAI API")
            # Load OpenAI configuration
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            self.openai_model = os.getenv("OPENAI_MODEL", model_name or "gpt-3.5-turbo")
            
            logger.info(f"OpenAI model: {self.openai_model}")
            
            if not self.openai_api_key:
                logger.warning("OpenAI API key not found. LLM operations will fail.")
            
            # Create async OpenAI client
            self.client = openai.AsyncOpenAI(
                api_key=self.openai_api_key,
            )
            # Store the model name
            self.model = self.openai_model
        
        # Default QA prompt template
        self.qa_prompt_template = """You are a helpful AI assistant. Use the following context to answer the question.
            
Context:
{context}

Question:
{question}

Answer the question based on the context provided. If the answer is not contained within the context, say "I don't have enough information to answer this question" and suggest related topics the user might want to ask about instead.
"""
    
    async def generate_answer_stream(self, question: str, documents: List[Any]) -> AsyncGenerator[str, None]:
        """
        Generate a streaming answer to a question using retrieved documents
        
        Args:
            question: User's question
            documents: Retrieved documents from vector store
            
        Yields:
            Streaming SSE formatted chunks
        """
        if not documents:
            no_info_msg = "I don't have enough information to answer this question."
            yield f"data: {json.dumps({'type': 'answer', 'content': no_info_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'content': []})}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        # Extract text and sources from documents (same logic as generate_answer)
        context_parts = []
        sources = []
        
        for i, doc in enumerate(documents):
            # Handle different document formats
            if isinstance(doc, dict):
                text = doc.get("text", "")
                metadata = doc.get("metadata", {})
                # If we have a score from our weighted search, log it
                score = doc.get("score", None)
                if score is not None:
                    logger.info(f"Document {i+1} score: {score}")
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
                
                # Include heading information if available
                heading = metadata.get("heading")
                if heading:
                    title = f"{title} - {heading}"
            
            # Truncate text for context
            if len(text) > 500:
                text = text[:500] + "..."
            
            context_parts.append(f"Source {i+1} ({title}): {text}")
            
            # Add unique sources
            if source and source not in sources:
                sources.append(source)
        
        context = "\n".join(context_parts)
        
        # Format the prompt (same as generate_answer)
        prompt = self.qa_prompt_template.format(
            context=context,
            question=question
        )
        
        # Send sources first
        yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"
        
        # Generate streaming answer with OpenAI API
        try:
            logger.info(f"Generating streaming answer for question: {question}")
            
            # Prepare request payload
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # Create parameters dict with shared parameters
            completion_params = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": 2500,
                "stream": True  # Enable streaming
            }
            
            # Detect if we're using an Azure endpoint by checking the base_url
            is_azure_endpoint = False
            if hasattr(self.client, 'base_url'):
                base_url_str = str(getattr(self.client, 'base_url', ''))
                is_azure_endpoint = 'azure' in base_url_str.lower()
            
            if is_azure_endpoint or self.use_azure:
                # Log Azure-specific details
                logger.info(f"Using Azure deployment for streaming: {self.azure_deployment}, API version: {self.azure_api_version}")
            else:
                # Log OpenAI-specific details
                logger.info(f"Using OpenAI model for streaming: {self.openai_model}")
            
            # Make streaming API call to appropriate OpenAI service
            stream = await self.client.chat.completions.create(**completion_params)
            
            async for chunk in stream:
                # Check if chunk has choices before accessing
                if not chunk.choices:
                    continue
                    
                choice = chunk.choices[0]
                
                # Check if there's content to stream
                if hasattr(choice, 'delta') and choice.delta.content is not None:
                    content = choice.delta.content
                    yield f"data: {json.dumps({'type': 'answer_chunk', 'content': content})}\n\n"
                
                # Check if streaming is complete
                if hasattr(choice, 'finish_reason') and choice.finish_reason is not None:
                    break
            
            # Send completion signal
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error generating streaming answer: {str(e)}")
            import traceback
            logger.error(f"Complete error: {traceback.format_exc()}")
            
            # Try to provide a more helpful error message based on API type
            error_msg = str(e)
            api_type = "Azure OpenAI" if self.use_azure else "OpenAI"
            
            if self.use_azure and "api version" in error_msg.lower():
                suggestion = "Try updating AZURE_OPENAI_API_VERSION in your .env file to a newer version like 2024-02-01 or 2024-12-01-preview."
                logger.error(f"API Version issue detected. {suggestion}")
                error_response = f"I encountered an API version compatibility error with {api_type}. {suggestion}"
            else:
                error_response = f"I encountered an error while generating the answer with {api_type}."
            
            yield f"data: {json.dumps({'type': 'error', 'content': error_response})}\n\n"
            yield "data: [DONE]\n\n"

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
                # If we have a score from our weighted search, log it
                score = doc.get("score", None)
                if score is not None:
                    logger.info(f"Document {i+1} score: {score}")
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
                
                # Include heading information if available
                heading = metadata.get("heading")
                if heading:
                    title = f"{title} - {heading}"
            elif hasattr(metadata, "url"):
                source = metadata.url
                title = getattr(metadata, "title", source or title)
            
            # Add to context - include metadata that might help LLM understand importance
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
        
        # Generate answer with OpenAI API
        try:
            logger.info(f"Generating answer for question: {question}")
            
            # Prepare request payload
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # Create parameters dict with shared parameters
            completion_params = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": 2500  # Use consistently for both Azure and OpenAI
            }
            
            # Detect if we're using an Azure endpoint by checking the base_url
            is_azure_endpoint = False
            if hasattr(self.client, 'base_url'):
                base_url_str = str(getattr(self.client, 'base_url', ''))
                is_azure_endpoint = 'azure' in base_url_str.lower()
            
            if is_azure_endpoint or self.use_azure:
                # Log Azure-specific details
                logger.info(f"Using Azure deployment: {self.azure_deployment}, API version: {self.azure_api_version}")
                logger.info(f"Request payload to Azure OpenAI: messages={messages}, max_completion_tokens=2500")
            else:
                # Log OpenAI-specific details
                logger.info(f"Using OpenAI model: {self.openai_model}")
                logger.info(f"Request payload to OpenAI: messages={messages}, max_completion_tokens=2500")
            
            # Make API call to appropriate OpenAI service with the right parameters
            response = await self.client.chat.completions.create(**completion_params)
            
            # Log API response
            api_type = "Azure OpenAI" if is_azure_endpoint or self.use_azure else "OpenAI"
            logger.info(f"{api_type} response: id={response.id}, model={response.model}, finish_reason={response.choices[0].finish_reason}, usage={response.usage}")
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            import traceback
            logger.error(f"Complete error: {traceback.format_exc()}")
            
            # Try to provide a more helpful error message based on API type
            error_msg = str(e)
            api_type = "Azure OpenAI" if self.use_azure else "OpenAI"
            
            if self.use_azure and "api version" in error_msg.lower():
                suggestion = "Try updating AZURE_OPENAI_API_VERSION in your .env file to a newer version like 2024-02-01 or 2024-12-01-preview."
                logger.error(f"API Version issue detected. {suggestion}")
                return {
                    "answer": f"I encountered an API version compatibility error with {api_type}. {suggestion}",
                    "sources": []
                }
            
            return {
                "answer": f"I encountered an error while generating the answer with {api_type}.",
                "sources": []
            }
