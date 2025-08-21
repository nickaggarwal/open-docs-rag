import os
import logging
import time
import asyncio
import json
import re
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
        
        # Enhanced QA prompt template with better structure
        self.qa_prompt_template = """You are a helpful AI documentation assistant. Use the provided context to answer the user's question comprehensively and accurately.

CONTEXT INFORMATION:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based on the context provided, prioritizing information from higher-relevance sources
- If multiple sources contain relevant information, synthesize them into a coherent response
- Include specific details, code examples, and step-by-step instructions when available
- If the context doesn't contain enough information, clearly state what's missing and suggest related topics
- Reference source sections when helpful for the user to find more details

ANSWER:"""
    
    def _calculate_semantic_relevance(self, question: str, document: Dict[str, Any]) -> float:
        """
        Calculate semantic relevance score between question and document
        
        Args:
            question: User's question
            document: Document to score
            
        Returns:
            Semantic relevance score (0.0 to 1.0)
        """
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        
        # Start with base embedding score if available
        base_score = document.get("score", 0.5)
        
        # Extract question keywords (remove common words)
        question_lower = question.lower()
        # Include important technical terms that might be short
        important_short_terms = {'io', 'in', 'out', 'api', 'app', 'py', 'js', 'ts', 'id'}
        question_keywords = []
        for word in re.findall(r'\b\w+\b', question_lower):
            if (len(word) > 2 or word in important_short_terms) and word not in {'the', 'and', 'are', 'how', 'what', 'where', 'when', 'why', 'with', 'for', 'can', 'you', 'use', 'get', 'any', 'all', 'but', 'not', 'has', 'had', 'has', 'was', 'were', 'been', 'have', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'than', 'then', 'this', 'that', 'they', 'them', 'their', 'there', 'here', 'from', 'into', 'onto', 'upon', 'over', 'under', 'also', 'just', 'only', 'even', 'more', 'most', 'much', 'many', 'some', 'such', 'very', 'well', 'make', 'made', 'take', 'took', 'come', 'came', 'give', 'gave', 'find', 'found', 'know', 'knew', 'tell', 'told', 'keep', 'kept', 'left', 'right', 'good', 'best', 'better', 'same', 'different', 'new', 'old', 'first', 'last', 'long', 'short', 'high', 'low', 'big', 'small', 'large', 'great'}:
                question_keywords.append(word)
        
        # Calculate keyword matching score
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in question_keywords if keyword in text_lower)
        keyword_score = min(0.3, keyword_matches * 0.05)
        
        # Boost for exact phrase matches
        phrase_boost = 0.0
        if len(question_keywords) > 1:
            # Check for multi-word phrases
            question_phrases = []
            words = question_lower.split()
            for i in range(len(words) - 1):
                if len(words[i]) > 2 and len(words[i + 1]) > 2:
                    phrase = f"{words[i]} {words[i + 1]}"
                    question_phrases.append(phrase)
            
            phrase_matches = sum(1 for phrase in question_phrases if phrase in text_lower)
            phrase_boost = min(0.2, phrase_matches * 0.1)
        
        # Context-specific boosts
        context_boost = 0.0
        
        # Schema-specific boost - very important for input/output questions
        schema_keywords = ['input', 'output', 'schema', 'accept', 'return', 'parameter', 'response', 'request']
        schema_matches = sum(1 for keyword in schema_keywords if keyword in question_lower and keyword in text_lower)
        if schema_matches > 0:
            context_boost += min(0.4, schema_matches * 0.15)  # High boost for schema relevance
        
        # Technical documentation patterns
        if any(tech_word in question_lower for tech_word in ['api', 'function', 'method', 'class', 'import', 'install', 'configure', 'app.py']):
            if any(tech_pattern in text_lower for tech_pattern in ['def ', 'class ', 'import ', 'pip install', 'npm install', '```', 'app.py', 'input_schema', 'output_schema']):
                context_boost += 0.15
        
        # Code example boost
        if 'example' in question_lower or 'how to' in question_lower:
            if '```' in text or 'def ' in text or 'function ' in text or 'import ' in text:
                context_boost += 0.1
        
        # URL relevance boost - prioritize concept docs over how-to guides for theoretical questions
        url = metadata.get("url", "")
        if 'concepts/' in url and any(concept_word in question_lower for concept_word in ['schema', 'configure', 'input', 'output']):
            context_boost += 0.25
        
        # Heading relevance boost
        heading = metadata.get("heading", "")
        if heading:
            heading_lower = heading.lower()
            heading_keyword_matches = sum(1 for keyword in question_keywords if keyword in heading_lower)
            if heading_keyword_matches > 0:
                context_boost += min(0.2, heading_keyword_matches * 0.1)
        
        # Calculate final score
        final_score = base_score + keyword_score + phrase_boost + context_boost
        return min(1.0, final_score)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using rough estimation"""
        return len(text) // 4  # Rough token estimation
    
    def _classify_query_intent(self, question: str) -> str:
        """
        Classify the user's query intent to prioritize document types
        
        Args:
            question: User's question
            
        Returns:
            Query intent classification
        """
        question_lower = question.lower()
        
        # API/Reference queries
        if any(word in question_lower for word in ['api', 'endpoint', 'parameter', 'response', 'schema', 'reference']):
            return "api_query"
        
        # Implementation/How-to queries
        if any(word in question_lower for word in ['how to', 'deploy', 'setup', 'configure', 'implement', 'example', 'tutorial']):
            return "implementation_query"
        
        # Conceptual queries
        if any(word in question_lower for word in ['what is', 'what are', 'explain', 'concept', 'overview', 'understand']):
            return "concept_query"
        
        # Default to implementation
        return "implementation_query"

    def _calculate_priority_score(self, doc: Dict[str, Any], query_intent: str) -> float:
        """
        Calculate priority score based on document type and query intent
        
        Args:
            doc: Document with metadata
            query_intent: Classified query intent
            
        Returns:
            Priority score (higher is better)
        """
        metadata = doc.get("metadata", {})
        doc_type = metadata.get("document_type", "unknown")
        priority_level = metadata.get("priority_level", 3)
        
        # Base priority (lower number = higher priority)
        base_score = (6 - priority_level) / 5.0  # Convert to 0.2-1.0 scale
        
        # Intent-based weighting - heavily favor concept documents
        intent_weights = {
            "concept_query": {
                "concept": 1.0, "api": 0.7, "quickstart": 0.8, 
                "tutorial": 0.5, "cookbook": 0.4
            },
            "implementation_query": {
                "concept": 0.95, "tutorial": 0.8, "cookbook": 0.7,  # Concept documents get very high weight
                "api": 0.6, "quickstart": 0.6
            },
            "api_query": {
                "concept": 1.0, "api": 0.8, "tutorial": 0.5,  # Concept documents get highest weight for API queries
                "cookbook": 0.4, "quickstart": 0.6
            }
        }
        
        type_weight = intent_weights.get(query_intent, {}).get(doc_type, 0.5)
        
        # Special boost for concept documents regardless of query type
        if doc_type == "concept":
            type_weight = min(1.0, type_weight * 1.3)  # 30% additional boost for all concept docs
        
        # Extra boost for specific schema/configuration concept documents
        if doc.get("artificial_concept_boost", False):
            type_weight = min(1.0, type_weight * 1.5)  # 50% additional boost for schema concepts
        
        # Additional boosts
        chunk_type = metadata.get("chunk_type", "unknown")
        chunk_boost = {
            "document_summary": 1.0,  # Increased from 0.8
            "section": 1.0,
            "subsection": 0.9,
            "code_example": 0.7 if query_intent == "implementation_query" else 0.5,
            "fine": 0.8,  # Good for detailed information
            "coarse": 0.9  # Good for high-level concepts
        }.get(chunk_type, 0.5)
        
        # Has code boost for implementation queries
        code_boost = 1.0
        if query_intent == "implementation_query" and metadata.get("has_code", False):
            code_boost = 1.2
        
        final_score = base_score * type_weight * chunk_boost * code_boost
        return min(final_score, 1.0)

    def _build_enhanced_context(self, question: str, documents: List[Any]) -> tuple[str, List[str]]:
        """
        Build enhanced context with priority-based document selection and better formatting
        
        Args:
            question: User's question
            documents: Retrieved documents
            
        Returns:
            Tuple of (formatted_context, sources)
        """
        if not documents:
            return "", []
        
        # Classify query intent
        query_intent = self._classify_query_intent(question)
        
        # Calculate combined scores (semantic + priority) with higher weight for priority
        enhanced_docs = []
        for doc in documents:
            semantic_score = self._calculate_semantic_relevance(question, doc)
            priority_score = self._calculate_priority_score(doc, query_intent)
            
            # Give more weight to priority score to favor concept documents (was 70/30, now 60/40)
            combined_score = (semantic_score * 0.6) + (priority_score * 0.4)
            
            doc_copy = doc.copy()
            doc_copy["enhanced_score"] = combined_score
            doc_copy["semantic_score"] = semantic_score
            doc_copy["priority_score"] = priority_score
            enhanced_docs.append(doc_copy)
        
        # Sort by combined score
        enhanced_docs.sort(key=lambda x: x["enhanced_score"], reverse=True)
        
        # Build context with more chunks and better organization
        context_sections = []
        sources = []
        
        # Group by document type and source for better organization
        type_groups = {}
        for doc in enhanced_docs[:20]:  # Consider top 20 instead of limiting to 4 sections
            metadata = doc.get("metadata", {})
            doc_type = metadata.get("document_type", "unknown")
            
            # If document type is unknown, try to classify it based on URL
            if doc_type == "unknown":
                url = metadata.get("url", "")
                if url:
                    # Quick classification based on URL patterns
                    if 'concepts' in url:
                        doc_type = "concept"
                        # Special boost for configuration/schema concepts
                        if 'configuring' in url or 'schema' in url or 'input-output' in url:
                            # Artificially boost the priority score for this specific type of concept
                            doc["artificial_concept_boost"] = True
                    elif 'how-to-guides' in url or 'tutorial' in url:
                        doc_type = "tutorial"
                    elif 'api-reference' in url or 'api' in url:
                        doc_type = "api"
                    elif 'quickstart' in url or 'getting-started' in url:
                        doc_type = "quickstart"
                    elif 'cookbook' in url or 'examples' in url:
                        doc_type = "cookbook"
                    else:
                        doc_type = "tutorial"  # Default fallback for unknown types
            
            logger.info(f"Document type: {doc_type}, URL: {metadata.get('url', 'N/A')}, enhanced_score: {doc.get('enhanced_score', 0):.3f}")
            
            if doc_type not in type_groups:
                type_groups[doc_type] = []
            type_groups[doc_type].append(doc)
        
        logger.info(f"Type groups: {list(type_groups.keys())}")
        
        # Prioritize document types based on query intent
        type_priority = {
            "concept_query": ["concept", "quickstart", "api", "tutorial", "cookbook"],
            "implementation_query": ["concept", "tutorial", "cookbook", "quickstart", "api"],  # Concepts first for implementation too
            "api_query": ["concept", "api", "tutorial", "quickstart", "cookbook"]  # Concepts first for API queries
        }.get(query_intent, ["concept", "tutorial", "api", "cookbook", "quickstart"])  # Default prioritizes concepts
        
        # Add any remaining types that weren't in the priority list
        for doc_type in type_groups.keys():
            if doc_type not in type_priority:
                type_priority.append(doc_type)
        
        logger.info(f"Query intent: {query_intent}, Type priority: {type_priority}")
        
        # Build context sections by type priority with enhanced weighting for concepts
        total_tokens = 0
        max_tokens = 15000  # Use more of the available context window
        section_num = 1
        
        for doc_type in type_priority:
            if doc_type not in type_groups:
                logger.info(f"No documents found for type: {doc_type}")
                continue
                
            logger.info(f"Processing {len(type_groups[doc_type])} documents of type: {doc_type}")
            
            # Give concept documents more slots and higher token allowance
            if doc_type == "concept":
                type_docs = type_groups[doc_type][:8]  # More concept docs (was 5)
                doc_token_limit = 3000  # Higher token limit per concept doc
            else:
                type_docs = type_groups[doc_type][:4]  # Fewer slots for other types (was 5)
                doc_token_limit = 2000  # Standard token limit
            
            for doc in type_docs:
                if total_tokens >= max_tokens:
                    break
                    
                metadata = doc.get("metadata", {})
                text = doc.get("text", "")
                
                # Estimate tokens for this document
                doc_tokens = self._count_tokens(text) if hasattr(self, '_count_tokens') else len(text) // 4
                
                # Allow concept documents to use more tokens
                if doc_type == "concept":
                    doc_tokens = min(doc_tokens, doc_token_limit)
                elif total_tokens + doc_tokens > max_tokens:
                    continue
                
                # Create section header with rich metadata
                source_url = metadata.get("url", "unknown")
                title = metadata.get("title", "Unknown Document")
                heading = metadata.get("heading", "")
                chunk_type = metadata.get("chunk_type", "")
                
                section_title = f"Section {section_num}: {title}"
                if heading:
                    section_title += f" - {heading}"
                
                context_sections.append(f"=== {section_title} ===")
                context_sections.append(f"Type: {doc_type.title()} ({chunk_type})")
                context_sections.append(f"Source: {source_url}")
                context_sections.append(f"Relevance: {doc['enhanced_score']:.2f} (Semantic: {doc['semantic_score']:.2f}, Priority: {doc['priority_score']:.2f})")
                
                # Add language info for code examples
                if chunk_type == "code_example":
                    language = metadata.get("language", "unknown")
                    context_sections.append(f"Language: {language}")
                
                context_sections.append("")
                
                # Add document content
                cleaned_text = text.strip()
                if cleaned_text:
                    context_sections.append(cleaned_text)
                    context_sections.append("")
                
                # Add source to list
                if source_url not in sources:
                    sources.append(source_url)
                
                total_tokens += doc_tokens
                section_num += 1
        
        context = "\n".join(context_sections)
        logger.info(f"Built context with {section_num-1} sections, ~{total_tokens} tokens, query intent: {query_intent}")
        
        return context, sources
    
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
        
        # Use enhanced context building
        context, sources = self._build_enhanced_context(question, documents)
        
        # Log document scores for debugging
        for i, doc in enumerate(documents):
            score = doc.get("score", None)
            if score is not None:
                logger.info(f"Document {i+1} original score: {score}")
                
        logger.info(f"Built enhanced context with {len(sources)} sources")
        
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
        
        # Use enhanced context building
        context, sources = self._build_enhanced_context(question, documents)
        
        # Log document scores for debugging
        for i, doc in enumerate(documents):
            score = doc.get("score", None)
            if score is not None:
                logger.info(f"Document {i+1} original score: {score}")
                
        logger.info(f"Built enhanced context with {len(sources)} sources")
        
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
