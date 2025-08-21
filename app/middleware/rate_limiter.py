import time
import asyncio
from collections import defaultdict, deque
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from typing import Dict, Callable, Optional
import os

logger = logging.getLogger(__name__)

class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with client-specific limits based on authentication tokens
    """
    
    def __init__(self, app):
        super().__init__(app)
        
        # Default rate limits (requests per minute)
        self.default_limits = {
            "/question": int(os.getenv("RATE_LIMIT_QUESTION", "10")),      # 10 requests per minute
            "/question_full": int(os.getenv("RATE_LIMIT_QUESTION_FULL", "10")),  # 10 requests per minute  
            "/crawl": int(os.getenv("RATE_LIMIT_CRAWL", "5")),            # 5 requests per minute
        }
        
        # Client-specific rate limits (can be configured via environment variables)
        # Format: RATE_LIMIT_CLIENT_{CLIENT_ID}_{ENDPOINT}=limit
        self.client_limits = self._load_client_limits()
        
        # Storage for tracking requests: {client_id: {endpoint: deque of timestamps}}
        self.request_history: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        
        # Window size in seconds (60 seconds = 1 minute)
        self.window_size = 60
        
        # Cleanup interval (clean old entries every 5 minutes)
        self.cleanup_interval = 300
        self.last_cleanup = time.time()
        
        logger.info(f"Rate limiter initialized with default limits: {self.default_limits}")
        if self.client_limits:
            logger.info(f"Client-specific limits configured: {len(self.client_limits)} clients")
    
    def _load_client_limits(self) -> Dict[str, Dict[str, int]]:
        """Load client-specific rate limits from environment variables"""
        client_limits = {}
        
        # Look for environment variables in format: RATE_LIMIT_CLIENT_{CLIENT_ID}_{ENDPOINT}
        for key, value in os.environ.items():
            if key.startswith("RATE_LIMIT_CLIENT_"):
                try:
                    # Parse: RATE_LIMIT_CLIENT_client123_question_full -> client_id="client123", endpoint="question_full"
                    parts = key.replace("RATE_LIMIT_CLIENT_", "").split("_")
                    if len(parts) >= 2:
                        # Handle multi-part client IDs and endpoints
                        # Find the endpoint part (known endpoints)
                        known_endpoints = ["question", "question_full", "crawl"]
                        endpoint = None
                        client_parts = []
                        
                        for i, part in enumerate(parts):
                            if part in known_endpoints:
                                endpoint = part
                                client_parts = parts[:i]
                                # Handle question_full case
                                if i + 1 < len(parts) and parts[i + 1] == "full":
                                    endpoint = "question_full"
                                break
                            elif part == "full" and i > 0 and parts[i-1] == "question":
                                # Already handled above
                                continue
                        
                        if endpoint and client_parts:
                            client_id = "_".join(client_parts)
                            limit = int(value)
                            
                            if client_id not in client_limits:
                                client_limits[client_id] = {}
                            
                            client_limits[client_id][f"/{endpoint}"] = limit
                            logger.info(f"Loaded client-specific limit: {client_id} -> /{endpoint}: {limit}/min")
                            
                except (ValueError, IndexError) as e:
                    logger.warning(f"Invalid rate limit configuration: {key}={value}, error: {e}")
        
        return client_limits
    
    def _extract_client_id(self, request: Request) -> str:
        """Extract client ID from request (using auth token or IP as fallback)"""
        # Try to get client ID from Authorization header
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]  # Remove "Bearer " prefix
            # Use first 8 characters of token as client ID for privacy
            return f"token_{token[:8]}"
        
        # Fallback to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip_{client_ip}"
    
    def _get_endpoint_pattern(self, path: str) -> Optional[str]:
        """Get the rate limit pattern for a given path"""
        # Direct matches
        if path in self.default_limits:
            return path
        
        # Pattern matching for endpoints with parameters
        if path.startswith("/job-status/"):
            return "/job-status"  # Could add limits for status checks
        
        return None
    
    def _cleanup_old_entries(self):
        """Remove old entries from request history"""
        current_time = time.time()
        cutoff_time = current_time - self.window_size
        
        for client_id in list(self.request_history.keys()):
            for endpoint in list(self.request_history[client_id].keys()):
                # Remove timestamps older than window_size
                request_times = self.request_history[client_id][endpoint]
                while request_times and request_times[0] < cutoff_time:
                    request_times.popleft()
                
                # Remove empty endpoint entries
                if not request_times:
                    del self.request_history[client_id][endpoint]
            
            # Remove empty client entries
            if not self.request_history[client_id]:
                del self.request_history[client_id]
    
    def _is_rate_limited(self, client_id: str, endpoint: str) -> bool:
        """Check if client has exceeded rate limit for endpoint"""
        current_time = time.time()
        cutoff_time = current_time - self.window_size
        
        # Get rate limit for this client and endpoint
        limit = self.default_limits.get(endpoint, 0)
        
        # Check for client-specific limits
        if client_id in self.client_limits and endpoint in self.client_limits[client_id]:
            limit = self.client_limits[client_id][endpoint]
        
        if limit <= 0:
            return False  # No limit configured
        
        # Get request history for this client and endpoint
        request_times = self.request_history[client_id][endpoint]
        
        # Remove old entries
        while request_times and request_times[0] < cutoff_time:
            request_times.popleft()
        
        # Check if limit is exceeded
        return len(request_times) >= limit
    
    def _record_request(self, client_id: str, endpoint: str):
        """Record a new request for rate limiting"""
        current_time = time.time()
        self.request_history[client_id][endpoint].append(current_time)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            # Periodic cleanup
            current_time = time.time()
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup_old_entries()
                self.last_cleanup = current_time
            
            # Get endpoint pattern
            endpoint = self._get_endpoint_pattern(request.url.path)
            
            # Skip rate limiting for non-configured endpoints
            if not endpoint:
                return await call_next(request)
            
            # Extract client ID
            client_id = self._extract_client_id(request)
            
            # Check rate limit
            if self._is_rate_limited(client_id, endpoint):
                # Get the applicable limit for error message
                limit = self.default_limits.get(endpoint, 0)
                if client_id in self.client_limits and endpoint in self.client_limits[client_id]:
                    limit = self.client_limits[client_id][endpoint]
                
                logger.warning(f"Rate limit exceeded for client {client_id} on {endpoint}: {limit}/min")
                
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "message": f"You have exceeded the rate limit of {limit} requests per minute for {endpoint}",
                        "client_id": client_id,
                        "endpoint": endpoint,
                        "limit": limit,
                        "window_seconds": self.window_size,
                        "retry_after": 60  # seconds
                    },
                    headers={"Retry-After": "60"}
                )
            
            # Record the request
            self._record_request(client_id, endpoint)
            
            # Process the request
            response = await call_next(request)
            
            # Add rate limit headers to response
            limit = self.default_limits.get(endpoint, 0)
            if client_id in self.client_limits and endpoint in self.client_limits[client_id]:
                limit = self.client_limits[client_id][endpoint]
            
            remaining = max(0, limit - len(self.request_history[client_id][endpoint]))
            
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Window"] = str(self.window_size)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in rate limiter middleware: {str(e)}")
            # Don't block requests if rate limiter fails
            return await call_next(request) 