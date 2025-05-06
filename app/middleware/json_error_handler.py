from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import json
import logging
from typing import Callable
from ..json_repair import JSONRepair

logger = logging.getLogger(__name__)

class JSONErrorHandlerMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, openai_api_key: str):
        super().__init__(app)
        self.json_repair = JSONRepair(openai_api_key)
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            # Try to read and parse the request body
            body = await request.body()
            if body:
                try:
                    json.loads(body)
                except json.JSONDecodeError as e:
                    # Analyze the error
                    error_context = self.json_repair.analyze_json_error(e)
                    
                    # Try to repair the JSON
                    repaired_json = self.json_repair.repair_json(body.decode(), error_context)
                    
                    if repaired_json is not None:
                        # Create a new request with repaired JSON
                        new_body = json.dumps(repaired_json).encode()
                        request._body = new_body
                        logger.info("Successfully repaired invalid JSON")
                    else:
                        # Return error response if repair failed
                        return JSONResponse(
                            status_code=400,
                            content={
                                "detail": [{
                                    "type": "json_invalid",
                                    "loc": error_context["loc"],
                                    "msg": error_context["msg"],
                                    "input": {},
                                    "ctx": error_context["ctx"]
                                }]
                            }
                        )
            
            # Process the request
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.error(f"Error in JSON error handler middleware: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={
                    "detail": [{
                        "type": "server_error",
                        "msg": "Internal server error",
                        "ctx": {"error": str(e)}
                    }]
                }
            ) 