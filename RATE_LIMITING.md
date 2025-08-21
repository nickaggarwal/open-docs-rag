# Rate Limiting Configuration

The RAG API includes client-specific rate limiting to control usage and prevent abuse.

## Default Rate Limits

By default, the following rate limits are applied (requests per minute):

- `/question`: 10 requests/minute
- `/question_full`: 10 requests/minute  
- `/crawl`: 5 requests/minute

## Environment Variables

### Default Limits
You can configure default rate limits using environment variables:

```bash
# Default rate limits (requests per minute)
RATE_LIMIT_QUESTION=10          # /question endpoint
RATE_LIMIT_QUESTION_FULL=10     # /question_full endpoint  
RATE_LIMIT_CRAWL=5              # /crawl endpoint
```

### Client-Specific Limits
Configure different limits for specific clients using their authentication tokens:

```bash
# Format: RATE_LIMIT_CLIENT_{CLIENT_ID}_{ENDPOINT}=limit
# Examples:
RATE_LIMIT_CLIENT_premium_user_question=50
RATE_LIMIT_CLIENT_premium_user_question_full=50
RATE_LIMIT_CLIENT_premium_user_crawl=20

RATE_LIMIT_CLIENT_basic_user_question=5
RATE_LIMIT_CLIENT_basic_user_question_full=5
RATE_LIMIT_CLIENT_basic_user_crawl=2
```

## How Client Identification Works

1. **Authentication Token**: If a request includes a `Bearer` token in the `Authorization` header, the first 8 characters of the token are used as the client ID (e.g., `token_asds.dsa`)

2. **IP Address Fallback**: If no token is provided, the client's IP address is used (e.g., `ip_192.168.1.1`)

## Rate Limit Headers

All responses include rate limit information in headers:

- `X-RateLimit-Limit`: The rate limit ceiling for the endpoint
- `X-RateLimit-Remaining`: Number of requests remaining in the current window
- `X-RateLimit-Window`: The time window in seconds (60 seconds)

## Rate Limit Exceeded Response

When rate limits are exceeded, the API returns a `429 Too Many Requests` response:

```json
{
    "detail": "Rate limit exceeded",
    "message": "You have exceeded the rate limit of 10 requests per minute for /question",
    "client_id": "token_asds.dsa",
    "endpoint": "/question",
    "limit": 10,
    "window_seconds": 60,
    "retry_after": 60
}
```

## Configuration Examples

### Basic Setup
```bash
# Set default limits for all clients
RATE_LIMIT_QUESTION=15
RATE_LIMIT_QUESTION_FULL=15
RATE_LIMIT_CRAWL=3
```

### Premium vs Basic Users
```bash
# Default limits (basic users)
RATE_LIMIT_QUESTION=5
RATE_LIMIT_QUESTION_FULL=5
RATE_LIMIT_CRAWL=1

# Premium user with token starting with "premium_"
RATE_LIMIT_CLIENT_premium_question=100
RATE_LIMIT_CLIENT_premium_question_full=100
RATE_LIMIT_CLIENT_premium_crawl=50

# Enterprise user with token starting with "enterprise_"
RATE_LIMIT_CLIENT_enterprise_question=1000
RATE_LIMIT_CLIENT_enterprise_question_full=1000
RATE_LIMIT_CLIENT_enterprise_crawl=200
```

### Development vs Production
```bash
# Development environment - more relaxed limits
RATE_LIMIT_QUESTION=100
RATE_LIMIT_QUESTION_FULL=100
RATE_LIMIT_CRAWL=20

# Production environment - stricter limits
RATE_LIMIT_QUESTION=10
RATE_LIMIT_QUESTION_FULL=10
RATE_LIMIT_CRAWL=2
```

## Features

- ✅ **Sliding Window**: Uses a 60-second sliding window for accurate rate limiting
- ✅ **Client-Specific**: Different limits per authentication token
- ✅ **Endpoint-Specific**: Different limits per API endpoint
- ✅ **Memory Efficient**: Automatic cleanup of old request history
- ✅ **Graceful Degradation**: If rate limiter fails, requests are not blocked
- ✅ **Informative Responses**: Clear error messages with retry information
- ✅ **Rate Limit Headers**: Standard HTTP headers for client awareness

## Monitoring

Rate limit events are logged with the following information:
- Client ID (token prefix or IP)
- Endpoint accessed
- Current limit
- When limits are exceeded

Example log:
```
WARNING:Rate limit exceeded for client token_asds.dsa on /question: 10/min
``` 