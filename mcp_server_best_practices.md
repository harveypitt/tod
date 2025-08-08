# FastMCP Server Development: Complete Best Practices Guide

## Table of Contents
1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Core Architecture Patterns](#core-architecture-patterns)
4. [Tool Development](#tool-development)
5. [Resource Management](#resource-management)
6. [Error Handling & Security](#error-handling--security)
7. [Testing & Quality Assurance](#testing--quality-assurance)
8. [Performance & Optimization](#performance--optimization)
9. [Production Deployment](#production-deployment)
10. [Advanced Patterns](#advanced-patterns)

## Overview

FastMCP 2.0 is a Python framework for building Model Context Protocol (MCP) servers and clients. MCP is "the USB-C port for AI" - a standardized protocol that connects Large Language Models (LLMs) to external resources and functionality.

### Key Principles
- **🚀 Fast**: High-performance, async-first design
- **🍀 Simple**: Minimal boilerplate, decorator-based API
- **🐍 Pythonic**: Leverages Python's type system and best practices
- **🔍 Comprehensive**: Full ecosystem with testing, deployment, and authentication

## Getting Started

### Basic Server Structure

```python
from fastmcp import FastMCP

# Create server instance
mcp = FastMCP("My MCP Server")

@mcp.tool
def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"

# Run the server
if __name__ == "__main__":
    mcp.run()
```

### Alternative Server Startup

```python
# Using FastMCP CLI (recommended for development)
# fastmcp run my_server.py:mcp

# Or specify different object names
# fastmcp run my_server.py:app
# fastmcp run my_server.py:server
```

## Core Architecture Patterns

### 1. Single Responsibility Principle
Each tool should have a clear, single purpose:

```python
# Good: Specific, focused functionality
@mcp.tool
def calculate_compound_interest(principal: float, rate: float, years: int) -> float:
    """Calculate compound interest for investment."""
    return principal * (1 + rate) ** years

# Avoid: Multiple unrelated operations in one tool
@mcp.tool
def do_everything(data: dict) -> dict:
    # Don't mix calculations, file operations, API calls, etc.
    pass
```

### 2. Stateless Design
Design tools to be stateless when possible:

```python
# Good: Stateless, deterministic
@mcp.tool
def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency string."""
    return f"${amount:.2f} {currency}"

# Problematic: Stateful operations
class ServerState:
    def __init__(self):
        self.user_sessions = {}  # This creates shared state

# Better: Pass state explicitly or use external storage
@mcp.tool
def get_user_profile(user_id: str, session_token: str) -> dict:
    """Get user profile with explicit authentication."""
    # Validate session_token and fetch user data
    pass
```

### 3. Configuration Management

```python
import os
from pydantic import BaseSettings, Field
from typing import Optional

class ServerConfig(BaseSettings):
    """Server configuration with validation."""
    
    # Required settings
    api_key: str = Field(..., env="API_KEY")
    
    # Optional with defaults
    debug: bool = Field(False, env="DEBUG")
    max_retries: int = Field(3, env="MAX_RETRIES")
    timeout: int = Field(30, env="TIMEOUT")
    
    # Optional settings
    external_api_url: Optional[str] = Field(None, env="EXTERNAL_API_URL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Usage
config = ServerConfig()
mcp = FastMCP("Configured Server", debug=config.debug)
```

## Tool Development

### 1. Type Annotations & Parameter Handling

```python
from typing import List, Dict, Optional, Union
from pydantic import Field
from typing_extensions import Annotated
from datetime import datetime, date

@mcp.tool
def advanced_search(
    query: Annotated[str, Field(description="Search query", min_length=1, max_length=200)],
    categories: Annotated[List[str], Field(description="Categories to search in")] = [],
    max_results: Annotated[int, Field(description="Maximum results", ge=1, le=100)] = 10,
    include_archived: Annotated[bool, Field(description="Include archived items")] = False,
    date_from: Annotated[Optional[date], Field(description="Start date filter")] = None,
    date_to: Annotated[Optional[date], Field(description="End date filter")] = None
) -> Dict[str, Union[List[Dict], int]]:
    """
    Perform advanced search with multiple filters.
    
    Returns:
        Dictionary containing search results and metadata
    """
    # Implementation with proper validation
    results = perform_search_logic(query, categories, max_results, include_archived, date_from, date_to)
    
    return {
        "results": results,
        "total_count": len(results),
        "query": query,
        "filters_applied": {
            "categories": categories,
            "date_range": [date_from, date_to] if date_from or date_to else None
        }
    }
```

### 2. Input Validation & Sanitization

```python
import re
from pathlib import Path
from fastmcp import ToolError

@mcp.tool
def read_file_safe(file_path: str) -> str:
    """Read file contents with security validation."""
    
    # Input validation
    if not file_path or not isinstance(file_path, str):
        raise ToolError("file_path must be a non-empty string")
    
    # Sanitize path
    try:
        path = Path(file_path).resolve()
    except (ValueError, OSError) as e:
        raise ToolError(f"Invalid file path: {e}")
    
    # Security checks
    allowed_dirs = [Path.cwd(), Path.home() / "documents"]
    if not any(path.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs):
        raise ToolError("Access denied: file outside allowed directories")
    
    # Size check
    if path.stat().st_size > 1024 * 1024:  # 1MB limit
        raise ToolError("File too large (max 1MB)")
    
    # Read file
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except (IOError, UnicodeDecodeError) as e:
        raise ToolError(f"Cannot read file: {e}")
    
    return content
```

### 3. Structured Output & Error Handling

```python
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"

class ProcessingResult(BaseModel):
    """Structured result for processing operations."""
    status: ProcessingStatus
    processed_items: int
    total_items: int
    errors: List[str] = []
    warnings: List[str] = []
    processing_time: float
    timestamp: datetime

@mcp.tool
def process_data_batch(data_items: List[Dict]) -> ProcessingResult:
    """Process a batch of data items with detailed results."""
    start_time = datetime.now()
    processed = 0
    errors = []
    warnings = []
    
    try:
        for i, item in enumerate(data_items):
            try:
                # Process individual item
                result = process_single_item(item)
                processed += 1
                
                if result.get("warning"):
                    warnings.append(f"Item {i}: {result['warning']}")
                    
            except Exception as e:
                errors.append(f"Item {i}: {str(e)}")
                continue
    
    except Exception as e:
        # Fatal error
        return ProcessingResult(
            status=ProcessingStatus.FAILED,
            processed_items=processed,
            total_items=len(data_items),
            errors=[f"Fatal error: {str(e)}"],
            warnings=warnings,
            processing_time=(datetime.now() - start_time).total_seconds(),
            timestamp=datetime.now()
        )
    
    # Determine status
    if errors and processed == 0:
        status = ProcessingStatus.FAILED
    elif errors:
        status = ProcessingStatus.PARTIAL
    else:
        status = ProcessingStatus.SUCCESS
    
    return ProcessingResult(
        status=status,
        processed_items=processed,
        total_items=len(data_items),
        errors=errors,
        warnings=warnings,
        processing_time=(datetime.now() - start_time).total_seconds(),
        timestamp=datetime.now()
    )
```

### 4. Async Tool Patterns

```python
import asyncio
import httpx
from typing import AsyncGenerator

@mcp.tool
async def fetch_api_data(url: str, headers: Dict[str, str] = None) -> Dict:
    """Fetch data from external API asynchronously."""
    
    timeout = httpx.Timeout(30.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(url, headers=headers or {})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise ToolError(f"HTTP error: {e}")
        except Exception as e:
            raise ToolError(f"Request failed: {e}")

@mcp.tool
async def parallel_processing(urls: List[str]) -> List[Dict]:
    """Process multiple URLs in parallel."""
    
    async def fetch_single(url: str) -> Dict:
        try:
            return await fetch_api_data(url)
        except ToolError as e:
            return {"url": url, "error": str(e)}
    
    # Process in parallel with concurrency limit
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
    
    async def bounded_fetch(url: str) -> Dict:
        async with semaphore:
            return await fetch_single(url)
    
    results = await asyncio.gather(*[bounded_fetch(url) for url in urls])
    return results
```

## Resource Management

### 1. Dynamic Resources

```python
@mcp.resource("user_profiles/{user_id}")
async def get_user_profile(user_id: str) -> str:
    """Get user profile as a resource."""
    
    # Validate user_id
    if not user_id.isalnum():
        raise ValueError("Invalid user ID format")
    
    # Fetch user data
    user_data = await fetch_user_from_db(user_id)
    if not user_data:
        raise ValueError(f"User {user_id} not found")
    
    # Return formatted profile
    return format_user_profile(user_data)

@mcp.resource("reports/{report_type}/{date_range}")
async def generate_report(report_type: str, date_range: str) -> str:
    """Generate dynamic reports based on type and date range."""
    
    valid_types = ["sales", "users", "performance"]
    if report_type not in valid_types:
        raise ValueError(f"Invalid report type. Must be one of: {valid_types}")
    
    # Parse date range
    try:
        start_date, end_date = parse_date_range(date_range)
    except ValueError as e:
        raise ValueError(f"Invalid date range format: {e}")
    
    # Generate report
    report_data = await generate_report_data(report_type, start_date, end_date)
    return format_report(report_data, report_type, start_date, end_date)
```

### 2. Resource Caching

```python
from functools import lru_cache
from datetime import datetime, timedelta
import asyncio

class AsyncLRUCache:
    """Async LRU cache with TTL support."""
    
    def __init__(self, maxsize: int = 128, ttl: int = 300):
        self.cache = {}
        self.access_times = {}
        self.maxsize = maxsize
        self.ttl = ttl
    
    async def get_or_set(self, key: str, coro_func):
        now = datetime.now()
        
        # Check if cached and not expired
        if key in self.cache:
            if now - self.access_times[key] < timedelta(seconds=self.ttl):
                self.access_times[key] = now
                return self.cache[key]
            else:
                # Expired, remove
                del self.cache[key]
                del self.access_times[key]
        
        # Not in cache or expired, compute
        result = await coro_func()
        
        # Add to cache (with size management)
        if len(self.cache) >= self.maxsize:
            # Remove oldest entry
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = result
        self.access_times[key] = now
        
        return result

# Usage in resources
resource_cache = AsyncLRUCache(maxsize=100, ttl=600)  # 10 min TTL

@mcp.resource("heavy_computation/{params}")
async def expensive_resource(params: str) -> str:
    """Cached expensive resource computation."""
    
    async def compute():
        # Simulate expensive operation
        await asyncio.sleep(2)
        return f"Computed result for {params}"
    
    return await resource_cache.get_or_set(f"compute_{params}", compute)
```

## Error Handling & Security

### 1. Comprehensive Error Handling

```python
import logging
from contextlib import asynccontextmanager
from fastmcp import ToolError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityError(ToolError):
    """Security-related tool error."""
    pass

class ValidationError(ToolError):
    """Input validation error."""
    pass

@asynccontextmanager
async def error_context(operation: str):
    """Context manager for consistent error handling."""
    try:
        logger.info(f"Starting {operation}")
        yield
        logger.info(f"Completed {operation}")
    except ValidationError as e:
        logger.warning(f"Validation error in {operation}: {e}")
        raise  # Re-raise for client
    except SecurityError as e:
        logger.error(f"Security error in {operation}: {e}")
        raise ToolError("Operation not permitted")  # Mask security details
    except Exception as e:
        logger.error(f"Unexpected error in {operation}: {e}", exc_info=True)
        raise ToolError(f"Internal error during {operation}")

@mcp.tool
async def secure_operation(user_input: str, api_key: str) -> Dict:
    """Example of secure operation with comprehensive error handling."""
    
    async with error_context("secure_operation"):
        # Input validation
        if not user_input or len(user_input) > 1000:
            raise ValidationError("user_input must be 1-1000 characters")
        
        # Security check
        if not verify_api_key(api_key):
            raise SecurityError("Invalid API key")
        
        # Rate limiting
        if await is_rate_limited(api_key):
            raise SecurityError("Rate limit exceeded")
        
        # Business logic
        result = await process_secure_data(user_input)
        
        return {"status": "success", "data": result}
```

### 2. Input Sanitization Patterns

```python
import html
import re
from urllib.parse import urlparse

def sanitize_html_input(text: str) -> str:
    """Sanitize HTML input to prevent XSS."""
    # Remove HTML tags and escape special characters
    text = re.sub(r'<[^>]+>', '', text)
    return html.escape(text)

def validate_url(url: str) -> bool:
    """Validate URL format and allowed schemes."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ['http', 'https'] and bool(parsed.netloc)
    except Exception:
        return False

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal."""
    # Remove directory separators and special chars
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.replace('..', '')
    return filename[:255]  # Limit length

@mcp.tool
def create_user_content(
    title: str,
    content: str,
    image_url: Optional[str] = None
) -> Dict:
    """Create user content with input sanitization."""
    
    # Sanitize inputs
    clean_title = sanitize_html_input(title)
    clean_content = sanitize_html_input(content)
    
    if image_url:
        if not validate_url(image_url):
            raise ValidationError("Invalid image URL")
    
    # Validate lengths
    if not clean_title or len(clean_title) > 200:
        raise ValidationError("Title must be 1-200 characters")
    
    if not clean_content or len(clean_content) > 5000:
        raise ValidationError("Content must be 1-5000 characters")
    
    return {
        "id": generate_content_id(),
        "title": clean_title,
        "content": clean_content,
        "image_url": image_url,
        "created_at": datetime.now().isoformat()
    }
```

## Testing & Quality Assurance

### 1. Comprehensive Testing Framework

```python
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastmcp import FastMCP, Client

# Test fixtures
@pytest.fixture
def test_server():
    """Create test server instance."""
    mcp = FastMCP("Test Server")
    
    @mcp.tool
    def test_add(a: int, b: int) -> int:
        """Test addition tool."""
        return a + b
    
    @mcp.tool
    async def test_async_operation(data: str) -> Dict:
        """Test async operation."""
        await asyncio.sleep(0.1)  # Simulate async work
        return {"processed": data.upper()}
    
    return mcp

@pytest.fixture
async def test_client(test_server):
    """Create test client."""
    client = Client(test_server)
    async with client:
        yield client

# Unit tests
class TestMCPTools:
    
    async def test_tool_basic_functionality(self, test_client):
        """Test basic tool functionality."""
        result = await test_client.call_tool("test_add", {"a": 2, "b": 3})
        assert result == 5
    
    async def test_tool_async_functionality(self, test_client):
        """Test async tool functionality."""
        result = await test_client.call_tool("test_async_operation", {"data": "hello"})
        assert result == {"processed": "HELLO"}
    
    async def test_tool_error_handling(self, test_client):
        """Test tool error handling."""
        with pytest.raises(Exception):
            await test_client.call_tool("test_add", {"a": "invalid", "b": 3})
    
    async def test_tool_parameter_validation(self, test_client):
        """Test parameter validation."""
        # Missing required parameter
        with pytest.raises(Exception):
            await test_client.call_tool("test_add", {"a": 2})

# Integration tests
class TestMCPIntegration:
    
    @patch('httpx.AsyncClient.get')
    async def test_external_api_integration(self, mock_get, test_client):
        """Test integration with external APIs."""
        # Mock external API response
        mock_response = Mock()
        mock_response.json.return_value = {"data": "test"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value.__aenter__.return_value = mock_response
        
        # Test the integration
        result = await test_client.call_tool("fetch_api_data", {
            "url": "https://api.example.com/data"
        })
        assert "data" in result
    
    async def test_database_integration(self, test_client):
        """Test database integration with mocks."""
        with patch('your_module.database_query') as mock_query:
            mock_query.return_value = [{"id": 1, "name": "test"}]
            
            result = await test_client.call_tool("get_users", {})
            assert len(result) == 1
            assert result[0]["name"] == "test"

# Performance tests
class TestMCPPerformance:
    
    async def test_tool_performance(self, test_client):
        """Test tool performance under load."""
        import time
        
        start_time = time.time()
        
        # Run multiple concurrent operations
        tasks = [
            test_client.call_tool("test_add", {"a": i, "b": i+1})
            for i in range(100)
        ]
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Verify results
        assert len(results) == 100
        assert all(results[i] == i + (i+1) for i in range(100))
        
        # Performance assertion (adjust threshold as needed)
        assert end_time - start_time < 1.0  # Should complete in under 1 second

# Property-based testing
from hypothesis import given, strategies as st

class TestMCPPropertyBased:
    
    @given(a=st.integers(), b=st.integers())
    async def test_addition_properties(self, a, b, test_client):
        """Property-based test for addition."""
        result = await test_client.call_tool("test_add", {"a": a, "b": b})
        assert result == a + b
        
        # Commutative property
        result_reversed = await test_client.call_tool("test_add", {"a": b, "b": a})
        assert result == result_reversed
```

### 2. Quality Assurance Checklist

```python
# quality_assurance.py

class MCPQualityChecker:
    """Quality assurance checker for MCP servers."""
    
    def __init__(self, mcp_server):
        self.server = mcp_server
        self.issues = []
    
    def check_tool_documentation(self):
        """Check if all tools have proper documentation."""
        for tool_name, tool_func in self.server.tools.items():
            if not tool_func.__doc__:
                self.issues.append(f"Tool '{tool_name}' missing docstring")
            
            # Check parameter documentation
            import inspect
            sig = inspect.signature(tool_func)
            for param_name, param in sig.parameters.items():
                if param.annotation == inspect.Parameter.empty:
                    self.issues.append(f"Tool '{tool_name}' parameter '{param_name}' missing type annotation")
    
    def check_error_handling(self):
        """Check if tools have proper error handling."""
        # This would involve static analysis or runtime checking
        # Implementation depends on specific requirements
        pass
    
    def check_security_practices(self):
        """Check security practices."""
        for tool_name, tool_func in self.server.tools.items():
            source = inspect.getsource(tool_func)
            
            # Check for common security issues
            if 'eval(' in source or 'exec(' in source:
                self.issues.append(f"Tool '{tool_name}' uses dangerous eval/exec")
            
            if 'os.system(' in source:
                self.issues.append(f"Tool '{tool_name}' uses potentially unsafe os.system")
    
    def generate_report(self) -> str:
        """Generate quality assurance report."""
        if not self.issues:
            return "✅ All quality checks passed!"
        
        report = "Quality Assurance Issues:\n"
        for issue in self.issues:
            report += f"- {issue}\n"
        
        return report

# Usage
def run_quality_checks(mcp_server):
    """Run quality checks on MCP server."""
    checker = MCPQualityChecker(mcp_server)
    checker.check_tool_documentation()
    checker.check_security_practices()
    return checker.generate_report()
```

## Performance & Optimization

### 1. Async Optimization Patterns

```python
import asyncio
from asyncio import Semaphore, Queue
from typing import Callable, Any

class AsyncTaskManager:
    """Manage async tasks with concurrency control."""
    
    def __init__(self, max_concurrency: int = 10):
        self.semaphore = Semaphore(max_concurrency)
        self.results = Queue()
    
    async def execute_with_limit(self, coro):
        """Execute coroutine with concurrency limit."""
        async with self.semaphore:
            return await coro
    
    async def batch_process(self, items: List[Any], processor: Callable) -> List[Any]:
        """Process items in batches with concurrency control."""
        tasks = [
            self.execute_with_limit(processor(item))
            for item in items
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Usage in tools
task_manager = AsyncTaskManager(max_concurrency=5)

@mcp.tool
async def process_urls_optimized(urls: List[str]) -> List[Dict]:
    """Process URLs with optimized concurrency."""
    
    async def fetch_url(url: str) -> Dict:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url, timeout=10.0)
                return {
                    "url": url,
                    "status": response.status_code,
                    "content_length": len(response.content)
                }
            except Exception as e:
                return {"url": url, "error": str(e)}
    
    results = await task_manager.batch_process(urls, fetch_url)
    return [r for r in results if not isinstance(r, Exception)]
```

### 2. Memory Management

```python
import gc
import psutil
from contextlib import asynccontextmanager

@asynccontextmanager
async def memory_monitor(operation: str, max_memory_mb: int = 500):
    """Monitor memory usage during operations."""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    try:
        yield
    finally:
        gc.collect()  # Force garbage collection
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_used = final_memory - initial_memory
        
        if memory_used > max_memory_mb:
            logger.warning(f"High memory usage in {operation}: {memory_used:.1f}MB")

@mcp.tool
async def memory_intensive_operation(large_dataset: List[Dict]) -> Dict:
    """Handle memory-intensive operations safely."""
    
    async with memory_monitor("data_processing", max_memory_mb=200):
        # Process data in chunks to manage memory
        chunk_size = 1000
        results = []
        
        for i in range(0, len(large_dataset), chunk_size):
            chunk = large_dataset[i:i + chunk_size]
            chunk_result = await process_data_chunk(chunk)
            results.extend(chunk_result)
            
            # Clear chunk from memory
            del chunk
            
            # Periodic garbage collection for long operations
            if i % (chunk_size * 10) == 0:
                gc.collect()
        
        return {"processed_count": len(results), "results": results}
```

### 3. Caching Strategies

```python
import redis
from functools import wraps
import pickle
import hashlib

class AdvancedCache:
    """Advanced caching with multiple backends."""
    
    def __init__(self, redis_url: str = None):
        self.memory_cache = {}
        self.redis_client = redis.from_url(redis_url) if redis_url else None
    
    def _generate_key(self, func_name: str, args, kwargs) -> str:
        """Generate cache key from function arguments."""
        key_data = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def get(self, key: str):
        """Get value from cache (memory first, then Redis)."""
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check Redis if available
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    value = pickle.loads(data)
                    # Store in memory cache for faster access
                    self.memory_cache[key] = value
                    return value
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")
        
        return None
    
    async def set(self, key: str, value, ttl: int = 300):
        """Set value in cache."""
        # Store in memory
        self.memory_cache[key] = value
        
        # Store in Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, pickle.dumps(value))
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")

def cached(ttl: int = 300):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = getattr(wrapper, '_cache', AdvancedCache())
            if not hasattr(wrapper, '_cache'):
                wrapper._cache = cache
            
            key = cache._generate_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = await cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# Usage
@mcp.tool
@cached(ttl=600)  # Cache for 10 minutes
async def expensive_computation(input_data: str) -> Dict:
    """Expensive computation with caching."""
    await asyncio.sleep(2)  # Simulate expensive work
    return {"result": f"processed_{input_data}", "timestamp": datetime.now().isoformat()}
```

## Production Deployment

### 1. Health Checks & Monitoring

```python
from datetime import datetime, timedelta
import asyncio
import json

class HealthChecker:
    """Health check system for MCP server."""
    
    def __init__(self, mcp_server):
        self.server = mcp_server
        self.last_check = datetime.now()
        self.health_status = {"status": "unknown"}
    
    async def check_server_health(self) -> Dict:
        """Comprehensive health check."""
        checks = {}
        overall_status = "healthy"
        
        # Check server responsiveness
        try:
            start_time = datetime.now()
            # Test a simple operation
            await asyncio.sleep(0.001)  # Minimal async operation
            response_time = (datetime.now() - start_time).total_seconds()
            
            checks["server_responsive"] = {
                "status": "pass",
                "response_time_ms": response_time * 1000
            }
        except Exception as e:
            checks["server_responsive"] = {"status": "fail", "error": str(e)}
            overall_status = "unhealthy"
        
        # Check memory usage
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            memory_percent = process.memory_percent()
            
            checks["memory"] = {
                "status": "pass" if memory_percent < 80 else "warn",
                "memory_mb": memory_mb,
                "memory_percent": memory_percent
            }
            
            if memory_percent > 90:
                overall_status = "unhealthy"
        except Exception as e:
            checks["memory"] = {"status": "fail", "error": str(e)}
        
        # Check external dependencies
        checks["external_deps"] = await self._check_external_dependencies()
        if checks["external_deps"]["status"] == "fail":
            overall_status = "unhealthy"
        
        self.health_status = {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "checks": checks,
            "uptime": str(datetime.now() - self.last_check)
        }
        
        return self.health_status
    
    async def _check_external_dependencies(self) -> Dict:
        """Check external service dependencies."""
        deps = []
        
        # Example: Check database connection
        try:
            # await database.execute("SELECT 1")
            deps.append({"service": "database", "status": "pass"})
        except Exception as e:
            deps.append({"service": "database", "status": "fail", "error": str(e)})
        
        # Example: Check external API
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("https://api.external-service.com/health", timeout=5.0)
                if response.status_code == 200:
                    deps.append({"service": "external_api", "status": "pass"})
                else:
                    deps.append({"service": "external_api", "status": "fail", "status_code": response.status_code})
        except Exception as e:
            deps.append({"service": "external_api", "status": "fail", "error": str(e)})
        
        overall = "pass" if all(dep["status"] == "pass" for dep in deps) else "fail"
        
        return {"status": overall, "dependencies": deps}

# Add health check tool to server
@mcp.tool
async def health_check() -> Dict:
    """Get server health status."""
    health_checker = HealthChecker(mcp)
    return await health_checker.check_server_health()
```

### 2. Logging & Observability

```python
import logging
import structlog
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

def setup_production_logging():
    """Set up production-grade logging."""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RotatingFileHandler("mcp_server.log", maxBytes=10*1024*1024, backupCount=5),
            logging.StreamHandler()
        ]
    )

# Usage in tools
logger = structlog.get_logger()

@mcp.tool
async def logged_operation(data: Dict) -> Dict:
    """Example of comprehensive logging in tool."""
    
    operation_id = generate_uuid()
    
    logger.info(
        "Operation started",
        operation_id=operation_id,
        operation="logged_operation",
        input_size=len(str(data))
    )
    
    try:
        start_time = datetime.now()
        
        # Process data
        result = await process_data(data)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(
            "Operation completed successfully",
            operation_id=operation_id,
            processing_time=processing_time,
            result_size=len(str(result))
        )
        
        return result
        
    except Exception as e:
        logger.error(
            "Operation failed",
            operation_id=operation_id,
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        raise ToolError(f"Operation failed: {operation_id}")
```

### 3. Configuration for Production

```python
# production_config.py
import os
from pydantic import BaseSettings, Field, validator
from typing import List, Optional
import secrets

class ProductionConfig(BaseSettings):
    """Production configuration with security and performance settings."""
    
    # Server settings
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    workers: int = Field(4, env="WORKERS")
    
    # Security
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    allowed_hosts: List[str] = Field(default_factory=list, env="ALLOWED_HOSTS")
    cors_origins: List[str] = Field(default_factory=list, env="CORS_ORIGINS")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(20, env="DATABASE_POOL_SIZE")
    
    # Redis cache
    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    cache_ttl: int = Field(300, env="CACHE_TTL")
    
    # External APIs
    external_api_key: str = Field(..., env="EXTERNAL_API_KEY")
    external_api_timeout: int = Field(30, env="EXTERNAL_API_TIMEOUT")
    
    # Performance
    max_request_size: int = Field(10 * 1024 * 1024, env="MAX_REQUEST_SIZE")  # 10MB
    max_concurrent_requests: int = Field(100, env="MAX_CONCURRENT_REQUESTS")
    
    # Monitoring
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    metrics_endpoint: str = Field("/metrics", env="METRICS_ENDPOINT")
    
    @validator('allowed_hosts', 'cors_origins', pre=True)
    def split_hosts(cls, v):
        if isinstance(v, str):
            return [host.strip() for host in v.split(',') if host.strip()]
        return v
    
    class Config:
        env_file = ".env.production"
        env_file_encoding = "utf-8"

# Usage
config = ProductionConfig()
mcp = FastMCP("Production Server", **config.dict())
```

## Advanced Patterns

### 1. Plugin System

```python
import importlib
from abc import ABC, abstractmethod
from typing import Dict, List, Type

class MCPPlugin(ABC):
    """Base class for MCP plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass
    
    @abstractmethod
    async def initialize(self, server: FastMCP) -> None:
        """Initialize plugin with server instance."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass

class PluginManager:
    """Manage MCP plugins."""
    
    def __init__(self, server: FastMCP):
        self.server = server
        self.plugins: Dict[str, MCPPlugin] = {}
    
    async def load_plugin(self, plugin_class: Type[MCPPlugin]):
        """Load and initialize a plugin."""
        plugin = plugin_class()
        await plugin.initialize(self.server)
        self.plugins[plugin.name] = plugin
    
    async def unload_plugin(self, plugin_name: str):
        """Unload a plugin."""
        if plugin_name in self.plugins:
            await self.plugins[plugin_name].cleanup()
            del self.plugins[plugin_name]
    
    async def load_plugins_from_config(self, plugin_configs: List[Dict]):
        """Load plugins from configuration."""
        for config in plugin_configs:
            module_name = config["module"]
            class_name = config["class"]
            
            try:
                module = importlib.import_module(module_name)
                plugin_class = getattr(module, class_name)
                await self.load_plugin(plugin_class)
            except Exception as e:
                logger.error(f"Failed to load plugin {class_name}: {e}")

# Example plugin
class DatabasePlugin(MCPPlugin):
    """Database integration plugin."""
    
    @property
    def name(self) -> str:
        return "database"
    
    async def initialize(self, server: FastMCP) -> None:
        """Initialize database connection and tools."""
        self.db_pool = await create_connection_pool()
        
        @server.tool
        async def query_database(sql: str, params: List = None) -> List[Dict]:
            """Execute database query."""
            async with self.db_pool.acquire() as conn:
                result = await conn.fetch(sql, *(params or []))
                return [dict(row) for row in result]
        
        @server.tool
        async def get_table_schema(table_name: str) -> Dict:
            """Get database table schema."""
            # Implementation here
            pass
    
    async def cleanup(self) -> None:
        """Close database connections."""
        if hasattr(self, 'db_pool'):
            await self.db_pool.close()

# Usage
plugin_manager = PluginManager(mcp)
await plugin_manager.load_plugin(DatabasePlugin)
```

### 2. Middleware System

```python
from functools import wraps
from typing import Callable, Any
import time

class MCPMiddleware:
    """Base middleware class."""
    
    async def before_tool_call(self, tool_name: str, arguments: Dict) -> Dict:
        """Called before tool execution."""
        return arguments
    
    async def after_tool_call(self, tool_name: str, result: Any) -> Any:
        """Called after tool execution."""
        return result
    
    async def on_error(self, tool_name: str, error: Exception) -> Exception:
        """Called when tool execution fails."""
        return error

class TimingMiddleware(MCPMiddleware):
    """Middleware for timing tool execution."""
    
    async def before_tool_call(self, tool_name: str, arguments: Dict) -> Dict:
        """Record start time."""
        arguments['_start_time'] = time.time()
        return arguments
    
    async def after_tool_call(self, tool_name: str, result: Any) -> Any:
        """Log execution time."""
        if isinstance(result, dict) and '_start_time' in result:
            execution_time = time.time() - result.pop('_start_time', 0)
            logger.info(f"Tool {tool_name} executed in {execution_time:.3f}s")
        return result

class ValidationMiddleware(MCPMiddleware):
    """Middleware for additional validation."""
    
    async def before_tool_call(self, tool_name: str, arguments: Dict) -> Dict:
        """Validate arguments."""
        # Add custom validation logic
        if 'sensitive_data' in arguments:
            if not self.is_authorized(arguments.get('user_id')):
                raise SecurityError("Unauthorized access to sensitive data")
        return arguments

def with_middleware(*middleware_classes):
    """Decorator to apply middleware to tools."""
    def decorator(tool_func):
        @wraps(tool_func)
        async def wrapper(*args, **kwargs):
            middleware_instances = [cls() for cls in middleware_classes]
            
            # Extract tool name and arguments
            tool_name = tool_func.__name__
            arguments = kwargs
            
            try:
                # Apply before middleware
                for middleware in middleware_instances:
                    arguments = await middleware.before_tool_call(tool_name, arguments)
                
                # Execute tool
                result = await tool_func(*args, **arguments)
                
                # Apply after middleware
                for middleware in reversed(middleware_instances):
                    result = await middleware.after_tool_call(tool_name, result)
                
                return result
                
            except Exception as e:
                # Apply error middleware
                for middleware in reversed(middleware_instances):
                    e = await middleware.on_error(tool_name, e)
                raise e
        
        return wrapper
    return decorator

# Usage
@mcp.tool
@with_middleware(TimingMiddleware, ValidationMiddleware)
async def protected_operation(data: Dict, user_id: str) -> Dict:
    """Tool with middleware protection."""
    return await process_sensitive_data(data, user_id)
```

### 3. Dynamic Tool Registration

```python
class DynamicToolManager:
    """Manage dynamic tool registration and updates."""
    
    def __init__(self, server: FastMCP):
        self.server = server
        self.dynamic_tools: Dict[str, Callable] = {}
    
    async def register_tool_from_config(self, config: Dict):
        """Register tool from configuration."""
        tool_name = config["name"]
        tool_code = config["code"]
        tool_description = config.get("description", "")
        
        # Create tool function dynamically
        namespace = {
            'asyncio': asyncio,
            'httpx': httpx,
            'json': json,
            'datetime': datetime,
            'logger': logger
        }
        
        exec(tool_code, namespace)
        tool_func = namespace[tool_name]
        
        # Add documentation
        if tool_description:
            tool_func.__doc__ = tool_description
        
        # Register with server
        self.server.tool(tool_func)
        self.dynamic_tools[tool_name] = tool_func
    
    async def unregister_tool(self, tool_name: str):
        """Unregister a dynamic tool."""
        if tool_name in self.dynamic_tools:
            # Remove from server (implementation depends on FastMCP internals)
            del self.dynamic_tools[tool_name]
    
    async def update_tool(self, tool_name: str, new_config: Dict):
        """Update existing tool."""
        await self.unregister_tool(tool_name)
        await self.register_tool_from_config(new_config)

# Example usage
dynamic_manager = DynamicToolManager(mcp)

# Register tool from configuration
tool_config = {
    "name": "dynamic_calculator",
    "description": "Dynamically created calculator tool",
    "code": """
async def dynamic_calculator(expression: str) -> float:
    '''Evaluate mathematical expression safely.'''
    import ast
    import operator
    
    # Safe evaluation implementation
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    
    def eval_expr(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return allowed_operators[type(node.op)](
                eval_expr(node.left), 
                eval_expr(node.right)
            )
        else:
            raise ValueError('Unsupported operation')
    
    return eval_expr(ast.parse(expression, mode='eval').body)
"""
}

await dynamic_manager.register_tool_from_config(tool_config)
```

## Summary

This comprehensive guide covers all essential aspects of building production-ready MCP servers with FastMCP:

1. **Foundation**: Basic server setup and configuration patterns
2. **Tools**: Advanced tool development with validation, error handling, and structured output  
3. **Resources**: Dynamic resource management and caching strategies
4. **Security**: Input sanitization, authentication, and secure coding practices
5. **Testing**: Comprehensive testing strategies including unit, integration, and performance tests
6. **Performance**: Async optimization, memory management, and caching
7. **Production**: Health checks, logging, monitoring, and deployment configurations
8. **Advanced**: Plugin systems, middleware, and dynamic tool registration

Following these patterns will ensure your MCP server is:
- ✅ **Reliable**: Proper error handling and testing
- ✅ **Secure**: Input validation and security best practices  
- ✅ **Performant**: Async optimization and caching
- ✅ **Maintainable**: Clear structure and comprehensive documentation
- ✅ **Production-ready**: Health checks, logging, and monitoring

Use this guide as a reference when building MCP servers that need to handle real-world production workloads with enterprise-grade reliability and security.