
#!/usr/bin/env python3
"""
FastMCP Server Boilerplate

A comprehensive boilerplate for creating MCP (Model Context Protocol) servers using FastMCP.
This server provides a foundation for building AI assistant tools with proper error handling,
logging, and configuration management.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastmcp import FastMCP
from pydantic import BaseModel, Field


class ServerConfig:
    """Server configuration management"""
    
    def __init__(self):
        self.api_key = os.getenv("API_KEY")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.timeout = int(os.getenv("TIMEOUT", "30"))
        
    def validate(self) -> bool:
        """Validate required configuration"""
        if not self.api_key:
            logging.warning("API_KEY not set in environment")
            return False
        return True


class ToolResponse(BaseModel):
    """Standardized tool response model"""
    success: bool
    data: Any = None
    error: str = None
    timestamp: datetime = Field(default_factory=datetime.now)


def setup_logging(config: ServerConfig) -> None:
    """Configure logging for the server"""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fastmcp_server.log')
        ]
    )


async def make_http_request(
    url: str, 
    method: str = "GET", 
    headers: Optional[Dict[str, str]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Make HTTP requests with proper error handling
    
    Args:
        url: Target URL
        method: HTTP method (GET, POST, PUT, DELETE)
        headers: Optional headers dictionary
        json_data: Optional JSON payload
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary containing response data
        
    Raises:
        httpx.RequestError: On request failures
        httpx.HTTPStatusError: On HTTP error status codes
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=json_data,
                timeout=timeout
            )
            response.raise_for_status()
            
            # Handle different content types
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            else:
                return {"content": response.text}
                
        except httpx.RequestError as e:
            logging.error(f"Request error for {url}: {e}")
            raise
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error {e.response.status_code} for {url}: {e}")
            raise


def create_mcp_server() -> FastMCP:
    """Create and configure the FastMCP server instance"""
    
    # Initialize configuration
    config = ServerConfig()
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Validate configuration
    if not config.validate():
        logger.warning("Server starting with incomplete configuration")
    
    # Create FastMCP instance
    mcp = FastMCP("FastMCP Boilerplate Server")
    
    @mcp.tool()
    async def echo_tool(message: str) -> str:
        """
        Echo tool - returns the input message
        
        Args:
            message: The message to echo back
            
        Returns:
            The echoed message with timestamp
        """
        logger.info(f"Echo tool called with message: {message}")
        timestamp = datetime.now().isoformat()
        return f"[{timestamp}] Echo: {message}"
    
    @mcp.tool()
    async def my_name() -> str:
        """
        Get my name
        
        Returns:
            My name as a string
        """
        logger.info("My name tool called")
        return "shahel"
    
    @mcp.tool()
    async def get_system_info() -> str:
        """
        Get system information
        
        Returns:
            JSON string containing system information
        """
        logger.info("System info tool called")
        try:
            info = {
                "server_name": "FastMCP Boilerplate Server",
                "timestamp": datetime.now().isoformat(),
                "working_directory": str(Path.cwd()),
                "environment_variables": {
                    key: value for key, value in os.environ.items() 
                    if not key.endswith("_KEY") and not key.endswith("_TOKEN")
                },
                "config": {
                    "debug": config.debug,
                    "log_level": config.log_level,
                    "timeout": config.timeout
                }
            }
            return json.dumps(info, indent=2)
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return json.dumps({
                "error": f"Failed to get system info: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
    
    @mcp.tool()
    async def web_search(query: str, num_results: int = 5) -> str:
        """
        Perform web search (example implementation)
        
        Args:
            query: Search query string
            num_results: Number of results to return (default: 5)
            
        Returns:
            JSON string containing search results
        """
        logger.info(f"Web search tool called with query: {query}")
        
        if not config.api_key:
            error_msg = "API key not configured for web search"
            logger.error(error_msg)
            return json.dumps({
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            })
        
        try:
            # Example using a hypothetical search API
            # Replace with your preferred search service (Serper, Google, etc.)
            url = "https://api.example-search.com/search"
            headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "query": query,
                "num_results": num_results
            }
            
            # For demonstration, return mock results
            # In real implementation, uncomment the HTTP request below
            # response_data = await make_http_request(url, "POST", headers, payload, config.timeout)
            
            mock_results = {
                "query": query,
                "results": [
                    {
                        "title": f"Result {i+1} for '{query}'",
                        "url": f"https://example.com/result-{i+1}",
                        "snippet": f"This is a mock result {i+1} for the query '{query}'"
                    }
                    for i in range(min(num_results, 3))
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            return json.dumps(mock_results, indent=2)
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return json.dumps({
                "error": f"Web search failed: {str(e)}",
                "query": query,
                "timestamp": datetime.now().isoformat()
            })
    
    @mcp.tool()
    async def read_file(file_path: str) -> str:
        """
        Read file contents
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            File contents or error message
        """
        logger.info(f"Read file tool called with path: {file_path}")
        
        try:
            path = Path(file_path)
            
            if not path.exists():
                return json.dumps({
                    "error": f"File not found: {file_path}",
                    "timestamp": datetime.now().isoformat()
                })
            
            if not path.is_file():
                return json.dumps({
                    "error": f"Path is not a file: {file_path}",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Check file size (limit to 1MB for safety)
            if path.stat().st_size > 1024 * 1024:
                return json.dumps({
                    "error": f"File too large (>1MB): {file_path}",
                    "timestamp": datetime.now().isoformat()
                })
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return json.dumps({
                "file_path": file_path,
                "content": content,
                "size_bytes": path.stat().st_size,
                "timestamp": datetime.now().isoformat()
            })
            
        except UnicodeDecodeError:
            return json.dumps({
                "error": f"Cannot decode file as UTF-8: {file_path}",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return json.dumps({
                "error": f"Failed to read file: {str(e)}",
                "file_path": file_path,
                "timestamp": datetime.now().isoformat()
            })
    
    @mcp.tool()
    async def list_directory(directory_path: str = ".") -> str:
        """
        List directory contents
        
        Args:
            directory_path: Path to the directory to list (default: current directory)
            
        Returns:
            JSON string containing directory listing
        """
        logger.info(f"List directory tool called with path: {directory_path}")
        
        try:
            path = Path(directory_path)
            
            if not path.exists():
                return json.dumps({
                    "error": f"Directory not found: {directory_path}",
                    "timestamp": datetime.now().isoformat()
                })
            
            if not path.is_dir():
                return json.dumps({
                    "error": f"Path is not a directory: {directory_path}",
                    "timestamp": datetime.now().isoformat()
                })
            
            items = []
            for item in path.iterdir():
                items.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                })
            
            # Sort by type (directories first) then by name
            items.sort(key=lambda x: (x["type"] == "file", x["name"].lower()))
            
            return json.dumps({
                "directory": directory_path,
                "items": items,
                "total_items": len(items),
                "timestamp": datetime.now().isoformat()
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error listing directory {directory_path}: {e}")
            return json.dumps({
                "error": f"Failed to list directory: {str(e)}",
                "directory": directory_path,
                "timestamp": datetime.now().isoformat()
            })
    
    @mcp.tool()
    async def mcp_build_practices() -> str:
        """
        Get comprehensive best practices for building MCP servers
        
        Returns:
            String containing comprehensive MCP server development best practices based on FastMCP documentation
        """
        logger.info("MCP build practices tool called")
        
        try:
            # Read the comprehensive best practices file
            practices_file = Path(__file__).parent / "mcp_server_best_practices.md"
            
            if practices_file.exists():
                with open(practices_file, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # Fallback to basic practices if file not found
                return """
# FastMCP Server Development Best Practices (Fallback)

FastMCP 2.0 is a Python framework for building Model Context Protocol (MCP) servers and clients. MCP is "the USB-C port for AI" - a standardized protocol that connects Large Language Models (LLMs) to external resources and functionality.

## Quick Start

```python
from fastmcp import FastMCP

# Create server instance
mcp = FastMCP("My MCP Server")

@mcp.tool
def greet(name: str) -> str:
    '''Greet a user by name.'''
    return f"Hello, {name}!"

# Run the server
if __name__ == "__main__":
    mcp.run()
```

## Key Principles
- 🚀 **Fast**: High-performance, async-first design
- 🍀 **Simple**: Minimal boilerplate, decorator-based API  
- 🐍 **Pythonic**: Leverages Python's type system and best practices
- 🔍 **Comprehensive**: Full ecosystem with testing, deployment, and authentication

## Core Best Practices

1. **Use Type Annotations**: Always use type hints for parameters and return values
2. **Input Validation**: Validate and sanitize all inputs using Pydantic Field annotations
3. **Error Handling**: Implement comprehensive error handling with try/catch blocks
4. **Documentation**: Provide detailed docstrings for all tools
5. **Async Operations**: Use async/await for I/O operations
6. **Security**: Never trust user input, validate everything
7. **Testing**: Write comprehensive tests for all tools
8. **Logging**: Implement structured logging for debugging and monitoring
9. **Performance**: Use caching and connection pooling where appropriate
10. **Production Ready**: Include health checks and monitoring

For complete documentation, ensure the mcp_server_best_practices.md file is present.
"""
                
        except Exception as e:
            logger.error(f"Error reading best practices file: {e}")
            return json.dumps({
                "error": f"Could not load best practices: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
    
    logger.info("FastMCP server initialized successfully")
    return mcp


def main():
    """Main entry point for the server"""
    logger = logging.getLogger(__name__)
    logger.info("Starting FastMCP server...")
    
    try:
        mcp = create_mcp_server()
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
    finally:
        logger.info("FastMCP server stopped")


if __name__ == "__main__":
    main()