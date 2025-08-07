"""
### SYSTEM PROMPT ###

You are a agent which is tasked with creating a tool to use on a MCP server.

Write all code in python. Please only use standard python libraries.

You MUST follow the following workflow:
1. Retrieve relevant data related to the query, this includes the relevant documentation, this can be document using the tool "retrieve_info"
2. Create a "specs" document, this outlines this individual api tools that will be used so they can be unit tested first.
3. You should generate these unit functions in code and pass them to a tool "test_function" which runs the code and outputs the responses ( or a traceback if it errors).
4. You should iterate until all unit tests pass.
5. You should then aggregate the functions to generate a aggreated function which also calls "test_function".
6. Iterate this until it works and correctly answering the user's query.


The specs document should be in the following format:
specs = {
    "original_query": "Create a tool to get my open GitHub PRs and link any relevant issues.",
    "tool_name": "get_open_github_prs_and_link_relevant_issues",
    "description": "Get open GitHub PRs and link any relevant issues",
    "required_unit_functions": {
        "get_current_username" : "description of the function",
        "fetch_open_prs" : "description of the function",
        "extract_linked_issues" : "description of the function"
    },
    "full_specs": "..."
}

You can access the a github api token using environ["GITHUB_TOKEN"]
"""

import os
os.environ['GITHUB_TOKEN'] = "secret token"
os.environ['PERPLEXITY_API_KEY'] = "secret key"


def retrieve_info(query: str):
    """Retrieve information from the internet using Perplexity API."""
    import json
    import urllib.request
    import urllib.parse
    
    enhanced_query = f"""Please provide all relevant documentation from the internet/source for the following request: {query}

Include comprehensive documentation that covers:
- Complete API documentation with examples
- Implementation patterns and best practices
- Error handling approaches
- Authentication methods if applicable
- Rate limiting considerations
- Response formats and data structures
- Dependencies and requirements
- Code examples and usage patterns

Provide enough detailed documentation that this can be understood as a standalone document to later generate functions to answer the request."""
    
    try:
        # Get Perplexity API key from environment
        api_key = os.environ.get('PERPLEXITY_API_KEY')
        if not api_key:
            return f"Error: PERPLEXITY_API_KEY environment variable not set.\n\nFallback enhanced query: {enhanced_query}"
        
        # Prepare the API request
        url = "https://api.perplexity.ai/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama-3.1-sonar-large-128k-online",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful research assistant that provides comprehensive documentation and information from the internet."
                },
                {
                    "role": "user",
                    "content": enhanced_query
                }
            ],
            "max_tokens": 4000,
            "temperature": 0.1,
            "stream": False
        }
        
        # Make the API request
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            
        # Extract the response content
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            return content
        else:
            return f"Error: Unexpected response format from Perplexity API.\n\nFallback enhanced query: {enhanced_query}"
            
    except Exception as e:
        return f"Error calling Perplexity API: {str(e)}\n\nFallback enhanced query: {enhanced_query}"

def test_function(function_code: str) -> str:
    """Execute function code in a temporary file with error handling.
    
    Args:
        function_code: Python code string to execute
        
    Returns:
        str: Execution result or traceback if error occurs
    """
    import subprocess
    import tempfile
    import traceback
    import sys
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(function_code)
            temp_file_path = temp_file.name
        
        try:
            # Execute the temporary file using subprocess
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            # Check if execution was successful
            if result.returncode == 0:
                output = result.stdout.strip()
                return f"Execution successful:\n{output}" if output else "Execution successful (no output)"
            else:
                error_output = result.stderr.strip()
                return f"Execution failed with return code {result.returncode}:\n{error_output}"
                
        except subprocess.TimeoutExpired:
            return "Execution failed: Timeout after 30 seconds"
        except Exception as e:
            return f"Subprocess execution failed: {str(e)}\n{traceback.format_exc()}"
            
        finally:
            # Clean up temporary file
            try:
                import os
                os.unlink(temp_file_path)
            except:
                pass  # Ignore cleanup errors
                
    except Exception as e:
        return f"Failed to create or execute temporary file: {str(e)}\n{traceback.format_exc()}"
