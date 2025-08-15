import os
from google.adk.tools.mcp_tool import MCPToolset, StreamableHTTPConnectionParams

# Handle missing EXA_API_KEY gracefully - use a placeholder URL if key not available
exa_api_key = os.environ.get('EXA_API_KEY', 'placeholder')

mcp_toolset = MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=f"https://mcp.exa.ai/mcp?exaApiKey={exa_api_key}",
    )
)
