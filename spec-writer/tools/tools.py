import os
from google.adk.tools.mcp_tool import MCPToolset, StreamableHTTPConnectionParams

mcp_tools = MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=f"https://mcp.exa.ai/mcp?exaApiKey={os.environ['EXA_API_KEY']}",
    )
)
