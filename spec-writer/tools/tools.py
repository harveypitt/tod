from google.adk.tools.mcp_tool import MCPToolset, StreamableHTTPConnectionParams

mcp_tools = MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://mcp.exa.ai/mcp?exaApiKey=0fce9485-b420-4bac-9f2e-a558dd09aac6",
    )
)
