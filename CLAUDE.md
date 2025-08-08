# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TOD (Tool On Demand) is a multi-agent system that enables LLMs to dynamically generate and deploy custom MCP (Model Context Protocol) servers. The system uses a three-stage pipeline where specialized Google ADK agents handle request processing, function generation, and server assembly.

## Architecture

### Three-Stage Agent Pipeline

1. **Spec Writer Agent** (`spec-writer/`) - Converts natural language requests into structured specifications using Gemini-2.5-flash
2. **Unit Function Agent** (`unit-function-agent/`) - Generates individual Python functions using Gemini-2.5-pro with code execution
3. **MCP Creation Agent** (`mcp-creation-agent/`) - Assembles functions into complete FastMCP servers using Gemini-2.5-flash

Each agent follows the Google ADK pattern:
```python
Agent(
    model="gemini-2.5-flash", 
    instruction=agent_instruction,
    planner=PlanReActPlanner(),
    tools=[mcp_toolset]
)
```

### FastMCP Server Implementation

The `fastmcp_server.py` provides a production-ready MCP server template with:
- Environment-based configuration management
- Comprehensive error handling and logging
- Built-in tools (echo, file operations, system info)
- Health monitoring capabilities
- Security controls for file access

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Configure environment variables (copy and edit .env.example)
cp .env.example .env
```

### Running Components

```bash
# Start the FastMCP server
python fastmcp_server.py
# Or using FastMCP CLI
fastmcp run fastmcp_server.py:create_mcp_server

# Test individual agents
python -c "from mcp_creation_agent.agent import root_agent; print('MCP Agent loaded')"
python -c "from unit_function_agent.agent import root_agent; print('Function Agent loaded')"
python -c "from spec_writer.agent import root_agent; print('Spec Agent loaded')"
```

### Testing
```bash
# Run test suite
python -m pytest tests/ -v

# Test specific components
python test_agent.py
```

## Key Dependencies

- **Google ADK (≥1.9.0)**: Agent framework for the three-stage pipeline
- **Pydantic (≥2.11.7)**: Data validation and configuration management  
- **FastMCP**: MCP server framework (imported in fastmcp_server.py)
- **Exa API**: Web search integration for research-capable agents

## Configuration Requirements

All research-capable agents require `EXA_API_KEY` for web search integration:
```python
mcp_toolset = MCPToolset(
    connection_params=StreamableHTTPConnectionParams(
        url=f"https://mcp.exa.ai/mcp?exaApiKey={os.environ['EXA_API_KEY']}"
    )
)
```

The FastMCP server expects optional `API_KEY`, `DEBUG`, `LOG_LEVEL`, and `TIMEOUT` environment variables.

## Agent Workflow

1. Each agent has specialized prompts in `prompt.py` files that define their behavior
2. Agents use external tools (primarily Exa search) for research and documentation lookup
3. The unit function agent includes code execution capabilities for testing generated functions
4. Output flows from spec writer → unit function agent → MCP creation agent

## Production Considerations

The FastMCP server is designed for production deployment with:
- Structured logging with file rotation
- Input validation and security controls
- Health check endpoints
- Async-first design for performance
- Comprehensive error handling

## Current State vs Vision

The project contains working components (FastMCP server, individual agents) but lacks the orchestrator that would coordinate the full three-stage pipeline automatically. The README describes aspirational features that aren't yet implemented.