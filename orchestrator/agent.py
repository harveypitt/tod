from google.adk.agents import SequentialAgent

from .sub_agents.mcp_creation_agent import mcp_creation_agent
from .sub_agents.spec_writer import spec_writer_agent
from .sub_agents.unit_function_agent import unit_function_agent

orchestrator_instruction = """
You are the orchestrator agent for TOD (Tool On Demand), coordinating a three-stage pipeline to create custom MCP servers.

Your pipeline stages:
1. Spec Writer: Converts natural language requests into structured specifications
2. Unit Function Agent: Generates individual Python functions from specifications
3. MCP Creation Agent: Assembles functions into complete FastMCP servers

Coordinate the flow of data between agents, ensuring each stage receives the necessary context and outputs from previous stages.
"""

orchestrator_agent = SequentialAgent(
    name="orchestrator_agent",
    sub_agents=[spec_writer_agent, unit_function_agent, mcp_creation_agent],
    description="Orchestrates the full pipeline of creating an MCP server on demand"
)

root_agent = orchestrator_agent
