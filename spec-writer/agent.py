from google.adk.agents import Agent

from .prompt import agent_instruction
from .tools.tools import mcp_tools

root_agent = Agent(
    model="gemini-2.5-flash",
    name="software_assistant",
    instruction=agent_instruction,
    tools=[mcp_tools],
)
