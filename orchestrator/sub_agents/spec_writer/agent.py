from google.adk import Agent
from google.adk.planners import PlanReActPlanner

from .prompt import agent_instruction
from .tools.tools import mcp_toolset

spec_writer_agent = Agent(
    model="gemini-2.5-flash",
    name="spec_writer",
    instruction=agent_instruction,
    planner=PlanReActPlanner(),
    output_key="mcp-spec",
    tools=[mcp_toolset]
)
