from google.adk import Agent
from google.adk.planners import PlanReActPlanner

from .prompt import agent_instruction
from .tools.tools import mcp_toolset

root_agent = Agent(
    model="gemini-2.5-flash",
    name="spec_writer",
    instruction=agent_instruction,
    planner=PlanReActPlanner(),
    tools=[mcp_toolset],
)
