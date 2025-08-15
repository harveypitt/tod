from google.adk import Agent
from google.adk.planners import PlanReActPlanner
from google.adk.code_executors import BuiltInCodeExecutor

from .prompt import agent_instruction
from .tools.tools import mcp_toolset

mcp_creation_agent = Agent(
    model="gemini-2.5-flash",
    name="mcp_creation_agent",
    instruction=agent_instruction,
    planner=PlanReActPlanner(),
    tools=[mcp_toolset]
)
