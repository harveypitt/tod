from google.adk import Agent
from google.adk.code_executors import BuiltInCodeExecutor

from .prompt import agent_instruction

unit_function_agent = Agent(
    model="gemini-2.5-pro",
    name="unit_function_agent",
    code_executor=BuiltInCodeExecutor(),
    output_key="unit_functions",
    instruction=agent_instruction
)
