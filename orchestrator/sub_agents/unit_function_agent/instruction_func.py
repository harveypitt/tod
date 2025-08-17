"""
Dynamic instruction function for unit function agent that reads context from session state.
"""
from google.adk.agents import callback_context as callback_context_module
from .prompt import agent_instruction


def get_unit_function_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the unit function agent instruction with context data injected."""
    
    # Get the mcp-spec from session state
    mcp_spec = context.state.get("mcp-spec", "No specification available")
    
    # Replace the template variable with actual data
    instruction_with_context = agent_instruction.replace(
        "{mcp-spec}", 
        str(mcp_spec)
    )
    
    return instruction_with_context