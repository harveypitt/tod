"""
Dynamic instruction function for MCP creation agent that reads context from session state.
"""
from google.adk.agents import callback_context as callback_context_module
from .prompt import agent_instruction


def get_mcp_creation_instruction(
    context: callback_context_module.ReadonlyContext,
) -> str:
    """Gets the MCP creation agent instruction with context data injected."""
    
    # Get both context variables from session state
    mcp_spec = context.state.get("mcp-spec", "No specification available")
    unit_functions = context.state.get("unit-functions", "No functions available")
    
    # Replace the template variables with actual data
    instruction_with_context = agent_instruction.replace(
        "{mcp-spec}", 
        str(mcp_spec)
    ).replace(
        "{unit-functions}",
        str(unit_functions)
    )
    
    return instruction_with_context