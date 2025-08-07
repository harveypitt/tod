import os
import tempfile
from typing import Dict, List
from pydantic import BaseModel
from google.adk.agents import Agent


class RequiredUnitFunctions(BaseModel):
    function_name: str
    description: str


class ExternalApisNeeded(BaseModel):
    api_name: str
    api_config: Dict[str, str]


class ToolSpecs(BaseModel):
    original_query: str
    tool_name: str
    description: str
    required_unit_functions: List[RequiredUnitFunctions]
    external_apis_needed: List[ExternalApisNeeded]
    full_specs: str


def generate_unit_function(function_name: str, description: str) -> dict:
    """Generate a Python unit function based on the provided name and description.

    Args:
        function_name (str): The name of the function to generate
        description (str): Description of what the function should do

    Returns:
        dict: Status and file path of the generated function, or error message
    """
    try:
        # Use /temp directory
        temp_dir = "temp"
        file_path = os.path.join(temp_dir, f"{function_name}.py")

        # Generate basic function template based on description
        function_code = f'''"""
Generated unit function: {function_name}
Description: {description}
"""

def {function_name}():
    """
    {description}

    Returns:
        dict: Result with status and data or error message
    """
    # TODO: Implement function logic based on description
    return {{
        "status": "success",
        "message": "Function {function_name} executed successfully",
        "data": None
    }}
'''

        # Write the function to file
        with open(file_path, 'w') as f:
            f.write(function_code)

        return {
            "status": "success",
            "message": f"Unit function '{function_name}' generated successfully",
            "file_path": file_path
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to generate unit function: {str(e)}"
        }


root_agent = Agent(
    name="unit_function_agent",
    model="gemini-2.0-flash",
    description="Agent to generate unit functions based on function name and description",
    instruction="You are a helpful agent that generates Python unit functions based on provided specifications.",
    tools=[generate_unit_function],
)
