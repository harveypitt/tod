from typing import List, Optional
from pydantic import BaseModel, Field


class UnitFunction(BaseModel):
    """Represents a single unit function in the tool specification."""
    name: str = Field(..., description="Name of the function")
    description: str = Field(..., description="What this function does")
    inputs: List[str] = Field(..., description="List of input parameters")
    outputs: str = Field(..., description="Description of the output")
    external_dependencies: List[str] = Field(..., description="List of external dependencies like APIs or libraries")


class ExternalAPI(BaseModel):
    """Represents an external API requirement."""
    api_name: str = Field(..., description="Name of the API")
    base_url: str = Field(..., description="Base URL of the API")
    auth_type: str = Field(..., description="Authentication type (e.g., 'bearer_token', 'api_key', 'oauth')")
    required_env_vars: List[str] = Field(..., description="List of required environment variables")
    rate_limits: str = Field(..., description="Rate limit information")


class ToolSpecification(BaseModel):
    """Complete specification for a MCP server tool."""
    tool_name: str = Field(..., description="Clear, descriptive tool name")
    description: str = Field(..., description="What this tool does and why it's needed")
    required_unit_functions: List[UnitFunction] = Field(..., description="List of modular functions with single responsibilities")
    external_apis_needed: List[ExternalAPI] = Field(..., description="List of external APIs required")
    environment_variables: List[str] = Field(..., description="List of all required environment variables")
    expected_workflow: str = Field(..., description="Step-by-step description of how the tool works")

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "tool_name": "github_pr_reviewer",
                "description": "A tool to assist in reviewing GitHub Pull Requests by fetching details, files, and allowing users to add comments or submit reviews.",
                "required_unit_functions": [
                    {
                        "name": "get_pr_details",
                        "description": "Fetches comprehensive details of a specific Pull Request",
                        "inputs": ["owner", "repo", "pull_number"],
                        "outputs": "JSON object containing PR details",
                        "external_dependencies": ["github_api"]
                    }
                ],
                "external_apis_needed": [
                    {
                        "api_name": "github_api",
                        "base_url": "https://api.github.com",
                        "auth_type": "bearer_token",
                        "required_env_vars": ["GITHUB_TOKEN"],
                        "rate_limits": "5000/hour (authenticated users)"
                    }
                ],
                "environment_variables": ["GITHUB_TOKEN"],
                "expected_workflow": "1. User provides repository details. 2. Tool fetches PR information. 3. User reviews and comments."
            }
        }