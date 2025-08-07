"""
Unit Function Agent for TOD (Tool On Demand) Project
Stage 2: Function Generation - Harvey & James's Team

This agent generates individual Python functions based on structured specifications
from Stage 1 (Request Processing). Generated functions include proper error handling,
authentication, type hints, and comprehensive documentation.
"""

import ast
import inspect
import json
import re
from typing import Dict, List, Any, Optional, Union
from google.adk.agents import Agent


def process_stage1_output(stage1_specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes Stage 1 output and generates all required unit functions.
    
    Args:
        stage1_specs (Dict[str, Any]): Stage 1 specifications containing:
            - parsed_request: Dict with tool_name, required_unit_functions, external_apis_needed
            - full_specs: String with detailed function specifications
    
    Returns:
        Dict[str, Any]: Contains all generated unit functions with their code and imports
    """
    try:
        parsed_request = stage1_specs.get("parsed_request", {})
        full_specs = stage1_specs.get("full_specs", "")
        
        required_functions = parsed_request.get("required_unit_functions", [])
        external_apis = parsed_request.get("external_apis_needed", [])
        
        if not required_functions:
            return {
                "status": "error",
                "error_message": "No required_unit_functions found in Stage 1 output"
            }
        
        # Parse individual function specs from full_specs
        function_specs = _parse_function_specs_from_full_specs(full_specs, required_functions, external_apis)
        
        generated_functions = {}
        all_imports = set()
        
        # Generate each required function
        for func_name in required_functions:
            if func_name not in function_specs:
                return {
                    "status": "error",
                    "error_message": f"Function specification not found for: {func_name}"
                }
            
            result = generate_unit_function(function_specs[func_name])
            
            if result.get("status") != "success":
                return result
            
            generated_functions[func_name] = {
                "code": result["function_code"],
                "imports": result["imports"]
            }
            
            # Collect all unique imports
            for import_line in result["imports"].split("\n"):
                if import_line.strip():
                    all_imports.add(import_line.strip())
        
        return {
            "status": "success",
            "tool_name": parsed_request.get("tool_name", "generated_tool"),
            "unit_functions": generated_functions,
            "combined_imports": "\n".join(sorted(all_imports)),
            "function_count": len(generated_functions)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Stage 1 processing failed: {str(e)}"
        }


def _parse_function_specs_from_full_specs(full_specs: str, required_functions: List[str], 
                                        external_apis: List[Dict]) -> Dict[str, Dict]:
    """
    Parses individual function specifications from the full_specs string.
    
    Args:
        full_specs (str): Full specifications text from Stage 1
        required_functions (List[str]): List of function names to generate
        external_apis (List[Dict]): External API configurations
    
    Returns:
        Dict[str, Dict]: Function specifications in the format expected by generate_unit_function
    """
    function_specs = {}
    
    # Extract authentication info from external APIs
    auth_config = {}
    if external_apis:
        for api_dict in external_apis:
            for api_name, api_config in api_dict.items():
                if isinstance(api_config, dict):
                    for key, env_var in api_config.items():
                        if key in ["api_token", "token", "key"]:
                            auth_config = {
                                "type": "api_key",
                                "env_var": env_var
                            }
    
    # Parse each function from the specs
    for func_name in required_functions:
        # Look for function pattern in specs: "1. **function_name(params)**: description"
        pattern = rf'\d+\.\s*\*\*{re.escape(func_name)}\(([^)]*)\)\*\*:\s*([^.]+\.)'
        match = re.search(pattern, full_specs)
        
        if match:
            params_str = match.group(1)
            description = match.group(2).strip()
            
            # Parse parameters
            parameters = _parse_parameters_from_string(params_str)
            
            # Determine return type based on description
            return_type = "Dict[str, Any]"  # Default return type
            if "returns username" in description.lower():
                return_type = "str"
            elif "returns list" in description.lower():
                return_type = "List[Dict[str, Any]]"
            
            function_specs[func_name] = {
                "function_name": func_name,
                "description": description,
                "parameters": parameters,
                "return_type": return_type,
                "external_apis": external_apis,
                "authentication": auth_config,
                "error_handling": ["requests.exceptions.RequestException", "ValueError", "ConnectionError"]
            }
        else:
            # Fallback for functions not found in specs
            function_specs[func_name] = _create_fallback_spec(func_name, external_apis, auth_config)
    
    return function_specs


def _parse_parameters_from_string(params_str: str) -> List[Dict[str, Any]]:
    """Parse parameters from function signature string."""
    parameters = []
    if not params_str.strip():
        return parameters
    
    # Split by comma and parse each parameter
    param_parts = [p.strip() for p in params_str.split(",")]
    
    for param_part in param_parts:
        if not param_part:
            continue
        
        # Simple parameter parsing - assumes format like "token", "username", etc.
        param_name = param_part
        param_type = "str"  # Default type
        
        # Infer type from parameter name
        if "token" in param_name.lower() or "key" in param_name.lower():
            param_type = "str"
        elif "count" in param_name.lower() or "page" in param_name.lower():
            param_type = "int"
        elif "body" in param_name.lower() or "text" in param_name.lower():
            param_type = "str"
        
        parameters.append({
            "name": param_name,
            "type": param_type,
            "description": f"The {param_name} parameter",
            "optional": False
        })
    
    return parameters


def _create_fallback_spec(func_name: str, external_apis: List[Dict], auth_config: Dict) -> Dict[str, Any]:
    """Create a fallback specification for functions not found in specs."""
    return {
        "function_name": func_name,
        "description": f"Generated function for {func_name}",
        "parameters": [
            {"name": "input_data", "type": "str", "description": "Input data for the function", "optional": False}
        ],
        "return_type": "Dict[str, Any]",
        "external_apis": external_apis,
        "authentication": auth_config,
        "error_handling": ["ValueError", "ConnectionError"]
    }


def generate_unit_function(specification: Dict[str, Any]) -> Dict[str, str]:
    """
    Generates a Python function based on the provided specification.
    
    Args:
        specification (Dict[str, Any]): Function specification containing:
            - function_name (str): Name of the function to generate
            - description (str): Description of what the function does
            - parameters (List[Dict]): List of parameter specifications
            - return_type (str): Expected return type
            - external_apis (List[Dict], optional): External APIs the function should use
            - authentication (Dict, optional): Authentication requirements
            - error_handling (List[str], optional): Specific error types to handle
    
    Returns:
        Dict[str, str]: Contains:
            - status: "success" or "error"
            - function_code: Generated function code (if successful)
            - imports: Required imports for the function
            - error_message: Error description (if failed)
    """
    try:
        # Validate required fields
        required_fields = ["function_name", "description", "parameters", "return_type"]
        for field in required_fields:
            if field not in specification:
                return {
                    "status": "error",
                    "error_message": f"Missing required field: {field}"
                }
        
        # Extract specification details
        func_name = specification["function_name"]
        description = specification["description"]
        parameters = specification["parameters"]
        return_type = specification["return_type"]
        external_apis = specification.get("external_apis", [])
        authentication = specification.get("authentication", {})
        error_handling = specification.get("error_handling", ["ValueError", "ConnectionError", "TimeoutError"])
        
        # Validate function name
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', func_name):
            return {
                "status": "error",
                "error_message": f"Invalid function name: {func_name}"
            }
        
        # Generate imports
        imports = _generate_imports(external_apis, authentication)
        
        # Generate function signature
        signature = _generate_function_signature(func_name, parameters, return_type)
        
        # Generate docstring
        docstring = _generate_docstring(description, parameters, return_type)
        
        # Generate function body
        function_body = _generate_function_body(
            parameters, external_apis, authentication, error_handling, return_type
        )
        
        # Combine all parts
        function_code = f'''{signature}
    """{docstring}"""
{function_body}'''
        
        # Validate generated code syntax
        try:
            ast.parse(imports + "\n\n" + function_code)
        except SyntaxError as e:
            return {
                "status": "error",
                "error_message": f"Generated code has syntax errors: {str(e)}"
            }
        
        return {
            "status": "success",
            "function_code": function_code,
            "imports": imports
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Function generation failed: {str(e)}"
        }


def validate_function_specification(specification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates a function specification for completeness and correctness.
    
    Args:
        specification (Dict[str, Any]): Function specification to validate
    
    Returns:
        Dict[str, Any]: Validation result with status and details
    """
    try:
        validation_errors = []
        warnings = []
        
        # Required field validation
        required_fields = ["function_name", "description", "parameters", "return_type"]
        for field in required_fields:
            if field not in specification:
                validation_errors.append(f"Missing required field: {field}")
        
        if validation_errors:
            return {
                "status": "invalid",
                "errors": validation_errors,
                "warnings": warnings
            }
        
        # Function name validation
        func_name = specification["function_name"]
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', func_name):
            validation_errors.append(f"Invalid function name: {func_name}")
        
        # Parameters validation
        parameters = specification["parameters"]
        if not isinstance(parameters, list):
            validation_errors.append("Parameters must be a list")
        else:
            for i, param in enumerate(parameters):
                if not isinstance(param, dict):
                    validation_errors.append(f"Parameter {i} must be a dictionary")
                    continue
                
                if "name" not in param:
                    validation_errors.append(f"Parameter {i} missing 'name' field")
                if "type" not in param:
                    validation_errors.append(f"Parameter {i} missing 'type' field")
                if "description" not in param:
                    warnings.append(f"Parameter {i} missing description")
        
        # API specification validation
        external_apis = specification.get("external_apis", [])
        for i, api in enumerate(external_apis):
            if not isinstance(api, dict):
                validation_errors.append(f"External API {i} must be a dictionary")
                continue
            
            if not api:
                warnings.append(f"External API {i} is empty")
        
        status = "invalid" if validation_errors else ("valid" if not warnings else "valid_with_warnings")
        
        return {
            "status": status,
            "errors": validation_errors,
            "warnings": warnings
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Validation failed: {str(e)}"
        }


def _generate_imports(external_apis: List[Dict], authentication: Dict) -> str:
    """Generate required imports based on external APIs and authentication."""
    imports = ["import json", "import os", "from typing import Dict, Any, Optional"]
    
    # Add requests if external APIs are used
    if external_apis:
        imports.append("import requests")
    
    # Add authentication-specific imports
    auth_type = authentication.get("type", "")
    if auth_type in ["oauth2", "jwt"]:
        imports.append("import base64")
        imports.append("from datetime import datetime, timedelta")
    
    return "\n".join(imports)


def _generate_function_signature(func_name: str, parameters: List[Dict], return_type: str) -> str:
    """Generate function signature with type hints."""
    param_strings = []
    
    for param in parameters:
        param_name = param["name"]
        param_type = param["type"]
        is_optional = param.get("optional", False)
        default_value = param.get("default")
        
        type_hint = param_type
        if is_optional:
            type_hint = f"Optional[{param_type}]"
        
        if default_value is not None:
            if isinstance(default_value, str):
                param_strings.append(f'{param_name}: {type_hint} = "{default_value}"')
            else:
                param_strings.append(f'{param_name}: {type_hint} = {default_value}')
        elif is_optional:
            param_strings.append(f'{param_name}: {type_hint} = None')
        else:
            param_strings.append(f'{param_name}: {type_hint}')
    
    params_str = ", ".join(param_strings)
    return f"def {func_name}({params_str}) -> {return_type}:"


def _generate_docstring(description: str, parameters: List[Dict], return_type: str) -> str:
    """Generate comprehensive docstring."""
    docstring_parts = [description, ""]
    
    if parameters:
        docstring_parts.append("Args:")
        for param in parameters:
            param_name = param["name"]
            param_type = param["type"]
            param_desc = param.get("description", "No description provided")
            optional_str = " (optional)" if param.get("optional", False) else ""
            docstring_parts.append(f"        {param_name} ({param_type}){optional_str}: {param_desc}")
        docstring_parts.append("")
    
    docstring_parts.extend([
        "Returns:",
        f"        {return_type}: Function result with status and data or error information"
    ])
    
    return "\n    ".join(docstring_parts)


def _generate_function_body(parameters: List[Dict], external_apis: List[Dict], 
                          authentication: Dict, error_handling: List[str], 
                          return_type: str) -> str:
    """Generate function body with error handling and API calls."""
    body_parts = []
    indent = "    "
    
    # Input validation
    body_parts.append(f"{indent}try:")
    body_parts.append(f"{indent*2}# Input validation")
    
    for param in parameters:
        param_name = param["name"]
        param_type = param["type"]
        is_optional = param.get("optional", False)
        
        if not is_optional:
            if param_type == "str":
                body_parts.append(f"{indent*2}if not {param_name} or not isinstance({param_name}, str):")
                body_parts.append(f"{indent*3}return {{'status': 'error', 'error_message': '{param_name} must be a non-empty string'}}")
            elif param_type == "int":
                body_parts.append(f"{indent*2}if not isinstance({param_name}, int):")
                body_parts.append(f"{indent*3}return {{'status': 'error', 'error_message': '{param_name} must be an integer'}}")
    
    body_parts.append("")
    
    # Authentication setup
    if authentication and external_apis:
        body_parts.append(f"{indent*2}# Authentication setup")
        auth_type = authentication.get("type", "")
        if auth_type == "api_key":
            key_env = authentication.get("env_var", "API_KEY")
            body_parts.append(f"{indent*2}api_key = os.getenv('{key_env}')")
            body_parts.append(f"{indent*2}if not api_key:")
            body_parts.append(f"{indent*3}return {{'status': 'error', 'error_message': 'API key not found in environment'}}")
            body_parts.append(f"{indent*2}headers = {{'Authorization': f'Bearer {{api_key}}'}}")
        elif auth_type == "basic":
            body_parts.append(f"{indent*2}username = os.getenv('API_USERNAME')")
            body_parts.append(f"{indent*2}password = os.getenv('API_PASSWORD')")
            body_parts.append(f"{indent*2}if not username or not password:")
            body_parts.append(f"{indent*3}return {{'status': 'error', 'error_message': 'Username/password not found in environment'}}")
            body_parts.append(f"{indent*2}auth = (username, password)")
        body_parts.append("")
    
    # API calls
    if external_apis:
        body_parts.append(f"{indent*2}# API calls")
        for i, api in enumerate(external_apis):
            api_name = list(api.keys())[0] if api else f"api_{i}"
            api_config = api.get(api_name, {})
            base_url = api_config.get("base_url", "https://api.example.com")
            
            body_parts.append(f"{indent*2}# {api_name.title()} API call")
            body_parts.append(f"{indent*2}url = f'{base_url}/endpoint'")
            
            if authentication:
                if authentication.get("type") == "api_key":
                    body_parts.append(f"{indent*2}response = requests.get(url, headers=headers, timeout=30)")
                elif authentication.get("type") == "basic":
                    body_parts.append(f"{indent*2}response = requests.get(url, auth=auth, timeout=30)")
                else:
                    body_parts.append(f"{indent*2}response = requests.get(url, timeout=30)")
            else:
                body_parts.append(f"{indent*2}response = requests.get(url, timeout=30)")
            
            body_parts.append(f"{indent*2}response.raise_for_status()")
            body_parts.append(f"{indent*2}data = response.json()")
            body_parts.append("")
    else:
        # Mock implementation for functions without external APIs
        body_parts.append(f"{indent*2}# Mock implementation - replace with actual logic")
        body_parts.append(f"{indent*2}result = {{'message': 'Function executed successfully'}}")
        body_parts.append("")
    
    # Return success
    if external_apis:
        body_parts.append(f"{indent*2}return {{'status': 'success', 'data': data}}")
    else:
        body_parts.append(f"{indent*2}return {{'status': 'success', 'data': result}}")
    
    body_parts.append("")
    
    # Error handling
    error_handlers = []
    if "requests.exceptions.RequestException" not in error_handling and external_apis:
        error_handling.append("requests.exceptions.RequestException")
    
    for error_type in error_handling:
        if error_type == "requests.exceptions.RequestException":
            error_handlers.append(f"{indent}except requests.exceptions.RequestException as e:")
            error_handlers.append(f"{indent*2}return {{'status': 'error', 'error_message': f'API request failed: {{str(e)}}'}}")
        elif error_type == "ValueError":
            error_handlers.append(f"{indent}except ValueError as e:")
            error_handlers.append(f"{indent*2}return {{'status': 'error', 'error_message': f'Invalid input: {{str(e)}}'}}")
        elif error_type == "ConnectionError":
            error_handlers.append(f"{indent}except ConnectionError as e:")
            error_handlers.append(f"{indent*2}return {{'status': 'error', 'error_message': f'Connection failed: {{str(e)}}'}}")
        else:
            error_handlers.append(f"{indent}except {error_type} as e:")
            error_handlers.append(f"{indent*2}return {{'status': 'error', 'error_message': f'{error_type}: {{str(e)}}'}}")
    
    # General exception handler
    error_handlers.extend([
        f"{indent}except Exception as e:",
        f"{indent*2}return {{'status': 'error', 'error_message': f'Unexpected error: {{str(e)}}'}}"
    ])
    
    body_parts.extend(error_handlers)
    
    return "\n".join(body_parts)


# Create the Unit Function Agent
unit_function_agent = Agent(
    name="unit_function_generator",
    model="gemini-2.0-flash", 
    description=(
        "Agent specialized in generating individual Python functions based on structured specifications. "
        "Part of TOD (Tool On Demand) Stage 2: Function Generation pipeline. Processes Stage 1 output "
        "and generates all required unit functions for tool assembly."
    ),
    instruction=(
        "You are a specialized function generation agent that creates production-ready Python functions "
        "based on detailed specifications from Stage 1 (Request Processing). You can process complete "
        "Stage 1 output to generate all required unit functions, or generate individual functions. "
        "Generate functions with proper error handling, type hints, comprehensive docstrings, and support "
        "for external API integrations. Always follow Python best practices and ensure generated code "
        "is secure and maintainable."
    ),
    tools=[process_stage1_output, generate_unit_function, validate_function_specification]
)