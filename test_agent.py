#!/usr/bin/env python3
"""
Test script for the unit-function-agent with example GitHub PR specifications.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'unit-function-agent'))

from agent import process_stage1_output, generate_unit_function, validate_function_specification
from example_specs import specs

def test_stage1_processing():
    """Test processing of Stage 1 output format."""
    print("=" * 60)
    print("TESTING STAGE 1 OUTPUT PROCESSING")
    print("=" * 60)
    
    # Test with the example GitHub PR specs
    result = process_stage1_output(specs)
    
    print(f"Status: {result.get('status')}")
    if result.get('status') == 'error':
        print(f"Error: {result.get('error_message')}")
        return False
    
    print(f"Tool Name: {result.get('tool_name')}")
    print(f"Function Count: {result.get('function_count')}")
    print(f"Generated Functions: {list(result.get('unit_functions', {}).keys())}")
    
    print("\nCombined Imports:")
    print("-" * 30)
    print(result.get('combined_imports', ''))
    
    print("\nGenerated Functions:")
    print("-" * 30)
    
    for func_name, func_data in result.get('unit_functions', {}).items():
        print(f"\n--- {func_name} ---")
        print(func_data.get('code', ''))
        print()
    
    return True

def test_individual_function_generation():
    """Test individual function generation."""
    print("=" * 60)
    print("TESTING INDIVIDUAL FUNCTION GENERATION")
    print("=" * 60)
    
    # Test specification for a GitHub function
    test_spec = {
        "function_name": "get_current_username",
        "description": "Get the current authenticated user's username from GitHub",
        "parameters": [
            {
                "name": "token",
                "type": "str",
                "description": "GitHub API token",
                "optional": False
            }
        ],
        "return_type": "Dict[str, Any]",
        "external_apis": [{"github": {"base_url": "https://api.github.com"}}],
        "authentication": {"type": "api_key", "env_var": "GITHUB_TOKEN"},
        "error_handling": ["requests.exceptions.RequestException", "ValueError"]
    }
    
    # Validate the specification first
    validation_result = validate_function_specification(test_spec)
    print(f"Validation Status: {validation_result.get('status')}")
    
    if validation_result.get('errors'):
        print(f"Validation Errors: {validation_result.get('errors')}")
        return False
    
    if validation_result.get('warnings'):
        print(f"Validation Warnings: {validation_result.get('warnings')}")
    
    # Generate the function
    result = generate_unit_function(test_spec)
    
    print(f"\nGeneration Status: {result.get('status')}")
    if result.get('status') == 'error':
        print(f"Error: {result.get('error_message')}")
        return False
    
    print("\nGenerated Imports:")
    print("-" * 20)
    print(result.get('imports', ''))
    
    print("\nGenerated Function Code:")
    print("-" * 20)
    print(result.get('function_code', ''))
    
    return True

def main():
    """Run all tests."""
    print("Unit Function Agent Test Suite")
    print("==============================")
    
    # Test Stage 1 processing
    stage1_success = test_stage1_processing()
    
    print("\n")
    
    # Test individual function generation
    individual_success = test_individual_function_generation()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Stage 1 Processing: {'✅ PASS' if stage1_success else '❌ FAIL'}")
    print(f"Individual Generation: {'✅ PASS' if individual_success else '❌ FAIL'}")
    
    if stage1_success and individual_success:
        print("\n🎉 All tests passed! Agent is compatible with Stage 1 output format.")
        return 0
    else:
        print("\n❌ Some tests failed. Agent needs fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())