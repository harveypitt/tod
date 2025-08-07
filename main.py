#!/usr/bin/env python3
"""
TOD (Tool On Demand) - Stage 2: Function Generation
Main entry point for the unit-function-agent that bypasses ADK YAML configuration issues.
"""

import sys
import os
import json

# Add the unit-function-agent to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'unit-function-agent'))

from agent import process_stage1_output, generate_unit_function, validate_function_specification, unit_function_agent
from example_specs import specs


def run_agent_directly():
    """Run the agent directly without ADK web interface."""
    print("=" * 60)
    print("TOD Stage 2: Unit Function Generator")
    print("=" * 60)
    
    # Example: Process Stage 1 output
    print("Processing example Stage 1 specifications...")
    result = process_stage1_output(specs)
    
    if result.get('status') == 'success':
        print(f"✅ Successfully generated {result.get('function_count')} functions")
        print(f"Tool: {result.get('tool_name')}")
        
        print("\nGenerated Functions:")
        for func_name in result.get('unit_functions', {}):
            print(f"  - {func_name}")
        
        print(f"\nCombined imports needed:")
        print(result.get('combined_imports', ''))
        
        return result
    else:
        print(f"❌ Error: {result.get('error_message')}")
        return None


def interactive_mode():
    """Interactive mode for testing individual function generation."""
    print("\n" + "=" * 60)
    print("Interactive Function Generator")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Process Stage 1 output")
        print("2. Generate individual function")
        print("3. Validate function specification")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            result = process_stage1_output(specs)
            print(json.dumps(result, indent=2))
        
        elif choice == "2":
            print("\nEnter function specification as JSON:")
            try:
                spec_input = input("Specification: ")
                spec = json.loads(spec_input)
                result = generate_unit_function(spec)
                print(json.dumps(result, indent=2))
            except json.JSONDecodeError:
                print("❌ Invalid JSON format")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        elif choice == "3":
            print("\nEnter function specification to validate as JSON:")
            try:
                spec_input = input("Specification: ")
                spec = json.loads(spec_input)
                result = validate_function_specification(spec)
                print(json.dumps(result, indent=2))
            except json.JSONDecodeError:
                print("❌ Invalid JSON format")
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1-4.")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        # Run the example processing
        result = run_agent_directly()
        
        if result:
            # Save output for Stage 3
            output_file = "stage2_output.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Output saved to {output_file} for Stage 3 processing")


if __name__ == "__main__":
    main()
