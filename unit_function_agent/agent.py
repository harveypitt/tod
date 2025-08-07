import os
import tempfile
import uuid
from datetime import datetime
from typing import Dict, List
import boto3
from botocore.exceptions import NoCredentialsError, BotoCoreError
from pydantic import BaseModel
from google.adk.agents import Agent

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file if it exists"""
    env_files = ['.env', '../.env']
    for env_file in env_files:
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
            break

load_env()


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



def upload_to_storage(file_path: str, filename: str) -> dict:
    """Upload file to Tigris storage on Fly.io.
    
    Args:
        file_path (str): Local path to the file
        filename (str): Name for the file in storage
        
    Returns:
        dict: Status and download URL or error message
    """
    try:
        # Tigris configuration from environment variables
        endpoint_url = os.getenv('AWS_ENDPOINT_URL_S3', 'https://fly.storage.tigris.dev')
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        bucket_name = os.getenv('BUCKET_NAME', 'tod-files')
        region = os.getenv('AWS_REGION', 'auto')
        
        # Create S3 client configured for Tigris
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        
        # Upload file without ACL (Tigris might not support ACLs)
        s3_client.upload_file(file_path, bucket_name, unique_filename)
        
        # Generate presigned URL for download (valid for 24 hours)
        download_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': unique_filename},
            ExpiresIn=86400  # 24 hours
        )
        
        return {
            "status": "success",
            "message": "File uploaded successfully",
            "download_url": download_url,
            "filename": unique_filename
        }
        
    except NoCredentialsError:
        return {
            "status": "error",
            "error_message": "Storage credentials not found. Check environment variables."
        }
    except BotoCoreError as e:
        return {
            "status": "error",
            "error_message": f"Storage connection error: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to upload file: {str(e)}"
        }


def generate_unit_function(function_name: str, description: str) -> dict:
    """Generate a Python unit function and upload it to storage.

    Args:
        function_name (str): The name of the function to generate
        description (str): Description of what the function should do

    Returns:
        dict: Status and download URL of the generated function, or error message
    """
    try:
        # Create temp directory if it doesn't exist
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, f"{function_name}.py")

        # Generate basic function template based on description
        function_code = f'''"""Generated unit function: {function_name}
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


if __name__ == "__main__":
    # Example usage
    result = {function_name}()
    print(result)
'''

        # Write the function to file
        with open(file_path, 'w') as f:
            f.write(function_code)

        # Upload to storage
        upload_result = upload_to_storage(file_path, f"{function_name}.py")
        
        # Clean up local file
        try:
            os.remove(file_path)
        except:
            pass  # Ignore cleanup errors
            
        if upload_result["status"] == "success":
            return {
                "status": "success",
                "message": f"Unit function '{function_name}' generated and uploaded successfully",
                "download_url": upload_result["download_url"],
                "filename": upload_result["filename"]
            }
        else:
            return upload_result

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to generate unit function: {str(e)}"
        }


root_agent = Agent(
    name="unit_function_agent",
    model="gemini-2.0-flash",
    description="Agent to generate unit functions and upload them to storage",
    instruction="You are a helpful agent that generates Python unit functions based on provided specifications and uploads them to cloud storage for download.",
    tools=[generate_unit_function, upload_to_storage],
)