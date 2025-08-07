import os
import sys
import json
import time
import requests
import tempfile
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel
import boto3
from botocore.exceptions import NoCredentialsError, BotoCoreError
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


@dataclass
class FirecrawlQuery:
    """Structured Firecrawl query with all necessary parameters"""
    url: str
    endpoint: str = "scrape"
    formats: List[str] = None
    extract_prompt: Optional[str] = None
    max_pages: Optional[int] = None
    include_paths: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.formats is None:
            self.formats = ["markdown", "links"]


class QueryConverter:
    """Convert natural language queries to Firecrawl-friendly format using Perplexity API"""
    
    def __init__(self, perplexity_api_key: str):
        self.api_key = perplexity_api_key
        self.api_url = "https://api.perplexity.ai/chat/completions"
        
    def _create_prompt(self, user_query: str) -> str:
        """Create a prompt for Perplexity to analyze the user query"""
        return f"""Find the EXACT main documentation URLs for the technologies mentioned in this query:

User Query: "{user_query}"

CRITICAL RULES:
1. Be LITERAL - Find documentation for the EXACT technologies mentioned, don't interpret or add words
2. Find MAIN documentation pages - not sub-pages or specific sections
3. Search for the PRIMARY official documentation URL for each technology
4. If user says "Firebase auth doc" - find Firebase Authentication docs
5. If user says "React Native doc" - find React Native main documentation
6. Parse conjunctions (and, &, also, plus) to identify SEPARATE technology requests

For each distinct technology mentioned:
- Search the web to find its main official documentation URL
- Use the root documentation page, not sub-sections
- Prefer /docs over /docs/getting-started or other sub-pages

RESPONSE FORMAT:
Return a JSON array where each object has:
- url: Main official documentation URL (search to find current URL)
- endpoint: "crawl" (for documentation crawling)  
- formats: ["markdown", "links"]
- max_pages: 10
- explanation: Just the technology name (e.g., "Firebase Authentication", "React Native")

Example for "Firebase auth doc and React Native doc":
[
  {{
    "url": "https://firebase.google.com/docs/auth",
    "endpoint": "crawl",
    "formats": ["markdown", "links"],
    "max_pages": 10,
    "explanation": "Firebase Authentication"
  }},
  {{
    "url": "https://reactnative.dev/docs",
    "endpoint": "crawl", 
    "formats": ["markdown", "links"],
    "max_pages": 10,
    "explanation": "React Native"
  }}
]

IMPORTANT: 
- Search for current URLs, don't guess
- Use main /docs pages when possible
- Keep explanations simple (just the technology name)
- Be exact - don't add "Official" or other words the user didn't say"""

    def convert_query(self, user_query: str) -> List[FirecrawlQuery]:
        """Convert user query to Firecrawl queries using Perplexity API. Returns a list of queries."""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "sonar",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at converting natural language queries into structured API parameters for web scraping with Firecrawl. Always return valid JSON."
                },
                {
                    "role": "user",
                    "content": self._create_prompt(user_query)
                }
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Extract JSON from the response
            array_start = content.find('[')
            array_end = content.rfind(']') + 1
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            queries = []
            
            # Check if response is an array
            if array_start != -1 and array_end > array_start and (array_start < json_start or json_start == -1):
                json_str = content[array_start:array_end]
                parsed_array = json.loads(json_str)
                
                # Create FirecrawlQuery objects for each item
                for parsed in parsed_array:
                    query = FirecrawlQuery(
                        url=parsed.get('url', ''),
                        endpoint=parsed.get('endpoint', 'scrape'),
                        formats=parsed.get('formats', ['markdown', 'links']),
                        extract_prompt=parsed.get('extract_prompt'),
                        max_pages=parsed.get('max_pages'),
                        include_paths=parsed.get('include_paths'),
                        exclude_paths=parsed.get('exclude_paths')
                    )
                    queries.append(query)
                
            # Otherwise, try single object
            elif json_start != -1 and json_end > json_start:
                json_str = content[json_start:json_end]
                parsed = json.loads(json_str)
                
                query = FirecrawlQuery(
                    url=parsed.get('url', ''),
                    endpoint=parsed.get('endpoint', 'scrape'),
                    formats=parsed.get('formats', ['markdown', 'links']),
                    extract_prompt=parsed.get('extract_prompt'),
                    max_pages=parsed.get('max_pages'),
                    include_paths=parsed.get('include_paths'),
                    exclude_paths=parsed.get('exclude_paths')
                )
                queries.append(query)
            else:
                raise ValueError("Could not parse JSON from Perplexity response")
            
            return queries
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling Perplexity API: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Error parsing JSON response: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")


class FirecrawlClient:
    """Client for interacting with Firecrawl API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FIRECRAWL_API_KEY')
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment variables")
        
        self.base_url = "https://api.firecrawl.dev/v1"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def transform_query_to_firecrawl_format(self, query: FirecrawlQuery) -> Dict:
        """Transform our FirecrawlQuery to Firecrawl API format"""
        
        payload = {
            'url': query.url
        }
        
        if query.endpoint == 'crawl':
            if query.max_pages:
                payload['limit'] = query.max_pages
            
            if query.formats:
                payload['scrapeOptions'] = {
                    'formats': query.formats
                }
            
            if query.include_paths:
                payload['includePaths'] = query.include_paths
            if query.exclude_paths:
                payload['excludePaths'] = query.exclude_paths
                
        elif query.endpoint == 'scrape':
            if query.formats:
                payload['formats'] = query.formats
        
        return payload
    
    def initiate_crawl(self, query: FirecrawlQuery) -> Dict:
        """Initiate a crawl job and return job info"""
        
        payload = self.transform_query_to_firecrawl_format(query)
        endpoint = f"{self.base_url}/{query.endpoint}"
        
        try:
            response = requests.post(endpoint, headers=self.headers, json=payload)
            
            # Handle rate limit errors specifically
            if response.status_code == 429:
                time.sleep(60)
                # Retry once after rate limit
                response = requests.post(endpoint, headers=self.headers, json=payload)
            
            response.raise_for_status()
            result = response.json()
            
            if query.endpoint == 'crawl':
                return {
                    'success': result.get('success', False),
                    'job_id': result.get('id'),
                    'status_url': result.get('url'),
                    'type': 'crawl',
                    'source_url': query.url
                }
            else:
                return {
                    'success': True,
                    'data': result.get('data', {}),
                    'type': 'scrape',
                    'source_url': query.url
                }
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                error_msg = "Rate limit exceeded. Try again later or upgrade your Firecrawl plan."
                return {
                    'success': False,
                    'error': error_msg,
                    'status_code': 429,
                    'source_url': query.url
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {e.response.status_code}: {str(e)}",
                    'source_url': query.url
                }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'source_url': query.url
            }
    
    def try_llms_txt(self, base_url: str) -> Dict:
        """Try to scrape /llms.txt file first as it's often curated for LLMs"""
        
        # Construct /llms.txt URL
        if base_url.endswith('/'):
            llms_url = f"{base_url}llms.txt"
        else:
            llms_url = f"{base_url}/llms.txt"
        
        # Create a scrape query for the llms.txt file
        llms_query = FirecrawlQuery(
            url=llms_url,
            endpoint="scrape",
            formats=["markdown"]
        )
        
        try:
            result = self.initiate_crawl(llms_query)
            
            if result.get('success') and result.get('data'):
                data = result.get('data', {})
                content = data.get('markdown', '')
                
                if content and len(content.strip()) > 50:  # Ensure meaningful content
                    return {
                        'success': True,
                        'data': [{'markdown': content, 'url': llms_url}],
                        'source_url': base_url,
                        'type': 'llms_txt'
                    }
                else:
                    return {'success': False, 'reason': 'content_too_short'}
            else:
                return {'success': False, 'reason': 'file_not_found'}
                
        except Exception as e:
            return {'success': False, 'reason': 'error', 'error': str(e)}
    
    def check_crawl_status(self, job_id: str) -> Dict:
        """Check the status of a crawl job"""
        
        endpoint = f"{self.base_url}/crawl/{job_id}"
        
        try:
            response = requests.get(endpoint, headers=self.headers)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def wait_for_crawl_completion(self, job_id: str, max_wait: int = 300, poll_interval: int = 5) -> Dict:
        """Wait for a crawl job to complete and return results"""
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = self.check_crawl_status(job_id)
            status = status_response.get('status', 'unknown')
            
            if status == 'completed':
                return {
                    'success': True,
                    'data': status_response.get('data', []),
                    'total': status_response.get('total', 0),
                    'completed': status_response.get('completed', 0)
                }
            
            elif status in ['failed', 'error']:
                return {
                    'success': False,
                    'error': status_response.get('error', 'Unknown error'),
                    'status': status
                }
            
            else:
                time.sleep(poll_interval)
        
        return {
            'success': False,
            'error': f'Crawl job timed out after {max_wait} seconds',
            'status': 'timeout'
        }
    
    def crawl_multiple(self, queries: List[FirecrawlQuery]) -> List[Dict]:
        """Crawl multiple URLs sequentially to respect API rate limits"""
        
        results = []
        
        # Smart rate limiting based on endpoint type
        scrape_delay = 7   # 7 seconds for scrape (Free plan: 10/min = 6s + buffer)
        crawl_delay = 65   # 65 seconds for crawl (Free plan: 1/min + buffer)
        
        last_request_type = None
        
        for i, query in enumerate(queries, 1):
            try:
                # Step 1: Try to get curated /llms.txt content first (uses scrape)
                llms_result = self.try_llms_txt(query.url)
                current_request_type = 'scrape'
                
                if llms_result.get('success'):
                    # Found /llms.txt content - use it directly
                    
                    # Apply scrape rate limiting if needed
                    if i > 1 and last_request_type == 'scrape':
                        time.sleep(scrape_delay)
                    
                    results.append(llms_result)
                    last_request_type = current_request_type
                    continue
                
                # Step 2: Fallback to regular crawl if /llms.txt failed
                current_request_type = 'crawl'
                
                # Apply appropriate rate limiting based on last request type
                if i > 1:
                    if last_request_type == 'crawl':
                        # crawl -> crawl: need full crawl delay
                        delay = crawl_delay
                    elif last_request_type == 'scrape':
                        # scrape -> crawl: need shorter delay
                        delay = max(scrape_delay, 10)  # at least 10s between different endpoints
                    
                    time.sleep(delay)
                
                job_info = self.initiate_crawl(query)
                
                if job_info.get('success') and job_info.get('type') == 'crawl':
                    # Wait for this crawl to complete before starting the next
                    crawl_result = self.wait_for_crawl_completion(job_info['job_id'])
                    crawl_result['source_url'] = job_info['source_url']
                    results.append(crawl_result)
                    
                elif job_info.get('type') == 'scrape':
                    # Scrape results are immediate
                    results.append(job_info)
                    current_request_type = 'scrape'
                else:
                    # Failed to initiate
                    results.append(job_info)
                
                last_request_type = current_request_type
                    
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e),
                    'source_url': query.url
                })
        
        return results
    
    def extract_markdown_content(self, crawl_results: List[Dict]) -> Dict[str, str]:
        """Extract markdown content from crawl results"""
        
        content_by_source = {}
        
        for result in crawl_results:
            source_url = result.get('source_url', 'Unknown')
            
            if not result.get('success'):
                content_by_source[source_url] = f"# Error\n\nFailed to crawl: {result.get('error', 'Unknown error')}\n"
                continue
            
            markdown_parts = []
            data = result.get('data', [])
            
            if isinstance(data, list):
                for page in data:
                    if isinstance(page, dict):
                        markdown = page.get('markdown', '')
                        if markdown:
                            page_url = page.get('url', '')
                            if page_url:
                                markdown_parts.append(f"## Source: {page_url}\n\n{markdown}")
                            else:
                                markdown_parts.append(markdown)
            
            content_by_source[source_url] = "\n\n---\n\n".join(markdown_parts) if markdown_parts else "No content found"
        
        return content_by_source


class MarkdownAggregator:
    """Aggregates multiple documentation sources into a single unified markdown file"""
    
    def generate_metadata(self, query: str, sources_count: int) -> str:
        """Generate metadata header"""
        
        metadata = f"""# Documentation Aggregate

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Query:** "{query}"  
**Sources:** {sources_count} documentation sources  

---
"""
        return metadata
    
    def aggregate(self, content_by_source: Dict[str, str], query: str) -> str:
        """Aggregate all content into a single unified markdown document"""
        
        # Generate metadata header
        metadata = self.generate_metadata(query, len(content_by_source))
        
        # Create sources list for reference
        sources_list = "\n## Sources\n\n"
        for i, source_url in enumerate(content_by_source.keys(), 1):
            title = self.extract_title_from_url(source_url)
            sources_list += f"{i}. **{title}**: [{source_url}]({source_url})\n"
        
        # Combine all content into one unified document
        combined_content = "\n## Documentation Content\n\n"
        
        for source_url, content in content_by_source.items():
            if content and content.strip() and "No content found" not in content:
                # Clean up the content - remove redundant headers
                cleaned_content = self.clean_content_for_unification(content, source_url)
                combined_content += cleaned_content + "\n\n"
        
        # Build final unified document
        document_parts = [
            metadata,
            sources_list,
            "\n---\n",
            combined_content
        ]
        
        return "\n".join(document_parts)
    
    def clean_content_for_unification(self, content: str, source_url: str) -> str:
        """Clean and prepare content for unified document"""
        
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip redundant "Source:" lines since we have sources section
            if line.startswith('## Source:'):
                continue
            
            # Skip excessive separators
            if line.strip() == '---':
                continue
                
            # Keep the content
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def extract_title_from_url(self, url: str) -> str:
        """Extract a readable title from URL"""
        
        if 'firebase.google.com/docs/auth' in url:
            return "Firebase Authentication Documentation"
        elif 'reactnative.dev' in url:
            return "React Native Documentation"
        elif 'nextjs.org/docs' in url:
            return "Next.js Documentation"
        elif 'react.dev' in url:
            return "React Documentation"
        elif 'vuejs.org' in url:
            return "Vue.js Documentation"
        elif 'angular.io' in url:
            return "Angular Documentation"
        elif 'nodejs.org' in url:
            return "Node.js Documentation"
        elif 'mongodb.com/docs' in url:
            return "MongoDB Documentation"
        else:
            parts = url.replace('https://', '').replace('http://', '').split('/')
            domain = parts[0].replace('www.', '')
            name = domain.split('.')[0].title()
            
            if 'docs' in url.lower():
                return f"{name} Documentation"
            else:
                return name


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


def retrieve_documentation(query: str) -> dict:
    """Retrieve documentation based on a natural language query and return a markdown file.

    Args:
        query (str): Natural language query describing what documentation to retrieve

    Returns:
        dict: Status and download URL of the generated markdown file, or error message
    """
    try:
        # Check environment variables
        perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        firecrawl_key = os.getenv('FIRECRAWL_API_KEY')
        
        if not perplexity_key or perplexity_key == 'your-perplexity-api-key-here':
            return {
                "status": "error",
                "error_message": "PERPLEXITY_API_KEY not found or not set. Please set your API key from https://www.perplexity.ai/settings/api"
            }
        
        if not firecrawl_key or firecrawl_key == 'your-firecrawl-api-key-here':
            return {
                "status": "error",
                "error_message": "FIRECRAWL_API_KEY not found or not set. Please set your API key from https://www.firecrawl.dev/"
            }
        
        # Step 1: Convert query to Firecrawl format using Perplexity
        converter = QueryConverter(perplexity_key)
        firecrawl_queries = converter.convert_query(query)
        
        # Step 2: Crawl documentation using Firecrawl
        firecrawl_client = FirecrawlClient(firecrawl_key)
        crawl_results = firecrawl_client.crawl_multiple(firecrawl_queries)
        
        # Step 3: Extract markdown content
        content_by_source = firecrawl_client.extract_markdown_content(crawl_results)
        
        successful_sources = sum(1 for result in crawl_results if result.get('success'))
        
        if successful_sources == 0:
            return {
                "status": "error",
                "error_message": "No sources were successfully crawled. Please check your API keys and try again."
            }
        
        # Step 4: Aggregate content into single markdown
        aggregator = MarkdownAggregator()
        aggregated_content = aggregator.aggregate(content_by_source, query)
        
        # Step 5: Save to temporary file and upload
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Create filename from query
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        clean_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_query = clean_query.replace(' ', '_')[:50]
        filename = f"{timestamp}_{clean_query}_docs.md"
        file_path = os.path.join(temp_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(aggregated_content)
        
        # Upload to storage
        upload_result = upload_to_storage(file_path, filename)
        
        # Clean up local file
        try:
            os.remove(file_path)
        except:
            pass  # Ignore cleanup errors
        
        if upload_result["status"] == "success":
            file_size = len(aggregated_content.encode('utf-8'))
            return {
                "status": "success",
                "message": f"Documentation retrieved and aggregated successfully from {successful_sources} sources",
                "download_url": upload_result["download_url"],
                "filename": upload_result["filename"],
                "file_size_bytes": file_size,
                "sources_crawled": successful_sources,
                "total_sources": len(crawl_results)
            }
        else:
            return upload_result

    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Failed to retrieve documentation: {str(e)}"
        }


root_agent = Agent(
    name="doc_retriever_agent",
    model="gemini-2.0-flash",
    description="Agent to retrieve and aggregate documentation from multiple sources based on natural language queries",
    instruction="You are a helpful agent that retrieves documentation from multiple sources based on natural language queries. You convert user queries into specific documentation URLs, crawl the content, and aggregate it into a single comprehensive markdown file that is uploaded to cloud storage.",
    tools=[retrieve_documentation, upload_to_storage],
)