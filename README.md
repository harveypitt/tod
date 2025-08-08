# TOD (Tool On Demand)

<div align="center">

```
 ████████  ██████  ██████  
 ╚══██╔══╝██╔═══██╗██╔══██╗
    ██║   ██║   ██║██║  ██║
    ██║   ██║   ██║██║  ██║
    ██║   ╚██████╔╝██████╔╝
    ╚═╝    ╚═════╝ ╚═════╝ 
```

<img src="TOD.png" alt="TOD - The AI Tool Factory Mascot" width="300"/>

**TOD - Tool On Demand**  
*AI agents that make MCP servers*

*Won a hackathon at Tessl*

</div>

## What This Is

Three-stage pipeline for generating MCP servers on demand. Built with Google ADK agents and FastMCP.

### Currently Implemented:
- **FastMCP Server Template**: Working server with file ops, system info, web search mock
- **Spec Writer Agent**: Takes requests, outputs structured specifications  
- **Unit Function Agent**: Generates individual Python functions with error handling
- **MCP Creation Agent**: Assembles functions into complete FastMCP servers

### Missing Components:
- Pipeline orchestrator to chain the three stages
- Integration layer between agents
- Automated deployment to hosting platforms

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone the repo
git clone https://github.com/your-repo/tod.git
cd tod

# Install dependencies with uv
uv sync

# Create environment file (optional - server works without it)
touch .env
# Add EXA_API_KEY=your_key if you plan to use the research agents
```

### 2. Run the Working FastMCP Server
```bash
# Start the MCP server (works immediately)
python fastmcp_server.py

# The server provides these working tools:
# - echo_tool: Echo messages with timestamps
# - my_name: Returns "shahel" 
# - get_system_info: System information in JSON
# - web_search: Mock web search (returns sample data)
# - read_file: Read file contents safely
# - list_directory: List directory contents
# - mcp_build_practices: Returns FastMCP best practices guide
```

### 3. Test Individual Agents (Requires EXA_API_KEY)
```bash
# These load but need orchestration to work together:
python -c "from mcp_creation_agent.agent import root_agent; print('MCP Creation Agent loaded')"
python -c "from unit_function_agent.agent import root_agent; print('Unit Function Agent loaded')"
python -c "from spec_writer.agent import root_agent; print('Spec Writer Agent loaded')"
```

## 🔧 Current Architecture

### Working Components:
- **fastmcp_server.py**: Complete FastMCP server implementation
- **mcp_server_best_practices.md**: Comprehensive development guidelines  
- **Individual Agents**: Three Google ADK agents with specialized prompts

### Agent Definitions (Need Orchestration):
- **spec-writer/**: Gemini-2.5-flash agent for creating specifications from natural language
- **unit-function-agent/**: Gemini-2.5-pro agent with code execution for generating functions  
- **mcp-creation-agent/**: Gemini-2.5-flash agent for assembling complete MCP servers

### Missing Pieces:
- **Orchestrator**: Coordinator to chain the three agents together
- **Integration Layer**: Code to pass outputs between pipeline stages
- **Deployment Automation**: Automatic deployment to hosting platforms

## Architecture

### Agent Pipeline:
1. **spec-writer**: Gemini-2.5-flash + Exa search → structured specifications
2. **unit-function-agent**: Gemini-2.5-pro + code executor → individual functions  
3. **mcp-creation-agent**: Gemini-2.5-flash + Exa search → complete MCP servers

### FastMCP Server Implementation:
- Environment-based configuration
- File operations with security controls
- Health monitoring and structured logging
- Production error handling patterns

## Dependencies

- **Python 3.12+**: Core language version
- **Google ADK (≥1.9.0)**: Agent framework for the three agents
- **Pydantic (≥2.11.7)**: Data validation and configuration
- **Boto3 (≥1.35.0)**: AWS integration (if needed)
- **FastMCP**: MCP server framework (imported in main server)
- **Exa API**: Web search for research-capable agents

## Configuration

### Required for Agents:
```bash
# Add to .env file
EXA_API_KEY=your_exa_api_key_here
```

### Optional for FastMCP Server:
```bash
API_KEY=your_api_key
DEBUG=true
LOG_LEVEL=INFO
TIMEOUT=30
```

## 🧪 Testing & Development

### FastMCP Server Testing
```bash
# Install dependencies
uv sync

# Test the FastMCP server
python fastmcp_server.py

# Run any existing test files
python test_agent.py  # if available
python -m pytest tests/ -v  # if test directory exists
```

### ADK Agent Testing (Google ADK Methods)

#### Interactive Web UI Testing
```bash
# Start the ADK web interface for interactive testing
adk web

# Opens browser at http://localhost:8000
# Test with prompts like:
# - "Create a specification for a weather API tool"
# - "Generate a function to read CSV files"
# - "Build an MCP server for GitHub integration"
```

#### Command Line Testing
```bash
# Test agents via command line (useful for headless environments)
adk run mcp_creation_agent
adk run unit_function_agent  
adk run spec_writer_agent

# Example test queries:
# - "I need a tool to fetch weather data"
# - "Create a function to validate email addresses"
# - "Build an MCP server with authentication"
```

#### Agent Loading Verification
```bash
# Verify individual agents load correctly (requires EXA_API_KEY)
python -c "from mcp_creation_agent.agent import root_agent; print('MCP Creation Agent loads successfully')"
python -c "from unit_function_agent.agent import root_agent; print('Unit Function Agent loads successfully')"
python -c "from spec_writer.agent import root_agent; print('Spec Writer Agent loads successfully')"
```

### Development Notes
- ADK testing runs entirely on your local machine
- Requires Python 3.10+ and proper Google Cloud credentials
- Use `gcloud auth application-default login` for authentication
- Web UI provides step-by-step execution tracing
- Command line interface good for automated testing workflows

### Development Setup
```bash
# Clone and setup
git clone https://github.com/your-repo/tod.git
cd tod
uv sync
cp .env.example .env  # Configure your environment variables
```

## 📋 TODO Checklist

### Core Development
- [ ] **Orchestrator Agent** - Master coordinator for the 3-stage pipeline
  - [ ] Request routing and validation
  - [ ] Stage coordination and error handling  
  - [ ] Result aggregation and quality assurance
  - [ ] Integration testing framework

### Deployment & Infrastructure  
- [ ] **Deployment Agent** - Automated MCP server deployment
  - [ ] [Fly.io MCP deployment](https://fly.io/docs/mcp/) integration
  - [ ] [Remote MCP server setup](https://fly.io/docs/blueprints/remote-mcp-servers/)
  - [ ] Authentication and security configuration
  - [ ] Health monitoring and auto-scaling

### Quality Assurance
- [ ] **Comprehensive Testing Suite**
  - [ ] Unit tests for all agents
  - [ ] Integration tests for pipeline flow
  - [ ] Performance benchmarking
  - [ ] Error scenario coverage

### Documentation & Examples
- [ ] **API Documentation** - Complete function and class documentation
- [ ] **Usage Examples** - Real-world scenarios and tutorials  
- [ ] **Integration Guides** - Claude, ChatGPT, and other LLM setup
- [ ] **Video Demos** - Walkthrough of key features

### Advanced Features
- [ ] **Tool Versioning** - Version control for generated tools
- [ ] **Tool Sharing** - Marketplace for generated MCP servers
- [ ] **Performance Optimization** - Caching and parallel processing
- [ ] **Multi-language Support** - Generate tools in JavaScript, Go, etc.

### Useful Links
- [Fly.io MCP Launch Guide](https://fly.io/blog/mcp-launch/)
- [MCP Protocol Specification](https://spec.modelcontextprotocol.io/)
- [FastMCP Documentation](https://fastmcp.com/)
- [Anthropic MCP Examples](https://github.com/anthropics/anthropic-quickstarts/tree/main/mcp)

---

## 📄 License

MIT License

Copyright (c) 2024 TOD Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

*Built during a hackathon by the teams: Sunny (Request Processing), Harvey & James (Function Generation), and Shahel (MCP Integration)*
