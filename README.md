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

🏆 *Winner of [AI Native Dev](https://ainativedev.io/) Hackathon*

</div>

## What This Is

Three-stage pipeline for generating MCP servers on demand. Built with Google ADK agents and FastMCP.

### Currently Implemented:
- **FastMCP Server Template**: Working server with file ops, system info, web search mock
- **Spec Writer Agent**: Takes requests, outputs structured specifications  
- **Unit Function Agent**: Generates individual Python functions with error handling
- **MCP Creation Agent**: Assembles functions into complete FastMCP servers
- **Orchestrator Pipeline**: SequentialAgent that coordinates all three stages with context passing

### In Development:
- **Pipeline Reliability**: Orchestrator works but needs improved error handling and stability
- **Context Optimization**: Fine-tuning ReadonlyContext data flow between agents
- **Deployment Integration**: Automated deployment to hosting platforms

## Why This Matters

### Token Efficiency 🎯
Instead of multiple back-and-forth conversations to build tools, TOD generates complete MCP servers in a single coordinated pipeline. No wasted tokens on repeated context or incremental refinements.

### Autonomous Tool Building
LLMs can extend their own capabilities without human intervention. Request a tool, get a working MCP server. No manual coding, configuration, or deployment steps.

### Production Ready Output
Generated servers include proper error handling, input validation, security controls, and monitoring. Not just proof-of-concept code, but deployment-ready implementations.

## The Vision

Today, TOD demonstrates three specialized agents that can generate MCP servers from natural language requests. Tomorrow, it enables a future where AI systems continuously extend their own capabilities—requesting, building, and deploying new tools as they encounter novel problems, sharing these capabilities across a network of interconnected agents, and ultimately evolving into truly autonomous systems that grow smarter through experience rather than just training data.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone the repo
git clone https://github.com/your-repo/tod.git
cd tod

# Install dependencies with uv
uv sync

# Copy and configure environment file
cp .env.example .env
# Edit .env with your API keys (see Configuration section below)
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

### 3. Test the Orchestrator Pipeline (Requires API Keys)
```bash
# Test the complete orchestrator (experimental - reliability improvements ongoing)
python -c "from orchestrator.agent import root_agent; print('Orchestrator loaded successfully')"

# The orchestrator coordinates:
# 1. Spec Writer: Natural language → JSON specifications
# 2. Unit Function: Specifications → Python functions  
# 3. MCP Creation: Functions → Complete FastMCP servers

# Note: Pipeline works but needs improved error handling and stability
```

## 🔧 Current Architecture

### Working Components:
- **fastmcp_server.py**: Complete FastMCP server implementation
- **mcp_server_best_practices.md**: Comprehensive development guidelines  
- **orchestrator/**: SequentialAgent pipeline with ReadonlyContext passing
- **Individual Agents**: Three Google ADK agents with specialized prompts and context access

### Orchestrator Pipeline:
- **orchestrator/agent.py**: SequentialAgent that coordinates all three stages
- **spec-writer/**: Gemini-2.5-flash agent → outputs to `mcp-spec` session state
- **unit-function-agent/**: Gemini-2.5-pro agent → reads `mcp-spec`, outputs `unit-functions`
- **mcp-creation-agent/**: Gemini-2.5-flash agent → reads both contexts, creates final server

### Current Status:
- ✅ **Context Passing**: ReadonlyContext implementation working
- ✅ **Agent Coordination**: SequentialAgent pipeline functional
- ⚠️ **Reliability**: Pipeline needs improved error handling for production use
- 🚧 **Deployment**: Automatic deployment integration in progress

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

Create a `.env` file by copying `.env.example`:

```bash
cp .env.example .env
```

### Required for Orchestrator:
```bash
# Google Gemini API configuration
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE

# Exa search API for research agents
EXA_API_KEY=your_exa_api_key_here

# ADK configuration (prevents context variable errors)
SERVER_NAME=orchestrator
APP_NAME=tod_orchestrator
USER_ID=user
SESSION_ID=session_001
ADK_LOG_LEVEL=INFO
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

#### Orchestrator Testing
```bash
# Verify orchestrator loads correctly (requires Google API key)
python -c "from orchestrator.agent import root_agent; print('Orchestrator loads successfully')"

# Test orchestrator via ADK web interface
adk web orchestrator

# The orchestrator coordinates the full pipeline:
# Input: Natural language request → Output: Complete MCP server
# Note: Currently experimental, reliability improvements ongoing
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

## 🚧 Orchestrator Status & Reliability

### Current Implementation (Experimental)
The orchestrator pipeline is **functional but requires stability improvements** before production use:

#### ✅ Working Features:
- **SequentialAgent Pipeline**: Coordinates all three stages in sequence
- **ReadonlyContext Passing**: Proper data flow between agents using session state
- **Dynamic Instructions**: Context-aware prompts that inject session data
- **Agent Loading**: All agents load successfully with proper configuration

#### ⚠️ Known Issues & Reliability Concerns:
- **Gemini API Errors**: Occasional 500 Internal Server errors during complex operations
- **Context Variable Resolution**: Template variables sometimes fail to resolve in edge cases  
- **Error Recovery**: Limited retry logic when individual agents fail
- **Session Management**: Long-running sessions may encounter state corruption
- **Resource Usage**: Complex multi-agent calls can hit API rate limits

#### 🔧 Reliability Improvements Needed:
- [ ] **Robust Error Handling**: Implement retry logic and graceful fallbacks
- [ ] **Request Throttling**: Add rate limiting between agent calls
- [ ] **Session Validation**: Verify context integrity before each stage
- [ ] **API Circuit Breakers**: Handle temporary API failures gracefully  
- [ ] **Monitoring & Logging**: Add comprehensive execution tracking
- [ ] **Fallback Modes**: Simple single-agent mode when pipeline fails
- [ ] **Integration Testing**: Comprehensive end-to-end test suite

### Usage Recommendations:
- **Development**: Safe to use for testing and experimentation
- **Production**: Wait for reliability improvements before critical deployments
- **Monitoring**: Watch logs carefully for API errors and context issues

## 📋 TODO Checklist

### Core Development (Pipeline Reliability)
- [ ] **Orchestrator Stability** - Critical reliability improvements
  - [ ] API error handling and retry logic
  - [ ] Session state validation and recovery
  - [ ] Request throttling and rate limiting
  - [ ] Comprehensive error monitoring

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

*Built during a hackathon by the teams: Sunny (Request Processing), Harvey & James (Function Generation), Shahel and Osman (MCP Integration) and Sachin (Product Management)*
