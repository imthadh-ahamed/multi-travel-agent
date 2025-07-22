# Development Guide 🛠️

This guide is for developers who want to extend, modify, or contribute to the Multi-Agent Travel Assistant System.

## 🏗️ Architecture Overview

```
multi-travel-agent/
├── agents/                 # Agent implementations by phase
│   ├── phase1/            # Basic agents (rule-based, prompt-based)
│   ├── phase2/            # Tool-using agents with memory
│   ├── phase3/            # Multi-agent systems
│   └── phase4/            # Production systems (coming soon)
├── tools/                 # External API integrations
├── memory/                # Memory systems and storage
├── data/                  # Sample data and knowledge base
├── ui/                    # User interfaces (future)
├── api/                   # Backend APIs (future)
└── tests/                 # Test suites (future)
```

## 🔧 Key Components

### 1. Base Agent Architecture

All agents inherit from `BaseAgent`:

```python
class BaseAgent(ABC):
    def __init__(self, agent_id: str, role: AgentRole, model: str = "gpt-4"):
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.capabilities = []
        self.tools = {}
        self.memory = HybridMemorySystem()
    
    @abstractmethod
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process a specific task"""
        pass
```

### 2. Tool System

Tools provide external functionality:

```python
class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        pass
```

### 3. Memory Systems

Multiple memory types for different needs:

- **ConversationSummaryMemory**: Summarizes long conversations
- **VectorMemory**: Semantic search using ChromaDB
- **UserPreferenceManager**: Tracks and learns user preferences
- **HybridMemorySystem**: Combines all memory types

### 4. Multi-Agent Coordination

Agents communicate via messages and task delegation:

```python
@dataclass
class AgentMessage:
    sender: str
    recipient: str
    message_type: str
    content: Any
    timestamp: datetime
    conversation_id: str
```

## 🚀 Adding New Agents

### Step 1: Create Agent Class

```python
class YourNewAgent(BaseAgent):
    def __init__(self):
        super().__init__("your_agent", AgentRole.YOUR_ROLE)
        self.capabilities = ["capability1", "capability2"]
    
    def get_system_prompt(self) -> str:
        return "Your agent's system prompt..."
    
    async def process_task(self, task: TaskRequest) -> TaskResult:
        # Your agent logic here
        pass
```

### Step 2: Register with Coordinator

```python
coordinator = CoordinatorAgent()
coordinator.register_agent(YourNewAgent())
```

### Step 3: Add to Menu System

Update `main.py` to include your agent in the interactive menu.

## 🔌 Adding New Tools

### Step 1: Implement Tool Interface

```python
class YourNewTool(Tool):
    @property
    def name(self) -> str:
        return "your_tool_name"
    
    @property
    def description(self) -> str:
        return "What your tool does"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Parameter description"}
            },
            "required": ["param1"]
        }
    
    async def execute(self, **kwargs) -> ToolResult:
        # Your tool logic
        return ToolResult(success=True, data=result)
```

### Step 2: Add to Agent

```python
class YourAgent(BaseAgent):
    def __init__(self):
        super().__init__("agent_id", AgentRole.ROLE)
        self.tools["your_tool_name"] = YourNewTool()
```

## 🧠 Extending Memory Systems

### Custom Memory Backend

```python
class YourMemorySystem(MemorySystem):
    def store(self, key: str, value: Any, metadata: Optional[Dict] = None):
        # Your storage logic
        pass
    
    def retrieve(self, query: str, limit: int = 5) -> List[Any]:
        # Your retrieval logic
        pass
```

### Adding to Hybrid System

```python
class ExtendedHybridMemory(HybridMemorySystem):
    def __init__(self):
        super().__init__()
        self.your_memory = YourMemorySystem()
```

## 🌐 Adding External APIs

### Step 1: Create API Client

```python
class YourAPIClient(ExternalAPI):
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.yourservice.com")
    
    async def health_check(self) -> bool:
        # Check if API is accessible
        pass
    
    async def your_api_method(self, param1: str) -> APIResponse:
        return await self._make_request("GET", "/endpoint", {"param": param1})
```

### Step 2: Create Tool Wrapper

```python
class YourAPITool(Tool):
    def __init__(self, api_key: str):
        self.client = YourAPIClient(api_key)
    
    async def execute(self, **kwargs) -> ToolResult:
        async with self.client:
            result = await self.client.your_api_method(kwargs["param1"])
            return ToolResult(success=result.success, data=result.data)
```

## 📊 Data Formats

### Task Request Format

```python
TaskRequest(
    task_id="unique_id",
    task_type="category",
    description="Human readable description",
    parameters={"key": "value"},
    priority=1,  # 1-5, 1 is highest
    dependencies=["other_task_id"]  # Optional
)
```

### Tool Result Format

```python
ToolResult(
    success=True,
    data={"your": "data"},
    error=None,
    metadata={"additional": "info"}
)
```

## 🔄 Phase Development Pattern

Each phase follows this pattern:

1. **Phase N Folder**: `agents/phaseN/`
2. **Main Implementation**: Core agent logic
3. **Supporting Classes**: Tools, memory, utilities
4. **Test Function**: `test_phaseN_system()`
5. **Integration**: Added to `main.py` menu

## 🧪 Testing Guidelines

### Unit Tests (Future)

```python
import pytest
from agents.phase1.rule_based_advisor import RuleBasedTravelAdvisor

def test_rule_based_advisor():
    advisor = RuleBasedTravelAdvisor()
    # Test logic here
```

### Integration Tests

```python
async def test_multi_agent_integration():
    coordinator = CoordinatorAgent()
    # Register agents
    # Test full workflow
```

## 📝 Code Style Guidelines

### 1. Naming Conventions
- Classes: `PascalCase`
- Functions/Variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Files: `snake_case.py`

### 2. Documentation
- All classes and public methods need docstrings
- Use type hints consistently
- Add inline comments for complex logic

### 3. Error Handling
- Always use try-catch for external APIs
- Return structured error responses
- Log errors appropriately

### 4. Async/Await
- Use async for I/O operations
- Prefer async libraries (aiohttp over requests)
- Handle async context properly

## 🚀 Performance Considerations

### 1. Memory Management
- Clean up large objects after use
- Use generators for large datasets
- Monitor memory usage in long-running agents

### 2. API Rate Limits
- Implement backoff strategies
- Cache responses when appropriate
- Batch requests when possible

### 3. Concurrent Operations
- Use asyncio for parallel API calls
- Avoid blocking operations in async functions
- Consider connection pooling

## 🔐 Security Best Practices

### 1. API Keys
- Never hardcode API keys
- Use environment variables
- Rotate keys regularly

### 2. Input Validation
- Validate all user inputs
- Sanitize data before API calls
- Use parameterized queries

### 3. Output Sanitization
- Filter sensitive information
- Validate LLM outputs
- Implement content filtering

## 📦 Deployment Considerations

### 1. Environment Setup
- Use virtual environments
- Pin dependency versions
- Document system requirements

### 2. Configuration Management
- Separate dev/prod configs
- Use environment-specific settings
- Implement feature flags

### 3. Monitoring & Logging
- Log all agent interactions
- Monitor API usage and costs
- Track performance metrics

## 🤝 Contributing

### 1. Fork & Branch
```bash
git fork <repo>
git checkout -b feature/your-feature
```

### 2. Development
- Follow coding standards
- Add tests for new features
- Update documentation

### 3. Pull Request
- Clear description of changes
- Link to relevant issues
- Ensure tests pass

## 📚 Resources

### Learning Materials
- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Docs](https://platform.openai.com/docs/)
- [AsyncIO Guide](https://docs.python.org/3/library/asyncio.html)

### Useful Tools
- **ChromaDB**: Vector database
- **Rich**: Terminal formatting
- **aiohttp**: Async HTTP client
- **pydantic**: Data validation

## 🎯 Future Roadmap

### Phase 4 Features (Planned)
- Auto-agent loops and self-improvement
- RAG (Retrieval Augmented Generation) 
- Production deployment configurations
- Web UI with real-time updates
- Database persistence
- User authentication

### Advanced Features (Future)
- Model fine-tuning capabilities
- Custom embedding models
- Advanced orchestration
- Plugin system
- Monitoring dashboard

---

**Happy developing! 🚀**

For questions or suggestions, please open an issue or contribute to the project.
