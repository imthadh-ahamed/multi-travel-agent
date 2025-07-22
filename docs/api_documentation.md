# API Documentation 📡

This document provides comprehensive API documentation for the Multi-Agent Travel Assistant System.

## 🌐 External API Integrations

### OpenWeatherMap API

**Purpose**: Real weather data for travel destinations

**Base URL**: `https://api.openweathermap.org/data/2.5/`

**Authentication**: API key in query parameter

#### Current Weather
```python
GET /weather?q={city}&appid={API_key}&units=metric

# Response
{
    "main": {
        "temp": 22.5,
        "feels_like": 21.8,
        "humidity": 65
    },
    "weather": [
        {
            "main": "Clear",
            "description": "clear sky"
        }
    ],
    "name": "London"
}
```

#### Weather Forecast
```python
GET /forecast?q={city}&appid={API_key}&units=metric

# Response
{
    "list": [
        {
            "dt": 1647345600,
            "main": {"temp": 20.5},
            "weather": [{"description": "partly cloudy"}]
        }
    ]
}
```

### Exchange Rate API

**Purpose**: Currency conversion for budget planning

**Base URL**: `https://api.exchangerate-api.com/v4/latest/`

**Authentication**: Free tier available

```python
GET /latest/{base_currency}

# Response
{
    "base": "USD",
    "rates": {
        "EUR": 0.85,
        "GBP": 0.73,
        "JPY": 110.0
    }
}
```

### Mock Flight API

**Purpose**: Simulated flight search functionality

**Implementation**: Local mock service

```python
# Search flights
flights = await flight_api.search_flights(
    origin="JFK",
    destination="LHR", 
    departure_date="2024-12-15",
    passengers=2
)

# Response format
{
    "flights": [
        {
            "airline": "British Airways",
            "flight_number": "BA112",
            "departure": "2024-12-15T10:30:00",
            "arrival": "2024-12-15T22:15:00",
            "duration": "7h 45m",
            "price": 850.00,
            "stops": 0
        }
    ]
}
```

### Mock Hotel API

**Purpose**: Simulated hotel search functionality

**Implementation**: Local mock service

```python
# Search hotels
hotels = await hotel_api.search_hotels(
    destination="London",
    check_in="2024-12-15",
    check_out="2024-12-18",
    guests=2,
    rooms=1
)

# Response format
{
    "hotels": [
        {
            "name": "The Savoy",
            "rating": 5,
            "price_per_night": 450.00,
            "amenities": ["wifi", "spa", "restaurant"],
            "distance_from_center": "0.5 km"
        }
    ]
}
```

## 🤖 Agent APIs

### Base Agent Interface

All agents implement this core interface:

```python
class BaseAgent(ABC):
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process a task and return results"""
        pass
    
    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        pass
    
    def get_system_prompt(self) -> str:
        """Return the agent's system prompt"""
        pass
```

### Rule-Based Travel Advisor

**Purpose**: Provides rule-based travel recommendations

```python
from agents.phase1.rule_based_advisor import RuleBasedTravelAdvisor

advisor = RuleBasedTravelAdvisor()

# Get destination recommendation
recommendation = advisor.get_destination_recommendation(
    budget=2000,
    travel_style="adventure",
    climate_preference="warm",
    season="summer"
)

# Response
{
    "destination": "Thailand",
    "confidence": 0.85,
    "reasoning": ["Budget friendly", "Adventure activities", "Warm climate"],
    "estimated_cost": 1800
}
```

### Prompt-Based Agent

**Purpose**: LLM-powered conversational travel agent

```python
from agents.phase1.prompt_based_agent import PromptBasedTravelAgent

agent = PromptBasedTravelAgent()

response = await agent.chat(
    "I want to plan a romantic getaway for 2 people under $3000"
)

# Response
{
    "response": "I'd recommend Paris or Prague for a romantic getaway...",
    "suggestions": ["Paris", "Prague", "Venice"],
    "estimated_budget": 2800
}
```

### Tool-Using Agent

**Purpose**: Agent with external tool capabilities

```python
from agents.phase2.tool_using_agent import ToolUsingAgent

agent = ToolUsingAgent()

# Add tools
agent.add_tool(WeatherTool())
agent.add_tool(FlightSearchTool())

result = await agent.process_request(
    "What's the weather like in Tokyo and find flights from NYC?"
)
```

### Multi-Agent System

**Purpose**: Coordinated specialist agents

```python
from agents.phase3.multi_agent_system import CoordinatorAgent

coordinator = CoordinatorAgent()

# Process complex travel request
result = await coordinator.plan_trip({
    "destination": "Japan",
    "budget": 5000,
    "duration": "10 days",
    "travelers": 2,
    "interests": ["culture", "food", "temples"]
})
```

## 🛠️ Tool APIs

### Weather Tool

```python
class WeatherTool(Tool):
    async def execute(self, city: str) -> ToolResult:
        """Get current weather for a city"""
        pass

# Usage
weather_tool = WeatherTool()
result = await weather_tool.execute(city="Tokyo")

# Result format
{
    "success": True,
    "data": {
        "temperature": 22.5,
        "condition": "Clear",
        "humidity": 65,
        "description": "Perfect weather for sightseeing!"
    }
}
```

### Flight Search Tool

```python
class FlightSearchTool(Tool):
    async def execute(self, origin: str, destination: str, 
                     departure_date: str, passengers: int = 1) -> ToolResult:
        """Search for flights"""
        pass

# Usage
flight_tool = FlightSearchTool()
result = await flight_tool.execute(
    origin="JFK",
    destination="NRT",
    departure_date="2024-12-15",
    passengers=2
)
```

### Hotel Search Tool

```python
class HotelSearchTool(Tool):
    async def execute(self, destination: str, check_in: str,
                     check_out: str, guests: int = 1) -> ToolResult:
        """Search for hotels"""
        pass
```

### Currency Converter Tool

```python
class CurrencyConverterTool(Tool):
    async def execute(self, amount: float, from_currency: str,
                     to_currency: str) -> ToolResult:
        """Convert currency amounts"""
        pass

# Usage
converter = CurrencyConverterTool()
result = await converter.execute(
    amount=1000,
    from_currency="USD",
    to_currency="EUR"
)
```

## 🧠 Memory System APIs

### Conversation Summary Memory

```python
from agents.phase2.memory_system import ConversationSummaryMemory

memory = ConversationSummaryMemory()

# Store conversation
await memory.store_conversation(
    conversation_id="conv_123",
    messages=[
        {"role": "user", "content": "I want to visit Japan"},
        {"role": "assistant", "content": "Great choice! When would you like to go?"}
    ]
)

# Get summary
summary = await memory.get_summary("conv_123")
```

### Vector Memory

```python
from agents.phase2.memory_system import VectorMemory

memory = VectorMemory()

# Store knowledge
await memory.store(
    key="japan_travel_tips",
    text="Japan is best visited in spring for cherry blossoms...",
    metadata={"topic": "travel", "destination": "japan"}
)

# Search
results = await memory.search("when to visit Japan", limit=5)
```

### User Preference Manager

```python
from agents.phase2.memory_system import UserPreferenceManager

prefs = UserPreferenceManager()

# Learn from interaction
await prefs.learn_from_interaction(
    user_id="user_123",
    interaction_data={
        "preferred_destinations": ["Japan", "Italy"],
        "budget_range": "luxury",
        "travel_style": "cultural"
    }
)

# Get preferences
user_prefs = await prefs.get_preferences("user_123")
```

## 📨 Message Formats

### Task Request

```python
@dataclass
class TaskRequest:
    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any]
    priority: int = 3  # 1-5, 1 is highest
    dependencies: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

# Example
task = TaskRequest(
    task_id="task_001",
    task_type="flight_search",
    description="Find flights from NYC to Tokyo",
    parameters={
        "origin": "JFK",
        "destination": "NRT",
        "departure_date": "2024-12-15",
        "passengers": 2
    },
    priority=1
)
```

### Task Result

```python
@dataclass
class TaskResult:
    task_id: str
    success: bool
    data: Any
    error: Optional[str] = None
    agent_id: Optional[str] = None
    execution_time: Optional[float] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Example
result = TaskResult(
    task_id="task_001",
    success=True,
    data={
        "flights": [...],
        "total_found": 15,
        "cheapest_price": 850.00
    },
    agent_id="flight_specialist",
    execution_time=2.3,
    tokens_used=1250
)
```

### Agent Message

```python
@dataclass
class AgentMessage:
    sender: str
    recipient: str
    message_type: str  # 'request', 'response', 'notification'
    content: Any
    timestamp: datetime
    conversation_id: str
    priority: int = 3
    requires_response: bool = False

# Example
message = AgentMessage(
    sender="coordinator",
    recipient="flight_specialist",
    message_type="request",
    content=task_request,
    timestamp=datetime.now(),
    conversation_id="conv_123",
    priority=1,
    requires_response=True
)
```

### Tool Result

```python
@dataclass
class ToolResult:
    success: bool
    data: Any
    error: Optional[str] = None
    tool_name: Optional[str] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# Example
result = ToolResult(
    success=True,
    data={"temperature": 22.5, "condition": "Clear"},
    tool_name="weather_tool",
    execution_time=0.8,
    metadata={"api_response_time": 0.6}
)
```

## 🔧 Configuration APIs

### Environment Variables

```python
# Required API keys
OPENAI_API_KEY=your_openai_api_key
OPENWEATHERMAP_API_KEY=your_weather_api_key

# Optional settings
TRAVEL_AGENT_LOG_LEVEL=INFO
TRAVEL_AGENT_MAX_TOKENS=4000
TRAVEL_AGENT_TEMPERATURE=0.7
TRAVEL_AGENT_MODEL=gpt-4

# Memory settings
CHROMA_PERSIST_DIRECTORY=./memory/chroma_db
MEMORY_SUMMARY_MAX_TOKENS=500
```

### Agent Configuration

```python
from dataclasses import dataclass

@dataclass
class AgentConfig:
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4000
    timeout: int = 30
    retry_attempts: int = 3
    enable_memory: bool = True
    memory_type: str = "hybrid"
```

## 📊 Response Codes

### Standard Success Codes
- `200`: Success
- `201`: Created (new resource)
- `202`: Accepted (async processing)

### Client Error Codes
- `400`: Bad Request (invalid parameters)
- `401`: Unauthorized (invalid API key)
- `404`: Not Found (resource doesn't exist)
- `429`: Too Many Requests (rate limited)

### Server Error Codes
- `500`: Internal Server Error
- `502`: Bad Gateway (external API error)
- `503`: Service Unavailable
- `504`: Gateway Timeout

## 🚀 Rate Limits

### OpenAI API
- **GPT-4**: 10,000 requests/min
- **GPT-3.5**: 60,000 requests/min
- **Embeddings**: 300,000 requests/min

### OpenWeatherMap API
- **Free Tier**: 1,000 calls/day
- **Paid Tier**: Up to 1,000,000 calls/month

### Internal Rate Limits
- **Tool Execution**: 100 calls/min per tool
- **Memory Operations**: 1,000 ops/min
- **Agent Messages**: 500 messages/min per agent

## 🔐 Authentication

### API Key Management

```python
# Environment-based authentication
import os
api_key = os.getenv("OPENAI_API_KEY")

# Secure key validation
def validate_api_key(key: str) -> bool:
    return key and len(key) > 20 and key.startswith("sk-")
```

### Request Headers

```python
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "User-Agent": "multi-travel-agent/1.0"
}
```

## 📝 Example Usage Patterns

### Basic Travel Planning

```python
# Initialize system
coordinator = CoordinatorAgent()

# Plan a trip
trip_plan = await coordinator.plan_trip({
    "destination": "Paris",
    "duration": "5 days",
    "budget": 3000,
    "travelers": 2,
    "interests": ["art", "food", "history"]
})

# Access results
print(f"Total cost: ${trip_plan.total_cost}")
print(f"Flights: {trip_plan.flights}")
print(f"Hotels: {trip_plan.hotels}")
print(f"Itinerary: {trip_plan.itinerary}")
```

### Weather Integration

```python
# Get weather for multiple destinations
weather_tool = WeatherTool()

destinations = ["Paris", "London", "Rome"]
weather_data = {}

for city in destinations:
    result = await weather_tool.execute(city=city)
    weather_data[city] = result.data
```

### Memory-Enhanced Conversations

```python
# Agent with memory
agent = ToolUsingAgent()
agent.memory = HybridMemorySystem()

# First interaction
response1 = await agent.chat("I love adventure travel")

# Later interaction (agent remembers preference)
response2 = await agent.chat("Suggest a destination for my next trip")
# Agent will consider adventure preference
```

---

**For additional API documentation or support, please refer to the development guide or open an issue in the repository.**
