"""
Phase 3: Advanced Agentic Behavior - Multi-Agent System

This module implements a sophisticated multi-agent system where specialized
agents collaborate to provide comprehensive travel assistance. It demonstrates
agent coordination, delegation, and collaborative problem-solving.

Learning Objectives:
- Design specialized agent roles
- Implement agent communication protocols
- Create a coordinator/planner agent
- Handle inter-agent data sharing
- Manage collaborative workflows
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import openai
from dotenv import load_dotenv

# Import from previous phases
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase2.tool_using_agent import Tool, ToolResult, WeatherTool, FlightSearchTool, HotelSearchTool
from phase2.memory_system import HybridMemorySystem

load_dotenv()


class AgentRole(Enum):
    """Different agent roles in the system"""
    COORDINATOR = "coordinator"
    FLIGHT_SPECIALIST = "flight_specialist"
    HOTEL_SPECIALIST = "hotel_specialist"
    WEATHER_ADVISOR = "weather_advisor"
    BUDGET_ADVISOR = "budget_advisor"
    ITINERARY_PLANNER = "itinerary_planner"


@dataclass
class AgentMessage:
    """Message passed between agents"""
    sender: str
    recipient: str
    message_type: str
    content: Any
    timestamp: datetime
    conversation_id: str
    requires_response: bool = False


@dataclass
class TaskRequest:
    """Request for an agent to perform a task"""
    task_id: str
    task_type: str
    description: str
    parameters: Dict[str, Any]
    priority: int
    deadline: Optional[datetime] = None
    dependencies: List[str] = None


@dataclass
class TaskResult:
    """Result from agent task execution"""
    task_id: str
    agent_id: str
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, agent_id: str, role: AgentRole, model: str = "gpt-4"):
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.message_queue: List[AgentMessage] = []
        self.capabilities: List[str] = []
        self.tools: Dict[str, Tool] = {}
        self.memory = HybridMemorySystem()
        
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass
    
    @abstractmethod
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process a specific task"""
        pass
    
    async def send_message(self, recipient: str, message_type: str, content: Any, 
                          conversation_id: str, requires_response: bool = False):
        """Send message to another agent"""
        message = AgentMessage(
            sender=self.agent_id,
            recipient=recipient,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            requires_response=requires_response
        )
        
        # In a real system, this would use a message broker
        # For now, we'll use the coordinator to route messages
        return message
    
    async def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and process a message from another agent"""
        self.message_queue.append(message)
        
        if message.requires_response:
            return await self._generate_response(message)
        
        return None
    
    async def _generate_response(self, message: AgentMessage) -> AgentMessage:
        """Generate response to a message"""
        response_content = f"Received {message.message_type} from {message.sender}"
        
        return AgentMessage(
            sender=self.agent_id,
            recipient=message.sender,
            message_type="response",
            content=response_content,
            timestamp=datetime.now(),
            conversation_id=message.conversation_id
        )
    
    async def use_llm(self, prompt: str, context: Optional[str] = None) -> str:
        """Use LLM to generate response"""
        try:
            messages = [{"role": "system", "content": self.get_system_prompt()}]
            
            if context:
                messages.append({"role": "system", "content": f"Context: {context}"})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"


class CoordinatorAgent(BaseAgent):
    """
    Main coordinator that delegates tasks to specialized agents
    """
    
    def __init__(self):
        super().__init__("coordinator", AgentRole.COORDINATOR)
        self.agents: Dict[str, BaseAgent] = {}
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.capabilities = [
            "task_delegation",
            "workflow_coordination", 
            "result_aggregation",
            "user_communication"
        ]
    
    def register_agent(self, agent: BaseAgent):
        """Register a specialized agent"""
        self.agents[agent.agent_id] = agent
    
    def get_system_prompt(self) -> str:
        return """You are a travel coordination assistant that manages a team of specialized travel agents. 

Your role is to:
1. Understand user requests and break them down into tasks
2. Delegate tasks to appropriate specialist agents
3. Coordinate between agents to ensure smooth workflow
4. Aggregate results and provide comprehensive responses
5. Handle any conflicts or issues between agents

Available specialists:
- Flight Specialist: Flight searches, airline recommendations, route planning
- Hotel Specialist: Accommodation searches, booking advice, location recommendations  
- Weather Advisor: Weather forecasts, climate information, seasonal advice
- Budget Advisor: Cost estimation, budget planning, money-saving tips
- Itinerary Planner: Activity planning, scheduling, local recommendations

Be efficient in task delegation and provide clear, actionable travel advice."""
    
    async def process_user_request(self, user_request: str, conversation_id: str) -> str:
        """Process a user request by coordinating with specialist agents"""
        try:
            # Analyze the request and determine required tasks
            tasks = await self._analyze_request(user_request)
            
            # Execute tasks in parallel where possible
            task_results = await self._execute_tasks(tasks, conversation_id)
            
            # Aggregate results and generate response
            response = await self._aggregate_results(user_request, task_results)
            
            # Store in memory
            self.memory.store_conversation(user_request, response, 
                                         list(task_results.keys()), conversation_id)
            
            return response
            
        except Exception as e:
            return f"❌ Error processing request: {str(e)}"
    
    async def _analyze_request(self, user_request: str) -> List[TaskRequest]:
        """Analyze user request and create tasks for specialist agents"""
        analysis_prompt = f"""
        Analyze this travel request and determine what tasks need to be performed:
        
        Request: "{user_request}"
        
        Available agents and their capabilities:
        - flight_specialist: Flight searches, airline info, routing
        - hotel_specialist: Hotel searches, accommodation advice
        - weather_advisor: Weather forecasts, climate information
        - budget_advisor: Cost estimation, budget planning
        - itinerary_planner: Activity planning, scheduling
        
        Return a JSON list of tasks needed, each with:
        - agent_id: which agent should handle it
        - task_type: type of task
        - description: what the agent should do
        - parameters: any specific parameters
        - priority: 1-5 (1=highest)
        
        Example:
        [
            {{
                "agent_id": "flight_specialist",
                "task_type": "flight_search",
                "description": "Search for flights from NYC to Paris",
                "parameters": {{"origin": "NYC", "destination": "Paris", "date": "2024-06-15"}},
                "priority": 1
            }}
        ]
        """
        
        try:
            response = await self.use_llm(analysis_prompt)
            
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                tasks_data = json.loads(response[json_start:json_end])
                
                tasks = []
                for i, task_data in enumerate(tasks_data):
                    tasks.append(TaskRequest(
                        task_id=f"task_{i}_{datetime.now().timestamp()}",
                        task_type=task_data.get("task_type", "general"),
                        description=task_data.get("description", ""),
                        parameters=task_data.get("parameters", {}),
                        priority=task_data.get("priority", 3),
                        dependencies=task_data.get("dependencies", [])
                    ))
                
                return tasks
            else:
                # Fallback: create a general task
                return [TaskRequest(
                    task_id=f"general_{datetime.now().timestamp()}",
                    task_type="general_assistance",
                    description=user_request,
                    parameters={"request": user_request},
                    priority=1
                )]
                
        except Exception as e:
            print(f"Error analyzing request: {e}")
            return []
    
    async def _execute_tasks(self, tasks: List[TaskRequest], conversation_id: str) -> Dict[str, TaskResult]:
        """Execute tasks using appropriate agents"""
        results = {}
        
        # Sort tasks by priority
        tasks.sort(key=lambda t: t.priority)
        
        # Execute tasks
        for task in tasks:
            agent_id = task.parameters.get("agent_id", "coordinator")
            
            if agent_id in self.agents:
                try:
                    result = await self.agents[agent_id].process_task(task)
                    results[task.task_id] = result
                except Exception as e:
                    results[task.task_id] = TaskResult(
                        task_id=task.task_id,
                        agent_id=agent_id,
                        success=False,
                        data=None,
                        error=str(e)
                    )
            else:
                # Handle with general reasoning
                result = await self._handle_general_task(task)
                results[task.task_id] = result
        
        return results
    
    async def _handle_general_task(self, task: TaskRequest) -> TaskResult:
        """Handle tasks that don't have a specific agent"""
        try:
            response = await self.use_llm(
                f"Help with this travel request: {task.description}",
                json.dumps(task.parameters)
            )
            
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                data=response
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                data=None,
                error=str(e)
            )
    
    async def _aggregate_results(self, user_request: str, task_results: Dict[str, TaskResult]) -> str:
        """Aggregate results from all agents into a coherent response"""
        successful_results = {
            task_id: result for task_id, result in task_results.items() 
            if result.success
        }
        
        if not successful_results:
            return "❌ I wasn't able to process your request successfully. Please try rephrasing it."
        
        # Prepare context for aggregation
        results_context = []
        for task_id, result in successful_results.items():
            results_context.append(f"Task {task_id}: {result.data}")
        
        aggregation_prompt = f"""
        User Request: "{user_request}"
        
        Results from specialist agents:
        {chr(10).join(results_context)}
        
        Please provide a comprehensive, well-structured response that:
        1. Directly addresses the user's request
        2. Integrates all the specialist information
        3. Provides actionable recommendations
        4. Is clear and easy to understand
        5. Includes specific details and prices when available
        
        Format the response with clear sections and use appropriate emojis.
        """
        
        return await self.use_llm(aggregation_prompt)
    
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process a coordination task"""
        try:
            response = await self.use_llm(task.description, json.dumps(task.parameters))
            
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                data=response
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                data=None,
                error=str(e)
            )


class FlightSpecialistAgent(BaseAgent):
    """Specialized agent for flight-related tasks"""
    
    def __init__(self):
        super().__init__("flight_specialist", AgentRole.FLIGHT_SPECIALIST)
        self.capabilities = ["flight_search", "route_planning", "airline_advice"]
        
        # Add flight search tool
        self.tools["search_flights"] = FlightSearchTool()
    
    def get_system_prompt(self) -> str:
        return """You are a flight specialist agent with expertise in:
        - Flight searches and comparisons
        - Airline recommendations
        - Route optimization
        - Travel timing advice
        - Booking strategies
        
        Provide detailed, accurate flight information with prices, schedules, and practical advice.
        Always consider factors like baggage policies, seat selection, and travel comfort."""
    
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process flight-related tasks"""
        try:
            if task.task_type == "flight_search":
                return await self._handle_flight_search(task)
            elif task.task_type == "route_planning":
                return await self._handle_route_planning(task)
            else:
                # General flight advice
                response = await self.use_llm(task.description, json.dumps(task.parameters))
                
                return TaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    success=True,
                    data=response
                )
                
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                data=None,
                error=str(e)
            )
    
    async def _handle_flight_search(self, task: TaskRequest) -> TaskResult:
        """Handle flight search tasks"""
        params = task.parameters
        
        # Use flight search tool
        result = await self.tools["search_flights"].execute(
            origin=params.get("origin", ""),
            destination=params.get("destination", ""),
            departure_date=params.get("departure_date", ""),
            return_date=params.get("return_date"),
            passengers=params.get("passengers", 1)
        )
        
        if result.success:
            # Enhance with flight specialist analysis
            analysis_prompt = f"""
            Analyze these flight options and provide expert recommendations:
            
            {json.dumps(result.data, indent=2)}
            
            Consider:
            - Best value for money
            - Comfort and convenience
            - Travel time efficiency
            - Airline reputation
            
            Provide a summary with your top 3 recommendations and reasoning.
            """
            
            analysis = await self.use_llm(analysis_prompt)
            
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                data={
                    "flight_options": result.data,
                    "specialist_analysis": analysis
                }
            )
        else:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                data=None,
                error=result.error
            )
    
    async def _handle_route_planning(self, task: TaskRequest) -> TaskResult:
        """Handle route planning tasks"""
        params = task.parameters
        
        route_prompt = f"""
        Plan the optimal flight route for this trip:
        
        Origin: {params.get('origin', 'Not specified')}
        Destination: {params.get('destination', 'Not specified')}
        Travel Dates: {params.get('dates', 'Not specified')}
        Budget: {params.get('budget', 'Not specified')}
        Preferences: {params.get('preferences', 'None')}
        
        Consider:
        - Direct vs connecting flights
        - Best airports to use
        - Timing optimization
        - Cost considerations
        - Seasonal factors
        
        Provide detailed routing recommendations.
        """
        
        analysis = await self.use_llm(route_prompt)
        
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            success=True,
            data=analysis
        )


class HotelSpecialistAgent(BaseAgent):
    """Specialized agent for hotel and accommodation tasks"""
    
    def __init__(self):
        super().__init__("hotel_specialist", AgentRole.HOTEL_SPECIALIST)
        self.capabilities = ["hotel_search", "accommodation_advice", "location_recommendations"]
        
        # Add hotel search tool
        self.tools["search_hotels"] = HotelSearchTool()
    
    def get_system_prompt(self) -> str:
        return """You are a hotel specialist agent with expertise in:
        - Hotel searches and comparisons
        - Accommodation recommendations
        - Location advice
        - Amenity analysis
        - Booking strategies
        
        Provide detailed accommodation advice considering location, amenities, value, and guest experiences."""
    
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process hotel-related tasks"""
        try:
            if task.task_type == "hotel_search":
                return await self._handle_hotel_search(task)
            elif task.task_type == "location_advice":
                return await self._handle_location_advice(task)
            else:
                response = await self.use_llm(task.description, json.dumps(task.parameters))
                
                return TaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    success=True,
                    data=response
                )
                
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                data=None,
                error=str(e)
            )
    
    async def _handle_hotel_search(self, task: TaskRequest) -> TaskResult:
        """Handle hotel search tasks"""
        params = task.parameters
        
        result = await self.tools["search_hotels"].execute(
            city=params.get("city", ""),
            check_in=params.get("check_in", ""),
            check_out=params.get("check_out", ""), 
            guests=params.get("guests", 2),
            max_price=params.get("max_price")
        )
        
        if result.success:
            analysis_prompt = f"""
            Analyze these hotel options and provide expert recommendations:
            
            {json.dumps(result.data, indent=2)}
            
            Consider:
            - Location convenience
            - Value for money
            - Amenities quality
            - Guest experience
            
            Provide top 3 recommendations with reasoning.
            """
            
            analysis = await self.use_llm(analysis_prompt)
            
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                data={
                    "hotel_options": result.data,
                    "specialist_analysis": analysis
                }
            )
        else:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                data=None,
                error=result.error
            )
    
    async def _handle_location_advice(self, task: TaskRequest) -> TaskResult:
        """Handle location recommendation tasks"""
        params = task.parameters
        
        location_prompt = f"""
        Provide accommodation location advice for:
        
        Destination: {params.get('destination', 'Not specified')}
        Trip Purpose: {params.get('purpose', 'Not specified')}
        Budget Level: {params.get('budget_level', 'Not specified')}
        Interests: {params.get('interests', 'Not specified')}
        
        Recommend:
        - Best neighborhoods to stay in
        - Proximity to attractions
        - Transportation considerations
        - Safety and convenience factors
        - Local experiences
        
        Provide detailed location recommendations.
        """
        
        analysis = await self.use_llm(location_prompt)
        
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            success=True,
            data=analysis
        )


class WeatherAdvisorAgent(BaseAgent):
    """Specialized agent for weather and climate advice"""
    
    def __init__(self):
        super().__init__("weather_advisor", AgentRole.WEATHER_ADVISOR)
        self.capabilities = ["weather_forecast", "climate_analysis", "seasonal_advice"]
        
        # Add weather tool if API key is available
        weather_api_key = os.getenv("WEATHER_API_KEY")
        if weather_api_key:
            self.tools["get_weather"] = WeatherTool(weather_api_key)
    
    def get_system_prompt(self) -> str:
        return """You are a weather and climate specialist agent with expertise in:
        - Weather forecasts and analysis
        - Climate patterns and seasonal variations
        - Travel weather planning
        - Packing recommendations
        - Activity planning based on weather
        
        Provide detailed weather insights that help travelers plan better trips."""
    
    async def process_task(self, task: TaskRequest) -> TaskResult:
        """Process weather-related tasks"""
        try:
            if task.task_type == "weather_forecast" and "get_weather" in self.tools:
                return await self._handle_weather_forecast(task)
            elif task.task_type == "seasonal_advice":
                return await self._handle_seasonal_advice(task)
            else:
                response = await self.use_llm(task.description, json.dumps(task.parameters))
                
                return TaskResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    success=True,
                    data=response
                )
                
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                data=None,
                error=str(e)
            )
    
    async def _handle_weather_forecast(self, task: TaskRequest) -> TaskResult:
        """Handle weather forecast tasks"""
        params = task.parameters
        
        result = await self.tools["get_weather"].execute(
            city=params.get("city", ""),
            country_code=params.get("country_code", ""),
            forecast_days=params.get("forecast_days", 5)
        )
        
        if result.success:
            analysis_prompt = f"""
            Analyze this weather data for travel planning:
            
            {json.dumps(result.data, indent=2)}
            
            Provide:
            - Travel weather summary
            - Packing recommendations
            - Activity suggestions based on weather
            - Any weather-related travel warnings
            - Best times for outdoor activities
            
            Make it practical and actionable for travelers.
            """
            
            analysis = await self.use_llm(analysis_prompt)
            
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                data={
                    "weather_data": result.data,
                    "travel_analysis": analysis
                }
            )
        else:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=False,
                data=None,
                error=result.error
            )
    
    async def _handle_seasonal_advice(self, task: TaskRequest) -> TaskResult:
        """Handle seasonal travel advice"""
        params = task.parameters
        
        seasonal_prompt = f"""
        Provide seasonal travel advice for:
        
        Destination: {params.get('destination', 'Not specified')}
        Travel Months: {params.get('months', 'Not specified')}
        Activities: {params.get('activities', 'Not specified')}
        
        Include:
        - Seasonal weather patterns
        - Best/worst times to visit
        - Seasonal activities and events
        - Packing recommendations
        - Crowd levels and pricing impacts
        - Climate considerations
        
        Provide comprehensive seasonal guidance.
        """
        
        analysis = await self.use_llm(seasonal_prompt)
        
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            success=True,
            data=analysis
        )


# Example usage and testing
async def test_multi_agent_system():
    """Test the multi-agent system"""
    print("🤖 Testing Multi-Agent Travel System")
    print("=" * 50)
    
    # Initialize coordinator and agents
    coordinator = CoordinatorAgent()
    
    # Register specialist agents
    coordinator.register_agent(FlightSpecialistAgent())
    coordinator.register_agent(HotelSpecialistAgent())
    coordinator.register_agent(WeatherAdvisorAgent())
    
    # Test user request
    user_request = "I want to plan a 5-day trip to Tokyo in March 2024. Budget is $3000. I need flights from New York and hotel recommendations."
    
    print(f"User Request: {user_request}")
    print("\n🔄 Processing with multi-agent system...")
    
    response = await coordinator.process_user_request(user_request, "test_session")
    
    print(f"\n✅ Multi-Agent Response:\n{response}")


if __name__ == "__main__":
    asyncio.run(test_multi_agent_system())
