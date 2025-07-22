"""
Phase 2: Intermediate Integration - Tool-Using Agent

This module implements an agent that can access external tools like weather APIs,
flight search, and other travel-related services. It demonstrates how to extend
agent capabilities with real-world data integration.

Learning Objectives:
- Integrate external APIs as tools
- Create tool abstractions and interfaces
- Handle API errors and rate limits
- Combine LLM reasoning with real-time data
"""

import os
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import openai
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Tool(ABC):
    """Abstract base class for agent tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the LLM"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters schema"""
        pass


class WeatherTool(Tool):
    """Tool for fetching weather information"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "http://api.openweathermap.org/data/2.5"
    
    @property
    def name(self) -> str:
        return "get_weather"
    
    @property
    def description(self) -> str:
        return "Get current weather and forecast for a city"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get weather for"
                },
                "country_code": {
                    "type": "string",
                    "description": "Optional country code (e.g., 'US', 'FR')"
                },
                "forecast_days": {
                    "type": "integer",
                    "description": "Number of forecast days (1-5)",
                    "default": 3
                }
            },
            "required": ["city"]
        }
    
    async def execute(self, city: str, country_code: str = "", forecast_days: int = 3) -> ToolResult:
        """Fetch weather data for a city"""
        try:
            location = f"{city},{country_code}" if country_code else city
            
            async with aiohttp.ClientSession() as session:
                # Current weather
                current_url = f"{self.base_url}/weather"
                current_params = {
                    "q": location,
                    "appid": self.api_key,
                    "units": "metric"
                }
                
                async with session.get(current_url, params=current_params) as response:
                    if response.status != 200:
                        return ToolResult(
                            success=False,
                            data=None,
                            error=f"Weather API error: {response.status}"
                        )
                    
                    current_data = await response.json()
                
                # Forecast
                forecast_url = f"{self.base_url}/forecast"
                forecast_params = {
                    "q": location,
                    "appid": self.api_key,
                    "units": "metric",
                    "cnt": forecast_days * 8  # 8 forecasts per day (3-hour intervals)
                }
                
                async with session.get(forecast_url, params=forecast_params) as response:
                    if response.status == 200:
                        forecast_data = await response.json()
                    else:
                        forecast_data = None
                
                weather_info = self._format_weather_data(current_data, forecast_data)
                
                return ToolResult(
                    success=True,
                    data=weather_info,
                    metadata={"location": location, "forecast_days": forecast_days}
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Weather tool error: {str(e)}"
            )
    
    def _format_weather_data(self, current: Dict, forecast: Optional[Dict]) -> Dict[str, Any]:
        """Format weather data into a readable structure"""
        result = {
            "current": {
                "location": f"{current['name']}, {current['sys']['country']}",
                "temperature": current['main']['temp'],
                "feels_like": current['main']['feels_like'],
                "humidity": current['main']['humidity'],
                "description": current['weather'][0]['description'].title(),
                "wind_speed": current['wind']['speed'],
                "visibility": current.get('visibility', 0) / 1000  # Convert to km
            }
        }
        
        if forecast:
            daily_forecasts = []
            for item in forecast['list'][::8]:  # Every 8th item (daily)
                daily_forecasts.append({
                    "date": item['dt_txt'].split()[0],
                    "temperature": item['main']['temp'],
                    "description": item['weather'][0]['description'].title(),
                    "humidity": item['main']['humidity'],
                    "wind_speed": item['wind']['speed']
                })
            
            result["forecast"] = daily_forecasts
        
        return result


class FlightSearchTool(Tool):
    """Tool for searching flights (mock implementation)"""
    
    @property
    def name(self) -> str:
        return "search_flights"
    
    @property
    def description(self) -> str:
        return "Search for flights between cities"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Origin city or airport code"
                },
                "destination": {
                    "type": "string",
                    "description": "Destination city or airport code"
                },
                "departure_date": {
                    "type": "string",
                    "description": "Departure date (YYYY-MM-DD)"
                },
                "return_date": {
                    "type": "string",
                    "description": "Return date (YYYY-MM-DD) for round trip"
                },
                "passengers": {
                    "type": "integer",
                    "description": "Number of passengers",
                    "default": 1
                }
            },
            "required": ["origin", "destination", "departure_date"]
        }
    
    async def execute(self, origin: str, destination: str, departure_date: str, 
                     return_date: str = None, passengers: int = 1) -> ToolResult:
        """Mock flight search implementation"""
        try:
            # Simulate API delay
            await asyncio.sleep(1)
            
            # Mock flight data
            flights = [
                {
                    "airline": "SkyWings Airlines",
                    "flight_number": "SW123",
                    "departure_time": "08:30",
                    "arrival_time": "14:45",
                    "duration": "6h 15m",
                    "price": 299,
                    "stops": 0,
                    "aircraft": "Boeing 737"
                },
                {
                    "airline": "Global Airways", 
                    "flight_number": "GA456",
                    "departure_time": "12:15",
                    "arrival_time": "19:30",
                    "duration": "7h 15m",
                    "price": 245,
                    "stops": 1,
                    "aircraft": "Airbus A320"
                },
                {
                    "airline": "Premium Air",
                    "flight_number": "PA789",
                    "departure_time": "18:00",
                    "arrival_time": "23:20",
                    "duration": "5h 20m", 
                    "price": 399,
                    "stops": 0,
                    "aircraft": "Boeing 787"
                }
            ]
            
            search_result = {
                "origin": origin,
                "destination": destination,
                "departure_date": departure_date,
                "return_date": return_date,
                "passengers": passengers,
                "flights": flights,
                "search_time": datetime.now().isoformat(),
                "currency": "USD"
            }
            
            return ToolResult(
                success=True,
                data=search_result,
                metadata={"total_flights": len(flights)}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Flight search error: {str(e)}"
            )


class HotelSearchTool(Tool):
    """Tool for searching hotels (mock implementation)"""
    
    @property
    def name(self) -> str:
        return "search_hotels"
    
    @property
    def description(self) -> str:
        return "Search for hotels in a city"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City to search hotels in"
                },
                "check_in": {
                    "type": "string",
                    "description": "Check-in date (YYYY-MM-DD)"
                },
                "check_out": {
                    "type": "string",
                    "description": "Check-out date (YYYY-MM-DD)"
                },
                "guests": {
                    "type": "integer",
                    "description": "Number of guests",
                    "default": 2
                },
                "max_price": {
                    "type": "number",
                    "description": "Maximum price per night"
                }
            },
            "required": ["city", "check_in", "check_out"]
        }
    
    async def execute(self, city: str, check_in: str, check_out: str, 
                     guests: int = 2, max_price: float = None) -> ToolResult:
        """Mock hotel search implementation"""
        try:
            await asyncio.sleep(0.5)  # Simulate API delay
            
            hotels = [
                {
                    "name": "Grand Plaza Hotel",
                    "rating": 4.5,
                    "price_per_night": 150,
                    "total_price": 450,  # 3 nights
                    "amenities": ["WiFi", "Pool", "Gym", "Restaurant", "Spa"],
                    "distance_from_center": "0.5 km",
                    "cancellation": "Free cancellation"
                },
                {
                    "name": "Boutique City Inn",
                    "rating": 4.2,
                    "price_per_night": 95,
                    "total_price": 285,
                    "amenities": ["WiFi", "Restaurant", "24h Reception"],
                    "distance_from_center": "1.2 km",
                    "cancellation": "Free cancellation until 24h"
                },
                {
                    "name": "Luxury Suites & Resort",
                    "rating": 4.8,
                    "price_per_night": 280,
                    "total_price": 840,
                    "amenities": ["WiFi", "Pool", "Gym", "Spa", "Concierge", "Valet"],
                    "distance_from_center": "2.0 km",
                    "cancellation": "Non-refundable"
                }
            ]
            
            # Filter by max price if specified
            if max_price:
                hotels = [h for h in hotels if h["price_per_night"] <= max_price]
            
            search_result = {
                "city": city,
                "check_in": check_in,
                "check_out": check_out,
                "guests": guests,
                "hotels": hotels,
                "search_time": datetime.now().isoformat(),
                "currency": "USD"
            }
            
            return ToolResult(
                success=True,
                data=search_result,
                metadata={"total_hotels": len(hotels)}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Hotel search error: {str(e)}"
            )


class ToolUsingAgent:
    """
    An agent that can use external tools to provide enhanced travel assistance
    """
    
    def __init__(self, model: str = "gpt-4"):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.tools: Dict[str, Tool] = {}
        self.conversation_history = []
        
        # Initialize tools
        self._setup_tools()
        
        # Add system message
        self.conversation_history.append({
            "role": "system",
            "content": self._create_system_prompt()
        })
    
    def _setup_tools(self):
        """Initialize available tools"""
        weather_api_key = os.getenv("WEATHER_API_KEY")
        if weather_api_key:
            self.tools["get_weather"] = WeatherTool(weather_api_key)
        
        self.tools["search_flights"] = FlightSearchTool()
        self.tools["search_hotels"] = HotelSearchTool()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt that includes tool descriptions"""
        tool_descriptions = []
        for tool in self.tools.values():
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        
        tools_text = "\n".join(tool_descriptions) if tool_descriptions else "No tools available"
        
        return f"""You are an advanced travel assistant with access to real-time tools and data. You can help users plan trips by providing personalized recommendations and accessing live information.

Available Tools:
{tools_text}

Capabilities:
1. **Real-time Weather**: Get current weather and forecasts for any destination
2. **Flight Search**: Find flights between cities with pricing and schedules  
3. **Hotel Search**: Search accommodations with filtering options
4. **Trip Planning**: Combine tool data with your knowledge to create comprehensive plans

Guidelines:
- Use tools when you need real-time or specific data
- Always explain what tools you're using and why
- Combine tool results with your travel expertise
- Provide actionable recommendations based on tool data
- Handle tool errors gracefully and suggest alternatives
- Be specific about dates, prices, and logistics

When using tools:
1. Clearly state what information you're fetching
2. Call the appropriate tool with correct parameters
3. Interpret and explain the results in context
4. Provide recommendations based on the data

You should proactively suggest using tools when they would be helpful for the user's request."""
    
    async def chat(self, user_message: str) -> str:
        """Process user message with tool integration"""
        try:
            # Add user message
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Prepare tools for OpenAI function calling
            tool_definitions = []
            for tool in self.tools.values():
                tool_definitions.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters
                    }
                })
            
            # Call OpenAI with tools
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=tool_definitions if tool_definitions else None,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=2000
            )
            
            assistant_message = response.choices[0].message
            
            # Handle tool calls
            if assistant_message.tool_calls:
                return await self._handle_tool_calls(assistant_message)
            else:
                # Regular response without tool use
                content = assistant_message.content
                self.conversation_history.append({
                    "role": "assistant",
                    "content": content
                })
                return content
                
        except Exception as e:
            return f"❌ Error processing request: {str(e)}"
    
    async def _handle_tool_calls(self, assistant_message) -> str:
        """Handle tool calls from the assistant"""
        # Add assistant message with tool calls
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in assistant_message.tool_calls
            ]
        })
        
        # Execute tools
        tool_results = []
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            if tool_name in self.tools:
                print(f"🔧 Using {tool_name} with parameters: {tool_args}")
                result = await self.tools[tool_name].execute(**tool_args)
                
                if result.success:
                    tool_message = json.dumps(result.data, indent=2)
                    print(f"✅ Tool {tool_name} executed successfully")
                else:
                    tool_message = f"Error: {result.error}"
                    print(f"❌ Tool {tool_name} failed: {result.error}")
            else:
                tool_message = f"Error: Tool {tool_name} not available"
                print(f"❌ Tool {tool_name} not found")
            
            # Add tool result to conversation
            self.conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_message
            })
        
        # Get final response from assistant
        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            temperature=0.7,
            max_tokens=2000
        )
        
        final_content = final_response.choices[0].message.content
        
        # Add final response to conversation
        self.conversation_history.append({
            "role": "assistant",
            "content": final_content
        })
        
        return final_content
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return list(self.tools.keys())
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = [self.conversation_history[0]]  # Keep system message


async def main():
    """Main CLI application"""
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        return
    
    agent = ToolUsingAgent()
    
    print("🛠️ Tool-Using Travel Agent")
    print("=" * 50)
    print(f"Available tools: {', '.join(agent.get_available_tools())}")
    print("\nI can help you plan trips using real-time data!")
    print("Try asking about weather, flights, or hotels.\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                agent.clear_conversation()
                print("🔄 Conversation cleared!\n")
                continue
            
            if not user_input:
                continue
            
            print("\n🤖 Assistant: ", end="", flush=True)
            response = await agent.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
