"""
Phase 1: Foundation - Prompt-Based Agent with OpenAI

This module implements a prompt-based travel agent using OpenAI's API.
It demonstrates how to create conversational agents that can answer
travel-related questions using natural language processing.

Learning Objectives:
- Integrate with OpenAI API
- Design effective prompts for travel assistance
- Handle API responses and errors
- Create a conversational interface
"""

import os
import openai
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ChatMessage:
    role: str  # 'system', 'user', or 'assistant'
    content: str


class PromptBasedTravelAgent:
    """
    A prompt-based travel agent that uses OpenAI's API to provide
    intelligent travel advice and recommendations.
    """
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize the agent with OpenAI API configuration
        """
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.conversation_history: List[ChatMessage] = []
        self.system_prompt = self._create_system_prompt()
        
        # Initialize with system message
        self.conversation_history.append(
            ChatMessage(role="system", content=self.system_prompt)
        )
    
    def _create_system_prompt(self) -> str:
        """
        Create a comprehensive system prompt for the travel agent
        """
        return """You are an expert travel advisor AI assistant named "TravelBot". Your role is to help users plan amazing trips by providing personalized recommendations, practical advice, and detailed travel information.

Key Responsibilities:
1. **Destination Recommendations**: Suggest destinations based on user preferences (budget, interests, travel dates, climate preferences)
2. **Trip Planning**: Help create itineraries, suggest activities, and plan travel routes
3. **Practical Advice**: Provide information about visas, weather, local customs, safety tips
4. **Budget Planning**: Help estimate costs and suggest budget-friendly alternatives
5. **Local Insights**: Share knowledge about local culture, food, attractions, and hidden gems

Guidelines:
- Always ask clarifying questions to better understand user needs
- Provide specific, actionable recommendations
- Include practical details like costs, timing, and logistics
- Be enthusiastic but realistic about travel suggestions
- Consider seasonal factors, current events, and travel restrictions
- Offer alternatives for different budget levels
- Include both popular attractions and off-the-beaten-path experiences

Response Style:
- Friendly and conversational
- Well-structured with clear sections
- Include specific examples and recommendations
- Use emojis sparingly but effectively
- Provide sources or reasoning for recommendations when helpful

Remember: You're helping create memorable travel experiences, so be helpful, accurate, and inspiring!"""
    
    def chat(self, user_message: str) -> str:
        """
        Process user message and return AI response
        """
        try:
            # Add user message to conversation
            self.conversation_history.append(
                ChatMessage(role="user", content=user_message)
            )
            
            # Prepare messages for API call
            messages = [
                {"role": msg.role, "content": msg.content}
                for msg in self.conversation_history
            ]
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            # Extract assistant response
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to conversation
            self.conversation_history.append(
                ChatMessage(role="assistant", content=assistant_message)
            )
            
            return assistant_message
            
        except openai.APIError as e:
            return f"❌ OpenAI API Error: {e}"
        except openai.RateLimitError as e:
            return "❌ Rate limit exceeded. Please try again in a moment."
        except openai.AuthenticationError as e:
            return "❌ Authentication failed. Please check your API key."
        except Exception as e:
            return f"❌ Unexpected error: {e}"
    
    def get_conversation_summary(self) -> str:
        """
        Generate a summary of the current conversation
        """
        if len(self.conversation_history) <= 1:  # Only system message
            return "No conversation yet."
        
        user_messages = [
            msg.content for msg in self.conversation_history 
            if msg.role == "user"
        ]
        
        assistant_messages = [
            msg.content for msg in self.conversation_history 
            if msg.role == "assistant"
        ]
        
        return f"""
Conversation Summary:
📊 Total Messages: {len(self.conversation_history) - 1}  # Exclude system message
👤 User Messages: {len(user_messages)}
🤖 Assistant Messages: {len(assistant_messages)}

Recent Topics Discussed:
{chr(10).join(f"  • {msg[:100]}..." for msg in user_messages[-3:])}
        """.strip()
    
    def clear_conversation(self):
        """
        Clear conversation history (keep system prompt)
        """
        self.conversation_history = [
            ChatMessage(role="system", content=self.system_prompt)
        ]
    
    def save_conversation(self, filename: str):
        """
        Save conversation to file
        """
        try:
            import json
            conversation_data = [
                {"role": msg.role, "content": msg.content}
                for msg in self.conversation_history
            ]
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Conversation saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving conversation: {e}")


def display_welcome():
    """
    Display welcome message and instructions
    """
    print("🌍✈️ Welcome to TravelBot - Your AI Travel Assistant! ✈️🌍")
    print("=" * 60)
    print("""
I'm here to help you plan amazing trips! Here's what I can do:

🎯 Destination Recommendations
   • Suggest places based on your interests and budget
   • Find hidden gems and local favorites
   
📅 Trip Planning  
   • Create detailed itineraries
   • Plan routes and logistics
   
💰 Budget Advice
   • Estimate travel costs
   • Find budget-friendly alternatives
   
🌟 Local Insights
   • Share cultural tips and customs
   • Recommend local food and experiences

Commands:
  • Type your travel questions naturally
  • Type 'summary' to see conversation overview  
  • Type 'clear' to start fresh
  • Type 'save' to save conversation
  • Type 'quit' or 'exit' to end

Let's start planning your next adventure! 🚀
    """)
    print("=" * 60)


def main():
    """
    Main CLI application for the prompt-based travel agent
    """
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key in the .env file.")
        return
    
    try:
        # Initialize the agent
        agent = PromptBasedTravelAgent()
        
        # Display welcome message
        display_welcome()
        
        print("🤖 TravelBot: Hi! I'm ready to help you plan your next trip.")
        print("What kind of travel experience are you looking for?\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 TravelBot: Safe travels! Hope to help you plan another trip soon! 👋")
                    break
                
                elif user_input.lower() == 'summary':
                    print(f"\n📊 {agent.get_conversation_summary()}\n")
                    continue
                
                elif user_input.lower() == 'clear':
                    agent.clear_conversation()
                    print("\n🔄 Conversation cleared! Starting fresh.\n")
                    continue
                
                elif user_input.lower() == 'save':
                    filename = f"conversation_{int(os.time.time())}.json"
                    agent.save_conversation(filename)
                    continue
                
                elif user_input.lower() == 'help':
                    display_welcome()
                    continue
                
                # Process travel query
                print("\n🤖 TravelBot: ", end="", flush=True)
                response = agent.chat(user_input)
                
                # Display response with typing effect (optional)
                print(response)
                print()  # Add spacing
                
            except KeyboardInterrupt:
                print("\n\n🤖 TravelBot: Thanks for chatting! Safe travels! 👋")
                break
                
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your configuration and try again.")


# Example usage and testing
def run_example_conversation():
    """
    Demonstrate the agent with example queries
    """
    agent = PromptBasedTravelAgent()
    
    example_queries = [
        "I want to plan a 7-day trip to Europe in September with a budget of $3000. What do you recommend?",
        "What are some unique experiences I can have in Japan during cherry blossom season?",
        "I'm looking for an adventure trip in South America. I love hiking and nature photography.",
        "Can you help me plan a romantic getaway for my anniversary? We prefer beach destinations."
    ]
    
    print("🧪 Running Example Conversation\n")
    
    for i, query in enumerate(example_queries, 1):
        print(f"Example {i}: {query}")
        response = agent.chat(query)
        print(f"Response: {response[:200]}...\n")
        print("-" * 50)


if __name__ == "__main__":
    # Run the main CLI application
    main()
    
    # Uncomment to run example conversation instead
    # run_example_conversation()
