# Quick Start Guide 🚀

Welcome to the Multi-Agent Travel Assistant System! This guide will help you get started quickly.

## 🎯 What You'll Build

A progressively complex AI agent system that evolves from simple rule-based bots to sophisticated multi-agent collaborations for travel planning.

## ⚡ Quick Setup (5 minutes)

### 1. Prerequisites
- Python 3.8+ installed
- Git (if cloning from repository)
- OpenAI API key (get one at https://platform.openai.com/api-keys)

### 2. Install & Setup
```bash
# Clone or download the project
git clone <your-repo-url>
cd multi-travel-agent

# Run automated setup
python setup.py
```

The setup script will:
- ✅ Check your Python version
- 📦 Install all dependencies
- 🔑 Help configure API keys
- 📁 Create necessary directories
- 🧪 Test basic functionality

### 3. Start Exploring
```bash
# Launch interactive menu
python main.py

# Or run specific phases
python main.py --phase 1    # Beginner level
python main.py --phase 2    # Intermediate level
python main.py --phase 3    # Advanced level
```

## 🎓 Learning Path

### Phase 1: Foundation (🟢 Beginner)
**What you'll learn:** Basic AI agent concepts, rule-based systems, prompt engineering

**Try this:**
```bash
python main.py --phase 1
```

**What it does:**
- Rule-based travel advisor with CLI interface
- Prompt-based agent using OpenAI API
- Basic user preference handling

### Phase 2: Integration (🟡 Intermediate) 
**What you'll learn:** Tool integration, memory systems, API usage

**Try this:**
```bash
python main.py --phase 2
```

**What it does:**
- Agents that use external APIs (weather, flights, hotels)
- Memory systems for user preferences
- Conversation history and summarization

### Phase 3: Collaboration (🟠 Advanced)
**What you'll learn:** Multi-agent systems, coordination, delegation

**Try this:**
```bash
python main.py --phase 3
```

**What it does:**
- Specialized agents (flight, hotel, weather experts)
- Coordinator agent that delegates tasks
- Inter-agent communication and collaboration

### Phase 4: Production (🔴 Expert)
**Coming soon:** Auto-loops, RAG, deployment-ready systems

## 🎯 Quick Examples

### Example 1: Rule-Based Advisor
```
You: "I want a tropical vacation under $2000"
Agent: Analyzes budget, climate preferences, suggests Bali or Thailand
```

### Example 2: Tool-Using Agent
```
You: "What's the weather like in Paris next week?"
Agent: Uses Weather API → Returns actual forecast + travel advice
```

### Example 3: Multi-Agent System
```
You: "Plan a 5-day Tokyo trip in March, $3000 budget"
System: 
- Flight Agent: Finds flights from your location
- Hotel Agent: Searches accommodations in best areas
- Weather Agent: Provides March weather forecast
- Coordinator: Combines all info into comprehensive plan
```

## 🔧 Configuration

### API Keys (Optional but Recommended)
- **OpenAI**: Required for AI features - Get at https://platform.openai.com/
- **Weather**: Optional for real weather data - Get at https://openweathermap.org/
- **News**: Optional for travel news - Get at https://newsapi.org/

### Environment File (.env)
```env
OPENAI_API_KEY=your_openai_key_here
WEATHER_API_KEY=your_weather_key_here  # Optional
NEWS_API_KEY=your_news_key_here        # Optional
```

## 🎮 Interactive Commands

When you run `python main.py`, you'll see:

```
1. Run Phase 1 - Rule-Based Advisor
2. Run Phase 1 - Prompt-Based Agent  
3. Run Phase 2 - Tool-Using Agent
4. Test Phase 2 - Memory Systems
5. Run Phase 3 - Multi-Agent System
6. Test External APIs
7. Check Environment
q. Quit
```

## 🚨 Troubleshooting

### "Import Error" or "Module not found"
```bash
pip install -r requirements.txt
```

### "OpenAI API Error"
- Check your API key in `.env` file
- Ensure you have credits in your OpenAI account
- Verify the key hasn't expired

### "ChromaDB Error" 
- This is normal if you haven't used Phase 2+ yet
- ChromaDB creates databases automatically on first use

### "Weather API Error"
- Weather features are optional
- System will work with mock data if API key not provided

## 🎯 What to Try First

1. **Start Simple**: Run Phase 1 rule-based advisor
2. **Add AI**: Try Phase 1 prompt-based agent with your OpenAI key
3. **Add Tools**: Explore Phase 2 with external API integration
4. **Go Advanced**: Experience Phase 3 multi-agent collaboration

## 🎉 You're Ready!

Run this command and start exploring:
```bash
python main.py
```

**Happy coding! 🌍✈️**

---

Need help? Check out:
- `README.md` for detailed documentation
- `agents/` directory for code examples  
- `data/` directory for sample data
- Issues section if you find bugs
