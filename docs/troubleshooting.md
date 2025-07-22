# Troubleshooting Guide 🔍

This guide helps you resolve common issues when working with the Multi-Agent Travel Assistant System.

## 🚨 Common Issues & Solutions

### Installation Issues

#### Issue: `pip install` failures
```bash
ERROR: Could not install packages due to an EnvironmentError
```

**Solutions:**
1. **Use virtual environment:**
   ```bash
   python -m venv travel_agent_env
   travel_agent_env\Scripts\activate  # Windows
   # source travel_agent_env/bin/activate  # macOS/Linux
   ```

2. **Upgrade pip:**
   ```bash
   python -m pip install --upgrade pip
   ```

3. **Install with user flag:**
   ```bash
   pip install --user -r requirements.txt
   ```

#### Issue: Python version compatibility
```bash
ERROR: Package requires Python '>=3.8' but you have '3.7'
```

**Solution:**
- Install Python 3.8 or higher from [python.org](https://python.org)
- Use pyenv for multiple Python versions:
  ```bash
  pyenv install 3.11.0
  pyenv local 3.11.0
  ```

### API Key Issues

#### Issue: OpenAI API authentication error
```bash
openai.error.AuthenticationError: Invalid API key provided
```

**Solutions:**
1. **Check API key format:**
   - Must start with `sk-`
   - Should be 51 characters long
   - No extra spaces or characters

2. **Set environment variable:**
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY="your-api-key-here"
   
   # Windows Command Prompt
   set OPENAI_API_KEY=your-api-key-here
   
   # macOS/Linux
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Create .env file:**
   ```bash
   # In project root
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

#### Issue: Weather API not working
```bash
KeyError: 'main' - Weather data not found
```

**Solutions:**
1. **Get free API key from OpenWeatherMap:**
   - Visit [openweathermap.org](https://openweathermap.org/api)
   - Sign up for free account
   - Get API key from dashboard

2. **Set weather API key:**
   ```bash
   $env:OPENWEATHERMAP_API_KEY="your-weather-api-key"
   ```

3. **Check city name format:**
   ```python
   # Good
   weather_tool.execute(city="London")
   weather_tool.execute(city="New York")
   
   # Bad
   weather_tool.execute(city="london")  # Case sensitive
   weather_tool.execute(city="NYC")     # Use full name
   ```

### Memory System Issues

#### Issue: ChromaDB connection error
```bash
chromadb.errors.ConnectionError: Could not connect to chroma
```

**Solutions:**
1. **Install ChromaDB properly:**
   ```bash
   pip install chromadb==0.4.15
   ```

2. **Clear corrupted database:**
   ```bash
   rm -rf memory/chroma_db  # Delete and recreate
   ```

3. **Check permissions:**
   ```bash
   # Ensure write permissions to memory folder
   mkdir -p memory/chroma_db
   ```

#### Issue: Memory persistence problems
```bash
FileNotFoundError: [Errno 2] No such file or directory: 'memory/conversations.json'
```

**Solution:**
```bash
# Create missing directories
mkdir memory
mkdir memory\chroma_db  # Windows
# mkdir -p memory/chroma_db  # macOS/Linux
```

### Agent Execution Issues

#### Issue: Async function errors
```bash
RuntimeError: asyncio.run() cannot be called from a running event loop
```

**Solutions:**
1. **In Jupyter notebooks:**
   ```python
   # Don't use asyncio.run() in notebooks
   # Use await directly in async cells
   
   result = await agent.process_task(task)  # Good
   # result = asyncio.run(agent.process_task(task))  # Bad
   ```

2. **In regular Python scripts:**
   ```python
   import asyncio
   
   async def main():
       result = await agent.process_task(task)
       return result
   
   if __name__ == "__main__":
       result = asyncio.run(main())
   ```

#### Issue: Tool execution timeouts
```bash
asyncio.TimeoutError: Task took longer than 30 seconds
```

**Solutions:**
1. **Increase timeout:**
   ```python
   agent.config.timeout = 60  # Increase to 60 seconds
   ```

2. **Check network connection:**
   ```python
   import aiohttp
   
   async def test_connection():
       async with aiohttp.ClientSession() as session:
           async with session.get("https://api.openai.com") as response:
               print(f"Status: {response.status}")
   ```

### Model and LLM Issues

#### Issue: Token limit exceeded
```bash
openai.error.InvalidRequestError: This model's maximum context length is 4097 tokens
```

**Solutions:**
1. **Use GPT-4 with higher limits:**
   ```python
   agent = PromptBasedTravelAgent(model="gpt-4")  # 8K tokens
   # Or use gpt-4-32k for 32K tokens
   ```

2. **Implement conversation summarization:**
   ```python
   if len(conversation) > 3000:  # Approximate token count
       summary = await memory.summarize_conversation(conversation)
       conversation = [summary] + conversation[-2:]  # Keep recent messages
   ```

3. **Truncate input:**
   ```python
   def truncate_text(text: str, max_tokens: int = 3000) -> str:
       # Rough approximation: 1 token ≈ 4 characters
       max_chars = max_tokens * 4
       return text[:max_chars] if len(text) > max_chars else text
   ```

#### Issue: Model not responding appropriately
```bash
Model keeps giving generic responses
```

**Solutions:**
1. **Improve system prompt:**
   ```python
   system_prompt = """
   You are a specialized travel assistant. Always:
   - Ask specific questions about budget, dates, preferences
   - Provide concrete recommendations with reasons
   - Include practical details like costs and logistics
   - Maintain conversation context
   """
   ```

2. **Add examples to prompt:**
   ```python
   prompt += """
   
   Example interaction:
   User: "I want to go somewhere warm"
   Assistant: "Great! To help you plan the perfect warm getaway, could you tell me:
   1. What's your approximate budget?
   2. How long are you planning to travel?
   3. Do you prefer beaches, cities, or cultural sites?
   
   Based on your preferences, I can suggest specific destinations like..."
   """
   ```

### Network and API Issues

#### Issue: Connection timeouts
```bash
aiohttp.ServerTimeoutError: Timeout on reading data from socket
```

**Solutions:**
1. **Increase timeout settings:**
   ```python
   import aiohttp
   
   timeout = aiohttp.ClientTimeout(total=60)  # 60 seconds
   async with aiohttp.ClientSession(timeout=timeout) as session:
       # Your requests here
   ```

2. **Implement retry logic:**
   ```python
   import asyncio
   
   async def retry_request(func, max_retries=3):
       for attempt in range(max_retries):
           try:
               return await func()
           except (aiohttp.ClientError, asyncio.TimeoutError) as e:
               if attempt == max_retries - 1:
                   raise e
               await asyncio.sleep(2 ** attempt)  # Exponential backoff
   ```

#### Issue: Rate limiting
```bash
openai.error.RateLimitError: Rate limit reached
```

**Solutions:**
1. **Implement backoff:**
   ```python
   import time
   import random
   
   def exponential_backoff(attempt: int) -> float:
       return min(60, (2 ** attempt) + random.uniform(0, 1))
   
   for attempt in range(5):
       try:
           response = await openai_client.chat.completions.create(...)
           break
       except openai.RateLimitError:
           wait_time = exponential_backoff(attempt)
           await asyncio.sleep(wait_time)
   ```

2. **Use token bucket pattern:**
   ```python
   class RateLimiter:
       def __init__(self, max_calls: int, time_window: int):
           self.max_calls = max_calls
           self.time_window = time_window
           self.calls = []
       
       async def acquire(self):
           now = time.time()
           self.calls = [call for call in self.calls if now - call < self.time_window]
           
           if len(self.calls) >= self.max_calls:
               sleep_time = self.time_window - (now - self.calls[0])
               await asyncio.sleep(sleep_time)
           
           self.calls.append(now)
   ```

### Performance Issues

#### Issue: Slow response times
```bash
Agent taking 30+ seconds to respond
```

**Solutions:**
1. **Use faster models:**
   ```python
   # GPT-3.5 is faster than GPT-4
   agent = PromptBasedTravelAgent(model="gpt-3.5-turbo")
   ```

2. **Optimize prompts:**
   ```python
   # Bad: Long, complex prompt
   prompt = "Please analyze this extremely detailed travel scenario..."
   
   # Good: Concise, specific prompt
   prompt = "Recommend 3 destinations for: budget $2000, warm weather, 7 days"
   ```

3. **Use parallel processing:**
   ```python
   import asyncio
   
   # Process multiple tasks concurrently
   tasks = [
       weather_tool.execute(city="Paris"),
       weather_tool.execute(city="London"),
       weather_tool.execute(city="Rome")
   ]
   results = await asyncio.gather(*tasks)
   ```

#### Issue: High memory usage
```bash
Process using excessive RAM
```

**Solutions:**
1. **Clear conversation history:**
   ```python
   # Periodically clear old conversations
   if len(agent.conversation_history) > 50:
       agent.conversation_history = agent.conversation_history[-20:]
   ```

2. **Use memory-efficient storage:**
   ```python
   # Store summaries instead of full conversations
   summary = await memory.summarize_conversation(full_conversation)
   memory.store_summary(conversation_id, summary)
   ```

## 🔍 Debugging Tips

### Enable Debug Logging

```python
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('travel_agent.log'),
        logging.StreamHandler()
    ]
)

# Enable specific loggers
logging.getLogger('openai').setLevel(logging.DEBUG)
logging.getLogger('chromadb').setLevel(logging.DEBUG)
```

### Test Individual Components

```python
# Test API connections
async def test_apis():
    # Test OpenAI
    try:
        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}]
        )
        print("✅ OpenAI API working")
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
    
    # Test Weather API
    try:
        weather_tool = WeatherTool()
        result = await weather_tool.execute(city="London")
        print("✅ Weather API working")
    except Exception as e:
        print(f"❌ Weather API error: {e}")

asyncio.run(test_apis())
```

### Monitor Token Usage

```python
def track_token_usage(func):
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        print(f"Function: {func.__name__}")
        print(f"Execution time: {end_time - start_time:.2f}s")
        if hasattr(result, 'usage'):
            print(f"Tokens used: {result.usage.total_tokens}")
        
        return result
    return wrapper
```

## 📋 Health Check Commands

Run these commands to verify system health:

```python
# Quick system check
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import openai
    print('✅ OpenAI installed')
except ImportError:
    print('❌ OpenAI not installed')

try:
    import chromadb
    print('✅ ChromaDB installed')
except ImportError:
    print('❌ ChromaDB not installed')

import os
if os.getenv('OPENAI_API_KEY'):
    print('✅ OpenAI API key found')
else:
    print('❌ OpenAI API key missing')
"
```

## 🆘 Getting Help

### 1. Check Logs
Look for error messages in:
- Console output
- `travel_agent.log` (if logging enabled)
- VS Code Problems panel

### 2. Verify Environment
```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "(openai|chromadb|langchain|aiohttp)"

# Check environment variables
echo $OPENAI_API_KEY  # macOS/Linux
echo $env:OPENAI_API_KEY  # Windows PowerShell
```

### 3. Test Basic Functionality
```python
# Run the main.py with test mode
python main.py
# Select option 1 (Rule-based) first to test basic functionality
```

### 4. Common Environment Fixes
```bash
# Reset virtual environment
deactivate
rm -rf travel_agent_env
python -m venv travel_agent_env
travel_agent_env\Scripts\activate
pip install -r requirements.txt

# Clear cache
pip cache purge
```

### 5. Contact Support
If issues persist:
1. Check existing GitHub issues
2. Create new issue with:
   - Python version
   - Operating system
   - Error messages
   - Steps to reproduce
   - Log files (if available)

---

**Remember**: Most issues are related to environment setup, API keys, or network connectivity. Work through the basics first! 🔧
