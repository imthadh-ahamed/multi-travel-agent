"""
Setup Script for Multi-Agent Travel Assistant System

This script helps users set up the project environment, install dependencies,
and configure API keys for the multi-agent travel system.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

console = Console()

# Style constants
ERROR_STYLE = "bold red"
WARNING_STYLE = "yellow"
SUCCESS_STYLE = "green"


def print_header():
    """Print setup header"""
    header = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           Multi-Agent Travel Assistant Setup                 ║
    ║                                                              ║
    ║  🛠️  Let's get your AI travel system ready! 🛠️              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(header, style="bold blue"))


def check_python_version():
    """Check if Python version is compatible"""
    console.print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        console.print("❌ Python 3.8+ is required. Please upgrade Python.", style=ERROR_STYLE)
        return False
    
    console.print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible", style=SUCCESS_STYLE)
    return True


def check_pip():
    """Check if pip is available"""
    console.print("📦 Checking pip...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True, text=True)
        console.print("✅ pip is available", style=SUCCESS_STYLE)
        return True
    except subprocess.CalledProcessError:
        console.print("❌ pip is not available. Please install pip.", style=ERROR_STYLE)
        return False


def install_dependencies():
    """Install required dependencies"""
    console.print("📚 Installing dependencies...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Installing packages...", total=None)
        
        try:
            # Install requirements
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True, check=True)
            
            progress.update(task, completed=True)
            console.print("✅ All dependencies installed successfully!", style=SUCCESS_STYLE)
            return True
            
        except subprocess.CalledProcessError as e:
            progress.update(task, completed=True)
            console.print("❌ Failed to install dependencies", style=ERROR_STYLE)
            console.print(f"Error: {e.stderr}", style="red")
            return False


def setup_environment_file():
    """Set up .env file with API keys"""
    console.print("🔑 Setting up environment configuration...")
    
    # Check if .env already exists
    if os.path.exists(".env"):
        if not Confirm.ask(".env file already exists. Do you want to reconfigure it?"):
            console.print("Skipping environment setup.", style=WARNING_STYLE)
            return True
    
    # Copy from example
    if os.path.exists(".env.example"):
        shutil.copy(".env.example", ".env")
        console.print("✅ Created .env file from template", style=SUCCESS_STYLE)
    else:
        console.print("❌ .env.example not found", style=ERROR_STYLE)
        return False
    
    # Get API keys from user
    console.print("\n🔐 Let's configure your API keys:")
    console.print("You can skip optional keys and add them later.\n")
    
    api_keys = {}
    
    # OpenAI API Key (required)
    console.print("🤖 OpenAI API Key (Required for AI features)")
    console.print("Get your key from: https://platform.openai.com/api-keys")
    openai_key = Prompt.ask("Enter your OpenAI API key", default="", show_default=False)
    if openai_key:
        api_keys["OPENAI_API_KEY"] = openai_key
    
    # Weather API Key (optional)
    console.print("\n🌤️  Weather API Key (Optional - for real weather data)")
    console.print("Get a free key from: https://openweathermap.org/api")
    if Confirm.ask("Do you want to configure Weather API?", default=False):
        weather_key = Prompt.ask("Enter your Weather API key", default="", show_default=False)
        if weather_key:
            api_keys["WEATHER_API_KEY"] = weather_key
    
    # News API Key (optional)
    console.print("\n📰 News API Key (Optional - for travel news)")
    console.print("Get a free key from: https://newsapi.org/")
    if Confirm.ask("Do you want to configure News API?", default=False):
        news_key = Prompt.ask("Enter your News API key", default="", show_default=False)
        if news_key:
            api_keys["NEWS_API_KEY"] = news_key
    
    # Update .env file
    if api_keys:
        with open(".env", "r") as f:
            content = f.read()
        
        for key, value in api_keys.items():
            content = content.replace(f"{key}=your_{key.lower()}_here", f"{key}={value}")
        
        with open(".env", "w") as f:
            f.write(content)
        
        console.print(f"✅ Configured {len(api_keys)} API key(s)", style=SUCCESS_STYLE)
    
    return True


def create_data_directories():
    """Create necessary data directories"""
    console.print("📁 Creating data directories...")
    
    directories = [
        "data/memory",
        "data/vector_store",
        "data/logs",
        "data/exports"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    console.print("✅ Data directories created", style=SUCCESS_STYLE)


def test_basic_functionality():
    """Test basic functionality"""
    console.print("🧪 Testing basic functionality...")
    
    try:
        # Test imports
        import openai
        import aiohttp
        from dotenv import load_dotenv
        
        # Load environment
        load_dotenv()
        
        # Check if OpenAI key is set
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            console.print("⚠️  OpenAI API key not configured - some features won't work", style=WARNING_STYLE)
        else:
            console.print("✅ OpenAI API key configured", style=SUCCESS_STYLE)
        
        console.print("✅ Basic functionality test passed", style=SUCCESS_STYLE)
        return True
        
    except ImportError as e:
        console.print(f"❌ Import error: {e}", style=ERROR_STYLE)
        return False
    except Exception as e:
        console.print(f"❌ Test failed: {e}", style=ERROR_STYLE)
        return False


def show_next_steps():
    """Show next steps after setup"""
    next_steps = """
🎉 Setup Complete! Here's what you can do next:

1. 🚀 Start the interactive menu:
   python main.py

2. 🎯 Run specific phases:
   python main.py --phase 1    # Rule-based agents
   python main.py --phase 2    # Tool-using agents  
   python main.py --phase 3    # Multi-agent system

3. 🔧 Test API integrations:
   python main.py --test-apis

4. 📚 Learn more:
   - Check out the README.md for detailed documentation
   - Explore the agents/ directory for code examples
   - Review the data/ directory for sample data

5. 🔑 Add more API keys later:
   - Edit the .env file to add optional API keys
   - Weather API: https://openweathermap.org/api
   - News API: https://newsapi.org/

Happy coding! 🌍✈️
    """
    
    console.print(Panel(next_steps, title="Next Steps", style="green"))


def main():
    """Main setup function"""
    print_header()
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    if not check_pip():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Setup environment
    if not setup_environment_file():
        return False
    
    # Create directories
    create_data_directories()
    
    # Test functionality
    if not test_basic_functionality():
        console.print("⚠️  Setup completed with warnings. Some features may not work properly.", style=WARNING_STYLE)
    else:
        console.print("🎉 Setup completed successfully!", style="bold green")
    
    # Show next steps
    show_next_steps()
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        console.print("\n👋 Setup cancelled by user.", style=WARNING_STYLE)
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ Setup failed with error: {e}", style=ERROR_STYLE)
        sys.exit(1)
