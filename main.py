"""
Multi-Agent Travel Assistant System - Main Runner

This script provides a unified interface to run different phases of the 
travel agent system. It allows you to progressively explore and test
each phase of development.
"""

import sys
import os
import asyncio
from typing import Dict, Any
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

console = Console()

# Style constants
ERROR_STYLE = "bold red"
WARNING_STYLE = "yellow"
SUCCESS_STYLE = "green"


def display_banner():
    """Display the application banner"""
    banner = Text("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              Multi-Agent Travel Assistant System             ║
    ║                                                              ║
    ║  🌍 Progressive AI Agent Development Learning Platform 🌍    ║
    ╚══════════════════════════════════════════════════════════════╝
    """, style="bold blue")
    
    console.print(banner)


def display_phase_menu():
    """Display the phase selection menu"""
    table = Table(title="Available Development Phases", show_header=True, header_style="bold magenta")
    table.add_column("Phase", style="dim", width=12)
    table.add_column("Level", width=12)
    table.add_column("Description", width=50)
    table.add_column("Features", width=30)
    
    table.add_row(
        "Phase 1", 
        "🟢 Beginner", 
        "Foundation - Basic agent concepts and rule-based systems",
        "Rule-based logic, CLI interface"
    )
    table.add_row(
        "Phase 2", 
        "🟡 Intermediate", 
        "Integration - Tool usage, memory, and API integration",
        "External APIs, Memory systems"
    )
    table.add_row(
        "Phase 3", 
        "🟠 Advanced", 
        "Multi-Agent - Collaborative agent systems",
        "Agent coordination, Delegation"
    )
    table.add_row(
        "Phase 4", 
        "🔴 Expert", 
        "Production - Auto-loops, RAG, and deployment",
        "Auto-agents, RAG, Production ready"
    )
    
    console.print(table)
    console.print()


def run_phase1_rule_based():
    """Run Phase 1: Rule-based travel advisor"""
    try:
        from agents.phase1.rule_based_advisor import main
        console.print("🟢 Starting Phase 1: Rule-Based Travel Advisor")
        console.print("=" * 60)
        main()
    except ImportError as e:
        console.print(f"❌ Error importing Phase 1 modules: {e}", style=ERROR_STYLE)
    except Exception as e:
        console.print(f"❌ Error running Phase 1: {e}", style=ERROR_STYLE)


def run_phase1_prompt_based():
    """Run Phase 1: Prompt-based agent"""
    try:
        from agents.phase1.prompt_based_agent import main
        console.print("🟢 Starting Phase 1: Prompt-Based Agent")
        console.print("=" * 60)
        main()
    except ImportError as e:
        console.print(f"❌ Error importing Phase 1 modules: {e}", style=ERROR_STYLE)
    except Exception as e:
        console.print(f"❌ Error running Phase 1: {e}", style=ERROR_STYLE)


async def run_phase2_tool_using():
    """Run Phase 2: Tool-using agent"""
    try:
        from agents.phase2.tool_using_agent import main
        console.print("🟡 Starting Phase 2: Tool-Using Agent")
        console.print("=" * 60)
        await main()
    except ImportError as e:
        console.print(f"❌ Error importing Phase 2 modules: {e}", style="bold red")
    except Exception as e:
        console.print(f"❌ Error running Phase 2: {e}", style="bold red")


async def run_phase2_memory_test():
    """Test Phase 2: Memory systems"""
    try:
        from agents.phase2.memory_system import test_memory_systems
        console.print("🟡 Testing Phase 2: Memory Systems")
        console.print("=" * 60)
        test_memory_systems()
    except ImportError as e:
        console.print(f"❌ Error importing Phase 2 modules: {e}", style=ERROR_STYLE)
    except Exception as e:
        console.print(f"❌ Error testing Phase 2: {e}", style=ERROR_STYLE)


async def run_phase3_multi_agent():
    """Run Phase 3: Multi-agent system"""
    try:
        from agents.phase3.multi_agent_system import test_multi_agent_system
        console.print("🟠 Starting Phase 3: Multi-Agent System")
        console.print("=" * 60)
        await test_multi_agent_system()
    except ImportError as e:
        console.print(f"❌ Error importing Phase 3 modules: {e}", style=ERROR_STYLE)
    except Exception as e:
        console.print(f"❌ Error running Phase 3: {e}", style=ERROR_STYLE)


async def run_phase4_expert():
    """Run Phase 4: Expert-level production system"""
    try:
        from agents.phase4.test_phase4 import main as phase4_main
        console.print("🔴 Starting Phase 4: Expert Production System")
        console.print("=" * 60)
        await phase4_main()
    except ImportError as e:
        console.print(f"❌ Error importing Phase 4 modules: {e}", style=ERROR_STYLE)
    except Exception as e:
        console.print(f"❌ Error running Phase 4: {e}", style=ERROR_STYLE)


async def test_apis():
    """Test external API integrations"""
    try:
        from tools.external_apis import test_apis
        console.print("🔧 Testing External API Integrations")
        console.print("=" * 60)
        await test_apis()
    except ImportError as e:
        console.print(f"❌ Error importing API modules: {e}", style=ERROR_STYLE)
    except Exception as e:
        console.print(f"❌ Error testing APIs: {e}", style=ERROR_STYLE)


def check_environment():
    """Check if the environment is properly configured"""
    console.print("🔍 Checking Environment Configuration...")
    
    # Check for .env file
    env_file = ".env"
    if os.path.exists(env_file):
        console.print("✅ .env file found", style="green")
    else:
        console.print("⚠️  .env file not found. Copy .env.example to .env and configure your API keys.", style=WARNING_STYLE)
    
    # Check for required dependencies
    missing_deps = []
    
    try:
        import openai
        console.print("✅ OpenAI library available", style=SUCCESS_STYLE)
    except ImportError:
        missing_deps.append("openai")
    
    try:
        import chromadb
        console.print("✅ ChromaDB library available", style=SUCCESS_STYLE)
    except ImportError:
        missing_deps.append("chromadb")
    
    try:
        import aiohttp
        console.print("✅ aiohttp library available", style=SUCCESS_STYLE)
    except ImportError:
        missing_deps.append("aiohttp")
    
    if missing_deps:
        console.print(f"❌ Missing dependencies: {', '.join(missing_deps)}", style=ERROR_STYLE)
        console.print("Run: pip install -r requirements.txt", style=WARNING_STYLE)
        return False
    
    # Check API keys
    api_keys = {
        "OPENAI_API_KEY": "OpenAI API (Required for most features)",
        "WEATHER_API_KEY": "Weather API (Optional - for real weather data)",
        "NEWS_API_KEY": "News API (Optional - for travel news)"
    }
    
    for key, description in api_keys.items():
        if os.getenv(key):
            console.print(f"✅ {key} configured", style=SUCCESS_STYLE)
        else:
            style = ERROR_STYLE if key == "OPENAI_API_KEY" else WARNING_STYLE
            console.print(f"⚠️  {key} not set - {description}", style=style)
    
    console.print()
    return True


async def interactive_menu():
    """Interactive menu for selecting phases and options"""
    display_banner()
    
    if not check_environment():
        console.print("❌ Environment not properly configured. Please fix the issues above.", style=ERROR_STYLE)
        return
    
    while True:
        display_phase_menu()
        
        console.print("Available Commands:", style="bold")
        console.print("1. Run Phase 1 - Rule-Based Advisor")
        console.print("2. Run Phase 1 - Prompt-Based Agent")
        console.print("3. Run Phase 2 - Tool-Using Agent")
        console.print("4. Test Phase 2 - Memory Systems")
        console.print("5. Run Phase 3 - Multi-Agent System")
        console.print("6. Run Phase 4 - Expert Production System")
        console.print("7. Test External APIs")
        console.print("8. Check Environment")
        console.print("q. Quit")
        console.print()
        
        choice = console.input("Select an option: ").strip().lower()
        
        if choice == 'q' or choice == 'quit':
            console.print("👋 Thank you for using the Multi-Agent Travel Assistant!")
            break
        elif choice == '1':
            run_phase1_rule_based()
        elif choice == '2':
            run_phase1_prompt_based()
        elif choice == '3':
            await run_phase2_tool_using()
        elif choice == '4':
            await run_phase2_memory_test()
        elif choice == '5':
            await run_phase3_multi_agent()
        elif choice == '6':
            await run_phase4_expert()
        elif choice == '7':
            await test_apis()
        elif choice == '8':
            check_environment()
        else:
            console.print("❌ Invalid choice. Please try again.", style=ERROR_STYLE)
        
        console.print()
        console.input("Press Enter to continue...")
        console.clear()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Multi-Agent Travel Assistant System")
    parser.add_argument("--phase", choices=["1", "2", "3", "4"], help="Run specific phase")
    parser.add_argument("--test-apis", action="store_true", help="Test API integrations")
    parser.add_argument("--check-env", action="store_true", help="Check environment configuration")
    parser.add_argument("--interactive", action="store_true", default=True, help="Run interactive menu (default)")
    
    args = parser.parse_args()
    
    if args.check_env:
        check_environment()
    elif args.test_apis:
        asyncio.run(test_apis())
    elif args.phase == "1":
        console.print("Phase 1 has multiple components. Use --interactive for selection.")
        asyncio.run(interactive_menu())
    elif args.phase == "2":
        asyncio.run(run_phase2_tool_using())
    elif args.phase == "3":
        asyncio.run(run_phase3_multi_agent())
    elif args.phase == "4":
        asyncio.run(run_phase4_expert())
    else:
        # Default to interactive menu
        asyncio.run(interactive_menu())


if __name__ == "__main__":
    main()
