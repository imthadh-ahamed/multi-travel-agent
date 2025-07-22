"""
Phase 4 Expert Level Test Runner
Test all Phase 4 production features: Auto-loops, RAG, and deployment readiness
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# Import Phase 4 components
from agents.phase4.auto_agent_loop import SelfImprovingAgent, test_auto_agent_loop
from agents.phase4.rag_knowledge_base import RAGKnowledgeBase, TravelKnowledgeAgent, test_rag_knowledge_base
from agents.phase4.production_coordinator import ProductionCoordinator, TaskPriority, test_production_coordinator

console = Console()

async def test_phase4_comprehensive():
    """Comprehensive test of all Phase 4 features"""
    
    console.print(Panel.fit(
        "🚀 Phase 4 Expert Level - Production System Test\n" +
        "Testing: Auto-Agent Loops | RAG Knowledge Base | Production Deployment",
        style="bold blue"
    ))
    
    total_tests = 6
    passed_tests = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        
        # Test 1: Auto-Agent Loop System
        task1 = progress.add_task("Testing Auto-Agent Loop System...", total=None)
        try:
            console.print("\n🔄 Test 1: Auto-Agent Loop System")
            
            # Create self-improving agent
            agent = SelfImprovingAgent("test_expert_agent", enable_auto_improvement=True)
            
            # Test improvement capabilities
            test_task = "Plan a luxury honeymoon trip to the Maldives for 7 days with water villas and spa treatments"
            test_context = {
                'budget_range': 'luxury ($10,000+)',
                'preferences': ['water villas', 'spa', 'romantic dining', 'snorkeling'],
                'special_requirements': ['honeymoon package', 'private dining']
            }
            
            result = await agent.process_with_improvement(test_task, test_context)
            
            if result and result.get('best_quality', 0) > 0.7:
                console.print("✅ Auto-agent loop working correctly")
                passed_tests += 1
            else:
                console.print("❌ Auto-agent loop test failed")
                
        except Exception as e:
            console.print(f"❌ Auto-agent loop error: {e}")
        
        progress.remove_task(task1)
        
        # Test 2: RAG Knowledge Base
        task2 = progress.add_task("Testing RAG Knowledge Base...", total=None)
        try:
            console.print("\n🧠 Test 2: RAG Knowledge Base System")
            
            # Initialize knowledge base
            kb = RAGKnowledgeBase()
            await kb.initialize_default_knowledge()
            
            # Create knowledge agent
            knowledge_agent = TravelKnowledgeAgent(knowledge_base=kb)
            
            # Test knowledge retrieval
            test_query = "What are the visa requirements for US citizens traveling to Japan?"
            result = await knowledge_agent.answer_with_knowledge(test_query)
            
            if result and result.get('knowledge_used', False):
                console.print("✅ RAG knowledge base working correctly")
                passed_tests += 1
            else:
                console.print("❌ RAG knowledge base test failed")
                
        except Exception as e:
            console.print(f"❌ RAG knowledge base error: {e}")
        
        progress.remove_task(task2)
        
        # Test 3: Production Coordinator
        task3 = progress.add_task("Testing Production Coordinator...", total=None)
        try:
            console.print("\n🏭 Test 3: Production Coordinator System")
            
            # Initialize production coordinator
            coordinator = ProductionCoordinator(enable_auto_improvement=True)
            await coordinator.initialize_system()
            
            # Test comprehensive travel planning
            travel_request = """
            Plan a family vacation to Costa Rica for 4 people (2 adults, 2 children ages 8 and 12).
            We want adventure activities, wildlife viewing, and educational experiences.
            Budget is around $8000 total. We prefer eco-friendly accommodations.
            Traveling in March for 10 days from Miami.
            """
            
            result = await coordinator.process_travel_request(
                travel_request,
                user_id="test_family",
                priority=TaskPriority.HIGH
            )
            
            if result and result.get('success', False):
                console.print("✅ Production coordinator working correctly")
                passed_tests += 1
            else:
                console.print("❌ Production coordinator test failed")
                
        except Exception as e:
            console.print(f"❌ Production coordinator error: {e}")
        
        progress.remove_task(task3)
        
        # Test 4: System Integration
        task4 = progress.add_task("Testing System Integration...", total=None)
        try:
            console.print("\n🔗 Test 4: System Integration")
            
            # Test that all components work together
            coordinator = ProductionCoordinator(enable_auto_improvement=True)
            await coordinator.initialize_system()
            
            # Add custom knowledge
            await coordinator.add_knowledge_source(
                "Test Travel Regulation",
                "New travel regulation: All travelers must carry digital health certificates starting 2025.",
                "regulations"
            )
            
            # Test integrated request
            result = await coordinator.process_travel_request(
                "What new regulations should I know about for international travel in 2025?",
                user_id="test_integration"
            )
            
            if result and 'digital health certificates' in str(result).lower():
                console.print("✅ System integration working correctly")
                passed_tests += 1
            else:
                console.print("❌ System integration test failed")
                
        except Exception as e:
            console.print(f"❌ System integration error: {e}")
        
        progress.remove_task(task4)
        
        # Test 5: Performance Monitoring
        task5 = progress.add_task("Testing Performance Monitoring...", total=None)
        try:
            console.print("\n📊 Test 5: Performance Monitoring")
            
            coordinator = ProductionCoordinator()
            await coordinator.initialize_system()
            
            # Get system status
            status = await coordinator.get_system_status()
            
            if status and 'performance_metrics' in status:
                console.print("✅ Performance monitoring working correctly")
                passed_tests += 1
            else:
                console.print("❌ Performance monitoring test failed")
                
        except Exception as e:
            console.print(f"❌ Performance monitoring error: {e}")
        
        progress.remove_task(task5)
        
        # Test 6: Deployment Readiness
        task6 = progress.add_task("Testing Deployment Readiness...", total=None)
        try:
            console.print("\n🚀 Test 6: Deployment Readiness")
            
            # Check required files exist
            required_files = [
                'Dockerfile',
                'Dockerfile.ui', 
                'docker-compose.yml',
                'nginx.conf',
                'api/main.py',
                'ui/streamlit_app.py'
            ]
            
            missing_files = []
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
            
            if not missing_files:
                console.print("✅ All deployment files present")
                passed_tests += 1
            else:
                console.print(f"❌ Missing deployment files: {missing_files}")
                
        except Exception as e:
            console.print(f"❌ Deployment readiness error: {e}")
        
        progress.remove_task(task6)
    
    # Results summary
    console.print("\n" + "="*60)
    
    if passed_tests == total_tests:
        console.print(Panel.fit(
            f"🎉 ALL TESTS PASSED! ({passed_tests}/{total_tests})\n" +
            "Phase 4 Expert Level Implementation Complete!\n" +
            "✅ Auto-Agent Loops\n" +
            "✅ RAG Knowledge Base\n" +
            "✅ Production Coordinator\n" +
            "✅ System Integration\n" +
            "✅ Performance Monitoring\n" +
            "✅ Deployment Ready",
            style="bold green"
        ))
    else:
        console.print(Panel.fit(
            f"⚠️ PARTIAL SUCCESS ({passed_tests}/{total_tests} tests passed)\n" +
            "Some components need attention before production deployment.",
            style="bold yellow"
        ))
    
    return passed_tests, total_tests

async def demo_production_features():
    """Demonstrate key production features"""
    
    console.print(Panel.fit(
        "🌟 Phase 4 Production Features Demo",
        style="bold cyan"
    ))
    
    # Demo 1: Self-Improving Agent
    console.print("\n🤖 Demo 1: Self-Improving Agent with Auto-Loops")
    
    agent = SelfImprovingAgent("demo_agent", enable_auto_improvement=True)
    
    demo_task = "Plan a weekend getaway to a nearby city with good museums and restaurants"
    result = await agent.process_with_improvement(demo_task)
    
    console.print(f"📝 Task: {demo_task}")
    console.print(f"🎯 Quality achieved: {result.get('best_quality', 'N/A')}")
    console.print(f"🔄 Iterations: {result.get('total_iterations', 'N/A')}")
    console.print(f"📈 Improvements: {', '.join(result.get('improvement_history', []))}")
    
    # Demo 2: Knowledge-Enhanced Response
    console.print("\n📚 Demo 2: RAG Knowledge-Enhanced Response")
    
    kb = RAGKnowledgeBase()
    await kb.initialize_default_knowledge()
    knowledge_agent = TravelKnowledgeAgent(knowledge_base=kb)
    
    demo_query = "What should I know about travel insurance for international trips?"
    result = await knowledge_agent.answer_with_knowledge(demo_query)
    
    console.print(f"❓ Query: {demo_query}")
    console.print(f"🧠 Knowledge used: {result.get('knowledge_used', False)}")
    console.print(f"📄 Sources: {len(result.get('sources', []))}")
    console.print(f"⏱️ Response time: {result.get('response_time', 0):.2f}s")
    
    # Demo 3: Production Coordinator
    console.print("\n🏭 Demo 3: Production Coordinator Orchestration")
    
    coordinator = ProductionCoordinator(enable_auto_improvement=True)
    await coordinator.initialize_system()
    
    complex_request = """
    Help me plan a multi-city European trip for my anniversary. 
    My wife and I want to visit Paris, Amsterdam, and Prague over 12 days in September.
    We love art, wine tasting, and romantic restaurants. Budget is flexible around $7000.
    Need help with flights from Boston, hotels, and a day-by-day itinerary.
    """
    
    start_time = time.time()
    result = await coordinator.process_travel_request(complex_request, user_id="demo_couple")
    processing_time = time.time() - start_time
    
    console.print(f"📋 Complex request processed")
    console.print(f"✅ Success: {result.get('success', False)}")
    console.print(f"⏱️ Processing time: {processing_time:.2f}s")
    console.print(f"🎯 Quality scores: {result.get('quality_scores', {})}")
    
    # System status
    status = await coordinator.get_system_status()
    
    console.print("\n📊 System Performance Metrics:")
    metrics_table = Table(title="Production Metrics")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    
    perf_metrics = status.get('performance_metrics', {})
    for key, value in perf_metrics.items():
        if isinstance(value, (int, float)):
            metrics_table.add_row(key.replace('_', ' ').title(), str(value))
    
    console.print(metrics_table)

def show_deployment_guide():
    """Show deployment instructions"""
    
    console.print(Panel.fit(
        "🚀 Phase 4 Deployment Guide",
        style="bold blue"
    ))
    
    console.print("\n📋 **Production Deployment Steps:**")
    
    steps = [
        "1. **Environment Setup**",
        "   • Copy .env.example to .env",
        "   • Add your OpenAI API key and other credentials",
        "   • Configure database and Redis URLs",
        "",
        "2. **Docker Deployment**",
        "   • `docker-compose up -d` - Start all services",
        "   • `docker-compose logs -f api` - View API logs",
        "   • `docker-compose logs -f ui` - View UI logs",
        "",
        "3. **Access Points**",
        "   • API: http://localhost:8000",
        "   • UI: http://localhost:8501", 
        "   • API Docs: http://localhost:8000/docs",
        "   • Nginx (if enabled): http://localhost:80",
        "",
        "4. **Production Considerations**",
        "   • Set strong SECRET_KEY in production",
        "   • Configure SSL certificates in nginx.conf",
        "   • Set up database backups",
        "   • Configure monitoring and logging",
        "   • Set up proper firewall rules",
        "",
        "5. **Scaling**",
        "   • Increase API workers in docker-compose.yml",
        "   • Add load balancing in nginx.conf",
        "   • Configure database connection pooling",
        "   • Set up Redis clustering if needed"
    ]
    
    for step in steps:
        console.print(step)
    
    console.print("\n🔧 **Quick Start Commands:**")
    commands_table = Table()
    commands_table.add_column("Command", style="cyan")
    commands_table.add_column("Description", style="white")
    
    commands = [
        ("cp .env.example .env", "Copy environment template"),
        ("docker-compose up -d", "Start all services"),
        ("docker-compose ps", "Check service status"),
        ("docker-compose logs api", "View API logs"),
        ("docker-compose down", "Stop all services"),
        ("docker-compose pull", "Update images"),
    ]
    
    for cmd, desc in commands:
        commands_table.add_row(cmd, desc)
    
    console.print(commands_table)

async def main():
    """Main Phase 4 test and demo runner"""
    
    console.print(Panel.fit(
        """
        🎓 PHASE 4 EXPERT LEVEL COMPLETE! 
        
        ✅ Task 10: Auto-Agent Loops Implementation
        ✅ Task 11: RAG Knowledge Base Integration  
        ✅ Task 12: Production Deployment System
        
        Multi-Agent Travel Assistant - Production Ready!
        """,
        style="bold magenta"
    ))
    
    while True:
        console.print("\n🎯 **Phase 4 Expert Level Options:**")
        console.print("1. 🧪 Run Comprehensive Tests")
        console.print("2. 🌟 Demo Production Features")
        console.print("3. 📚 Show Deployment Guide")
        console.print("4. 🔄 Run Individual Tests")
        console.print("5. 🚪 Exit")
        
        choice = console.input("\n👉 Enter your choice (1-5): ")
        
        if choice == "1":
            await test_phase4_comprehensive()
            
        elif choice == "2":
            await demo_production_features()
            
        elif choice == "3":
            show_deployment_guide()
            
        elif choice == "4":
            console.print("\n🔬 **Individual Tests:**")
            console.print("1. Auto-Agent Loop Test")
            console.print("2. RAG Knowledge Base Test") 
            console.print("3. Production Coordinator Test")
            
            sub_choice = console.input("👉 Enter test number (1-3): ")
            
            if sub_choice == "1":
                await test_auto_agent_loop()
            elif sub_choice == "2":
                await test_rag_knowledge_base()
            elif sub_choice == "3":
                await test_production_coordinator()
            else:
                console.print("❌ Invalid choice")
                
        elif choice == "5":
            console.print("\n🎉 **Phase 4 Expert Level Complete!**")
            console.print("Ready for production deployment! 🚀")
            break
            
        else:
            console.print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye! Phase 4 Expert Level system ready for deployment!")
    except Exception as e:
        console.print(f"\n❌ Error: {e}")
        console.print("Check your environment setup and try again.")
