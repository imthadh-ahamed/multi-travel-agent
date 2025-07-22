"""
Production Coordinator for Phase 4
Combines auto-agent loops with RAG knowledge base for production-ready system
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import uuid
import openai

from .auto_agent_loop import SelfImprovingAgent, AutoAgentLoop, LoopStatus
from .rag_knowledge_base import RAGKnowledgeBase, TravelKnowledgeAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProductionTask:
    """Enhanced task for production system"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: str = ""
    task_type: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # Results and metadata
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    agent_assignments: List[str] = field(default_factory=list)
    execution_log: List[Dict] = field(default_factory=list)
    quality_score: float = 0.0
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dependent_tasks: List[str] = field(default_factory=list)
    
    def add_log_entry(self, message: str, level: str = "INFO", agent_id: str = None):
        """Add entry to execution log"""
        self.execution_log.append({
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'level': level,
            'agent_id': agent_id
        })
    
    def get_execution_time(self) -> Optional[float]:
        """Get task execution time in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

class ProductionCoordinator:
    """
    Production-ready coordinator that combines all Phase 4 capabilities:
    - Auto-agent loops for continuous improvement
    - RAG knowledge base for informed decisions
    - Task prioritization and routing
    - Performance monitoring and optimization
    """
    
    def __init__(self, 
                 model: str = "gpt-4",
                 enable_auto_improvement: bool = True,
                 max_concurrent_tasks: int = 10):
        self.coordinator_id = "production_coordinator"
        self.model = model
        self.enable_auto_improvement = enable_auto_improvement
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Initialize core components
        self.knowledge_base = RAGKnowledgeBase()
        self.knowledge_agent = TravelKnowledgeAgent(knowledge_base=self.knowledge_base)
        
        # Initialize specialized agents
        self.agents = self._initialize_agents()
        
        # Task management
        self.active_tasks: Dict[str, ProductionTask] = {}
        self.completed_tasks: Dict[str, ProductionTask] = {}
        self.task_queue: List[ProductionTask] = []
        
        # Performance monitoring
        self.performance_metrics = {
            'total_tasks_processed': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'average_execution_time': 0.0,
            'average_quality_score': 0.0,
            'agent_utilization': {},
            'knowledge_base_queries': 0,
            'auto_improvements_applied': 0
        }
        
        # Configuration
        self.config = {
            'task_timeout': 300,  # 5 minutes
            'quality_threshold': 0.8,
            'max_retries': 3,
            'enable_learning': True,
            'parallel_execution': True
        }
    
    def _initialize_agents(self) -> Dict[str, SelfImprovingAgent]:
        """Initialize specialized agents for different tasks"""
        agents = {}
        
        # Flight specialist with auto-improvement
        agents['flight_specialist'] = SelfImprovingAgent(
            agent_id="flight_specialist",
            model=self.model,
            enable_auto_improvement=self.enable_auto_improvement
        )
        
        # Hotel specialist with auto-improvement
        agents['hotel_specialist'] = SelfImprovingAgent(
            agent_id="hotel_specialist", 
            model=self.model,
            enable_auto_improvement=self.enable_auto_improvement
        )
        
        # Budget advisor with auto-improvement
        agents['budget_advisor'] = SelfImprovingAgent(
            agent_id="budget_advisor",
            model=self.model,
            enable_auto_improvement=self.enable_auto_improvement
        )
        
        # Itinerary planner with auto-improvement
        agents['itinerary_planner'] = SelfImprovingAgent(
            agent_id="itinerary_planner",
            model=self.model,
            enable_auto_improvement=self.enable_auto_improvement
        )
        
        # General travel advisor
        agents['general_advisor'] = SelfImprovingAgent(
            agent_id="general_advisor",
            model=self.model,
            enable_auto_improvement=self.enable_auto_improvement
        )
        
        return agents
    
    async def initialize_system(self):
        """Initialize the production system"""
        logger.info("🚀 Initializing Production Coordinator System...")
        
        try:
            # Initialize knowledge base with default travel information
            await self.knowledge_base.initialize_default_knowledge()
            
            # Initialize agent performance tracking
            for agent_id in self.agents:
                self.performance_metrics['agent_utilization'][agent_id] = {
                    'tasks_assigned': 0,
                    'tasks_completed': 0,
                    'average_quality': 0.0,
                    'total_execution_time': 0.0
                }
            
            logger.info("✅ Production system initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize production system: {e}")
            raise
    
    async def process_travel_request(self, 
                                   user_request: str,
                                   user_id: str = "anonymous",
                                   session_id: str = None,
                                   priority: TaskPriority = TaskPriority.MEDIUM) -> Dict[str, Any]:
        """
        Process a comprehensive travel request using all production capabilities
        """
        session_id = session_id or str(uuid.uuid4())
        
        try:
            # Create main task
            main_task = ProductionTask(
                user_id=user_id,
                session_id=session_id,
                task_type="comprehensive_travel_planning",
                description=user_request,
                priority=priority,
                parameters={'original_request': user_request}
            )
            
            main_task.add_log_entry(f"Received travel request: {user_request[:100]}...")
            
            # Analyze request and create subtasks
            subtasks = await self._analyze_and_create_subtasks(main_task)
            
            # Execute subtasks with coordination
            results = await self._execute_coordinated_planning(main_task, subtasks)
            
            # Synthesize final response
            final_response = await self._synthesize_travel_plan(main_task, results)
            
            # Update performance metrics
            self._update_performance_metrics(main_task)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Failed to process travel request: {e}")
            return {
                'success': False,
                'error': str(e),
                'user_request': user_request,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_and_create_subtasks(self, main_task: ProductionTask) -> List[ProductionTask]:
        """Analyze request and create appropriate subtasks"""
        user_request = main_task.description
        
        # Use knowledge base to understand the request better
        knowledge_context = await self.knowledge_base.get_context_for_query(
            f"travel planning for: {user_request}"
        )
        
        # Create analysis task
        analysis_prompt = f"""
        Analyze this travel request and determine what specific tasks are needed:
        
        Request: {user_request}
        
        Knowledge Context:
        {knowledge_context}
        
        Determine which of these tasks are needed (respond with JSON):
        - flight_search: Search for flights
        - hotel_search: Find accommodation
        - budget_planning: Calculate costs and budget
        - itinerary_planning: Create day-by-day itinerary
        - destination_info: Provide destination information
        - documentation_check: Check visa/passport requirements
        - weather_check: Get weather information
        
        Return JSON with task types as keys and task descriptions as values.
        Only include tasks that are clearly needed based on the request.
        """
        
        try:
            client = openai.AsyncOpenAI()
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3
            )
            
            # Parse the response to extract needed tasks
            task_analysis = json.loads(response.choices[0].message.content)
            
            subtasks = []
            for task_type, description in task_analysis.items():
                subtask = ProductionTask(
                    user_id=main_task.user_id,
                    session_id=main_task.session_id,
                    task_type=task_type,
                    description=description,
                    priority=main_task.priority,
                    parameters={'parent_task': main_task.task_id, 'context': knowledge_context}
                )
                subtasks.append(subtask)
            
            main_task.add_log_entry(f"Created {len(subtasks)} subtasks: {list(task_analysis.keys())}")
            return subtasks
            
        except Exception as e:
            logger.error(f"Failed to analyze request: {e}")
            # Fallback to default subtasks
            return self._create_default_subtasks(main_task)
    
    def _create_default_subtasks(self, main_task: ProductionTask) -> List[ProductionTask]:
        """Create default subtasks when analysis fails"""
        default_tasks = [
            ('destination_info', 'Provide general destination information'),
            ('budget_planning', 'Create budget estimates'),
            ('itinerary_planning', 'Suggest activities and itinerary')
        ]
        
        subtasks = []
        for task_type, description in default_tasks:
            subtask = ProductionTask(
                user_id=main_task.user_id,
                session_id=main_task.session_id,
                task_type=task_type,
                description=description,
                priority=main_task.priority,
                parameters={'parent_task': main_task.task_id}
            )
            subtasks.append(subtask)
        
        return subtasks
    
    async def _execute_coordinated_planning(self, 
                                          main_task: ProductionTask,
                                          subtasks: List[ProductionTask]) -> Dict[str, Any]:
        """Execute subtasks with coordination and auto-improvement"""
        results = {}
        
        # Execute tasks based on dependencies and priorities
        for subtask in sorted(subtasks, key=lambda t: t.priority.value, reverse=True):
            try:
                subtask.status = TaskStatus.IN_PROGRESS
                subtask.started_at = datetime.now()
                
                # Get appropriate agent for task
                agent = self._get_agent_for_task(subtask.task_type)
                
                # Prepare enhanced context with knowledge
                context = await self._prepare_enhanced_context(subtask)
                
                # Execute with auto-improvement if enabled
                if self.enable_auto_improvement:
                    result = await agent.process_with_improvement(
                        subtask.description, 
                        context
                    )
                else:
                    result = await agent._basic_process(subtask.description, context)
                
                # Update task with results
                subtask.result = result
                subtask.status = TaskStatus.COMPLETED
                subtask.completed_at = datetime.now()
                subtask.quality_score = result.get('best_quality', 0.8)
                
                results[subtask.task_type] = result
                
                # Update agent utilization metrics
                agent_id = agent.agent_id
                self.performance_metrics['agent_utilization'][agent_id]['tasks_completed'] += 1
                
                subtask.add_log_entry(f"Task completed by {agent_id}", "INFO", agent_id)
                
            except Exception as e:
                subtask.status = TaskStatus.FAILED
                subtask.error = str(e)
                subtask.add_log_entry(f"Task failed: {e}", "ERROR")
                logger.error(f"Subtask {subtask.task_type} failed: {e}")
        
        return results
    
    def _get_agent_for_task(self, task_type: str) -> SelfImprovingAgent:
        """Get the most appropriate agent for a task type"""
        agent_mapping = {
            'flight_search': 'flight_specialist',
            'hotel_search': 'hotel_specialist', 
            'budget_planning': 'budget_advisor',
            'itinerary_planning': 'itinerary_planner',
            'destination_info': 'general_advisor',
            'documentation_check': 'general_advisor',
            'weather_check': 'general_advisor'
        }
        
        agent_id = agent_mapping.get(task_type, 'general_advisor')
        return self.agents[agent_id]
    
    async def _prepare_enhanced_context(self, task: ProductionTask) -> Dict[str, Any]:
        """Prepare enhanced context using knowledge base"""
        base_context = task.parameters.copy()
        
        # Get relevant knowledge for the task
        knowledge_query = f"{task.task_type} {task.description}"
        knowledge_context = await self.knowledge_base.get_context_for_query(knowledge_query)
        
        # Enhance context with knowledge
        base_context.update({
            'knowledge_context': knowledge_context,
            'task_type': task.task_type,
            'session_id': task.session_id,
            'user_id': task.user_id,
            'quality_target': self.config['quality_threshold']
        })
        
        self.performance_metrics['knowledge_base_queries'] += 1
        
        return base_context
    
    async def _synthesize_travel_plan(self, 
                                    main_task: ProductionTask,
                                    subtask_results: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final comprehensive travel plan"""
        try:
            # Prepare synthesis context
            synthesis_context = {
                'original_request': main_task.description,
                'subtask_results': subtask_results,
                'user_id': main_task.user_id,
                'session_id': main_task.session_id
            }
            
            # Use knowledge agent to create comprehensive response
            synthesis_query = f"""
            Create a comprehensive travel plan based on these results:
            
            Original Request: {main_task.description}
            
            Available Information:
            {json.dumps(subtask_results, indent=2, default=str)}
            
            Please synthesize this into a cohesive, actionable travel plan.
            """
            
            final_response = await self.knowledge_agent.answer_with_knowledge(
                synthesis_query,
                include_sources=True
            )
            
            # Enhance with metadata
            final_response.update({
                'task_id': main_task.task_id,
                'session_id': main_task.session_id,
                'user_id': main_task.user_id,
                'processing_time': main_task.get_execution_time(),
                'subtasks_completed': len([r for r in subtask_results.values() if r]),
                'quality_scores': {k: v.get('best_quality', 0.8) for k, v in subtask_results.items()},
                'system_info': {
                    'coordinator_id': self.coordinator_id,
                    'auto_improvement_enabled': self.enable_auto_improvement,
                    'knowledge_base_used': True
                },
                'timestamp': datetime.now().isoformat(),
                'success': True
            })
            
            # Mark main task as completed
            main_task.result = final_response
            main_task.status = TaskStatus.COMPLETED
            main_task.completed_at = datetime.now()
            main_task.quality_score = sum(v.get('best_quality', 0.8) for v in subtask_results.values()) / len(subtask_results)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Failed to synthesize travel plan: {e}")
            return {
                'success': False,
                'error': f"Failed to synthesize plan: {str(e)}",
                'partial_results': subtask_results,
                'timestamp': datetime.now().isoformat()
            }
    
    def _update_performance_metrics(self, task: ProductionTask):
        """Update system performance metrics"""
        self.performance_metrics['total_tasks_processed'] += 1
        
        if task.status == TaskStatus.COMPLETED:
            self.performance_metrics['successful_tasks'] += 1
            
            # Update average execution time
            exec_time = task.get_execution_time() or 0
            total_time = self.performance_metrics['average_execution_time'] * (self.performance_metrics['successful_tasks'] - 1)
            self.performance_metrics['average_execution_time'] = (total_time + exec_time) / self.performance_metrics['successful_tasks']
            
            # Update average quality score
            total_quality = self.performance_metrics['average_quality_score'] * (self.performance_metrics['successful_tasks'] - 1)
            self.performance_metrics['average_quality_score'] = (total_quality + task.quality_score) / self.performance_metrics['successful_tasks']
            
        elif task.status == TaskStatus.FAILED:
            self.performance_metrics['failed_tasks'] += 1
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        agent_summaries = {}
        for agent_id, agent in self.agents.items():
            agent_summaries[agent_id] = agent.get_improvement_summary()
        
        knowledge_stats = self.knowledge_agent.get_performance_stats()
        
        return {
            'coordinator_id': self.coordinator_id,
            'system_health': 'healthy' if self.performance_metrics['successful_tasks'] > self.performance_metrics['failed_tasks'] else 'degraded',
            'performance_metrics': self.performance_metrics.copy(),
            'agent_status': agent_summaries,
            'knowledge_base_status': knowledge_stats,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'configuration': self.config.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    async def add_knowledge_source(self, 
                                 title: str,
                                 content: str,
                                 category: str = "custom") -> str:
        """Add new knowledge to the system"""
        try:
            doc_id = await self.knowledge_base.add_document(
                title=title,
                content=content,
                category=category,
                source="user_provided"
            )
            
            logger.info(f"Added new knowledge: {title}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            raise

# Example usage and testing
async def test_production_coordinator():
    """Test the production coordinator system"""
    print("🚀 Testing Production Coordinator System...")
    
    # Initialize coordinator
    coordinator = ProductionCoordinator(enable_auto_improvement=True)
    await coordinator.initialize_system()
    
    # Test comprehensive travel request
    test_request = """
    I want to plan a 10-day trip to Japan for 2 people. 
    We love cultural experiences, great food, and temples. 
    Our budget is around $6000 total. We're thinking of going in April for cherry blossoms.
    We need help with flights from New York, accommodation recommendations, 
    and a detailed itinerary. Also need to know about visa requirements.
    """
    
    print(f"📝 Processing request: {test_request[:100]}...")
    
    # Process the request
    result = await coordinator.process_travel_request(
        test_request, 
        user_id="test_user",
        priority=TaskPriority.HIGH
    )
    
    print(f"✅ Request processed successfully: {result.get('success', False)}")
    print(f"📋 Response length: {len(str(result.get('answer', '')))}")
    print(f"⏱️ Processing time: {result.get('processing_time', 'N/A')}s")
    print(f"🎯 Quality scores: {result.get('quality_scores', {})}")
    
    # Get system status
    status = await coordinator.get_system_status()
    print(f"\n📊 System Status:")
    print(f"Health: {status['system_health']}")
    print(f"Total tasks: {status['performance_metrics']['total_tasks_processed']}")
    print(f"Success rate: {status['performance_metrics']['successful_tasks']}/{status['performance_metrics']['total_tasks_processed']}")
    print(f"Knowledge queries: {status['performance_metrics']['knowledge_base_queries']}")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_production_coordinator())
