"""
Task 10: Auto-Agent Loop Implementation
Goal: Agents independently re-evaluate and iterate on their outputs
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Set
from enum import Enum
import json
import logging
from pathlib import Path
import openai
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoopStatus(Enum):
    """Status of the auto-agent loop"""
    IDLE = "idle"
    RUNNING = "running"
    EVALUATING = "evaluating"
    IMPROVING = "improving"
    COMPLETED = "completed"
    FAILED = "failed"

class ImprovementType(Enum):
    """Types of improvements an agent can make"""
    PROMPT_OPTIMIZATION = "prompt_optimization"
    TOOL_USAGE = "tool_usage"
    MEMORY_ENHANCEMENT = "memory_enhancement"
    WORKFLOW_REFINEMENT = "workflow_refinement"
    RESPONSE_QUALITY = "response_quality"

@dataclass
class LoopIteration:
    """Represents one iteration in the auto-agent loop"""
    iteration_id: str
    timestamp: datetime
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    quality_score: float
    improvements_made: List[str] = field(default_factory=list)
    feedback: Optional[str] = None
    execution_time: float = 0.0
    token_usage: int = 0
    cost: float = 0.0

@dataclass
class ImprovementPlan:
    """Plan for improving agent performance"""
    improvement_type: ImprovementType
    description: str
    implementation: Callable
    expected_improvement: float
    risk_level: float
    priority: int

class QualityEvaluator:
    """Evaluates the quality of agent outputs"""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0.1)
        
    async def evaluate_response(self, task: str, response: str, context: Dict = None) -> float:
        """Evaluate response quality on scale 0-1"""
        try:
            evaluation_prompt = f"""
            Evaluate the quality of this travel assistant response on a scale of 0.0 to 1.0.
            
            Task: {task}
            Response: {response}
            Context: {json.dumps(context or {}, indent=2)}
            
            Evaluation criteria:
            - Accuracy and relevance (30%)
            - Completeness and detail (25%)
            - Helpfulness and actionability (25%)
            - Clarity and organization (20%)
            
            Return ONLY a single number between 0.0 and 1.0.
            """
            
            result = await self.llm.ainvoke([SystemMessage(content=evaluation_prompt)])
            score = float(result.content.strip())
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return 0.5  # Default neutral score

class AutoAgentLoop:
    """
    Implements autonomous agent improvement loops
    Agents re-evaluate and iterate on their outputs based on quality metrics
    """
    
    def __init__(self, 
                 agent_id: str,
                 max_iterations: int = 5,
                 quality_threshold: float = 0.8,
                 improvement_threshold: float = 0.05):
        self.agent_id = agent_id
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.improvement_threshold = improvement_threshold
        
        self.status = LoopStatus.IDLE
        self.iterations: List[LoopIteration] = []
        self.quality_evaluator = QualityEvaluator()
        self.improvement_history: List[ImprovementPlan] = []
        
        # Performance tracking
        self.performance_metrics = {
            'average_quality': 0.0,
            'total_iterations': 0,
            'successful_improvements': 0,
            'total_execution_time': 0.0
        }
    
    async def run_loop(self, 
                      initial_task: str, 
                      initial_context: Dict,
                      agent_function: Callable) -> Dict[str, Any]:
        """
        Run the auto-improvement loop
        """
        self.status = LoopStatus.RUNNING
        best_result = None
        best_quality = 0.0
        
        try:
            for i in range(self.max_iterations):
                iteration_id = f"{self.agent_id}_iter_{i+1}_{datetime.now().isoformat()}"
                start_time = datetime.now()
                
                logger.info(f"Starting iteration {i+1}/{self.max_iterations}")
                
                # Execute agent function
                result = await agent_function(initial_task, initial_context)
                
                # Evaluate quality
                self.status = LoopStatus.EVALUATING
                quality_score = await self.quality_evaluator.evaluate_response(
                    initial_task, str(result), initial_context
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Create iteration record
                iteration = LoopIteration(
                    iteration_id=iteration_id,
                    timestamp=start_time,
                    input_data={'task': initial_task, 'context': initial_context},
                    output_data=result,
                    quality_score=quality_score,
                    execution_time=execution_time
                )
                
                self.iterations.append(iteration)
                
                # Track best result
                if quality_score > best_quality:
                    best_quality = quality_score
                    best_result = result
                
                # Check if quality threshold met
                if quality_score >= self.quality_threshold:
                    logger.info(f"Quality threshold reached: {quality_score:.3f}")
                    break
                
                # Plan improvements for next iteration
                self.status = LoopStatus.IMPROVING
                improvements = await self._plan_improvements(iteration)
                
                if improvements:
                    # Apply improvements to context for next iteration
                    initial_context = await self._apply_improvements(
                        initial_context, improvements
                    )
                    iteration.improvements_made = [imp.description for imp in improvements]
                else:
                    logger.info("No viable improvements found, stopping loop")
                    break
            
            self.status = LoopStatus.COMPLETED
            self._update_performance_metrics()
            
            return {
                'best_result': best_result,
                'best_quality': best_quality,
                'total_iterations': len(self.iterations),
                'improvement_history': [imp.description for imp in self.improvement_history],
                'performance_metrics': self.performance_metrics
            }
            
        except Exception as e:
            self.status = LoopStatus.FAILED
            logger.error(f"Auto-agent loop failed: {e}")
            raise
    
    async def _plan_improvements(self, iteration: LoopIteration) -> List[ImprovementPlan]:
        """Plan improvements based on iteration results"""
        improvements = []
        
        # Analyze what went wrong
        if iteration.quality_score < 0.6:
            # Low quality - need significant improvements
            improvements.extend([
                ImprovementPlan(
                    improvement_type=ImprovementType.PROMPT_OPTIMIZATION,
                    description="Enhance system prompt with more specific instructions",
                    implementation=self._improve_prompt,
                    expected_improvement=0.2,
                    risk_level=0.1,
                    priority=1
                ),
                ImprovementPlan(
                    improvement_type=ImprovementType.RESPONSE_QUALITY,
                    description="Add response validation and formatting",
                    implementation=self._improve_response_format,
                    expected_improvement=0.15,
                    risk_level=0.05,
                    priority=2
                )
            ])
        
        elif iteration.quality_score < 0.8:
            # Medium quality - fine-tuning needed
            improvements.append(
                ImprovementPlan(
                    improvement_type=ImprovementType.WORKFLOW_REFINEMENT,
                    description="Optimize workflow and add validation steps",
                    implementation=self._refine_workflow,
                    expected_improvement=0.1,
                    risk_level=0.03,
                    priority=1
                )
            )
        
        # Sort by priority and expected improvement
        improvements.sort(key=lambda x: (x.priority, -x.expected_improvement))
        
        return improvements[:3]  # Limit to top 3 improvements
    
    async def _apply_improvements(self, 
                                context: Dict, 
                                improvements: List[ImprovementPlan]) -> Dict:
        """Apply planned improvements to the context"""
        improved_context = context.copy()
        
        for improvement in improvements:
            try:
                improved_context = await improvement.implementation(improved_context)
                self.improvement_history.append(improvement)
                logger.info(f"Applied improvement: {improvement.description}")
            except Exception as e:
                logger.error(f"Failed to apply improvement {improvement.description}: {e}")
        
        return improved_context
    
    async def _improve_prompt(self, context: Dict) -> Dict:
        """Improve the system prompt"""
        context['system_prompt_enhancement'] = """
        Additional instructions for high-quality responses:
        1. Provide specific, actionable recommendations
        2. Include relevant details like costs, timing, and logistics
        3. Organize information clearly with bullet points or sections
        4. Address potential concerns or alternatives
        5. Use a helpful, professional tone
        """
        return context
    
    async def _improve_response_format(self, context: Dict) -> Dict:
        """Improve response formatting"""
        context['response_format'] = {
            'structure': 'Use clear sections with headers',
            'details': 'Include specific costs, times, and practical information',
            'validation': 'Double-check all recommendations for accuracy'
        }
        return context
    
    async def _refine_workflow(self, context: Dict) -> Dict:
        """Refine the workflow process"""
        context['workflow_enhancements'] = {
            'validation_steps': True,
            'cross_reference_data': True,
            'include_alternatives': True,
            'add_confidence_scores': True
        }
        return context
    
    def _update_performance_metrics(self):
        """Update performance tracking metrics"""
        if not self.iterations:
            return
        
        total_quality = sum(iter.quality_score for iter in self.iterations)
        self.performance_metrics.update({
            'average_quality': total_quality / len(self.iterations),
            'total_iterations': len(self.iterations),
            'successful_improvements': len(self.improvement_history),
            'total_execution_time': sum(iter.execution_time for iter in self.iterations)
        })
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning process"""
        if not self.iterations:
            return {'status': 'no_data'}
        
        quality_trend = [iter.quality_score for iter in self.iterations]
        
        return {
            'quality_trend': quality_trend,
            'quality_improvement': quality_trend[-1] - quality_trend[0] if len(quality_trend) > 1 else 0,
            'most_effective_improvements': [
                imp.description for imp in sorted(
                    self.improvement_history, 
                    key=lambda x: x.expected_improvement, 
                    reverse=True
                )[:3]
            ],
            'performance_metrics': self.performance_metrics,
            'recommendations': self._generate_learning_recommendations()
        }
    
    def _generate_learning_recommendations(self) -> List[str]:
        """Generate recommendations for future improvements"""
        recommendations = []
        
        if self.performance_metrics['average_quality'] < 0.7:
            recommendations.append("Focus on improving prompt engineering and response quality")
        
        if len(self.improvement_history) < 2:
            recommendations.append("Experiment with more improvement strategies")
        
        if self.performance_metrics['total_execution_time'] > 30:
            recommendations.append("Optimize for faster execution times")
        
        return recommendations

class SelfImprovingAgent:
    """
    An agent that uses auto-loops to continuously improve its performance
    """
    
    def __init__(self, 
                 agent_id: str,
                 model: str = "gpt-4",
                 enable_auto_improvement: bool = True):
        self.agent_id = agent_id
        self.model = model
        self.enable_auto_improvement = enable_auto_improvement
        
        self.llm = ChatOpenAI(model=model, temperature=0.7)
        self.auto_loop = AutoAgentLoop(agent_id) if enable_auto_improvement else None
        
        # Learning storage
        self.learned_patterns: Dict[str, Any] = {}
        self.improvement_log: List[Dict] = []
    
    async def process_with_improvement(self, 
                                     task: str, 
                                     context: Dict = None) -> Dict[str, Any]:
        """Process task with automatic improvement if enabled"""
        if not self.enable_auto_improvement or not self.auto_loop:
            return await self._basic_process(task, context or {})
        
        # Use auto-improvement loop
        result = await self.auto_loop.run_loop(
            task, 
            context or {}, 
            self._enhanced_process
        )
        
        # Learn from the experience
        await self._learn_from_iteration(task, result)
        
        return result
    
    async def _basic_process(self, task: str, context: Dict) -> Dict[str, Any]:
        """Basic processing without improvement loop"""
        try:
            # Apply learned patterns
            enhanced_context = self._apply_learned_patterns(context)
            
            # Generate response
            messages = [
                SystemMessage(content=self._get_system_prompt(enhanced_context)),
                HumanMessage(content=task)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            return {
                'response': response.content,
                'context': enhanced_context,
                'agent_id': self.agent_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Basic processing failed: {e}")
            raise
    
    async def _enhanced_process(self, task: str, context: Dict) -> Dict[str, Any]:
        """Enhanced processing with improvements applied"""
        try:
            # Apply both learned patterns and loop improvements
            enhanced_context = self._apply_learned_patterns(context)
            
            # Build enhanced system prompt
            system_prompt = self._get_system_prompt(enhanced_context)
            
            # Add improvement enhancements
            if 'system_prompt_enhancement' in context:
                system_prompt += "\n" + context['system_prompt_enhancement']
            
            # Generate response with enhancements
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Apply response formatting if specified
            formatted_response = response.content
            if 'response_format' in context:
                formatted_response = self._format_response(
                    formatted_response, 
                    context['response_format']
                )
            
            # Validate if workflow enhancements enabled
            if context.get('workflow_enhancements', {}).get('validation_steps'):
                formatted_response = await self._validate_response(
                    formatted_response, task
                )
            
            return {
                'response': formatted_response,
                'context': enhanced_context,
                'agent_id': self.agent_id,
                'timestamp': datetime.now().isoformat(),
                'enhancements_applied': list(context.keys())
            }
            
        except Exception as e:
            logger.error(f"Enhanced processing failed: {e}")
            raise
    
    def _get_system_prompt(self, context: Dict) -> str:
        """Get system prompt with context enhancements"""
        base_prompt = f"""
        You are {self.agent_id}, a specialized travel assistant agent.
        
        Your role is to provide high-quality, helpful travel recommendations and assistance.
        
        Guidelines:
        - Provide specific, actionable advice
        - Include relevant details like costs, timing, and practical information
        - Consider user preferences and constraints
        - Organize information clearly
        - Be helpful and professional
        """
        
        # Add context-specific enhancements
        if context.get('user_preferences'):
            base_prompt += f"\nUser preferences: {context['user_preferences']}"
        
        if context.get('constraints'):
            base_prompt += f"\nConstraints to consider: {context['constraints']}"
        
        return base_prompt
    
    def _apply_learned_patterns(self, context: Dict) -> Dict:
        """Apply previously learned patterns to enhance context"""
        enhanced_context = context.copy()
        
        # Apply learned patterns based on task type
        for pattern_key, pattern_data in self.learned_patterns.items():
            if pattern_key in str(context):
                enhanced_context.update(pattern_data.get('enhancements', {}))
        
        return enhanced_context
    
    def _format_response(self, response: str, format_config: Dict) -> str:
        """Format response according to configuration"""
        if format_config.get('structure') == 'Use clear sections with headers':
            # Add basic structure if not present
            if '##' not in response and '\n\n' in response:
                lines = response.split('\n\n')
                formatted = "## Travel Recommendations\n\n"
                for i, line in enumerate(lines):
                    if line.strip():
                        formatted += f"### {chr(65+i)}. {line}\n\n"
                return formatted
        
        return response
    
    async def _validate_response(self, response: str, original_task: str) -> str:
        """Validate and potentially enhance the response"""
        # Basic validation checks
        if len(response) < 50:
            response += "\n\nNote: This is a brief response. Please let me know if you need more detailed information."
        
        # Check if it addresses the task
        if 'travel' in original_task.lower() and 'travel' not in response.lower():
            response = f"Travel Recommendation:\n\n{response}"
        
        return response
    
    async def _learn_from_iteration(self, task: str, result: Dict):
        """Learn from successful patterns"""
        if result.get('best_quality', 0) > 0.8:
            # Extract patterns from successful iterations
            task_type = self._categorize_task(task)
            
            if task_type not in self.learned_patterns:
                self.learned_patterns[task_type] = {
                    'successful_count': 0,
                    'enhancements': {}
                }
            
            self.learned_patterns[task_type]['successful_count'] += 1
            
            # Store the improvement for future use
            self.improvement_log.append({
                'timestamp': datetime.now().isoformat(),
                'task_type': task_type,
                'quality_achieved': result.get('best_quality'),
                'improvements_used': result.get('improvement_history', [])
            })
    
    def _categorize_task(self, task: str) -> str:
        """Categorize task for pattern learning"""
        task_lower = task.lower()
        
        if 'flight' in task_lower:
            return 'flight_planning'
        elif 'hotel' in task_lower:
            return 'accommodation'
        elif 'itinerary' in task_lower:
            return 'itinerary_planning'
        elif 'budget' in task_lower:
            return 'budget_planning'
        elif 'weather' in task_lower:
            return 'weather_inquiry'
        else:
            return 'general_travel'
    
    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of agent improvements"""
        if not self.auto_loop:
            return {'auto_improvement': 'disabled'}
        
        return {
            'agent_id': self.agent_id,
            'learning_insights': self.auto_loop.get_learning_insights(),
            'learned_patterns': list(self.learned_patterns.keys()),
            'total_improvements': len(self.improvement_log),
            'auto_loop_status': self.auto_loop.status.value
        }

# Example usage and testing functions
async def test_auto_agent_loop():
    """Test the auto-agent loop functionality"""
    print("🔄 Testing Auto-Agent Loop System...")
    
    # Create self-improving agent
    agent = SelfImprovingAgent("test_travel_agent")
    
    # Test task
    test_task = "Plan a 5-day trip to Japan for 2 people with a budget of $4000"
    test_context = {
        'user_preferences': ['cultural sites', 'good food', 'comfortable hotels'],
        'constraints': ['no very long flights', 'vegetarian options needed']
    }
    
    # Process with improvement
    result = await agent.process_with_improvement(test_task, test_context)
    
    print(f"✅ Task completed with quality: {result.get('best_quality', 'N/A')}")
    print(f"🔧 Iterations: {result.get('total_iterations', 'N/A')}")
    print(f"📈 Improvements: {', '.join(result.get('improvement_history', []))}")
    
    # Get learning summary
    summary = agent.get_improvement_summary()
    print(f"🧠 Learning insights: {summary}")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_auto_agent_loop())
