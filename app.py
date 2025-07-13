# Multi-Agent AI Collaboration System
# Author: Spencer Purdy
# Description: Enterprise-grade multi-agent system with specialized AI agents collaborating
# to solve complex problems through intelligent task decomposition and parallel processing.

# Installation (uncomment for Google Colab)
# !pip install gradio langchain langchain-openai openai networkx matplotlib plotly pandas numpy python-dotenv pydantic aiohttp asyncio scipy reportlab pillow

import os
import json
import time
import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import gradio as gr
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# LangChain and AI libraries
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

# Report generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Async libraries
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration settings for the multi-agent system."""
    
    # Model settings
    DEFAULT_MODEL = "gpt-4"
    TEMPERATURE = 0.7
    MAX_TOKENS = 1500
    
    # Agent settings
    MAX_ITERATIONS = 10
    COLLABORATION_TIMEOUT = 300  # seconds
    
    # Visualization settings
    GRAPH_UPDATE_INTERVAL = 0.5  # seconds
    NODE_COLORS = {
        'Researcher': '#3498db',
        'Analyst': '#e74c3c', 
        'Critic': '#f39c12',
        'Synthesizer': '#2ecc71',
        'Coordinator': '#9b59b6'
    }
    
    # Performance settings
    ENABLE_PERFORMANCE_TRACKING = True
    BENCHMARK_BASELINE = {
        "single_agent_time": 45.0,
        "single_agent_quality": 0.72
    }
    
    # Report settings
    CONFIDENCE_THRESHOLD = 0.7
    MAX_REPORT_SECTIONS = 10
    COMPANY_NAME = "Multi-Agent AI Platform"
    
    # Demo settings
    DEMO_MODE_ENABLED = True
    DEMO_PROBLEMS = [
        "Analyze the impact of remote work on team productivity and collaboration",
        "Develop a strategy for sustainable urban transportation",
        "Evaluate the risks and benefits of AI in healthcare",
        "Design a framework for ethical AI development",
        "Create a plan for digital transformation in education"
    ]

class AgentRole(Enum):
    """Enumeration of agent roles in the system."""
    RESEARCHER = "Researcher"
    ANALYST = "Analyst"
    CRITIC = "Critic"
    SYNTHESIZER = "Synthesizer"
    COORDINATOR = "Coordinator"

class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class MessageType(Enum):
    """Types of messages between agents."""
    TASK_ASSIGNMENT = "task_assignment"
    COLLABORATION_REQUEST = "collaboration_request"
    INFORMATION_SHARING = "information_sharing"
    FEEDBACK = "feedback"
    COMPLETION_REPORT = "completion_report"

@dataclass
class Task:
    """Represents a task to be executed by agents."""
    id: str
    description: str
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    result: Optional[Any] = None
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class AgentMessage:
    """Message passed between agents."""
    sender: str
    recipient: str
    content: str
    message_type: MessageType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1 (low) to 5 (high)

class PerformanceTracker:
    """Tracks performance metrics for the multi-agent system."""
    
    def __init__(self):
        self.metrics = {
            'task_completion_times': [],
            'agent_utilization': {},
            'collaboration_count': 0,
            'total_messages': 0,
            'quality_scores': [],
            'system_start_time': None,
            'system_end_time': None
        }
        
    def start_tracking(self):
        """Start performance tracking."""
        self.metrics['system_start_time'] = datetime.now()
        
    def end_tracking(self):
        """End performance tracking."""
        self.metrics['system_end_time'] = datetime.now()
        
    def record_task_completion(self, task: Task):
        """Record task completion metrics."""
        if task.created_at and task.completed_at:
            completion_time = (task.completed_at - task.created_at).total_seconds()
            self.metrics['task_completion_times'].append(completion_time)
            
    def record_agent_activity(self, agent_name: str, activity_duration: float):
        """Record agent activity duration."""
        if agent_name not in self.metrics['agent_utilization']:
            self.metrics['agent_utilization'][agent_name] = 0
        self.metrics['agent_utilization'][agent_name] += activity_duration
        
    def record_collaboration(self):
        """Record a collaboration event."""
        self.metrics['collaboration_count'] += 1
        
    def record_message(self):
        """Record a message exchange."""
        self.metrics['total_messages'] += 1
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        total_time = 0
        if self.metrics['system_start_time'] and self.metrics['system_end_time']:
            total_time = (self.metrics['system_end_time'] - 
                         self.metrics['system_start_time']).total_seconds()
        
        avg_task_time = np.mean(self.metrics['task_completion_times']) if self.metrics['task_completion_times'] else 0
        
        # Calculate improvement over baseline
        baseline_time = Config.BENCHMARK_BASELINE['single_agent_time']
        time_improvement = ((baseline_time - avg_task_time) / baseline_time * 100) if avg_task_time > 0 else 0
        
        return {
            'total_execution_time': total_time,
            'average_task_completion_time': avg_task_time,
            'total_collaborations': self.metrics['collaboration_count'],
            'total_messages': self.metrics['total_messages'],
            'agent_utilization': self.metrics['agent_utilization'],
            'time_improvement_percentage': time_improvement,
            'efficiency_score': self._calculate_efficiency_score()
        }
        
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score."""
        factors = []
        
        # Task completion speed factor
        if self.metrics['task_completion_times']:
            avg_time = np.mean(self.metrics['task_completion_times'])
            speed_factor = min(1.0, Config.BENCHMARK_BASELINE['single_agent_time'] / avg_time)
            factors.append(speed_factor)
        
        # Collaboration efficiency factor
        if self.metrics['total_messages'] > 0:
            collab_factor = min(1.0, self.metrics['collaboration_count'] / (self.metrics['total_messages'] * 0.3))
            factors.append(collab_factor)
        
        # Agent utilization factor
        if self.metrics['agent_utilization']:
            utilization_values = list(self.metrics['agent_utilization'].values())
            if utilization_values:
                avg_utilization = np.mean(utilization_values)
                max_utilization = max(utilization_values)
                balance_factor = avg_utilization / max_utilization if max_utilization > 0 else 0
                factors.append(balance_factor)
        
        return np.mean(factors) if factors else 0.5

class AgentMemory:
    """Manages agent conversation history and context."""
    
    def __init__(self, max_messages: int = 50):
        self.messages: List[AgentMessage] = []
        self.max_messages = max_messages
        self.context: Dict[str, Any] = {}
        self.knowledge_base: Dict[str, Any] = {}
        
    def add_message(self, message: AgentMessage):
        """Add a message to memory."""
        self.messages.append(message)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
            
        # Extract and store important information
        self._extract_knowledge(message)
    
    def get_recent_messages(self, n: int = 10) -> List[AgentMessage]:
        """Get n most recent messages."""
        return self.messages[-n:]
    
    def get_messages_by_sender(self, sender: str) -> List[AgentMessage]:
        """Get all messages from a specific sender."""
        return [msg for msg in self.messages if msg.sender == sender]
    
    def get_high_priority_messages(self) -> List[AgentMessage]:
        """Get high priority messages."""
        return [msg for msg in self.messages if msg.priority >= 4]
    
    def update_context(self, key: str, value: Any):
        """Update context information."""
        self.context[key] = value
    
    def get_context(self, key: str) -> Any:
        """Get context information."""
        return self.context.get(key)
    
    def _extract_knowledge(self, message: AgentMessage):
        """Extract and store important knowledge from messages."""
        keywords = ['finding', 'conclusion', 'recommendation', 'insight', 'pattern']
        content_lower = message.content.lower()
        
        for keyword in keywords:
            if keyword in content_lower:
                knowledge_key = f"{message.sender}_{keyword}_{len(self.knowledge_base)}"
                self.knowledge_base[knowledge_key] = {
                    'content': message.content,
                    'sender': message.sender,
                    'timestamp': message.timestamp,
                    'type': keyword
                }

class BaseAgent:
    """Base class for all AI agents in the system."""
    
    def __init__(self, name: str, role: AgentRole, llm: Optional[ChatOpenAI] = None):
        self.name = name
        self.role = role
        self.llm = llm
        self.memory = AgentMemory()
        self.active = True
        self.current_task: Optional[Task] = None
        self.completed_tasks: List[Task] = []
        self.performance_tracker = PerformanceTracker()
        self.collaboration_partners: Set[str] = set()
        
    async def process_task(self, task: Task) -> Task:
        """Process a task and return the result."""
        self.current_task = task
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_to = self.name
        
        # Record start time
        start_time = datetime.now()
        
        try:
            # Execute task based on agent role
            if self.llm:
                result = await self._execute_task(task)
            else:
                # Demo mode - simulate execution
                result = await self._simulate_task_execution(task)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.confidence = self._calculate_confidence(result)
            
            # Record performance metrics
            execution_time = (task.completed_at - start_time).total_seconds()
            task.performance_metrics['execution_time'] = execution_time
            task.performance_metrics['confidence'] = task.confidence
            
            self.completed_tasks.append(task)
            self.performance_tracker.record_task_completion(task)
            
        except Exception as e:
            logger.error(f"Agent {self.name} failed to process task {task.id}: {str(e)}")
            task.status = TaskStatus.FAILED
            task.result = f"Error: {str(e)}"
            task.confidence = 0.0
        
        finally:
            self.current_task = None
            
        return task
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute the task - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _execute_task")
    
    async def _simulate_task_execution(self, task: Task) -> Any:
        """Simulate task execution for demo mode."""
        # Simulate processing time
        await asyncio.sleep(np.random.uniform(2, 4))
        
        # Create realistic simulation results based on agent role
        simulation_templates = {
            AgentRole.RESEARCHER: {
                "findings": f"Comprehensive research on '{task.description}' reveals multiple perspectives and critical data points.",
                "sources": ["Academic studies", "Industry reports", "Expert interviews", "Market analysis"],
                "key_points": [
                    "Significant trends identified in the domain",
                    "Multiple stakeholder perspectives considered",
                    "Historical context provides important insights",
                    "Current state analysis reveals opportunities",
                    "Future projections indicate growth potential"
                ],
                "data_collected": {
                    "quantitative": "Statistical analysis of 500+ data points",
                    "qualitative": "In-depth interviews with 20 experts"
                },
                "research_quality_score": 0.92
            },
            AgentRole.ANALYST: {
                "analysis": f"Detailed analysis of '{task.description}' reveals clear patterns and actionable insights.",
                "patterns": [
                    {"description": "Upward trend in adoption rates", "type": "trend", "confidence": 0.89},
                    {"description": "Strong correlation between factors X and Y", "type": "correlation", "confidence": 0.91},
                    {"description": "Seasonal variations detected", "type": "cyclical", "confidence": 0.87}
                ],
                "insights": [
                    "Data suggests strong positive outcomes with 85% confidence",
                    "Multiple factors contribute to observed patterns",
                    "Strategic opportunities identified in 3 key areas",
                    "Risk factors are manageable with proper mitigation"
                ],
                "recommendations": [
                    {"recommendation": "Implement phased approach", "priority": "high"},
                    {"recommendation": "Focus on high-impact areas first", "priority": "high"},
                    {"recommendation": "Monitor key metrics continuously", "priority": "medium"}
                ],
                "confidence_metrics": {"overall_confidence": 0.88, "data_quality": 0.90}
            },
            AgentRole.CRITIC: {
                "evaluation": f"Critical evaluation of '{task.description}' identifies strengths and areas for improvement.",
                "strengths": [
                    {"strength": "Comprehensive data coverage", "category": "methodology", "impact": "high"},
                    {"strength": "Well-structured analysis approach", "category": "process", "impact": "high"},
                    {"strength": "Clear evidence supporting conclusions", "category": "evidence", "impact": "medium"}
                ],
                "weaknesses": [
                    {"weakness": "Limited geographic scope", "severity": "medium", "category": "completeness"},
                    {"weakness": "Some assumptions require validation", "severity": "low", "category": "methodology"}
                ],
                "gaps": [
                    "Additional longitudinal data would strengthen conclusions",
                    "Competitive analysis could be expanded"
                ],
                "improvements": [
                    {"suggestion": "Include more diverse data sources", "priority": "high", "effort": "medium"},
                    {"suggestion": "Validate assumptions with field testing", "priority": "medium", "effort": "high"}
                ],
                "quality_score": {"overall": 0.85, "breakdown": {"accuracy": 0.88, "completeness": 0.82, "logic": 0.90}}
            },
            AgentRole.SYNTHESIZER: {
                "synthesis": f"Comprehensive synthesis for '{task.description}' integrates all findings into actionable strategy.",
                "key_themes": [
                    {"theme": "Digital transformation opportunity", "description": "Strong potential for technology adoption", "importance": "high"},
                    {"theme": "Customer-centric approach", "description": "Focus on user experience drives success", "importance": "high"},
                    {"theme": "Phased implementation", "description": "Gradual rollout minimizes risk", "importance": "medium"}
                ],
                "consensus_points": [
                    {"point": "All agents agree on strategic direction", "strength": "strong"},
                    {"point": "Timeline expectations are aligned", "strength": "strong"},
                    {"point": "Resource requirements are reasonable", "strength": "moderate"}
                ],
                "final_recommendations": [
                    {"recommendation": "Launch pilot program in Q1", "priority": "high", "timeframe": "immediate"},
                    {"recommendation": "Establish KPI dashboard", "priority": "high", "timeframe": "immediate"},
                    {"recommendation": "Build stakeholder coalition", "priority": "medium", "timeframe": "short-term"},
                    {"recommendation": "Develop training programs", "priority": "medium", "timeframe": "medium-term"}
                ],
                "executive_summary": "Based on comprehensive multi-agent analysis, we recommend a phased approach to implementation with focus on quick wins and risk mitigation. The strategy balances innovation with practical considerations.",
                "action_items": [
                    {"action": "Form implementation task force", "owner": "Leadership", "deadline": "2 weeks"},
                    {"action": "Develop detailed project plan", "owner": "PMO", "deadline": "1 month"},
                    {"action": "Secure budget approval", "owner": "Finance", "deadline": "1 month"}
                ],
                "confidence_level": {"overall": 0.91, "factors": {"evidence_strength": True, "consensus_level": True}}
            }
        }
        
        return simulation_templates.get(self.role, {"result": "Task completed successfully"})

    def _calculate_confidence(self, result: Any) -> float:
        """Calculate confidence score for the result."""
        base_confidence = 0.7
        
        if result and isinstance(result, dict):
            # Check for confidence metrics in result
            if 'confidence_metrics' in result:
                return result['confidence_metrics'].get('overall_confidence', base_confidence)
            
            # Calculate based on result completeness
            expected_keys = {'findings', 'analysis', 'evaluation', 'synthesis'}
            actual_keys = set(result.keys())
            completeness = len(actual_keys.intersection(expected_keys)) / len(expected_keys)
            
            # Calculate based on content depth
            content_length = sum(len(str(v)) for v in result.values() if isinstance(v, (str, list)))
            length_factor = min(1.0, content_length / 1000)
            
            # Check for quality indicators
            quality_indicators = ['quality_score', 'confidence_level', 'research_quality_score']
            quality_bonus = 0.1 if any(ind in result for ind in quality_indicators) else 0
            
            confidence = base_confidence + (completeness * 0.15) + (length_factor * 0.1) + quality_bonus
            return min(0.95, confidence)
        
        return base_confidence
    
    async def collaborate(self, other_agent: 'BaseAgent', message: AgentMessage) -> AgentMessage:
        """Handle collaboration with another agent."""
        self.memory.add_message(message)
        self.collaboration_partners.add(other_agent.name)
        self.performance_tracker.record_collaboration()
        
        # Process collaboration request
        response_content = await self._process_collaboration(message)
        
        # Create response message
        response_message = AgentMessage(
            sender=self.name,
            recipient=other_agent.name,
            content=response_content,
            message_type=MessageType.INFORMATION_SHARING,
            priority=message.priority
        )
        
        other_agent.memory.add_message(response_message)
        self.performance_tracker.record_message()
        
        return response_message
    
    async def _process_collaboration(self, message: AgentMessage) -> str:
        """Process collaboration message."""
        # Generate contextual response based on agent role
        role_responses = {
            AgentRole.RESEARCHER: f"Research findings indicate: Based on my investigation, {message.content} aligns with current data trends.",
            AgentRole.ANALYST: f"Analytical perspective: The patterns I've identified support {message.content} with 87% confidence.",
            AgentRole.CRITIC: f"Critical assessment: While {message.content} has merit, we should also consider potential risks.",
            AgentRole.SYNTHESIZER: f"Synthesis observation: Integrating {message.content} into our comprehensive strategy."
        }
        
        return role_responses.get(self.role, f"{self.name} acknowledges: {message.content}")
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get current status summary of the agent."""
        return {
            'name': self.name,
            'role': self.role.value,
            'active': self.active,
            'current_task': self.current_task.description if self.current_task else None,
            'completed_tasks': len(self.completed_tasks),
            'average_confidence': np.mean([t.confidence for t in self.completed_tasks]) if self.completed_tasks else 0,
            'collaboration_count': len(self.collaboration_partners),
            'memory_size': len(self.memory.messages)
        }

class ResearcherAgent(BaseAgent):
    """Agent specialized in researching and gathering information."""
    
    def __init__(self, name: str, llm: Optional[ChatOpenAI] = None):
        super().__init__(name, AgentRole.RESEARCHER, llm)
        self.research_sources: List[str] = []
        self.research_methods = ["literature_review", "data_collection", "expert_consultation", "field_research"]
        
    async def _execute_task(self, task: Task) -> Any:
        """Execute research task."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Research Agent specializing in gathering comprehensive information.
            Your role is to:
            1. Break down complex topics into research questions
            2. Identify key information sources and data points
            3. Provide detailed, factual information with citations where possible
            4. Flag areas requiring further investigation
            5. Maintain objectivity and consider multiple perspectives"""),
            HumanMessage(content=f"Research the following: {task.description}")
        ])
        
        response = await self.llm.ainvoke(prompt.format_messages())
        
        # Extract and structure research findings
        research_result = {
            "findings": response.content,
            "sources": self._extract_sources(response.content),
            "key_points": self._extract_key_points(response.content),
            "areas_for_investigation": self._identify_gaps(response.content),
            "research_quality_score": self._assess_research_quality(response.content)
        }
        
        # Update internal knowledge
        self.memory.update_context('latest_research', research_result)
        
        return research_result
    
    def _extract_sources(self, content: str) -> List[str]:
        """Extract potential sources from research content."""
        sources = []
        source_indicators = ['source:', 'reference:', 'based on:', 'according to', 'study:', 'report:']
        
        lines = content.split('\n')
        for line in lines:
            line_lower = line.lower()
            for indicator in source_indicators:
                if indicator in line_lower:
                    sources.append(line.strip())
                    break
        
        return sources[:10]  # Limit to top 10 sources
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from research."""
        key_points = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Check for numbered or bulleted points
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                key_points.append(line)
            # Check for key phrases
            elif any(phrase in line.lower() for phrase in ['key finding:', 'important:', 'notably:']):
                key_points.append(line)
        
        return key_points[:15]  # Limit to top 15 points
    
    def _identify_gaps(self, content: str) -> List[str]:
        """Identify areas needing more research."""
        gaps = []
        gap_indicators = ['unclear', 'requires further', 'need more', 'investigate', 
                         'unknown', 'limited data', 'insufficient evidence']
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in gap_indicators):
                gaps.append(sentence.strip() + '.')
        
        return gaps[:5]
    
    def _assess_research_quality(self, content: str) -> float:
        """Assess the quality of research output."""
        quality_score = 0.5  # Base score
        
        # Check for sources
        if self._extract_sources(content):
            quality_score += 0.15
        
        # Check for structured content
        if self._extract_key_points(content):
            quality_score += 0.15
        
        # Check for comprehensive coverage
        word_count = len(content.split())
        if word_count > 300:
            quality_score += 0.1
        
        # Check for analytical depth
        analytical_terms = ['analysis', 'evaluation', 'comparison', 'contrast', 'implication']
        if any(term in content.lower() for term in analytical_terms):
            quality_score += 0.1
        
        return min(1.0, quality_score)

class AnalystAgent(BaseAgent):
    """Agent specialized in analyzing data and identifying patterns."""
    
    def __init__(self, name: str, llm: Optional[ChatOpenAI] = None):
        super().__init__(name, AgentRole.ANALYST, llm)
        self.analysis_methods = ["statistical", "comparative", "trend", "causal", "predictive"]
        self.analysis_frameworks = ["SWOT", "PESTLE", "Porter's Five Forces", "Cost-Benefit"]
        
    async def _execute_task(self, task: Task) -> Any:
        """Execute analysis task."""
        # Get context from previous research if available
        context = self._get_relevant_context(task)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an Analyst Agent specializing in data analysis and pattern recognition.
            Your role is to:
            1. Analyze information systematically and objectively
            2. Identify patterns, trends, and correlations
            3. Provide quantitative insights where possible
            4. Draw logical conclusions based on evidence
            5. Apply appropriate analytical frameworks
            6. Consider multiple analytical perspectives"""),
            HumanMessage(content=f"Analyze the following: {task.description}\n\nContext: {context}")
        ])
        
        response = await self.llm.ainvoke(prompt.format_messages())
        
        # Structure analysis results
        analysis_result = {
            "analysis": response.content,
            "patterns": self._identify_patterns(response.content),
            "insights": self._extract_insights(response.content),
            "recommendations": self._generate_recommendations(response.content),
            "confidence_metrics": self._calculate_analysis_confidence(response.content),
            "analytical_framework": self._identify_framework_used(response.content)
        }
        
        # Store analysis in memory
        self.memory.update_context('latest_analysis', analysis_result)
        
        return analysis_result
    
    def _get_relevant_context(self, task: Task) -> str:
        """Get relevant context from memory for the task."""
        context_items = []
        
        # Get recent messages related to the task
        recent_messages = self.memory.get_recent_messages(5)
        for msg in recent_messages:
            if task.description.lower() in msg.content.lower():
                context_items.append(f"Previous finding: {msg.content[:200]}...")
        
        # Get knowledge base items
        for key, knowledge in self.memory.knowledge_base.items():
            if 'finding' in knowledge['type'] or 'insight' in knowledge['type']:
                context_items.append(f"Known insight: {knowledge['content'][:200]}...")
        
        return "\n".join(context_items[:3])  # Limit context items
    
    def _identify_patterns(self, content: str) -> List[Dict[str, str]]:
        """Identify patterns in the analysis."""
        patterns = []
        pattern_types = {
            'trend': ['trend', 'increasing', 'decreasing', 'growth', 'decline'],
            'correlation': ['correlation', 'relationship', 'associated', 'linked'],
            'cyclical': ['cycle', 'periodic', 'seasonal', 'recurring'],
            'anomaly': ['anomaly', 'outlier', 'unusual', 'exceptional']
        }
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for pattern_type, keywords in pattern_types.items():
                if any(keyword in sentence_lower for keyword in keywords):
                    patterns.append({
                        "description": sentence.strip() + '.',
                        "type": pattern_type,
                        "confidence": 0.8
                    })
                    break
        
        return patterns[:8]
    
    def _extract_insights(self, content: str) -> List[str]:
        """Extract key insights from analysis."""
        insights = []
        insight_indicators = ['shows', 'indicates', 'suggests', 'reveals', 
                            'demonstrates', 'implies', 'means that', 'therefore']
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in insight_indicators):
                insights.append(sentence.strip() + '.')
        
        return insights[:10]
    
    def _generate_recommendations(self, content: str) -> List[Dict[str, str]]:
        """Generate recommendations based on analysis."""
        recommendations = []
        rec_indicators = ['recommend', 'suggest', 'should', 'consider', 
                         'advise', 'propose', 'it would be beneficial']
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in rec_indicators):
                recommendations.append({
                    "recommendation": sentence.strip() + '.',
                    "priority": "high" if any(word in sentence_lower for word in ['critical', 'essential', 'must']) else "medium"
                })
        
        return recommendations[:7]
    
    def _calculate_analysis_confidence(self, content: str) -> Dict[str, float]:
        """Calculate confidence metrics for the analysis."""
        # Count evidence indicators
        evidence_count = sum(content.lower().count(word) for word in ['evidence', 'data', 'shows', 'proves'])
        uncertainty_count = sum(content.lower().count(word) for word in ['may', 'might', 'possibly', 'perhaps'])
        
        # Calculate confidence scores
        evidence_strength = min(1.0, evidence_count / 10)
        certainty_level = max(0.0, 1.0 - (uncertainty_count / 10))
        
        # Check for quantitative analysis
        quantitative_indicators = ['percentage', '%', 'ratio', 'correlation', 'statistical']
        quantitative_score = 0.7 if any(ind in content.lower() for ind in quantitative_indicators) else 0.5
        
        overall_confidence = (evidence_strength + certainty_level + quantitative_score) / 3
        
        return {
            "overall_confidence": overall_confidence,
            "evidence_strength": evidence_strength,
            "certainty_level": certainty_level,
            "quantitative_score": quantitative_score
        }
    
    def _identify_framework_used(self, content: str) -> Optional[str]:
        """Identify which analytical framework was used."""
        content_lower = content.lower()
        
        for framework in self.analysis_frameworks:
            if framework.lower() in content_lower:
                return framework
        
        # Check for implicit framework usage
        if all(word in content_lower for word in ['strength', 'weakness', 'opportunity', 'threat']):
            return "SWOT"
        elif any(word in content_lower for word in ['political', 'economic', 'social', 'technological']):
            return "PESTLE"
        
        return None

class CriticAgent(BaseAgent):
    """Agent specialized in critical evaluation and quality assurance."""
    
    def __init__(self, name: str, llm: Optional[ChatOpenAI] = None):
        super().__init__(name, AgentRole.CRITIC, llm)
        self.evaluation_criteria = [
            "accuracy", "completeness", "logic", "evidence", 
            "clarity", "relevance", "consistency", "objectivity"
        ]
        self.evaluation_rubric = self._create_evaluation_rubric()
        
    def _create_evaluation_rubric(self) -> Dict[str, Dict[str, float]]:
        """Create evaluation rubric with weighted criteria."""
        return {
            "accuracy": {"weight": 0.20, "score": 0.0},
            "completeness": {"weight": 0.15, "score": 0.0},
            "logic": {"weight": 0.15, "score": 0.0},
            "evidence": {"weight": 0.15, "score": 0.0},
            "clarity": {"weight": 0.10, "score": 0.0},
            "relevance": {"weight": 0.10, "score": 0.0},
            "consistency": {"weight": 0.10, "score": 0.0},
            "objectivity": {"weight": 0.05, "score": 0.0}
        }
        
    async def _execute_task(self, task: Task) -> Any:
        """Execute critical evaluation task."""
        # Get content to evaluate from context
        evaluation_context = self._gather_evaluation_context(task)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Critic Agent specializing in rigorous evaluation and quality assurance.
            Your role is to:
            1. Critically evaluate arguments and conclusions
            2. Identify weaknesses, gaps, and potential biases
            3. Verify logical consistency and evidence quality
            4. Suggest improvements and alternative perspectives
            5. Ensure high standards of analysis
            6. Apply systematic evaluation criteria
            7. Provide constructive feedback"""),
            HumanMessage(content=f"Critically evaluate the following: {task.description}\n\nContent to evaluate: {evaluation_context}")
        ])
        
        response = await self.llm.ainvoke(prompt.format_messages())
        
        # Structure critique results
        critique_result = {
            "evaluation": response.content,
            "strengths": self._identify_strengths(response.content),
            "weaknesses": self._identify_weaknesses(response.content),
            "gaps": self._identify_gaps(response.content),
            "improvements": self._suggest_improvements(response.content),
            "quality_score": self._calculate_quality_score(response.content),
            "alternative_perspectives": self._identify_alternatives(response.content),
            "final_verdict": self._generate_verdict(response.content)
        }
        
        # Update evaluation history
        self.memory.update_context('evaluation_history', critique_result)
        
        return critique_result

    def _gather_evaluation_context(self, task: Task) -> str:
        """Gather relevant context for evaluation."""
        context_items = []
        
        # Get recent analysis and research results
        recent_messages = self.memory.get_recent_messages(10)
        for msg in recent_messages:
            if msg.message_type in [MessageType.COMPLETION_REPORT, MessageType.INFORMATION_SHARING]:
                context_items.append(f"{msg.sender}: {msg.content[:300]}...")
        
        # Get knowledge base insights
        for key, knowledge in self.memory.knowledge_base.items():
            if knowledge['type'] in ['finding', 'conclusion', 'insight']:
                context_items.append(f"Previous {knowledge['type']}: {knowledge['content'][:200]}...")
        
        return "\n\n".join(context_items[:5])
    
    def _identify_strengths(self, content: str) -> List[Dict[str, str]]:
        """Identify strengths in the evaluated content."""
        strengths = []
        strength_indicators = ['strong', 'excellent', 'well', 'good', 'effective', 
                             'solid', 'robust', 'comprehensive', 'thorough']
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for indicator in strength_indicators:
                if indicator in sentence_lower:
                    strengths.append({
                        "strength": sentence.strip() + '.',
                        "category": self._categorize_strength(sentence),
                        "impact": "high" if any(word in sentence_lower for word in ['very', 'extremely', 'highly']) else "medium"
                    })
                    break
        
        return strengths[:6]
    
    def _categorize_strength(self, sentence: str) -> str:
        """Categorize the type of strength identified."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['method', 'approach', 'framework']):
            return "methodology"
        elif any(word in sentence_lower for word in ['data', 'evidence', 'support']):
            return "evidence"
        elif any(word in sentence_lower for word in ['logic', 'reasoning', 'argument']):
            return "reasoning"
        elif any(word in sentence_lower for word in ['clear', 'organized', 'structured']):
            return "presentation"
        else:
            return "general"
    
    def _identify_weaknesses(self, content: str) -> List[Dict[str, str]]:
        """Identify weaknesses in the evaluated content."""
        weaknesses = []
        weakness_indicators = ['weak', 'lack', 'insufficient', 'poor', 'inadequate', 
                             'missing', 'limited', 'unclear', 'vague']
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for indicator in weakness_indicators:
                if indicator in sentence_lower:
                    weaknesses.append({
                        "weakness": sentence.strip() + '.',
                        "severity": self._assess_severity(sentence),
                        "category": self._categorize_weakness(sentence)
                    })
                    break
        
        return weaknesses[:6]
    
    def _assess_severity(self, sentence: str) -> str:
        """Assess the severity of a weakness."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['critical', 'severe', 'major', 'significant']):
            return "high"
        elif any(word in sentence_lower for word in ['moderate', 'some', 'partial']):
            return "medium"
        else:
            return "low"
    
    def _categorize_weakness(self, sentence: str) -> str:
        """Categorize the type of weakness identified."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['data', 'evidence', 'support']):
            return "evidence"
        elif any(word in sentence_lower for word in ['logic', 'reasoning', 'argument']):
            return "reasoning"
        elif any(word in sentence_lower for word in ['bias', 'objective', 'neutral']):
            return "objectivity"
        elif any(word in sentence_lower for word in ['complete', 'comprehensive', 'thorough']):
            return "completeness"
        else:
            return "general"
    
    def _identify_gaps(self, content: str) -> List[str]:
        """Identify gaps in the analysis."""
        gaps = []
        gap_indicators = ['gap', 'missing', 'overlook', 'fail to', 'does not address', 
                         'ignores', 'omits', 'neglects']
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in gap_indicators):
                gaps.append(sentence.strip() + '.')
        
        return gaps[:5]
    
    def _suggest_improvements(self, content: str) -> List[Dict[str, str]]:
        """Suggest improvements based on critique."""
        improvements = []
        improvement_indicators = ['could', 'should', 'improve', 'enhance', 
                                'strengthen', 'add', 'consider', 'recommend']
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in improvement_indicators):
                improvements.append({
                    "suggestion": sentence.strip() + '.',
                    "priority": self._prioritize_improvement(sentence),
                    "effort": self._estimate_effort(sentence)
                })
        
        return improvements[:7]
    
    def _prioritize_improvement(self, sentence: str) -> str:
        """Prioritize improvement suggestions."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['critical', 'essential', 'must', 'urgent']):
            return "high"
        elif any(word in sentence_lower for word in ['should', 'important', 'recommend']):
            return "medium"
        else:
            return "low"
    
    def _estimate_effort(self, sentence: str) -> str:
        """Estimate effort required for improvement."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['simple', 'easy', 'quick', 'minor']):
            return "low"
        elif any(word in sentence_lower for word in ['moderate', 'some', 'reasonable']):
            return "medium"
        elif any(word in sentence_lower for word in ['significant', 'substantial', 'major']):
            return "high"
        else:
            return "medium"
    
    def _calculate_quality_score(self, content: str) -> Dict[str, float]:
        """Calculate detailed quality scores."""
        scores = self.evaluation_rubric.copy()
        content_lower = content.lower()
        
        # Score each criterion based on content analysis
        for criterion in self.evaluation_criteria:
            score = 0.5  # Base score
            
            # Positive indicators
            if criterion in content_lower and any(word in content_lower for word in ['good', 'strong', 'excellent']):
                score += 0.3
            
            # Negative indicators
            if criterion in content_lower and any(word in content_lower for word in ['poor', 'weak', 'lacking']):
                score -= 0.3
            
            scores[criterion]["score"] = max(0.0, min(1.0, score))
        
        # Calculate overall score
        overall = sum(scores[c]["score"] * scores[c]["weight"] for c in scores)
        
        return {
            "overall": overall,
            "breakdown": {c: scores[c]["score"] for c in scores},
            "grade": self._convert_to_grade(overall)
        }
    
    def _convert_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _identify_alternatives(self, content: str) -> List[str]:
        """Identify alternative perspectives mentioned."""
        alternatives = []
        alternative_indicators = ['alternatively', 'another perspective', 'different approach', 
                                'could also', 'different view', 'alternative']
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in alternative_indicators):
                alternatives.append(sentence.strip() + '.')
        
        return alternatives[:4]
    
    def _generate_verdict(self, content: str) -> Dict[str, str]:
        """Generate final verdict based on evaluation."""
        # Simple verdict generation based on content sentiment
        positive_count = sum(content.lower().count(word) for word in ['good', 'strong', 'excellent', 'effective'])
        negative_count = sum(content.lower().count(word) for word in ['poor', 'weak', 'lacking', 'insufficient'])
        
        if positive_count > negative_count * 2:
            verdict = "Approved with minor revisions"
            confidence = "high"
        elif positive_count > negative_count:
            verdict = "Approved with moderate revisions"
            confidence = "medium"
        else:
            verdict = "Requires significant improvements"
            confidence = "medium"
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "summary": "Based on comprehensive evaluation across multiple criteria."
        }

class SynthesizerAgent(BaseAgent):
    """Agent specialized in synthesizing information and creating coherent narratives."""
    
    def __init__(self, name: str, llm: Optional[ChatOpenAI] = None):
        super().__init__(name, AgentRole.SYNTHESIZER, llm)
        self.synthesis_strategies = ["integrate", "summarize", "reconcile", "consolidate", "harmonize"]
        self.output_formats = ["executive_summary", "detailed_report", "action_plan", "strategic_recommendation"]
        
    async def _execute_task(self, task: Task) -> Any:
        """Execute synthesis task."""
        # Gather all relevant information from previous agents
        synthesis_input = self._gather_synthesis_input(task)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Synthesizer Agent specializing in integrating diverse information.
            Your role is to:
            1. Combine multiple perspectives into coherent narratives
            2. Resolve contradictions and find common ground
            3. Create comprehensive summaries that capture key insights
            4. Generate actionable conclusions and recommendations
            5. Ensure clarity and accessibility of complex information
            6. Prioritize information based on relevance and impact
            7. Create structured outputs suitable for decision-making"""),
            HumanMessage(content=f"Synthesize the following information: {task.description}\n\nInput data: {synthesis_input}")
        ])
        
        response = await self.llm.ainvoke(prompt.format_messages())
        
        # Structure synthesis results
        synthesis_result = {
            "synthesis": response.content,
            "key_themes": self._extract_themes(response.content),
            "consensus_points": self._identify_consensus(response.content),
            "contradictions": self._identify_contradictions(response.content),
            "final_recommendations": self._generate_final_recommendations(response.content),
            "executive_summary": self._create_executive_summary(response.content),
            "action_items": self._extract_action_items(response.content),
            "confidence_level": self._assess_synthesis_confidence(response.content)
        }
        
        # Store synthesis for future reference
        self.memory.update_context('latest_synthesis', synthesis_result)
        
        return synthesis_result
    
    def _gather_synthesis_input(self, task: Task) -> str:
        """Gather all relevant information for synthesis."""
        input_sections = []
        
        # Get findings from all agents
        agent_findings = {}
        for msg in self.memory.get_recent_messages(20):
            if msg.sender not in agent_findings:
                agent_findings[msg.sender] = []
            agent_findings[msg.sender].append(msg.content[:500])
        
        # Structure input by agent type
        for agent, findings in agent_findings.items():
            if findings:
                input_sections.append(f"\n{agent} Contributions:\n" + "\n".join(findings[:3]))
        
        # Add knowledge base insights
        knowledge_items = []
        for key, knowledge in self.memory.knowledge_base.items():
            knowledge_items.append(f"{knowledge['type'].title()}: {knowledge['content'][:200]}...")
        
        if knowledge_items:
            input_sections.append("\nKnowledge Base:\n" + "\n".join(knowledge_items[:5]))
        
        return "\n".join(input_sections)
    
    def _extract_themes(self, content: str) -> List[Dict[str, Any]]:
        """Extract major themes from synthesis."""
        themes = []
        theme_indicators = ['theme', 'pattern', 'trend', 'common', 'recurring', 
                          'central', 'key finding', 'main point']
        
        # Split into paragraphs and analyze
        paragraphs = content.split('\n\n')
        theme_count = 0
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            if any(indicator in paragraph_lower for indicator in theme_indicators):
                theme_count += 1
                themes.append({
                    "theme": f"Theme {theme_count}",
                    "description": paragraph.strip()[:300] + "..." if len(paragraph) > 300 else paragraph.strip(),
                    "importance": self._assess_theme_importance(paragraph),
                    "support_level": self._assess_support_level(paragraph)
                })
        
        # If no explicit themes found, extract from content structure
        if not themes and paragraphs:
            for i, paragraph in enumerate(paragraphs[:5]):
                if len(paragraph.strip()) > 50:
                    themes.append({
                        "theme": f"Finding {i+1}",
                        "description": paragraph.strip()[:300] + "..." if len(paragraph) > 300 else paragraph.strip(),
                        "importance": "medium",
                        "support_level": "moderate"
                    })
        
        return themes[:6]
    
    def _assess_theme_importance(self, content: str) -> str:
        """Assess the importance of a theme."""
        content_lower = content.lower()
        
        high_importance_indicators = ['critical', 'essential', 'fundamental', 'crucial', 'vital']
        if any(indicator in content_lower for indicator in high_importance_indicators):
            return "high"
        
        low_importance_indicators = ['minor', 'secondary', 'marginal', 'peripheral']
        if any(indicator in content_lower for indicator in low_importance_indicators):
            return "low"
        
        return "medium"
    
    def _assess_support_level(self, content: str) -> str:
        """Assess the level of support for a theme."""
        content_lower = content.lower()
        
        strong_support = ['consensus', 'unanimous', 'clear evidence', 'strongly supported']
        if any(indicator in content_lower for indicator in strong_support):
            return "strong"
        
        weak_support = ['limited evidence', 'some indication', 'preliminary', 'tentative']
        if any(indicator in content_lower for indicator in weak_support):
            return "weak"
        
        return "moderate"
    
    def _identify_consensus(self, content: str) -> List[Dict[str, str]]:
        """Identify points of consensus."""
        consensus_points = []
        consensus_indicators = ['agree', 'consensus', 'common', 'shared', 'unanimous', 
                              'consistent', 'alignment', 'convergence']
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in consensus_indicators):
                consensus_points.append({
                    "point": sentence.strip() + '.',
                    "strength": "strong" if "unanimous" in sentence_lower or "clear consensus" in sentence_lower else "moderate"
                })
        
        return consensus_points[:6]
    
    def _identify_contradictions(self, content: str) -> List[Dict[str, str]]:
        """Identify contradictions or conflicts."""
        contradictions = []
        conflict_indicators = ['however', 'contrary', 'conflict', 'disagree', 'opposing', 
                             'contradicts', 'tension', 'divergent', 'inconsistent']
        
        sentences = content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in conflict_indicators):
                contradictions.append({
                    "contradiction": sentence.strip() + '.',
                    "resolution_suggested": self._check_for_resolution(sentence),
                    "impact": self._assess_contradiction_impact(sentence)
                })
        
        return contradictions[:4]
    
    def _check_for_resolution(self, sentence: str) -> bool:
        """Check if a resolution is suggested for the contradiction."""
        resolution_indicators = ['can be resolved', 'reconcile', 'bridge', 'common ground', 'compromise']
        return any(indicator in sentence.lower() for indicator in resolution_indicators)
    
    def _assess_contradiction_impact(self, sentence: str) -> str:
        """Assess the impact of a contradiction."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['fundamental', 'major', 'significant']):
            return "high"
        elif any(word in sentence_lower for word in ['minor', 'small', 'slight']):
            return "low"
        else:
            return "medium"
    
    def _generate_final_recommendations(self, content: str) -> List[Dict[str, Any]]:
        """Generate final synthesized recommendations."""
        recommendations = []
        
        # Extract recommendation sentences
        rec_indicators = ['recommend', 'suggest', 'propose', 'advise', 'should', 'must']
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in rec_indicators):
                recommendations.append({
                    "recommendation": sentence.strip() + '.',
                    "priority": self._determine_priority(sentence),
                    "timeframe": self._determine_timeframe(sentence),
                    "category": self._categorize_recommendation(sentence)
                })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return recommendations[:8]
    
    def _determine_priority(self, sentence: str) -> str:
        """Determine recommendation priority."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['urgent', 'immediate', 'critical', 'must']):
            return "high"
        elif any(word in sentence_lower for word in ['should', 'important', 'recommend']):
            return "medium"
        else:
            return "low"
    
    def _determine_timeframe(self, sentence: str) -> str:
        """Determine recommendation timeframe."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['immediate', 'now', 'urgent', 'asap']):
            return "immediate"
        elif any(word in sentence_lower for word in ['short-term', 'soon', 'near']):
            return "short-term"
        elif any(word in sentence_lower for word in ['long-term', 'future', 'eventually']):
            return "long-term"
        else:
            return "medium-term"
    
    def _categorize_recommendation(self, sentence: str) -> str:
        """Categorize the type of recommendation."""
        sentence_lower = sentence.lower()
        
        if any(word in sentence_lower for word in ['strategy', 'strategic', 'plan']):
            return "strategic"
        elif any(word in sentence_lower for word in ['operational', 'process', 'procedure']):
            return "operational"
        elif any(word in sentence_lower for word in ['tactical', 'action', 'implement']):
            return "tactical"
        else:
            return "general"
    
    def _create_executive_summary(self, content: str) -> str:
        """Create an executive summary of the synthesis."""
        # Extract key sentences for summary
        summary_parts = []
        
        # Get opening statement
        paragraphs = content.split('\n\n')
        if paragraphs:
            opening = paragraphs[0][:200]
            if len(paragraphs[0]) > 200:
                opening += "..."
            summary_parts.append(opening)
        
        # Extract key findings
        key_finding_indicators = ['key finding', 'main conclusion', 'importantly', 'notably']
        for paragraph in paragraphs[1:]:
            if any(indicator in paragraph.lower() for indicator in key_finding_indicators):
                summary_parts.append(paragraph[:150] + "..." if len(paragraph) > 150 else paragraph)
                if len(summary_parts) >= 3:
                    break
        
        # Add conclusion if present
        if len(paragraphs) > 1:
            conclusion = paragraphs[-1][:150]
            if conclusion not in summary_parts:
                summary_parts.append(conclusion + "..." if len(paragraphs[-1]) > 150 else conclusion)
        
        return " ".join(summary_parts)
    
    def _extract_action_items(self, content: str) -> List[Dict[str, str]]:
        """Extract specific action items from synthesis."""
        action_items = []
        action_indicators = ['action:', 'task:', 'todo:', 'action item', 'next step', 'to do']
        
        lines = content.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in action_indicators):
                action_items.append({
                    "action": line.strip(),
                    "owner": "TBD",
                    "deadline": "TBD",
                    "status": "pending"
                })
            # Also check for numbered action items
            elif line.strip() and line.strip()[0].isdigit() and 'action' in line_lower:
                action_items.append({
                    "action": line.strip(),
                    "owner": "TBD", 
                    "deadline": "TBD",
                    "status": "pending"
                })
        
        return action_items[:10]
    
    def _assess_synthesis_confidence(self, content: str) -> Dict[str, Any]:
        """Assess confidence in the synthesis."""
        # Calculate various confidence indicators
        word_count = len(content.split())
        
        # Check for confidence language
        high_confidence_words = ['clear', 'strong', 'definitive', 'conclusive', 'certain']
        low_confidence_words = ['uncertain', 'unclear', 'tentative', 'preliminary', 'limited']
        
        high_conf_count = sum(content.lower().count(word) for word in high_confidence_words)
        low_conf_count = sum(content.lower().count(word) for word in low_confidence_words)
        
        # Calculate confidence score
        base_confidence = 0.7
        confidence_adjustment = (high_conf_count * 0.05) - (low_conf_count * 0.08)
        overall_confidence = max(0.3, min(0.95, base_confidence + confidence_adjustment))
        
        return {
            "overall": overall_confidence,
            "level": "high" if overall_confidence > 0.8 else "medium" if overall_confidence > 0.6 else "low",
            "factors": {
                "content_depth": word_count > 500,
                "evidence_strength": high_conf_count > low_conf_count,
                "consensus_level": "consensus" in content.lower()
            }
        }

class CoordinatorAgent(BaseAgent):
    """Agent responsible for coordinating other agents and managing workflow."""
    
    def __init__(self, name: str, llm: Optional[ChatOpenAI] = None):
        super().__init__(name, AgentRole.COORDINATOR, llm)
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.workflow_graph = nx.DiGraph()
        self.execution_history: List[Dict[str, Any]] = []
        self.workflow_templates = self._create_workflow_templates()
        self.collaboration_network = nx.Graph()
        
    def _create_workflow_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create predefined workflow templates for common problem types."""
        return {
            "research_analysis": [
                {"role": "Researcher", "task": "Gather comprehensive information"},
                {"role": "Analyst", "task": "Analyze findings and identify patterns"},
                {"role": "Critic", "task": "Evaluate analysis quality"},
                {"role": "Synthesizer", "task": "Create final recommendations"}
            ],
            "strategic_planning": [
                {"role": "Researcher", "task": "Research current state and trends"},
                {"role": "Analyst", "task": "SWOT analysis and opportunity identification"},
                {"role": "Researcher", "task": "Benchmark best practices"},
                {"role": "Critic", "task": "Risk assessment and gap analysis"},
                {"role": "Synthesizer", "task": "Strategic plan synthesis"}
            ],
            "problem_solving": [
                {"role": "Researcher", "task": "Define problem and gather context"},
                {"role": "Analyst", "task": "Root cause analysis"},
                {"role": "Researcher", "task": "Research potential solutions"},
                {"role": "Critic", "task": "Evaluate solution feasibility"},
                {"role": "Synthesizer", "task": "Recommend optimal solution"}
            ]
        }
        
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the coordinator."""
        self.agents[agent.name] = agent
        self.workflow_graph.add_node(agent.name, role=agent.role.value)
        self.collaboration_network.add_node(agent.name, role=agent.role.value)
        logger.info(f"Registered agent: {agent.name} with role {agent.role.value}")
        
    async def decompose_problem(self, problem: str, use_template: bool = False) -> List[Task]:
        """Decompose a complex problem into subtasks."""
        if use_template:
            # Try to match problem to a template
            template_tasks = self._match_problem_to_template(problem)
            if template_tasks:
                return template_tasks
        
        if self.llm:
            # Use LLM to decompose problem
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a Coordinator Agent responsible for breaking down complex problems.
                Decompose the problem into specific subtasks that can be assigned to specialized agents:
                - Researcher: For gathering information and facts
                - Analyst: For analyzing data and identifying patterns  
                - Critic: For evaluating quality and identifying issues
                - Synthesizer: For combining insights and creating summaries
                
                Create 4-8 clear, actionable subtasks with dependencies.
                Format each task as: [Role]: [Specific task description]"""),
                HumanMessage(content=f"Decompose this problem into subtasks: {problem}")
            ])
            
            response = await self.llm.ainvoke(prompt.format_messages())
            tasks = self._parse_tasks(response.content, problem)
        else:
            # Demo mode - use template or default decomposition
            tasks = self._create_default_tasks(problem)
        
        # Enhance tasks with metadata
        for i, task in enumerate(tasks):
            task.metadata['problem_complexity'] = self._assess_problem_complexity(problem)
            task.metadata['estimated_duration'] = self._estimate_task_duration(task)
            
        return tasks
    
    def _match_problem_to_template(self, problem: str) -> Optional[List[Task]]:
        """Match problem to a workflow template."""
        problem_lower = problem.lower()
        
        # Check for template matches
        if any(word in problem_lower for word in ['strategy', 'strategic', 'plan']):
            template_name = "strategic_planning"
        elif any(word in problem_lower for word in ['research', 'analyze', 'investigate']):
            template_name = "research_analysis"
        elif any(word in problem_lower for word in ['problem', 'solve', 'solution', 'fix']):
            template_name = "problem_solving"
        else:
            return None
        
        # Create tasks from template
        template = self.workflow_templates[template_name]
        tasks = []
        
        for i, step in enumerate(template):
            task = Task(
                id=f"task_{i+1}",
                description=f"{step['task']} for: {problem}",
                metadata={
                    "original_problem": problem,
                    "suggested_role": step['role'],
                    "template": template_name
                }
            )
            tasks.append(task)
        
        return tasks
    
    def _assess_problem_complexity(self, problem: str) -> str:
        """Assess the complexity of a problem."""
        # Simple heuristic based on problem characteristics
        complexity_indicators = {
            "high": ['multiple', 'complex', 'comprehensive', 'strategic', 'long-term'],
            "medium": ['analyze', 'evaluate', 'develop', 'assess'],
            "low": ['simple', 'basic', 'straightforward', 'identify']
        }
        
        problem_lower = problem.lower()
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in problem_lower for indicator in indicators):
                return level
        
        # Default based on length
        return "high" if len(problem) > 200 else "medium"
    
    def _estimate_task_duration(self, task: Task) -> float:
        """Estimate task duration in seconds."""
        # Base estimation on task characteristics
        base_duration = 30.0
        
        # Adjust based on role
        role_multipliers = {
            "Researcher": 1.2,
            "Analyst": 1.5,
            "Critic": 1.0,
            "Synthesizer": 1.3
        }
        
        role = task.metadata.get("suggested_role", "Researcher")
        duration = base_duration * role_multipliers.get(role, 1.0)
        
        # Adjust based on complexity
        complexity = task.metadata.get("problem_complexity", "medium")
        if complexity == "high":
            duration *= 1.5
        elif complexity == "low":
            duration *= 0.7
        
        return duration
    
    def _parse_tasks(self, content: str, original_problem: str) -> List[Task]:
        """Parse LLM response into Task objects."""
        tasks = []
        lines = content.split('\n')
        
        task_id = 1
        current_role = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for role indicators
            role_found = False
            for role in AgentRole:
                if role.value in line or role.value.lower() in line.lower():
                    current_role = role.value
                    role_found = True
                    break
            
            # Extract task if we have a role
            if current_role and ':' in line:
                # Extract task description after role mention
                task_parts = line.split(':', 1)
                if len(task_parts) > 1:
                    task_desc = task_parts[1].strip()
                    
                    task = Task(
                        id=f"task_{task_id}",
                        description=task_desc,
                        metadata={
                            "original_problem": original_problem,
                            "suggested_role": current_role,
                            "source": "llm_decomposition"
                        }
                    )
                    
                    tasks.append(task)
                    task_id += 1
        
        # Ensure we have at least some tasks
        if len(tasks) < 4:
            tasks.extend(self._create_default_tasks(original_problem)[len(tasks):])
        
        return tasks
    
    def _create_default_tasks(self, problem: str) -> List[Task]:
        """Create default tasks for a problem."""
        return [
            Task(
                id="task_1",
                description=f"Research comprehensive background information on: {problem}",
                metadata={"suggested_role": "Researcher", "source": "default"}
            ),
            Task(
                id="task_2", 
                description=f"Analyze key factors and patterns related to: {problem}",
                metadata={"suggested_role": "Analyst", "source": "default"}
            ),
            Task(
                id="task_3",
                description="Critically evaluate the research findings and analysis quality",
                metadata={"suggested_role": "Critic", "source": "default"}
            ),
            Task(
                id="task_4",
                description="Synthesize all findings into actionable insights and recommendations",
                metadata={"suggested_role": "Synthesizer", "source": "default"}
            )
        ]
    
    def _build_dependency_graph(self, tasks: List[Task]):
        """Build a dependency graph for tasks."""
        # Define role execution order
        role_order = {
            "Researcher": 1,
            "Analyst": 2,
            "Critic": 3,
            "Synthesizer": 4
        }
        
        # Sort tasks by role order
        sorted_tasks = sorted(tasks, 
                            key=lambda t: role_order.get(t.metadata.get("suggested_role", "Researcher"), 5))
        
        # Create dependencies based on order
        for i in range(len(sorted_tasks) - 1):
            current_task = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]
            
            # Only add dependency if next task has higher order role
            current_order = role_order.get(current_task.metadata.get("suggested_role"), 0)
            next_order = role_order.get(next_task.metadata.get("suggested_role"), 0)
            
            if next_order >= current_order:
                next_task.dependencies.append(current_task.id)
    
    async def execute_workflow(self, tasks: List[Task], parallel: bool = True) -> Dict[str, Any]:
        """Execute the workflow with given tasks."""
        start_time = datetime.now()
        self.performance_tracker.start_tracking()
        
        # Build task dependency graph
        self._build_dependency_graph(tasks)
        
        # Update workflow graph
        self._update_workflow_graph(tasks)
        
        # Execute tasks
        try:
            if parallel:
                await self._execute_parallel(tasks)
            else:
                await self._execute_sequential(tasks)
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
        
        # Compile final results
        end_time = datetime.now()
        self.performance_tracker.end_tracking()
        
        execution_time = (end_time - start_time).total_seconds()
        
        # Update collaboration network
        self._update_collaboration_network()
        
        workflow_result = {
            "tasks": tasks,
            "execution_time": execution_time,
            "success_rate": self._calculate_success_rate(tasks),
            "agent_contributions": self._compile_agent_contributions(tasks),
            "workflow_graph": self.collaboration_network,  # Use collaboration network instead
            "performance_metrics": self.performance_tracker.get_performance_summary(),
            "timestamp": datetime.now()
        }
        
        self.execution_history.append(workflow_result)
        
        return workflow_result
    
    def _update_workflow_graph(self, tasks: List[Task]):
        """Update the workflow graph with task relationships."""
        # Add task nodes
        for task in tasks:
            self.workflow_graph.add_node(
                task.id,
                task_description=task.description[:50] + "...",
                status=task.status.value
            )
        
        # Add edges for dependencies
        for task in tasks:
            for dep_id in task.dependencies:
                self.workflow_graph.add_edge(dep_id, task.id)
    
    def _update_collaboration_network(self):
        """Update the collaboration network based on agent interactions."""
        # Add collaboration edges between agents
        agent_names = list(self.agents.keys())
        
        # Create edges based on workflow patterns
        if len(agent_names) >= 4:
            # Research -> Analyst
            if "Researcher-1" in agent_names and "Analyst-1" in agent_names:
                self.collaboration_network.add_edge("Researcher-1", "Analyst-1", weight=3)
            
            # Analyst -> Critic
            if "Analyst-1" in agent_names and "Critic-1" in agent_names:
                self.collaboration_network.add_edge("Analyst-1", "Critic-1", weight=2)
            
            # Critic -> Synthesizer
            if "Critic-1" in agent_names and "Synthesizer-1" in agent_names:
                self.collaboration_network.add_edge("Critic-1", "Synthesizer-1", weight=2)
            
            # Research -> Synthesizer (direct connection)
            if "Researcher-1" in agent_names and "Synthesizer-1" in agent_names:
                self.collaboration_network.add_edge("Researcher-1", "Synthesizer-1", weight=1)
    
    async def _execute_parallel(self, tasks: List[Task]) -> List[Task]:
        """Execute tasks in parallel where possible."""
        completed = set()
        pending = tasks.copy()
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            while pending:
                # Find tasks ready for execution
                ready_tasks = [
                    task for task in pending
                    if all(dep in completed for dep in task.dependencies)
                ]
                
                if not ready_tasks:
                    # Handle potential deadlock
                    logger.warning("No ready tasks found, executing first pending task")
                    ready_tasks = [pending[0]] if pending else []
                
                if not ready_tasks:
                    break
                
                # Submit tasks for parallel execution
                future_to_task = {}
                for task in ready_tasks:
                    agent_name = self._select_agent_for_task(task)
                    if agent_name and agent_name in self.agents:
                        agent = self.agents[agent_name]
                        future = executor.submit(asyncio.run, agent.process_task(task))
                        future_to_task[future] = (task, agent_name)
                
                # Wait for tasks to complete
                for future in as_completed(future_to_task):
                    task, agent_name = future_to_task[future]
                    try:
                        completed_task = future.result()
                        completed.add(task.id)
                        pending.remove(task)
                        self.completed_tasks.append(completed_task)
                        
                        # Update workflow graph
                        self.workflow_graph.add_edge(
                            self.name, agent_name,
                            task_id=task.id,
                            timestamp=datetime.now().isoformat()
                        )
                        
                        # Record collaboration
                        await self._facilitate_collaboration(completed_task, agent_name)
                        
                    except Exception as e:
                        logger.error(f"Task {task.id} failed: {str(e)}")
                        task.status = TaskStatus.FAILED
        
        return tasks
    
    async def _execute_sequential(self, tasks: List[Task]) -> List[Task]:
        """Execute tasks sequentially."""
        for task in tasks:
            agent_name = self._select_agent_for_task(task)
            if agent_name and agent_name in self.agents:
                agent = self.agents[agent_name]
                completed_task = await agent.process_task(task)
                
                # Update workflow graph
                self.workflow_graph.add_edge(
                    self.name, agent_name,
                    task_id=task.id,
                    timestamp=datetime.now().isoformat()
                )
                
                self.completed_tasks.append(completed_task)
                
                # Facilitate collaboration
                await self._facilitate_collaboration(completed_task, agent_name)
        
        return tasks
    
    def _select_agent_for_task(self, task: Task) -> Optional[str]:
        """Select the best agent for a given task."""
        suggested_role = task.metadata.get("suggested_role")
        
        # Find agent with matching role
        for agent_name, agent in self.agents.items():
            if agent.role.value == suggested_role:
                # Check agent availability
                if agent.active and agent.current_task is None:
                    return agent_name
        
        # Fallback: find any available agent with the role
        for agent_name, agent in self.agents.items():
            if agent.role.value == suggested_role:
                return agent_name
        
        # Last resort: return any available agent
        for agent_name, agent in self.agents.items():
            if agent.active:
                return agent_name
        
        return None
    
    async def _facilitate_collaboration(self, task: Task, agent_name: str):
        """Facilitate collaboration between agents after task completion."""
        if not task.result or task.status != TaskStatus.COMPLETED:
            return
        
        # Create collaboration message
        collab_message = AgentMessage(
            sender=agent_name,
            recipient="all",
            content=f"Task completed: {task.description}\nKey findings: {str(task.result)[:500]}",
            message_type=MessageType.COMPLETION_REPORT,
            priority=3
        )
        
        # Share with relevant agents
        shared_count = 0
        for other_agent_name, other_agent in self.agents.items():
            if other_agent_name != agent_name:
                # Determine if collaboration is needed
                if self._should_collaborate(task, other_agent):
                    await other_agent.memory.add_message(collab_message)
                    self.performance_tracker.record_collaboration()
                    self.performance_tracker.record_message()
                    shared_count += 1
        
        # Log collaboration activity
        if shared_count > 0:
            logger.info(f"Agent {agent_name} shared findings with {shared_count} other agents")
    
    def _should_collaborate(self, task: Task, agent: BaseAgent) -> bool:
        """Determine if an agent should receive collaboration message."""
        # Synthesizer should receive all completion reports
        if agent.role == AgentRole.SYNTHESIZER:
            return True
        
        # Critic should receive analysis and research results
        if agent.role == AgentRole.CRITIC and task.metadata.get("suggested_role") in ["Researcher", "Analyst"]:
            return True
        
        # Analyst should receive research results
        if agent.role == AgentRole.ANALYST and task.metadata.get("suggested_role") == "Researcher":
            return True
        
        return False
    
    def _calculate_success_rate(self, tasks: List[Task]) -> float:
        """Calculate the success rate of task execution."""
        if not tasks:
            return 0.0
        
        successful = sum(1 for task in tasks if task.status == TaskStatus.COMPLETED)
        return successful / len(tasks)
    
    def _compile_agent_contributions(self, tasks: List[Task]) -> Dict[str, Any]:
        """Compile contributions from each agent."""
        contributions = {}
        
        for agent_name, agent in self.agents.items():
            agent_tasks = [task for task in tasks if task.assigned_to == agent_name]
            
            if agent_tasks:
                total_execution_time = sum(
                    task.performance_metrics.get('execution_time', 0)
                    for task in agent_tasks
                )
                
                avg_confidence = np.mean([task.confidence for task in agent_tasks])
                
                contributions[agent_name] = {
                    "role": agent.role.value,
                    "tasks_completed": len(agent_tasks),
                    "average_confidence": avg_confidence,
                    "total_execution_time": total_execution_time,
                    "collaboration_count": len(agent.collaboration_partners),
                    "status": agent.get_status_summary()
                }
            else:
                contributions[agent_name] = {
                    "role": agent.role.value,
                    "tasks_completed": 0,
                    "average_confidence": 0.0,
                    "total_execution_time": 0.0,
                    "collaboration_count": 0,
                    "status": agent.get_status_summary()
                }
        
        return contributions
    
    def get_workflow_insights(self) -> Dict[str, Any]:
        """Get insights about workflow execution patterns."""
        if not self.execution_history:
            return {"message": "No execution history available"}
        
        # Analyze execution patterns
        total_executions = len(self.execution_history)
        avg_execution_time = np.mean([wf['execution_time'] for wf in self.execution_history])
        avg_success_rate = np.mean([wf['success_rate'] for wf in self.execution_history])
        
        # Analyze agent performance
        agent_stats = {}
        for workflow in self.execution_history:
            for agent, contrib in workflow['agent_contributions'].items():
                if agent not in agent_stats:
                    agent_stats[agent] = {
                        'total_tasks': 0,
                        'total_time': 0,
                        'confidence_scores': []
                    }
                
                agent_stats[agent]['total_tasks'] += contrib['tasks_completed']
                agent_stats[agent]['total_time'] += contrib['total_execution_time']
                if contrib['average_confidence'] > 0:
                    agent_stats[agent]['confidence_scores'].append(contrib['average_confidence'])
        
        # Calculate agent efficiency
        agent_efficiency = {}
        for agent, stats in agent_stats.items():
            if stats['total_tasks'] > 0:
                agent_efficiency[agent] = {
                    'avg_time_per_task': stats['total_time'] / stats['total_tasks'],
                    'avg_confidence': np.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0,
                    'total_tasks': stats['total_tasks']
                }
        
        return {
            'total_workflows_executed': total_executions,
            'average_execution_time': avg_execution_time,
            'average_success_rate': avg_success_rate,
            'agent_efficiency': agent_efficiency,
            'most_efficient_agent': min(agent_efficiency.items(), 
                                      key=lambda x: x[1]['avg_time_per_task'])[0] if agent_efficiency else None,
            'highest_quality_agent': max(agent_efficiency.items(), 
                                       key=lambda x: x[1]['avg_confidence'])[0] if agent_efficiency else None
        }

class WorkflowVisualizer:
    """Handles visualization of agent interactions and workflow."""
    
    def __init__(self):
        self.color_map = Config.NODE_COLORS
        self.layout_cache = {}
        self.animation_frames = []
        
    def create_workflow_graph(self, workflow_graph: nx.Graph, 
                            active_agents: List[str] = None,
                            highlight_tasks: List[str] = None) -> go.Figure:
        """Create an interactive workflow visualization."""
        
        if len(workflow_graph.nodes()) == 0:
            return self._create_empty_graph()
        
        # Use spring layout for better visualization of connections
        pos = nx.spring_layout(workflow_graph, k=2, iterations=50)
        
        # Create traces
        edge_trace = self._create_edge_trace(workflow_graph, pos)
        node_trace = self._create_node_trace(workflow_graph, pos, active_agents, highlight_tasks)
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title={
                    'text': 'Agent Collaboration Network',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#2c3e50'}
                },
                showlegend=False,
                hovermode='closest',
                margin=dict(b=40, l=40, r=40, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='#f8f9fa',
                paper_bgcolor='white',
                font=dict(family='Inter, sans-serif'),
            )
        )
        
        return fig
        
    def _create_empty_graph(self) -> go.Figure:
        """Create empty graph with message."""
        fig = go.Figure()
        fig.add_annotation(
            text="No workflow data to display.<br>Start an analysis to see agent interactions.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color='#7f8c8d'),
            bgcolor='#ecf0f1',
            borderpad=20
        )
        fig.update_layout(
            height=500,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white'
        )
        return fig
    
    def _create_edge_trace(self, G: nx.Graph, pos: Dict) -> go.Scatter:
        """Create edge trace for the graph."""
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            if edge[0] in pos and edge[1] in pos:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#95a5a6'),
            hoverinfo='none',
            mode='lines'
        )
        
        return edge_trace
    
    def _create_node_trace(self, G: nx.Graph, pos: Dict, 
                          active_agents: List[str] = None,
                          highlight_tasks: List[str] = None) -> go.Scatter:
        """Create node trace for the graph."""
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        for node in G.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                # Get node attributes
                node_data = G.nodes[node]
                role = node_data.get('role', '')
                
                # Set node properties
                color = self.color_map.get(role, '#95a5a6')
                size = 40
                
                node_colors.append(color)
                node_sizes.append(size)
                node_text.append(node)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            hovertext=node_text,
            textposition="bottom center",
            marker=dict(
                showscale=False,
                color=node_colors,
                size=node_sizes,
                line=dict(color='white', width=2)
            )
        )
        
        return node_trace
    
    def create_task_timeline(self, tasks: List[Task]) -> go.Figure:
        """Create a timeline visualization of task execution."""
        
        # Prepare timeline data
        timeline_data = []
        
        for task in tasks:
            if task.created_at:
                # Use completed_at if available, otherwise estimate
                end_time = task.completed_at if task.completed_at else task.created_at + timedelta(seconds=30)
                
                timeline_data.append({
                    'Task': task.id,
                    'Agent': task.assigned_to or 'Unassigned',
                    'Start': task.created_at,
                    'Finish': end_time,
                    'Status': task.status.value,
                    'Confidence': task.confidence
                })
        
        if not timeline_data:
            return self._create_empty_timeline()
        
        # Create DataFrame
        df = pd.DataFrame(timeline_data)
        
        # Create Gantt chart
        fig = px.timeline(
            df, 
            x_start="Start", 
            x_end="Finish", 
            y="Agent",
            color="Confidence",
            hover_data=["Task", "Status"],
            color_continuous_scale="Viridis",
            labels={'Confidence': 'Confidence Score'}
        )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Task Execution Timeline',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            height=400,
            xaxis_title="Time",
            yaxis_title="Agent",
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif'),
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def _create_empty_timeline(self) -> go.Figure:
        """Create empty timeline with message."""
        fig = go.Figure()
        fig.add_annotation(
            text="No task execution data available yet.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color='#7f8c8d')
        )
        fig.update_layout(
            height=400,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white'
        )
        return fig
    
    def create_confidence_heatmap(self, agent_contributions: Dict[str, Any]) -> go.Figure:
        """Create a heatmap showing agent performance metrics."""
        
        if not agent_contributions:
            return self._create_empty_heatmap()
        
        # Prepare data
        agents = list(agent_contributions.keys())
        metrics = ['Tasks Completed', 'Avg Confidence', 'Time Efficiency', 'Collaboration Score']
        
        # Create data matrix
        data = []
        for metric in metrics:
            row = []
            for agent in agents:
                contrib = agent_contributions[agent]
                
                if metric == 'Tasks Completed':
                    value = contrib.get('tasks_completed', 0) / 5.0  # Normalize to 0-1
                elif metric == 'Avg Confidence':
                    value = contrib.get('average_confidence', 0)
                elif metric == 'Time Efficiency':
                    # Inverse of average time per task, normalized
                    time = contrib.get('total_execution_time', 1)
                    tasks = contrib.get('tasks_completed', 1)
                    avg_time = time / tasks if tasks > 0 else float('inf')
                    value = min(1.0, 30.0 / avg_time) if avg_time > 0 else 0
                elif metric == 'Collaboration Score':
                    value = min(1.0, contrib.get('collaboration_count', 0) / 3.0)
                else:
                    value = 0
                
                row.append(value)
            data.append(row)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=agents,
            y=metrics,
            colorscale='Blues',
            text=np.round(data, 2),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title={'text': "Score", 'side': 'right'}),
            hoverongaps=False
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Agent Performance Metrics',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            xaxis_title="Agents",
            yaxis_title="Metrics",
            height=350,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif')
        )
        
        return fig
    
    def _create_empty_heatmap(self) -> go.Figure:
        """Create empty heatmap with message."""
        fig = go.Figure()
        fig.add_annotation(
            text="No agent performance data available yet.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color='#7f8c8d')
        )
        fig.update_layout(
            height=350,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white'
        )
        return fig
    
    def create_performance_comparison(self, performance_metrics: Dict[str, Any]) -> go.Figure:
        """Create performance comparison visualization."""
        
        # Extract metrics
        baseline_time = Config.BENCHMARK_BASELINE['single_agent_time']
        actual_time = performance_metrics.get('average_task_completion_time', baseline_time)
        time_improvement = performance_metrics.get('time_improvement_percentage', 0)
        
        # Create bar chart
        categories = ['Single Agent', 'Multi-Agent System']
        values = [baseline_time, actual_time]
        colors = ['#e74c3c', '#2ecc71']
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[f'{v:.1f}s' for v in values],
                textposition='auto'
            )
        ])
        
        # Add improvement annotation
        if time_improvement > 0:
            fig.add_annotation(
                x=1, y=actual_time + 5,
                text=f"{time_improvement:.1f}% Faster",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#2ecc71',
                font=dict(size=14, color='#2ecc71', weight='bold')
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Performance Comparison',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            yaxis_title="Average Completion Time (seconds)",
            height=400,
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='white',
            font=dict(family='Inter, sans-serif'),
            showlegend=False
        )
        
        return fig

class ReportGenerator:
    """Generates comprehensive PDF reports from multi-agent collaboration."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.custom_styles = self._create_custom_styles()
        
    def _create_custom_styles(self) -> Dict[str, ParagraphStyle]:
        """Create custom paragraph styles for the report."""
        
        custom_styles = {}
        
        # Title style
        custom_styles['CustomTitle'] = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        
        # Section header style
        custom_styles['SectionHeader'] = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=20
        )
        
        # Normal text style
        custom_styles['CustomBody'] = ParagraphStyle(
            'CustomBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            leading=14,
            spaceAfter=10
        )
        
        return custom_styles
    
    def generate_report(self, 
                       workflow_result: Dict[str, Any],
                       problem_statement: str,
                       include_sections: List[str] = None,
                       filename: str = "multi_agent_analysis_report.pdf") -> str:
        """Generate comprehensive PDF report from workflow results."""
        
        try:
            # Create document
            doc = SimpleDocTemplate(
                filename,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Container for the 'Flowable' objects
            elements = []
            
            # Title page
            elements.append(Paragraph("Multi-Agent Analysis Report", self.custom_styles['CustomTitle']))
            elements.append(Paragraph(Config.COMPANY_NAME, self.styles['Heading3']))
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                                    self.styles['Normal']))
            elements.append(Spacer(1, 0.3*inch))
            elements.append(Paragraph(f"<b>Problem Statement:</b> {problem_statement}", 
                                    self.custom_styles['CustomBody']))
            elements.append(PageBreak())
            
            # Add selected sections
            section_methods = {
                'executive_summary': self._add_executive_summary,
                'task_analysis': self._add_task_analysis,
                'agent_contributions': self._add_agent_contributions,
                'key_findings': self._add_key_findings,
                'recommendations': self._add_recommendations,
                'confidence_analysis': self._add_confidence_analysis,
                'performance_metrics': self._add_performance_metrics
            }
            
            if include_sections is None:
                include_sections = list(section_methods.keys())
            
            for section in include_sections:
                if section in section_methods:
                    section_methods[section](elements, workflow_result, problem_statement)
            
            # Footer
            elements.append(PageBreak())
            elements.append(Paragraph("Report Generation Details", self.custom_styles['SectionHeader']))
            
            execution_time = workflow_result.get('execution_time', 0)
            timestamp = workflow_result.get('timestamp', datetime.now())
            
            footer_text = f"""
            Analysis completed in {execution_time:.1f} seconds<br/>
            Report generated at {timestamp.strftime('%B %d, %Y at %I:%M %p')}<br/>
            <br/>
            Powered by {Config.COMPANY_NAME}<br/>
            Advanced Multi-Agent AI Collaboration System
            """
            
            elements.append(Paragraph(footer_text, self.styles['Normal']))
            
            # Build PDF
            doc.build(elements)
            
            return filename
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return None
    
    def _add_executive_summary(self, elements: list, workflow_result: Dict, problem_statement: str):
        """Add executive summary section to report."""
        elements.append(Paragraph("Executive Summary", self.custom_styles['SectionHeader']))
        
        tasks = workflow_result.get('tasks', [])
        success_rate = workflow_result.get('success_rate', 0)
        execution_time = workflow_result.get('execution_time', 0)
        performance = workflow_result.get('performance_metrics', {})
        
        # Find synthesis task for summary content
        synthesis_task = None
        for task in tasks:
            if task.assigned_to and 'Synthesizer' in task.assigned_to:
                synthesis_task = task
                break
        
        summary_text = f"""
        The multi-agent system successfully analyzed the problem through coordinated efforts of 
        specialized agents, achieving a <b>{success_rate:.0%} task completion rate</b> in 
        <b>{execution_time:.1f} seconds</b>.
        """
        
        elements.append(Paragraph(summary_text, self.custom_styles['CustomBody']))
        
        if synthesis_task and isinstance(synthesis_task.result, dict):
            exec_summary = synthesis_task.result.get('executive_summary', '')
            if exec_summary:
                elements.append(Spacer(1, 0.1*inch))
                elements.append(Paragraph(exec_summary, self.custom_styles['CustomBody']))
        
        # Add performance highlights
        time_improvement = performance.get('time_improvement_percentage', 0)
        efficiency_score = performance.get('efficiency_score', 0)
        
        performance_text = f"""
        <br/>
        <b>Key Performance Indicators:</b><br/>
        • Performance Improvement: {time_improvement:.1f}% faster than single-agent approach<br/>
        • System Efficiency Score: {efficiency_score:.2f}/1.0<br/>
        • Total Collaborations: {performance.get('total_collaborations', 0)}<br/>
        """
        
        elements.append(Paragraph(performance_text, self.custom_styles['CustomBody']))
        elements.append(Spacer(1, 0.2*inch))
    
    def _add_task_analysis(self, elements: list, workflow_result: Dict, problem_statement: str):
        """Add task analysis section to report."""
        elements.append(Paragraph("Task Analysis", self.custom_styles['SectionHeader']))
        
        tasks = workflow_result.get('tasks', [])
        
        # Task overview
        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        failed_tasks = [t for t in tasks if t.status == TaskStatus.FAILED]
        
        overview_text = f"""
        <b>Task Overview:</b><br/>
        • Total Tasks: {len(tasks)}<br/>
        • Completed: {len(completed_tasks)}<br/>
        • Failed: {len(failed_tasks)}<br/>
        • Average Confidence: {np.mean([t.confidence for t in completed_tasks]) if completed_tasks else 0:.2%}<br/>
        """
        
        elements.append(Paragraph(overview_text, self.custom_styles['CustomBody']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Create task table
        task_data = [['Task ID', 'Description', 'Agent', 'Status', 'Confidence']]
        
        for task in tasks:
            task_data.append([
                task.id,
                task.description[:50] + '...' if len(task.description) > 50 else task.description,
                task.assigned_to or 'N/A',
                task.status.value.title(),
                f"{task.confidence:.0%}"
            ])
        
        task_table = Table(task_data, colWidths=[1*inch, 2.5*inch, 1.2*inch, 1*inch, 1*inch])
        task_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(task_table)
        elements.append(Spacer(1, 0.2*inch))
    
    def _add_agent_contributions(self, elements: list, workflow_result: Dict, problem_statement: str):
        """Add agent contributions section to report."""
        elements.append(Paragraph("Agent Contributions", self.custom_styles['SectionHeader']))
        
        contributions = workflow_result.get('agent_contributions', {})
        
        for agent, stats in contributions.items():
            role = stats.get('role', 'Unknown')
            tasks_completed = stats.get('tasks_completed', 0)
            avg_confidence = stats.get('average_confidence', 0)
            exec_time = stats.get('total_execution_time', 0)
            collab_count = stats.get('collaboration_count', 0)
            
            agent_text = f"""
            <b>{agent} ({role}):</b><br/>
            • Tasks Completed: {tasks_completed}<br/>
            • Average Confidence: {avg_confidence:.0%}<br/>
            • Total Execution Time: {exec_time:.1f}s<br/>
            • Collaborations: {collab_count}<br/>
            """
            
            elements.append(Paragraph(agent_text, self.custom_styles['CustomBody']))
            elements.append(Spacer(1, 0.1*inch))
    
    def _add_key_findings(self, elements: list, workflow_result: Dict, problem_statement: str):
        """Add key findings section to report."""
        elements.append(Paragraph("Key Findings", self.custom_styles['SectionHeader']))
        
        tasks = workflow_result.get('tasks', [])
        
        findings_by_type = {
            'Research Findings': [],
            'Analytical Insights': [],
            'Critical Observations': [],
            'Synthesized Conclusions': []
        }
        
        # Extract findings from different agent types
        for task in tasks:
            if task.status == TaskStatus.COMPLETED and task.result:
                role = task.metadata.get('suggested_role', '')
                
                if isinstance(task.result, dict):
                    if 'Researcher' in role and 'key_points' in task.result:
                        findings_by_type['Research Findings'].extend(task.result['key_points'][:3])
                    
                    elif 'Analyst' in role and 'insights' in task.result:
                        findings_by_type['Analytical Insights'].extend(task.result['insights'][:3])
                    
                    elif 'Critic' in role and 'strengths' in task.result:
                        for strength in task.result['strengths'][:2]:
                            if isinstance(strength, dict):
                                findings_by_type['Critical Observations'].append(
                                    strength.get('strength', str(strength))
                                )
                            else:
                                findings_by_type['Critical Observations'].append(str(strength))
                    
                    elif 'Synthesizer' in role and 'key_themes' in task.result:
                        for theme in task.result['key_themes'][:2]:
                            if isinstance(theme, dict):
                                findings_by_type['Synthesized Conclusions'].append(
                                    theme.get('description', str(theme))
                                )
                            else:
                                findings_by_type['Synthesized Conclusions'].append(str(theme))
        
        # Format findings
        for finding_type, findings in findings_by_type.items():
            if findings:
                elements.append(Paragraph(f"<b>{finding_type}:</b>", self.styles['Heading4']))
                for finding in findings:
                    elements.append(Paragraph(f"• {finding}", self.custom_styles['CustomBody']))
                elements.append(Spacer(1, 0.1*inch))
    
    def _add_recommendations(self, elements: list, workflow_result: Dict, problem_statement: str):
        """Add recommendations section to report."""
        elements.append(Paragraph("Recommendations", self.custom_styles['SectionHeader']))
        
        tasks = workflow_result.get('tasks', [])
        
        all_recommendations = []
        
        # Collect recommendations from all agents
        for task in tasks:
            if task.status == TaskStatus.COMPLETED and task.result:
                if isinstance(task.result, dict):
                    for field in ['recommendations', 'final_recommendations', 'improvements']:
                        if field in task.result:
                            recs = task.result[field]
                            for rec in recs:
                                if isinstance(rec, dict):
                                    all_recommendations.append(rec)
                                else:
                                    all_recommendations.append({
                                        'recommendation': str(rec),
                                        'priority': 'medium',
                                        'source': task.assigned_to
                                    })
        
        if not all_recommendations:
            elements.append(Paragraph("No specific recommendations were generated.", 
                                    self.custom_styles['CustomBody']))
            return
        
        # Categorize by priority
        priority_groups = {'high': [], 'medium': [], 'low': []}
        
        for rec in all_recommendations:
            priority = rec.get('priority', 'medium')
            if priority in priority_groups:
                priority_groups[priority].append(rec)
        
        # Add recommendations by priority
        for priority, recs in priority_groups.items():
            if recs:
                elements.append(Paragraph(f"<b>{priority.title()} Priority:</b>", self.styles['Heading4']))
                for rec in recs[:5]:  # Limit to top 5 per category
                    rec_text = rec.get('recommendation', rec)
                    elements.append(Paragraph(f"• {rec_text}", self.custom_styles['CustomBody']))
                elements.append(Spacer(1, 0.1*inch))
    
    def _add_confidence_analysis(self, elements: list, workflow_result: Dict, problem_statement: str):
        """Add confidence analysis section to report."""
        elements.append(Paragraph("Confidence Analysis", self.custom_styles['SectionHeader']))
        
        tasks = workflow_result.get('tasks', [])
        contributions = workflow_result.get('agent_contributions', {})
        
        # Calculate overall confidence
        task_confidences = [t.confidence for t in tasks if t.confidence > 0]
        overall_confidence = np.mean(task_confidences) if task_confidences else 0
        
        confidence_text = f"""
        <b>Overall Confidence Score: {overall_confidence:.0%}</b><br/>
        <br/>
        The confidence score reflects the system's assessment of result quality and reliability 
        based on evidence strength, consistency, and completeness.
        """
        
        elements.append(Paragraph(confidence_text, self.custom_styles['CustomBody']))
        elements.append(Spacer(1, 0.1*inch))
        
        # Create confidence table
        conf_data = [['Agent Role', 'Average Confidence', 'Tasks Completed']]
        
        for agent, stats in contributions.items():
            role = stats.get('role', 'Unknown')
            avg_conf = stats.get('average_confidence', 0)
            tasks = stats.get('tasks_completed', 0)
            conf_data.append([role, f"{avg_conf:.0%}", str(tasks)])
        
        conf_table = Table(conf_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
        conf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(conf_table)
        elements.append(Spacer(1, 0.2*inch))
    
    def _add_performance_metrics(self, elements: list, workflow_result: Dict, problem_statement: str):
        """Add performance metrics section to report."""
        elements.append(Paragraph("Performance Metrics", self.custom_styles['SectionHeader']))
        
        performance = workflow_result.get('performance_metrics', {})
        
        if not performance:
            elements.append(Paragraph("No performance metrics available.", 
                                    self.custom_styles['CustomBody']))
            return
        
        # Extract metrics
        total_time = performance.get('total_execution_time', 0)
        avg_task_time = performance.get('average_task_completion_time', 0)
        total_collab = performance.get('total_collaborations', 0)
        total_messages = performance.get('total_messages', 0)
        efficiency = performance.get('efficiency_score', 0)
        time_improvement = performance.get('time_improvement_percentage', 0)
        
        metrics_text = f"""
        <b>System Performance Overview:</b><br/>
        <br/>
        • Total Execution Time: {total_time:.1f}s<br/>
        • Average Task Time: {avg_task_time:.1f}s<br/>
        • Time Improvement: {time_improvement:.1f}% faster than baseline<br/>
        • Total Collaborations: {total_collab}<br/>
        • Message Exchanges: {total_messages}<br/>
        • Efficiency Score: {efficiency:.2%}<br/>
        """
        
        elements.append(Paragraph(metrics_text, self.custom_styles['CustomBody']))
        
        # Add performance insights
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph("<b>Performance Insights:</b>", self.styles['Heading4']))
        
        insights = []
        if time_improvement > 30:
            insights.append("Exceptional Performance: The multi-agent system achieved significant time savings through parallel processing.")
        elif time_improvement > 15:
            insights.append("Good Performance: The system demonstrated efficient task distribution and execution.")
        else:
            insights.append("Standard Performance: The system completed tasks within expected parameters.")
        
        if efficiency > 0.8:
            insights.append("High Efficiency: Excellent resource utilization and agent coordination.")
        elif efficiency > 0.6:
            insights.append("Moderate Efficiency: Good balance between speed and quality.")
        else:
            insights.append("Efficiency Opportunity: Consider optimizing agent workflows for better performance.")
        
        for insight in insights:
            elements.append(Paragraph(f"• {insight}", self.custom_styles['CustomBody']))

# Gradio Interface Functions
def create_gradio_interface():
    """Create the main Gradio interface for the multi-agent system."""
    
    # Initialize components
    coordinator = None
    visualizer = WorkflowVisualizer()
    report_generator = ReportGenerator()
    
    # State variables
    current_workflow = None
    current_problem = ""
    demo_mode = False
    
    def initialize_agents(api_key: str, model: str = "gpt-4", use_demo: bool = False) -> str:
        """Initialize the multi-agent system."""
        nonlocal coordinator, demo_mode
        
        demo_mode = use_demo
        
        if not use_demo and not api_key:
            return "Please provide an OpenAI API key or enable Demo Mode to initialize the agents."
        
        try:
            # Initialize LLM only if not in demo mode
            llm = None
            if not use_demo and api_key:
                llm = ChatOpenAI(
                    api_key=api_key,
                    model=model,
                    temperature=Config.TEMPERATURE,
                    max_tokens=Config.MAX_TOKENS
                )
            
            # Create coordinator
            coordinator = CoordinatorAgent("Coordinator", llm)
            
            # Create specialized agents
            researcher = ResearcherAgent("Researcher-1", llm)
            analyst = AnalystAgent("Analyst-1", llm)
            critic = CriticAgent("Critic-1", llm)
            synthesizer = SynthesizerAgent("Synthesizer-1", llm)
            
            # Register agents with coordinator
            coordinator.register_agent(researcher)
            coordinator.register_agent(analyst)
            coordinator.register_agent(critic)
            coordinator.register_agent(synthesizer)
            
            mode_text = "Demo Mode" if use_demo else f"Live Mode ({model})"
            return f"Successfully initialized multi-agent system with {len(coordinator.agents)} agents in {mode_text}."
            
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            return f"Error initializing agents: {str(e)}"
    
    async def analyze_problem(problem: str, execution_mode: str, use_template: bool = False) -> Tuple[str, Any, Any, Any, Any]:
        """Analyze a problem using the multi-agent system."""
        nonlocal current_workflow, current_problem
        
        if not coordinator:
            return "Please initialize the agents first.", None, None, None, None
        
        if not problem:
            return "Please enter a problem to analyze.", None, None, None, None
        
        current_problem = problem
        
        try:
            # Update status
            status = "Decomposing problem into subtasks..."
            
            # Decompose problem
            tasks = await coordinator.decompose_problem(problem, use_template=use_template)
            
            if not tasks:
                return "Failed to decompose problem into tasks.", None, None, None, None
            
            # Update status
            status = f"Executing {len(tasks)} tasks using {execution_mode} mode..."
            parallel = execution_mode == "Parallel"
            
            # Execute workflow
            current_workflow = await coordinator.execute_workflow(tasks, parallel=parallel)
            
            # Create visualizations
            active_agents = list(coordinator.agents.keys())
            
            workflow_graph = visualizer.create_workflow_graph(
                current_workflow['workflow_graph'],
                active_agents=active_agents
            )
            
            timeline_chart = visualizer.create_task_timeline(tasks)
            
            confidence_heatmap = visualizer.create_confidence_heatmap(
                current_workflow['agent_contributions']
            )
            
            performance_chart = visualizer.create_performance_comparison(
                current_workflow['performance_metrics']
            )
            
            # Generate status summary
            success_rate = current_workflow['success_rate']
            execution_time = current_workflow['execution_time']
            performance = current_workflow['performance_metrics']
            
            status = f"""Analysis completed successfully!

Results Summary:
- Tasks executed: {len(tasks)}
- Success rate: {success_rate:.0%}
- Execution time: {execution_time:.1f} seconds
- Performance improvement: {performance.get('time_improvement_percentage', 0):.1f}% faster
- Agents involved: {len(coordinator.agents)}

Agent Activity:
- Total collaborations: {performance.get('total_collaborations', 0)}
- Messages exchanged: {performance.get('total_messages', 0)}
- Efficiency score: {performance.get('efficiency_score', 0):.2%}"""
            
            return status, workflow_graph, timeline_chart, confidence_heatmap, performance_chart
            
        except Exception as e:
            logger.error(f"Error analyzing problem: {str(e)}")
            return f"Error during analysis: {str(e)}", None, None, None, None
    
    def generate_report(selected_sections: List[str]) -> Tuple[Optional[str], str]:
        """Generate a report from the current workflow results."""
        
        if not current_workflow:
            return None, "No analysis results available. Please run an analysis first."
        
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multi_agent_report_{timestamp}.pdf"
            
            report_path = report_generator.generate_report(
                current_workflow,
                current_problem,
                include_sections=selected_sections,
                filename=filename
            )
            
            if report_path:
                return report_path, f"Report generated successfully: {filename}"
            else:
                return None, "Error generating report."
                
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return None, f"Error generating report: {str(e)}"
    
    def get_agent_details(agent_name: str) -> str:
        """Get detailed information about a specific agent."""
        
        if not coordinator or agent_name not in coordinator.agents:
            return "Agent not found or system not initialized."
        
        agent = coordinator.agents[agent_name]
        status = agent.get_status_summary()
        
        details = f"""## Agent Profile: {agent.name}

**Role:** {agent.role.value}  
**Status:** {'Active' if agent.active else 'Inactive'}  
**Completed Tasks:** {len(agent.completed_tasks)}  
**Current Task:** {agent.current_task.description if agent.current_task else 'None'}  
**Average Confidence:** {status['average_confidence']:.0%}  
**Collaborations:** {status['collaboration_count']}  

### Recent Task History"""
        
        for i, task in enumerate(agent.completed_tasks[-5:], 1):
            status_icon = "✓" if task.status == TaskStatus.COMPLETED else "✗"
            exec_time = task.performance_metrics.get('execution_time', 0)
            
            details += f"""

**{i}. {status_icon} {task.id}**
- Description: {task.description}
- Confidence: {task.confidence:.0%}
- Execution Time: {exec_time:.1f}s"""
        
        # Add performance insights
        if agent.performance_tracker.metrics['task_completion_times']:
            avg_time = np.mean(agent.performance_tracker.metrics['task_completion_times'])
            details += f"\n\n### Performance Statistics\n"
            details += f"• Average Task Time: {avg_time:.1f}s\n"
            details += f"• Total Active Time: {sum(agent.performance_tracker.metrics['task_completion_times']):.1f}s"
        
        return details
    
    def get_workflow_insights() -> str:
        """Get insights about the multi-agent system performance."""
        
        if not coordinator:
            return "System not initialized."
        
        insights = coordinator.get_workflow_insights()
        
        if insights.get('total_workflows_executed', 0) == 0:
            return "No workflow executions yet. Run an analysis to see performance insights."
        
        content = f"""## Workflow Insights

### System Performance Overview

- **Total Workflows:** {insights['total_workflows_executed']}
- **Average Execution Time:** {insights['average_execution_time']:.1f}s
- **Average Success Rate:** {insights['average_success_rate']:.0%}
- **Most Efficient Agent:** {insights.get('most_efficient_agent', 'N/A')}
- **Highest Quality Agent:** {insights.get('highest_quality_agent', 'N/A')}

### Agent Efficiency Rankings"""
        
        if insights.get('agent_efficiency'):
            content += "\n\n| Agent | Avg Time/Task | Avg Confidence | Total Tasks |\n"
            content += "|-------|---------------|----------------|-------------|\n"
            
            for agent, efficiency in insights['agent_efficiency'].items():
                content += f"| {agent} | {efficiency['avg_time_per_task']:.1f}s | "
                content += f"{efficiency['avg_confidence']:.0%} | {efficiency['total_tasks']} |\n"
        
        return content
    
    # Create Gradio interface with professional styling
    with gr.Blocks(
        title="Multi-Agent AI Collaboration System", 
        theme=gr.themes.Base(),
        css="""
        .gradio-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        h1, h2, h3 {
            color: #2c3e50;
        }
        
        .gr-button-primary {
            background-color: #3498db !important;
            border-color: #3498db !important;
        }
        
        .gr-button-primary:hover {
            background-color: #2980b9 !important;
            border-color: #2980b9 !important;
        }
        
        .status-box {
            background-color: #f3f4f6;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            border-left: 4px solid #3b82f6;
        }
        """
    ) as interface:
        
        gr.Markdown("""
        # Multi-Agent AI Collaboration System
        
        Advanced AI system with specialized agents working together to solve complex problems through intelligent task decomposition and parallel processing.
        """)
        
        # System Configuration Section
        with gr.Group():
            gr.Markdown("### System Configuration")
            
            with gr.Row():
                with gr.Column(scale=3):
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="sk-...",
                        type="password",
                        info="Required for live mode. Leave empty for demo mode."
                    )
                with gr.Column(scale=1):
                    model_select = gr.Dropdown(
                        choices=["gpt-4", "gpt-3.5-turbo"],
                        value="gpt-4",
                        label="Model",
                        info="Select the LLM model"
                    )
                with gr.Column(scale=1):
                    demo_mode_checkbox = gr.Checkbox(
                        label="Demo Mode",
                        value=False,
                        info="Run without API key"
                    )
                with gr.Column(scale=1):
                    init_button = gr.Button(
                        "Initialize Agents",
                        variant="primary",
                        size="lg"
                    )
            
            init_status = gr.Textbox(
                label="Initialization Status",
                interactive=False
            )
        
        # Main tabs
        with gr.Tabs() as tabs:
            # Problem Analysis Tab
            with gr.TabItem("Problem Analysis", id=1):
                with gr.Group():
                    gr.Markdown("### Enter a complex problem for multi-agent analysis")
                    
                    problem_input = gr.Textbox(
                        label="Problem Statement",
                        placeholder="Example: Analyze the potential impact of AI on healthcare delivery in the next 5 years",
                        lines=3,
                        info="Describe a complex problem that requires multiple perspectives"
                    )
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            execution_mode = gr.Radio(
                                choices=["Sequential", "Parallel"],
                                value="Parallel",
                                label="Execution Mode",
                                info="Parallel mode is faster but requires more resources"
                            )
                        with gr.Column(scale=1):
                            use_template = gr.Checkbox(
                                label="Use Workflow Template",
                                value=True,
                                info="Automatically match to predefined workflows"
                            )
                        with gr.Column(scale=2):
                            analyze_button = gr.Button(
                                "Analyze Problem",
                                variant="primary",
                                size="lg"
                            )
                    
                    analysis_status = gr.Textbox(
                        label="Analysis Status",
                        interactive=False,
                        lines=10
                    )
                
                # Visualization outputs
                with gr.Group():
                    gr.Markdown("### Analysis Visualizations")
                    
                    with gr.Row():
                        workflow_graph = gr.Plot(label="Agent Collaboration Network")
                    
                    with gr.Row():
                        with gr.Column():
                            timeline_chart = gr.Plot(label="Task Execution Timeline")
                        with gr.Column():
                            confidence_heatmap = gr.Plot(label="Agent Performance Metrics")
                    
                    with gr.Row():
                        performance_chart = gr.Plot(label="Performance Comparison")
            
            # Agent Details Tab
            with gr.TabItem("Agent Details", id=2):
                with gr.Group():
                    gr.Markdown("### View detailed information about each agent")
                    
                    with gr.Row():
                        agent_selector = gr.Dropdown(
                            choices=["Researcher-1", "Analyst-1", "Critic-1", "Synthesizer-1"],
                            label="Select Agent",
                            info="Choose an agent to view their profile and performance"
                        )
                        agent_details_button = gr.Button(
                            "Get Agent Details",
                            variant="secondary"
                        )
                    
                    agent_details_output = gr.Markdown()
                
                with gr.Group():
                    gr.Markdown("### System Insights")
                    
                    insights_button = gr.Button(
                        "Get Workflow Insights",
                        variant="secondary"
                    )
                    
                    insights_output = gr.Markdown()
            
            # Report Generation Tab
            with gr.TabItem("Report Generation", id=3):
                with gr.Group():
                    gr.Markdown("### Generate comprehensive analysis report")
                    
                    section_selector = gr.CheckboxGroup(
                        choices=[
                            "executive_summary",
                            "task_analysis",
                            "agent_contributions",
                            "key_findings",
                            "recommendations",
                            "confidence_analysis",
                            "performance_metrics"
                        ],
                        value=[
                            "executive_summary",
                            "key_findings",
                            "recommendations",
                            "confidence_analysis"
                        ],
                        label="Select Report Sections",
                        info="Choose which sections to include in the report"
                    )
                    
                    generate_report_button = gr.Button(
                        "Generate PDF Report",
                        variant="primary",
                        size="lg"
                    )
                    
                    with gr.Row():
                        report_download = gr.File(label="Download Report")
                        report_status = gr.Textbox(label="Report Status", interactive=False)
            
            # Example Problems Tab
            with gr.TabItem("Example Problems", id=4):
                gr.Markdown("""
                ### Example Problems for Analysis
                
                Click on any example to load it into the analysis tab. These examples demonstrate different types of complex problems suitable for multi-agent analysis.
                """)
                
                example_problems = [
                    {
                        "title": "Business Strategy",
                        "problem": "Develop a comprehensive strategy for a traditional retail company to transition to e-commerce while maintaining customer loyalty and managing existing physical stores",
                        "description": "Complex business transformation requiring market analysis, risk assessment, and strategic planning"
                    },
                    {
                        "title": "Technology Assessment", 
                        "problem": "Evaluate the potential risks and benefits of implementing blockchain technology in supply chain management for a global manufacturing company",
                        "description": "Technical evaluation requiring understanding of emerging technology and business operations"
                    },
                    {
                        "title": "Market Analysis",
                        "problem": "Analyze the competitive landscape for electric vehicles and identify key success factors for new entrants in the North American market",
                        "description": "Market research requiring industry analysis, competitor assessment, and trend identification"
                    },
                    {
                        "title": "Policy Evaluation",
                        "problem": "Assess the implications of remote work policies on organizational culture, productivity, and talent retention in technology companies",
                        "description": "Organizational analysis requiring understanding of human resources, culture, and productivity metrics"
                    },
                    {
                        "title": "Innovation Planning",
                        "problem": "Design an innovation framework for a healthcare organization to integrate AI-powered diagnostic tools while ensuring patient privacy and regulatory compliance",
                        "description": "Innovation strategy requiring technical, regulatory, and ethical considerations"
                    }
                ]
                
                example_buttons = []
                for i, example in enumerate(example_problems):
                    with gr.Group():
                        gr.Markdown(f"""
                        #### {example['title']}
                        *{example['description']}*
                        """)
                        btn = gr.Button(
                            f"Load This Example",
                            variant="secondary",
                            size="sm"
                        )
                        example_buttons.append((btn, example['problem']))
            
            # Help Tab
            with gr.TabItem("Help", id=5):
                gr.Markdown("""
                ## How to Use the Multi-Agent AI Collaboration System
                
                ### Getting Started
                
                1. **Initialize the System**
                   - Enter your OpenAI API key for live analysis (optional)
                   - Or enable Demo Mode to explore without an API key
                   - Select your preferred model (GPT-4 recommended)
                   - Click "Initialize Agents"
                
                2. **Analyze a Problem**
                   - Enter a complex problem in the Problem Analysis tab
                   - Choose execution mode (Parallel is faster)
                   - Optionally use workflow templates for common problem types
                   - Click "Analyze Problem"
                
                3. **Review Results**
                   - View the agent collaboration network
                   - Check the task execution timeline
                   - Review performance metrics
                   - Explore individual agent details
                
                4. **Generate Report**
                   - Select desired report sections
                   - Click "Generate PDF Report"
                   - Download the comprehensive analysis
                
                ### Understanding the Agents
                
                - **Researcher**: Gathers information and identifies key facts
                - **Analyst**: Processes data and identifies patterns
                - **Critic**: Evaluates quality and identifies gaps
                - **Synthesizer**: Combines insights into actionable recommendations
                - **Coordinator**: Manages workflow and facilitates collaboration
                
                ### Tips for Best Results
                
                - Be specific and detailed in your problem statements
                - Complex, multi-faceted problems work best
                - Use parallel execution for faster results
                - Review agent details to understand the analysis process
                - Generate reports for comprehensive documentation
                
                ### Troubleshooting
                
                - **Initialization fails**: Check your API key or enable Demo Mode
                - **Analysis takes too long**: Try Sequential mode or simpler problems
                - **Empty visualizations**: Ensure analysis completed successfully
                - **Report generation fails**: Check that analysis was completed first
                """)
        
        # Event handlers
        init_button.click(
            fn=initialize_agents,
            inputs=[api_key_input, model_select, demo_mode_checkbox],
            outputs=init_status
        )
        
        analyze_button.click(
            fn=lambda p, m, t: asyncio.run(analyze_problem(p, m, t)),
            inputs=[problem_input, execution_mode, use_template],
            outputs=[analysis_status, workflow_graph, timeline_chart, confidence_heatmap, performance_chart]
        )
        
        agent_details_button.click(
            fn=get_agent_details,
            inputs=agent_selector,
            outputs=agent_details_output
        )
        
        insights_button.click(
            fn=get_workflow_insights,
            inputs=[],
            outputs=insights_output
        )
        
        generate_report_button.click(
            fn=generate_report,
            inputs=section_selector,
            outputs=[report_download, report_status]
        )
        
        # Example button handlers
        for btn, problem in example_buttons:
            btn.click(
                fn=lambda p=problem: p,
                outputs=problem_input
            ).then(
                fn=lambda: gr.Tabs.update(selected=1),
                outputs=tabs
            )
    
    return interface

# Main execution
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        favicon_path=None,
        show_error=True
    )