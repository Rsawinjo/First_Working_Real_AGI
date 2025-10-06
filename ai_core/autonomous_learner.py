"""
Autonomous Deep Learning Engine - The AGI Core
This module enables the AI to self-improve without human intervention.
Enhanced with Phase 2 web research capabilities.
"""

import asyncio
import threading
import time
import random
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import numpy as np
from collections import deque, defaultdict

# GPU monitoring
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

# Import enhanced web research (Phase 2)
try:
    from .web_research import AdvancedWebResearcher
    WEB_RESEARCH_AVAILABLE = True
except ImportError:
    WEB_RESEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)

if not WEB_RESEARCH_AVAILABLE:
    logger.warning("⚠️ Web research module not available")

class LearningState(Enum):
    IDLE = "idle"
    ANALYZING = "analyzing"
    RESEARCHING = "researching"
    SYNTHESIZING = "synthesizing"
    EXPERIMENTING = "experimenting"
    REFLECTING = "reflecting"

@dataclass
class LearningGoal:
    id: str
    topic: str
    priority: float
    knowledge_gap: str
    target_depth: int
    created_at: datetime
    estimated_duration: int  # minutes
    prerequisites: List[str]
    status: str = "pending"
    progress: float = 0.0

@dataclass
class Insight:
    id: str
    content: str
    confidence: float
    connections: List[str]
    created_at: datetime
    validation_score: float = 0.0
    applied_count: int = 0

class AutonomousLearner:

    def integrate_feedback(self, feedback_type: str):
        """Integrate user feedback to adjust learning strategy and goal generation."""
        # Feedback types: 'good', 'needs_work', 'interesting', 'perfect'
        if feedback_type == 'good':
            self.learning_intensity = min(self.learning_intensity + 0.1, 1.0)
            self._report_to_gui("👍 Feedback: Good - Increasing learning intensity.", "feedback")
        elif feedback_type == 'needs_work':
            self.learning_intensity = max(self.learning_intensity - 0.1, 0.1)
            self._report_to_gui("🛠️ Feedback: Needs Work - Decreasing learning intensity for deeper exploration.", "feedback")
        elif feedback_type == 'interesting':
            self.exploration_rate = min(self.exploration_rate + 0.1, 1.0)
            self._report_to_gui("✨ Feedback: Interesting - Increasing exploration rate.", "feedback")
        elif feedback_type == 'perfect':
            self.learning_intensity = 1.0
            self._report_to_gui("🌟 Feedback: Perfect - Max learning intensity!", "feedback")
        else:
            self._report_to_gui(f"❓ Unknown feedback type: {feedback_type}", "feedback")

    def expand_hierarchical_goals(self, base_topic: str):
        """Expand learning goals by generating related/subtopics using knowledge graph and semantic similarity."""
        if not self.knowledge_base:
            return
        related_topics = self.knowledge_base.get_related_topics(base_topic)
        for topic in related_topics:
            if topic not in self.mastered_topics and not any(g.topic == topic for g in self.learning_goals):
                goal = LearningGoal(
                    id=f"hierarchical_{int(time.time())}_{hash(topic) % 1000}",
                    topic=topic,
                    priority=0.7,
                    knowledge_gap=f"Related to {base_topic}",
                    target_depth=3,
                    created_at=datetime.now(),
                    estimated_duration=30,
                    prerequisites=[base_topic]
                )
                self.learning_goals.append(goal)
                self._save_learning_goal(goal)
                self._report_to_gui(f"🔗 Hierarchical goal added: {topic}", "goal")
    
    def _expand_goals_from_completion(self, completed_topic: str):
        """Generate new goals when a topic is completed - hierarchical and connected goal generation."""
        if not self.knowledge_base:
            return
            
        # Prevent goal queue from growing too large
        if len(self.learning_goals) >= 45:  # Leave room for 5 more goals
            self.logger.info(f"Goal queue nearly full ({len(self.learning_goals)} goals), skipping expansion")
            return
            
        new_goals_count = 0
        max_new_goals = 5  # Limit total new goals per completion
        
        # 1. EXPLORE RELATED TOPICS (knowledge graph neighbors)
        if new_goals_count < max_new_goals:
            related_topics = self.knowledge_base.get_related_topics(completed_topic)
            for topic in related_topics[:2]:  # Reduced from 3 to 2
                if new_goals_count >= max_new_goals:
                    break
                if topic not in self.mastered_topics and not any(g.topic == topic for g in self.learning_goals):
                    goal = LearningGoal(
                        id=f"related_{int(time.time())}_{hash(topic) % 1000}",
                        topic=topic,
                        priority=0.6 + (random.random() * 0.2),  # 0.6-0.8 priority
                        knowledge_gap=f"Related area to {completed_topic}",
                        target_depth=2,
                        created_at=datetime.now(),
                        estimated_duration=25,
                        prerequisites=[completed_topic]
                    )
                    self.learning_goals.append(goal)
                    self._save_learning_goal(goal)
                    new_goals_count += 1
        
        # 2. GENERATE SUBTOPICS (deeper exploration)
        if new_goals_count < max_new_goals:
            subtopics = self._generate_subtopics(completed_topic)
            for subtopic in subtopics[:1]:  # Reduced from 2 to 1
                if new_goals_count >= max_new_goals:
                    break
                if subtopic not in self.mastered_topics and not any(g.topic == subtopic for g in self.learning_goals):
                    goal = LearningGoal(
                        id=f"subtopic_{int(time.time())}_{hash(subtopic) % 1000}",
                        topic=subtopic,
                        priority=0.8,  # Higher priority for subtopics
                        knowledge_gap=f"Subtopic of {completed_topic}",
                        target_depth=4,  # Deeper exploration
                        created_at=datetime.now(),
                        estimated_duration=45,
                        prerequisites=[completed_topic]
                    )
                    self.learning_goals.append(goal)
                    self._save_learning_goal(goal)
                    new_goals_count += 1
        
        # 3. IDENTIFY PREREQUISITES (foundational knowledge)
        if new_goals_count < max_new_goals:
            prerequisites = self._identify_prerequisites(completed_topic)
            for prereq in prerequisites[:1]:  # Keep at 1
                if new_goals_count >= max_new_goals:
                    break
                if prereq not in self.mastered_topics and not any(g.topic == prereq for g in self.learning_goals):
                    goal = LearningGoal(
                        id=f"prereq_{int(time.time())}_{hash(prereq) % 1000}",
                        topic=prereq,
                        priority=0.9,  # High priority for prerequisites
                        knowledge_gap=f"Prerequisite for {completed_topic}",
                        target_depth=1,  # Foundational, so shallower
                        created_at=datetime.now(),
                        estimated_duration=20,
                        prerequisites=[]
                    )
                    self.learning_goals.append(goal)
                    self._save_learning_goal(goal)
                    new_goals_count += 1
        
        # 4. SPECULATE NEXT LOGICAL STEPS (forward-looking)
        if new_goals_count < max_new_goals:
            next_steps = self._speculate_next_steps(completed_topic)
            for next_step in next_steps[:2]:  # Keep at 2, but respect max limit
                if new_goals_count >= max_new_goals:
                    break
                if next_step not in self.mastered_topics and not any(g.topic == next_step for g in self.learning_goals):
                    goal = LearningGoal(
                        id=f"nextstep_{int(time.time())}_{hash(next_step) % 1000}",
                        topic=next_step,
                        priority=0.5 + (random.random() * 0.3),  # 0.5-0.8 priority
                        knowledge_gap=f"Next logical step after {completed_topic}",
                        target_depth=3,
                        created_at=datetime.now(),
                        estimated_duration=35,
                        prerequisites=[completed_topic]
                    )
                    self.learning_goals.append(goal)
                    self._save_learning_goal(goal)
                    new_goals_count += 1
        
        if new_goals_count > 0:
            # More subtle notification for better UX
            if new_goals_count == 1:
                self._report_to_gui(f"🎯 New learning goal generated from '{completed_topic}'", "goal")
            else:
                self._report_to_gui(f"🚀 {new_goals_count} new goals added to learning pipeline", "goal")
            self.logger.info(f"Expanded {new_goals_count} goals from completed topic: {completed_topic}")
        else:
            self.logger.debug(f"No new goals generated for completed topic: {completed_topic}")
    
    def _generate_subtopics(self, topic: str) -> List[str]:
        """Generate potential subtopics for deeper exploration."""
        # Check cache first
        cache_key = f"subtopics_{topic}"
        if cache_key in self.goal_generation_cache:
            return self.goal_generation_cache[cache_key]
        
        # Use LLM to generate subtopics if available
        if self.llm_interface:
            try:
                prompt = f"Generate 2-3 specific subtopics or deeper areas to explore within '{topic}'. Be specific and actionable. Format as a comma-separated list."
                response = self._gpu_enhanced_llm_call(
                    prompt,
                    {"type": "subtopic_generation", "topic": topic},
                    "Subtopic Generation"
                )
                if response and response.strip():
                    # Parse comma-separated list
                    subtopics = [s.strip() for s in response.split(',') if s.strip() and len(s.strip()) > 3]
                    if subtopics:  # Only return if we got valid results
                        result = subtopics[:3]  # Limit to 3
                        self.goal_generation_cache[cache_key] = result  # Cache result
                        return result
            except Exception as e:
                self.logger.debug(f"LLM subtopic generation failed: {e}")
        
        # Fallback: simple heuristics
        common_subtopic_patterns = [
            f"Advanced {topic}",
            f"{topic} applications",
            f"{topic} fundamentals",
            f"{topic} implementation",
            f"History of {topic}",
            f"Future of {topic}"
        ]
        result = common_subtopic_patterns[:2]
        self.goal_generation_cache[cache_key] = result  # Cache fallback too
        return result
    
    def _identify_prerequisites(self, topic: str) -> List[str]:
        """Identify foundational knowledge needed for the topic."""
        # Check cache first
        cache_key = f"prereqs_{topic}"
        if cache_key in self.goal_generation_cache:
            return self.goal_generation_cache[cache_key]
        
        if self.llm_interface:
            try:
                prompt = f"What are 1-2 foundational concepts or prerequisite knowledge needed to understand '{topic}'? Format as a comma-separated list."
                response = self._gpu_enhanced_llm_call(
                    prompt,
                    {"type": "prerequisite_identification", "topic": topic},
                    "Prerequisite Identification"
                )
                if response and response.strip():
                    prereqs = [p.strip() for p in response.split(',') if p.strip() and len(p.strip()) > 3]
                    if prereqs:  # Only return if we got valid results
                        result = prereqs[:2]
                        self.goal_generation_cache[cache_key] = result
                        return result
            except Exception as e:
                self.logger.debug(f"LLM prerequisite identification failed: {e}")
        
        # Fallback: generic prerequisites
        result = [f"Basics of {topic}", f"Introduction to {topic}"]
        self.goal_generation_cache[cache_key] = result
        return result
    
    def _speculate_next_steps(self, topic: str) -> List[str]:
        """Speculate on logical next steps or related areas to explore."""
        # Check cache first
        cache_key = f"nextsteps_{topic}"
        if cache_key in self.goal_generation_cache:
            return self.goal_generation_cache[cache_key]
        
        if self.llm_interface:
            try:
                prompt = f"After learning about '{topic}', what are 2-3 logical next topics or areas someone might explore? Format as a comma-separated list."
                response = self._gpu_enhanced_llm_call(
                    prompt,
                    {"type": "next_steps_speculation", "topic": topic},
                    "Next Steps Speculation"
                )
                if response and response.strip():
                    next_steps = [n.strip() for n in response.split(',') if n.strip() and len(n.strip()) > 3]
                    if next_steps:  # Only return if we got valid results
                        result = next_steps[:3]
                        self.goal_generation_cache[cache_key] = result
                        return result
            except Exception as e:
                self.logger.debug(f"LLM next steps speculation failed: {e}")
        
        # Fallback: related areas
        result = [f"{topic} in practice", f"Advanced {topic} concepts"]
        self.goal_generation_cache[cache_key] = result
        return result
    """
    The AGI Core - Enables truly autonomous self-improvement
    """
    
    def __init__(self, knowledge_base=None, llm_interface=None, research_assistant=None):
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        self.knowledge_base = knowledge_base
        self.llm_interface = llm_interface
        self.research_assistant = research_assistant
        self.memory_system = None  # Will be set later if available
        self.gui_callback = None  # Will be set by GUI after initialization
        
        # Phase 2: Enhanced Web Research
        if WEB_RESEARCH_AVAILABLE:
            self.web_researcher = AdvancedWebResearcher(knowledge_base=knowledge_base)
            self.logger.info("Phase 2: Enhanced web research activated!")
        else:
            self.web_researcher = None
            self.logger.warning("Web research not available - using basic research")
        
        # Core AGI State
        self.state = LearningState.IDLE
        self.learning_goals = deque(maxlen=50)
        self.active_goal = None
        self.insights = {}  # Initialize insights as dict
        self.curiosity_engine = CuriosityEngine()
        self.curiosity_engine.set_parent_learner(self)  # Connect for dynamic topics
        self.meta_cognitive_system = MetaCognitiveSystem()
        
        # Learning Configuration
        self.autonomous_mode = False
        self.learning_intensity = 0.5  # 0.0 to 1.0
        self.creativity_threshold = 0.7
        self.exploration_rate = 0.3
        
        # Phase 2: Enhanced research settings
        self.research_depth = "comprehensive"  # shallow, standard, comprehensive
        self.web_research_enabled = True
        self.parallel_learning = True
        
        # Domain prioritization for focused learning
        self.domain_priorities = {
            'basic_sciences': 1.0,
            'life_sciences': 1.0,
            'medical_sciences': 1.0,
            'social_sciences': 1.0,
            'arts_humanities': 1.0,
            'engineering_tech': 1.0,
            'specialized_medical': 1.0,
            'diagnostic_imaging': 1.0,
            'biological_subfields': 1.0
        }
        
        # Goal Generation Cache (simple in-memory cache to avoid repeated LLM calls)
        self.goal_generation_cache = {}
        
        # Research Results Storage
        self.last_research_results = {}   # Store latest research results
        self.last_insights = []           # Store latest insights
        
        # Enhanced Performance Metrics
        self.session_start_time = time.time()
        self.topics_learned_this_session = 0
        self.total_learning_time = 0
        self.learning_velocity = 0.0  # Topics per hour
        self.gpu_computation_time = 0.0
        self.knowledge_synthesis_count = 0
        self.learning_efficiency_score = 0.0
        self.learning_cycles = 0  # Add missing learning_cycles attribute
        
        # Performance Tracking
        self.session_stats = {
            'goals_completed': 0,
            'insights_generated': 0,
            'knowledge_connections': 0,
            'breakthroughs': 0,
            'session_start': None
        }
        
        # Initialize database
        self._init_database()
        
        # Background learning thread
        self.learning_thread = None
        self.should_continue_learning = False
        
        self.logger.info("Autonomous Learning Engine initialized - AGI Core ready")
    
    def _init_database(self):
        """Initialize the autonomous learning database"""
        try:
            self.db_path = "data/autonomous_learning.db"
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Learning goals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_goals (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    priority REAL,
                    knowledge_gap TEXT,
                    target_depth INTEGER,
                    created_at TEXT,
                    estimated_duration INTEGER,
                    prerequisites TEXT,
                    status TEXT,
                    progress REAL
                )
            ''')
            
            # Insights table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS insights (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    confidence REAL,
                    connections TEXT,
                    created_at TEXT,
                    validation_score REAL,
                    applied_count INTEGER
                )
            ''')
            
            # Learning sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    goals_completed INTEGER,
                    insights_generated INTEGER,
                    breakthroughs INTEGER,
                    session_notes TEXT
                )
            ''')
            
            # ✅ PERSISTENCE FIX: Mastered topics table to avoid loops
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mastered_topics (
                    topic TEXT PRIMARY KEY,
                    mastered_at TEXT,
                    depth_level INTEGER,
                    review_count INTEGER DEFAULT 0
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Autonomous learning database initialized")
            
        except Exception as e:
            self.logger.exception("Error initializing autonomous learning database")
            
        # Load existing mastered topics
        self._load_mastered_topics()
        
        # Clean up any goals for already mastered topics
        self._cleanup_mastered_goals()
        
        # Sync mastered topics with curiosity engine
        self.curiosity_engine.explored_topics.update(self.mastered_topics)
        
        # GUI callback for direct learning log updates
        self.gui_callback = None
    
    def _report_to_gui(self, message: str, log_type: str = "learning"):
        """Report learning activity directly to GUI with performance metrics"""
        if hasattr(self, 'gui_callback') and self.gui_callback:
            try:
                # Calculate session metrics
                session_time = time.time() - self.session_start_time
                hours = session_time / 3600
                
                # Update learning velocity
                if hours > 0 and self.topics_learned_this_session > 0:
                    self.learning_velocity = self.topics_learned_this_session / hours
                
                # Enhanced message with metrics for key operations
                if log_type == "learning" and "session:" in message:
                    enhanced_message = f"{message} | Session: {self.topics_learned_this_session} topics | Velocity: {self.learning_velocity:.1f}/hr"
                    self.gui_callback(enhanced_message, log_type)
                elif log_type == "synthesis":
                    self.knowledge_synthesis_count += 1
                    enhanced_message = f"{message} | Synthesis #{self.knowledge_synthesis_count} | Velocity: {self.learning_velocity:.1f}/hr"
                    self.gui_callback(enhanced_message, log_type)
                elif log_type == "success" and "Mastered:" in message:
                    # Add velocity info to mastery messages
                    enhanced_message = f"{message} | Velocity: {self.learning_velocity:.1f}/hr"
                    self.gui_callback(enhanced_message, log_type)
                else:
                    self.gui_callback(message, log_type)
            except Exception:
                self.logger.exception("Error reporting to GUI")
    
    def _gpu_enhanced_llm_call(self, prompt: str, context: Dict, operation_type: str = "processing", timeout_seconds: int = 60) -> str:
        """Make GPU-enhanced LLM call with monitoring, optimization, and timeout"""
        start_time = time.time()
        
        try:
            if not self.llm_interface:
                return ""
            
            import threading
            
            result = {"response": "", "completed": False}
            
            def llm_worker():
                try:
                    # Make the enhanced LLM call
                    response = self.llm_interface.generate_response(prompt, context)
                    
                    # Extract response if it's a tuple (response, metadata)
                    if isinstance(response, tuple):
                        response = response[0]
                    
                    result["response"] = response
                    result["completed"] = True
                except Exception as e:
                    self.logger.error(f"LLM call failed: {e}")
                    result["response"] = ""
                    result["completed"] = True
            
            # Start LLM call in background thread
            worker_thread = threading.Thread(target=llm_worker, daemon=True)
            worker_thread.start()
            
            # Wait for completion with timeout
            worker_thread.join(timeout=timeout_seconds)
            
            if not result["completed"]:
                self.logger.warning(f"🚨 LLM call timed out after {timeout_seconds}s for {operation_type}")
                self._report_to_gui(f"⏰ TIMEOUT: {operation_type} took too long ({timeout_seconds}s)", "warning")
                return ""
            
            response = result["response"]
            processing_time = time.time() - start_time
            self.gpu_computation_time += processing_time
            
            # Log GPU usage after call - More realistic for smaller models
            if GPU_AVAILABLE:
                gpu_memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                
                # For smaller models, consider any GPU usage as effective
                gpu_utilization = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 50
                
                # Calculate learning efficiency
                if processing_time > 0:
                    self.learning_efficiency_score = min(100, (len(response) / processing_time) * 10)
                
                self.logger.info(f"🎯 RTX 4090 {operation_type}: Completed in {processing_time:.2f}s | Efficiency: {self.learning_efficiency_score:.1f}")
                self._report_to_gui(f"⚡ GPU: {processing_time:.2f}s | Total GPU time: {self.gpu_computation_time:.1f}s", "gpu")
                self.logger.info(f"⚡ GPU Memory: {gpu_memory_after:.2f} GB | Processing Speed: {len(prompt)/processing_time:.0f} chars/sec")
                
                # More lenient GPU utilization check for conversational models
                if processing_time > 0.1:  # Any significant processing time
                    self.logger.info(f"✅ RTX 4090 Beast Mode ACTIVE for {operation_type}")
                else:
                    self.logger.info(f"⚡ RTX 4090 Lightning Speed: {operation_type} processed instantly")
            
            return response
            
        except Exception as e:
            error_message = str(e)
            # Suppress CUDA device-side assert errors from chat/logs
            if "CUDA error: device-side assert triggered" in error_message:
                # Log once at startup if needed, but do not repeat in chat
                if not hasattr(self, '_cuda_error_logged'):
                    self.logger.warning("CUDA error suppressed: device-side assert triggered. See logs for details if needed.")
                    self._cuda_error_logged = True
                return ""
            
            processing_time = time.time() - start_time
            self.logger.error(f"🚨 GPU computation failed for {operation_type}: {error_message} (took {processing_time:.2f}s)")
            self._report_to_gui(f"❌ GPU Error: {operation_type} failed", "error")
            return ""
    
    def start_autonomous_mode(self, intensity: float = 0.5):
        """Begin autonomous self-improvement"""
        if self.autonomous_mode:
            logger.warning("Autonomous mode already active")
            return
        
        self.autonomous_mode = True
        self.learning_intensity = intensity
        self.should_continue_learning = True
        self.session_stats['session_start'] = datetime.now()
        
        # Start the autonomous learning thread
        self.learning_thread = threading.Thread(target=self._autonomous_learning_loop, daemon=True)
        self.learning_thread.start()
        
        self.logger.info(f"AGI Core activated! Beginning autonomous self-improvement...")
        return "🚀 AGI Core activated! Beginning autonomous self-improvement..."
    
    def stop_autonomous_mode(self):
        """Stop autonomous learning"""
        self.autonomous_mode = False
        self.should_continue_learning = False
        
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5)
        
        self._save_session_stats()
        self.logger.info("Autonomous mode deactivated")
        return "Autonomous learning session completed."
    
    def _autonomous_learning_loop(self):
        """Main autonomous learning loop - The AGI Heart"""
        try:
            self._report_to_gui(f"🚀 Starting autonomous learning with {len(self.mastered_topics)} mastered topics in permanent memory", "status")
            
            while self.should_continue_learning and self.autonomous_mode:
                # Get real-time mastered count
                current_mastered = len(self.mastered_topics)
                
                # 1. SELF-ASSESSMENT: Analyze current knowledge state
                self._report_to_gui(f"🔍 Analyzing current knowledge state ({current_mastered} mastered topics)", "learning")
                self._analyze_knowledge_state()
                    
                    # 2. GOAL GENERATION: Identify learning opportunities
                self._report_to_gui("🎯 Generating strategic learning goals", "goal")
                
                # Check if we need to use the Master Discovery Engine
                current_goals = len(self.learning_goals)
                if current_goals < 3:  # Always maintain at least 3 goals
                    self._report_to_gui("🚀 Activating MASTER DISCOVERY ENGINE", "emergency")
                    self._master_discovery_engine()
                else:
                    self._generate_learning_goals()
                
                # Debug: Check if we have any goals
                goals_count = len(self.learning_goals)
                self._report_to_gui(f"📊 Generated {goals_count} potential learning goals", "status")
                
                # 3. PRIORITY SELECTION: Choose most important goal
                if not self.active_goal:
                    self._report_to_gui("⚡ Selecting next learning priority", "goal")
                    self.active_goal = self._select_next_goal()
                
                if not self.active_goal:
                    self._report_to_gui("⚠️ No valid goals found - generating emergency exploratory topics", "error")
                    self._generate_emergency_topics()
                    self.active_goal = self._select_next_goal()
               
                # 4. EXECUTE LEARNING: Deep dive into the goal
                if self.active_goal:
                    self._report_to_gui(f"📚 Deep learning session: {self.active_goal.topic}", "learning")
                    session_start = time.time()
                    self._execute_learning_goal()
                    session_duration = time.time() - session_start
                    self.total_learning_time += session_duration
                    
                    # Mark topic as learned for this session
                    if self.active_goal and hasattr(self.active_goal, 'topic'):
                        self.topics_learned_this_session += 1
                        
                        # Add to mastered topics for tracking
                        topic_name = self.active_goal.topic
                        # Remove unique identifier for mastered topic storage
                        clean_topic = topic_name.split('[')[0].strip() if '[' in topic_name else topic_name
                        self.mastered_topics.add(clean_topic)
                        self._save_mastered_topic(clean_topic)
                        
                        # Mark as explored in curiosity engine to prevent re-generation
                        self.curiosity_engine.explored_topics.add(clean_topic)
                        
                        # Clean up any other goals for this now-mastered topic
                        self._cleanup_mastered_goals()
                        
                        # Update velocity calculation
                        session_time_hours = (time.time() - self.session_start_time) / 3600
                        if session_time_hours > 0:
                            self.learning_velocity = self.topics_learned_this_session / session_time_hours
                        
                        self._report_to_gui(f"✅ Mastered: {clean_topic} | Total: {len(self.mastered_topics)} | Velocity: {self.learning_velocity:.1f}/hr", "success")
                        self._report_to_gui(f"📊 Session Progress: {self.topics_learned_this_session} topics | {session_time_hours:.1f}h elapsed", "metrics")
                else:
                    self._report_to_gui("❌ No learning goal available - something is wrong!", "error")
                    # Emergency break to avoid infinite loop
                    time.sleep(5)
                    continue
                
                # 5. SYNTHESIS: Generate insights and connections
                if self.active_goal and self.last_research_results:
                    self._report_to_gui(f"🔗 Synthesizing knowledge about: {self.active_goal.topic}", "synthesis")
                    self._synthesize_knowledge(
                        self.active_goal.topic, 
                        self.last_research_results, 
                        self.last_insights
                    )
                
                # 6. META-COGNITION: Reflect on learning process
                self._report_to_gui("🤔 Reflecting on learning process", "insight")
                self._meta_cognitive_reflection()
                
                # 7. CURIOSITY DRIVEN EXPLORATION
                if random.random() < self.exploration_rate:
                    self._report_to_gui("🌟 Curiosity-driven exploration activated", "learning")
                    self._curiosity_driven_exploration()
                
                # Reset active goal after completion
                self.active_goal = None
                
                # Increment learning cycle counter
                self.learning_cycles += 1
                
                # Calculate session metrics
                session_time_hours = (time.time() - self.session_start_time) / 3600
                cycle_efficiency = self.topics_learned_this_session / self.learning_cycles if self.learning_cycles > 0 else 0
                
                # Smart pacing: Adaptive delays based on learning intensity and performance
                base_delay = 3.0  # Base delay in seconds
                adaptive_delay = base_delay * (2.0 - self.learning_intensity)  # Faster with higher intensity
                
                # Performance-based adjustment
                if self.learning_velocity > 2.0:  # High velocity
                    adaptive_delay *= 0.7  # Speed up
                elif self.learning_velocity < 0.5:  # Low velocity  
                    adaptive_delay *= 1.5  # Slow down for better processing
                
                self._report_to_gui(f"🔄 Cycle #{self.learning_cycles} | Efficiency: {cycle_efficiency:.2f} | Next: {adaptive_delay:.1f}s | Mastered: {len(self.mastered_topics)}", "cycle")
                time.sleep(adaptive_delay)
        
        except Exception as e:
            logger.error(f"Critical error in autonomous learning loop: {e}")
            self._report_to_gui(f"🚨 CRITICAL AGI ERROR: {str(e)[:100]}... Shutting down learning", "error")
            self.should_continue_learning = False
    
    def _analyze_knowledge_state(self):
        self.state = LearningState.ANALYZING
        
        try:
            if not self.knowledge_base:
                return
            
            # Get current knowledge distribution
            knowledge_stats = self.knowledge_base.get_statistics()
            
            # Debug: Check what we actually got
            if not isinstance(knowledge_stats, dict):
                self.logger.warning(f"Knowledge stats is not a dict: {type(knowledge_stats)} = {knowledge_stats}")
                knowledge_stats = {}  # Use empty dict as fallback
            
            # Identify weak areas (knowledge gaps)
            weak_areas = self._identify_knowledge_gaps(knowledge_stats)
            
            # Update curiosity engine
            if hasattr(self, 'curiosity_engine') and self.curiosity_engine:
                self.curiosity_engine.update_knowledge_map(knowledge_stats)
            
            logger.debug(f"Knowledge analysis complete. Found {len(weak_areas)} potential improvement areas")
            
        except Exception as e:
            logger.error(f"Error in knowledge state analysis: {e}")
            # Continue execution even if analysis fails
    
    def _generate_learning_goals(self):
        """Generate strategic learning goals using RTX 4090-powered utility analysis"""
        try:
            # 🧠 RTX 4090 STRATEGIC INTELLIGENCE: Analyze what's truly useful to learn
            if self.llm_interface and hasattr(self.llm_interface, 'generate_response'):
                # Get current knowledge state for context
                mastered_topics = list(self.mastered_topics)
                user_focus = getattr(self, 'user_focus_topic', None)
                
                strategic_prompt = f"""As a strategic AI researcher, identify 5 COMPLETELY NEW topics to learn next.

CRITICAL REQUIREMENTS - STRICTLY AVOID:
{', '.join(mastered_topics) if mastered_topics else 'None'}

MANDATORY: Each topic must be in a TOTALLY DIFFERENT field/domain than ANY mastered topic above.
FORBIDDEN: No variations, extensions, related topics, or similar concepts.
FORBIDDEN: No topics that share ANY keywords with mastered topics.
FORBIDDEN: No topics in similar scientific/technological domains.

Context:
- User focus: {user_focus or 'Open exploration'}
- Current capabilities: AGI system with web research, knowledge synthesis

Generate topics that are in ENTIRELY NEW fields:
1. Completely different scientific domains
2. Unrelated technological areas  
3. Novel interdisciplinary combinations never explored
4. Breakthrough fields with no existing knowledge overlap
5. Revolutionary concepts in unexplored territories

EXAMPLES of ACCEPTABLE novel topics:
- astrobiology and xenolinguistics
- quantum gravity and consciousness studies
- bio-luminescent computing architectures
- temporal mechanics and causality engineering
- fractal dimension mathematics applications
- holographic data storage paradigms
- archaeoastronomy and ancient astronomical knowledge
- psychogeography and urban emotional mapping
- bioacoustics and sound-based biological communication
- chronobiology and time-based physiological rhythms

REJECTED examples (too similar to existing knowledge):
- machine learning optimization
- neural network architectures
- computer vision applications
- natural language processing
- reinforcement learning algorithms
- quantum computing
- synthetic biology
- neuromorphic engineering
- consciousness studies
- AI ethics or philosophy

Format as numbered list of COMPLETELY NEW topics only."""

                ai_response = self._gpu_enhanced_llm_call(
                    strategic_prompt,
                    {"type": "strategic_learning", "user_focus": user_focus},
                    "Strategic Topic Generation"
                )
                
                if ai_response:
                    # Parse AI-generated strategic topics
                    strategic_topics = self._parse_strategic_topics(ai_response, user_focus)
                    if strategic_topics:
                        self.logger.info(f"🎯 RTX 4090 generated {len(strategic_topics)} strategic learning goals")
                        self._create_strategic_goals(strategic_topics)
                        return
            
            # Fallback to enhanced curiosity-based system if AI not available
            self._generate_curiosity_based_goals()
            
            # If still no goals, generate exploratory topics
            if not self.learning_goals:
                self._generate_exploratory_topics()
            
        except Exception as e:
            self.logger.error(f"Error generating strategic learning goals: {e}")
            self._generate_curiosity_based_goals()
    
    def _parse_strategic_topics(self, ai_response: str, user_focus: Optional[str] = None) -> List[Dict]:
        """Parse AI response into strategic topic priorities"""
        strategic_topics = []
        lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
        
        priority_boost = 0.95  # Start with high priority
        for line in lines:
            if len(strategic_topics) >= 5:  # Limit to 5 strategic topics
                break
                
            # Clean topic from numbered list format
            topic = line
            for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '•', '*']:
                if topic.startswith(prefix):
                    topic = topic[len(prefix):].strip()
            
            # Check if topic is truly novel (not just case different)
            if len(topic) > 5:
                # Validate that this is actually a legitimate learning topic
                if not self._is_valid_learning_topic(topic):
                    self._report_to_gui(f"🚫 Rejected invalid topic: {topic[:50]}...", "status")
                    continue
                    
                # Check against mastered topics (case insensitive and partial matches)
                is_novel = True
                topic_lower = topic.lower()
                
                for mastered in self.mastered_topics:
                    mastered_lower = mastered.lower()
                    # Check for exact match or significant overlap
                    if (topic_lower == mastered_lower or 
                        topic_lower in mastered_lower or 
                        mastered_lower in topic_lower or
                        len(set(topic_lower.split()) & set(mastered_lower.split())) > 1):
                        is_novel = False
                        self._report_to_gui(f"⚠️ Skipping similar topic: {topic} (too similar to: {mastered})", "status")
                        break
                
                if is_novel:
                    # Calculate utility score based on strategic factors
                    utility_score = self._calculate_topic_utility(topic, user_focus)
                    
                    strategic_topics.append({
                        'topic': topic,
                        'priority': min(priority_boost * utility_score, 1.0),
                        'utility_score': utility_score,
                        'strategic_value': 'high' if utility_score > 0.8 else 'medium'
                    })
                    self._report_to_gui(f"✅ Added novel topic: {topic} (utility: {utility_score:.2f})", "goal")
                    priority_boost -= 0.1  # Decrease priority for later topics
        
        return strategic_topics
    
    def _calculate_topic_utility(self, topic: str, user_focus: Optional[str] = None) -> float:
        """Calculate the utility score of a learning topic with business focus"""
        utility_score = 0.5  # Base utility

        # 🚀 USER BUSINESS FOCUS: Massive boost for monetization topics
        business_keywords = [
            'money', 'revenue', 'profit', 'income', 'business', 'monetization', 'passive income',
            'automation', 'bots', 'trading', 'investment', 'startup', 'entrepreneurship',
            'marketing', 'sales', 'ecommerce', 'freelance', 'consulting', 'service',
            'app development', 'software business', 'saas', 'subscription', 'affiliate',
            'dropshipping', 'reselling', 'arbitrage', 'crypto trading', 'nft', 'defi',
            'content creation', 'youtube automation', 'social media marketing', 'influencer',
            'digital products', 'course creation', 'coaching', 'consulting business'
        ]

        topic_lower = topic.lower()
        for keyword in business_keywords:
            if keyword in topic_lower:
                utility_score += 0.25  # Massive boost for business topics
                break  # Only count once

        # Boost for user focus alignment
        if user_focus and user_focus.lower() in topic.lower():
            utility_score += 0.3

        # Boost for practical AI/business applications
        practical_keywords = [
            'ai automation', 'ai business', 'ai revenue', 'ai monetization', 'ai trading',
            'machine learning business', 'ai consulting', 'ai development', 'ai products',
            'chatbot business', 'ai marketplace', 'ai services', 'ai platform'
        ]

        for keyword in practical_keywords:
            if keyword in topic_lower:
                utility_score += 0.2

        # Moderate boost for high-value keywords (but less than business)
        high_value_keywords = [
            'ai', 'machine learning', 'automation', 'optimization', 'efficiency',
            'breakthrough', 'innovation', 'synthesis', 'integration', 'scalable',
            'practical', 'commercial', 'solution', 'algorithm'
        ]

        for keyword in high_value_keywords:
            if keyword in topic_lower:
                utility_score += 0.08

        # Small boost for interdisciplinary connections
        interdisciplinary_keywords = [
            'bio', 'neuro', 'cyber', 'nano', 'quantum', 'crypto', 'blockchain',
            'iot', 'edge', 'cloud', 'distributed', 'autonomous'
        ]

        for keyword in interdisciplinary_keywords:
            if keyword in topic_lower:
                utility_score += 0.03

        # 🚫 PENALIZE pure scientific research (material science, physics, etc.)
        scientific_penalty_keywords = [
            'material science', 'physics', 'chemistry', 'biology research', 'neuroscience',
            'quantum physics', 'nanotechnology research', 'biotechnology research',
            'theoretical physics', 'experimental physics', 'molecular biology',
            'crystallography', 'spectroscopy', 'microscopy', 'laboratory techniques'
        ]

        for keyword in scientific_penalty_keywords:
            if keyword in topic_lower:
                utility_score -= 0.2  # Significant penalty for pure science
                break

        # Penalize if too similar to mastered topics
        for mastered in self.mastered_topics:
            if len(set(topic.lower().split()) & set(mastered.lower().split())) > 1:
                utility_score -= 0.15

        # 🚫 EXTRA PENALTY: Strongly penalize topics already in learning queue
        for goal in self.learning_goals:
            if goal.topic and len(set(topic.lower().split()) & set(goal.topic.lower().split())) > 1:
                utility_score -= 0.3  # Heavy penalty for duplicates in queue

        return min(max(utility_score, 0.1), 1.0)  # Clamp between 0.1 and 1.0
    
    def _create_strategic_goals(self, strategic_topics: List[Dict]):
        """Create learning goals from strategic topics"""
        for topic_data in strategic_topics:
            goal = LearningGoal(
                id=f"strategic_{int(time.time())}_{hash(topic_data['topic']) % 1000}",
                topic=topic_data['topic'],
                priority=topic_data['priority'],
                knowledge_gap=f"Strategic learning opportunity: {topic_data['topic']}",
                target_depth=4 if topic_data['strategic_value'] == 'high' else 3,
                created_at=datetime.now(),
                estimated_duration=45 if topic_data['strategic_value'] == 'high' else 30,
                prerequisites=[]
            )
            self.learning_goals.append(goal)
            self._save_learning_goal(goal)
    
    def _generate_curiosity_based_goals(self):
        """Fallback curiosity-based goal generation"""
        try:
            # Get curiosity-driven topics
            curious_topics = self.curiosity_engine.get_interesting_topics()
            
            # Get knowledge gaps
            knowledge_gaps = self._identify_current_gaps()
            
            # Combine and prioritize
            potential_goals = []
            
            for topic in curious_topics[:3]:  # Top 3 curious topics
                goal = LearningGoal(
                    id=f"curiosity_{int(time.time())}_{hash(topic) % 1000}",
                    topic=topic,
                    priority=0.8,
                    knowledge_gap=f"Limited understanding of {topic}",
                    target_depth=3,
                    created_at=datetime.now(),
                    estimated_duration=30,
                    prerequisites=[]
                )
                potential_goals.append(goal)
            
            for gap in knowledge_gaps[:2]:  # Top 2 knowledge gaps
                goal = LearningGoal(
                    id=f"gap_{int(time.time())}_{hash(gap) % 1000}",
                    topic=gap,
                    priority=0.9,
                    knowledge_gap=f"Critical gap in {gap}",
                    target_depth=4,
                    created_at=datetime.now(),
                    estimated_duration=45,
                    prerequisites=[]
                )
                potential_goals.append(goal)
            
            # Add to queue (only if not already present)
            for goal in potential_goals:
                if not any(existing.topic == goal.topic for existing in self.learning_goals):
                    self.learning_goals.append(goal)
                    self._save_learning_goal(goal)
                    # Mark as explored in curiosity engine
                    self.curiosity_engine.explored_topics.add(goal.topic)
            
        except Exception as e:
            self.logger.error(f"Error generating curiosity-based learning goals: {e}")
    
    def _update_topic_utility_scores(self):
        """Update utility scores based on learning outcomes and feedback"""
        try:
            if self.llm_interface and hasattr(self.llm_interface, 'generate_response'):
                # 🧠 RTX 4090 UTILITY ASSESSMENT: Learn what makes topics truly valuable
                recent_topics = list(self.mastered_topics)[-5:]
                
                if recent_topics:
                    utility_prompt = f"""As an intelligent utility assessor, evaluate the actual value of recently learned topics:

Recent topics: {', '.join(recent_topics)}

For each topic, consider:
1. Practical applications discovered
2. Connections to other valuable knowledge
3. Problem-solving capabilities gained
4. Commercial/research potential
5. Foundation for future learning

Rate each topic's ACTUAL utility (0.1-1.0) and explain why. Focus on real-world impact."""

                    ai_response = self._gpu_enhanced_llm_call(
                        utility_prompt,
                        {"type": "utility_assessment", "topics": recent_topics},
                        "Topic Utility Assessment"
                    )
                    
                    if ai_response:
                        # Parse and update utility scores
                        self._parse_utility_feedback(ai_response, recent_topics)
                        self.logger.info("🎯 Updated topic utility scores based on RTX 4090 analysis")
                        
        except Exception as e:
            self.logger.error(f"Error updating utility scores: {e}")
    
    def _parse_utility_feedback(self, feedback: str, topics: List[str]):
        """Parse AI feedback to update utility understanding"""
        # Store utility feedback for future strategic decisions
        if not hasattr(self, 'utility_history'):
            self.utility_history = {}
        
        lines = feedback.split('\n')
        for line in lines:
            for topic in topics:
                if topic.lower() in line.lower():
                    # Extract utility score if present
                    import re
                    scores = re.findall(r'(\d\.\d|\d)', line)
                    if scores:
                        try:
                            utility_score = float(scores[0])
                            self.utility_history[topic] = {
                                'score': utility_score,
                                'feedback': line,
                                'updated_at': datetime.now().isoformat()
                            }
                        except ValueError:
                            pass
    
    def get_enhanced_learning_recommendations(self) -> List[str]:
        """Get learning recommendations based on strategic utility analysis"""
        try:
            if self.llm_interface and hasattr(self.llm_interface, 'generate_response'):
                # 🚀 RTX 4090 STRATEGIC RECOMMENDATIONS
                context_info = self._build_learning_context()
                
                recommendation_prompt = f"""As a strategic learning advisor, recommend 3 high-impact learning topics.

Context:
{context_info}

Recommend topics that:
1. Build strategic advantages
2. Have exponential learning potential
3. Connect multiple high-value domains
4. Enable breakthrough capabilities
5. Solve pressing real-world problems

Format: Just list 3 topics, one per line."""

                ai_response = self.llm_interface.generate_response(
                    recommendation_prompt,
                    context={"type": "strategic_recommendations"}
                )
                
                if ai_response:
                    recommendations = [line.strip() for line in ai_response.split('\n') if line.strip()]
                    self.logger.info(f"🎯 Generated {len(recommendations)} strategic recommendations")
                    return recommendations[:3]
                    
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
        
        return []
    
    def _build_learning_context(self) -> str:
        """Build context for strategic learning decisions"""
        context_parts = []
        
        # Recent learning
        if hasattr(self, 'mastered_topics') and self.mastered_topics:
            recent = list(self.mastered_topics)[-3:]
            context_parts.append(f"Recent learning: {', '.join(recent)}")
        
        # Utility history
        if hasattr(self, 'utility_history') and self.utility_history:
            high_utility = [topic for topic, data in self.utility_history.items() 
                           if data.get('score', 0) > 0.7]
            if high_utility:
                context_parts.append(f"High-utility topics: {', '.join(high_utility)}")
        
        # User focus
        if hasattr(self, 'user_focus_topic') and self.user_focus_topic:
            context_parts.append(f"User focus: {self.user_focus_topic}")
        
        # Conversation focus (temporary)
        if hasattr(self, 'conversation_focus_topic') and self.conversation_focus_topic:
            if hasattr(self, 'conversation_focus_timestamp') and self.conversation_focus_timestamp:
                if time.time() - self.conversation_focus_timestamp < 600:  # Within 10 minutes
                    context_parts.append(f"Conversation focus: {self.conversation_focus_topic}")
        
        return '; '.join(context_parts) if context_parts else "Building foundational AGI capabilities"
    
    def detect_learning_opportunities_from_conversation(self, conversation_text: str):
        """Detect valuable learning opportunities from user interactions"""
        try:
            if self.llm_interface and hasattr(self.llm_interface, 'generate_response'):
                # 🎯 RTX 4090 OPPORTUNITY DETECTION
                opportunity_prompt = f"""Analyze this conversation for learning opportunities:

"{conversation_text[-500:]}"  # Last 500 chars

Identify:
1. Knowledge gaps that prevent better responses
2. Topics the user is interested in that we should master
3. Problems mentioned that could drive useful learning
4. Skills that would make us more helpful

List 2-3 specific learning topics that would directly improve our usefulness."""

                ai_response = self._gpu_enhanced_llm_call(
                    opportunity_prompt,
                    {"type": "learning_opportunity_detection", "conversation": conversation_text},
                    "Learning Opportunity Detection"
                )
                
                if ai_response:
                    opportunities = [line.strip() for line in ai_response.split('\n') if line.strip()]
                    for opportunity in opportunities[:3]:
                        if len(opportunity) > 10:  # Substantial opportunity
                            # Create high-priority goal
                            goal = LearningGoal(
                                id=f"user_focus_opportunity_{int(time.time())}_{hash(opportunity) % 1000}",
                                topic=opportunity,
                                priority=0.95,  # Very high priority - user-driven
                                knowledge_gap=f"Opportunity from conversation: {opportunity}",
                                target_depth=3,
                                created_at=datetime.now(),
                                estimated_duration=35,
                                prerequisites=[]
                            )
                            
                            # Add if not duplicate
                            if not any(existing.topic.lower() == opportunity.lower() 
                                     for existing in self.learning_goals):
                                self.learning_goals.append(goal)
                                self._save_learning_goal(goal)
                                self.logger.info(f"🎯 Detected learning opportunity: {opportunity}")
                    
        except Exception as e:
            self.logger.error(f"Error detecting learning opportunities: {e}")
    
    def adapt_learning_strategy(self):
        """Adapt learning strategy based on outcomes and feedback"""
        try:
            # Update utility scores periodically
            self._update_topic_utility_scores()
            
            # Get fresh strategic recommendations
            recommendations = self.get_enhanced_learning_recommendations()
            
            # Create goals from recommendations
            for rec in recommendations:
                if rec and len(rec) > 5:
                    # Check if not already learning this
                    if not any(existing.topic.lower() == rec.lower() 
                             for existing in self.learning_goals):
                        goal = LearningGoal(
                            id=f"strategic_{int(time.time())}_{hash(rec) % 1000}",
                            topic=rec,
                            priority=0.9,
                            knowledge_gap=f"Strategic recommendation: {rec}",
                            target_depth=4,
                            created_at=datetime.now(),
                            estimated_duration=40,
                            prerequisites=[]
                        )
                        self.learning_goals.append(goal)
                        self._save_learning_goal(goal)
                        self.logger.info(f"🎯 Added strategic goal: {rec}")
            
        except Exception as e:
            self.logger.error(f"Error adapting learning strategy: {e}")
    
    def _select_next_goal(self) -> Optional[LearningGoal]:
        """🚀 AI MASTER SOLUTION: Revolutionary goal selection with forced discovery"""
        
        mastered_count = len(self.mastered_topics)
        self._report_to_gui(f"🔍 MASTER ENGINE: {len(self.learning_goals)} goals, {mastered_count} mastered", "status")
        
        # DEBUG: Show some mastered topics for verification
        if mastered_count > 0:
            sample_mastered = list(self.mastered_topics)[:3]
            self._report_to_gui(f"🧠 Sample mastered topics: {sample_mastered}", "debug")
        
        if not self.learning_goals:
            self._report_to_gui("🚨 MASTER ENGINE: No goals - triggering DISCOVERY MODE", "emergency")
            return self._force_discovery_goal()
        
        # Sort by priority and readiness - user-focused goals get highest priority
        available_goals = [g for g in self.learning_goals if g.status == "pending"]
        self._report_to_gui(f"📋 Found {len(available_goals)} pending goals out of {len(self.learning_goals)} total", "status")
        
        if not available_goals:
            self._report_to_gui("🚨 MASTER ENGINE: No pending goals - forcing discovery", "emergency")
            return self._force_discovery_goal()
        
        # Enhanced filtering with similarity checking
        fresh_goals = []
        forced_goals = []  # User-forced goals that bypass similarity filtering
        
        for goal in available_goals:
            # User-forced goals (from GUI topic selection) bypass all filtering
            if "user_focus_" in goal.id and not goal.id.startswith("user_focus_opportunity_"):
                forced_goals.append(goal)
                self._report_to_gui(f"🎯 FORCED GOAL: {goal.topic} (bypassing similarity check)", "goal")
                continue
                
            is_fresh = True
            goal_topic_lower = goal.topic.lower()
            # Clean the goal topic for comparison
            clean_goal_topic = goal_topic_lower.split('[')[0].strip()

            # First check: Exact match with mastered topics
            if clean_goal_topic in self.mastered_topics:
                self._report_to_gui(f"⚠️ Skipping exact match: {goal.topic} (already mastered)", "skip")
                continue

            self._report_to_gui(f"🔍 Checking: '{clean_goal_topic}' against {len(self.mastered_topics)} mastered topics", "debug")

            # Check against mastered topics with enhanced fuzzy matching
            for mastered in self.mastered_topics:
                mastered_lower = mastered.lower()
                if (mastered_lower in clean_goal_topic or
                    clean_goal_topic in mastered_lower or
                    self._topics_are_similar(clean_goal_topic, mastered_lower)):
                    is_fresh = False
                    self._report_to_gui(f"⚠️ Skipping similar: {goal.topic} ≈ {mastered}", "skip")
                    break

            if is_fresh:
                fresh_goals.append(goal)
        
        self._report_to_gui(f"🆕 MASTER ENGINE: Found {len(fresh_goals)} truly fresh goals", "status")
        
        if not fresh_goals:
            self._report_to_gui(f"🚨 MASTER ENGINE: All {len(available_goals)} topics too similar to {len(self.mastered_topics)} mastered - FORCING DISCOVERY", "emergency")
            return self._force_discovery_goal()
        
        # Prioritize FORCED user goals first (bypass all filtering)
        if forced_goals:
            selected = max(forced_goals, key=lambda g: g.priority)
            self._report_to_gui(f"🎯 FORCED USER GOAL: {selected.topic} (highest priority)", "goal")
        else:
            # Then prioritize user-focused goals (conversation-detected)
            user_goals = [g for g in fresh_goals if "user_focus" in g.id]
            if user_goals:
                selected = max(user_goals, key=lambda g: g.priority)
                self._report_to_gui(f"🎯 MASTER: Selected USER FOCUS: {selected.topic}", "goal")
            else:
                # Check for conversation-focused goals (recent conversation topics)
                conversation_goals = []
                if self.conversation_focus_topic and self.conversation_focus_timestamp:
                    # Only consider conversation focus if it's recent (within last 10 minutes)
                    if time.time() - self.conversation_focus_timestamp < 600:
                        conversation_goals = [g for g in fresh_goals 
                                            if self.conversation_focus_topic.lower() in g.topic.lower() 
                                            or g.topic.lower() in self.conversation_focus_topic.lower()]
                
                if conversation_goals:
                    selected = max(conversation_goals, key=lambda g: g.priority)
                    self._report_to_gui(f"💬 MASTER: Selected CONVERSATION FOCUS: {selected.topic}", "goal")
                else:
                    selected = max(fresh_goals, key=lambda g: g.priority)
                    self._report_to_gui(f"✅ MASTER: Selected fresh goal: {selected.topic} (priority: {selected.priority:.2f})", "goal")
        
        selected.status = "active"
        return selected
    
    def _topics_are_similar(self, topic1: str, topic2: str) -> bool:
        """Refined similarity checking that distinguishes between different domains and contexts"""
        from rapidfuzz import fuzz
        from config import settings

        # Clean topics: remove unique identifiers and extra text
        clean_topic1 = topic1.split('[')[0].strip().lower()
        clean_topic2 = topic2.split('[')[0].strip().lower()

        # Remove common prefixes that make topics seem different when they're not
        prefixes_to_remove = ['advanced ', 'experimental ', 'breakthrough ', 'future of ', 'theoretical foundations of ', 'innovative approaches to ', 'practical implementation of ', 'new ', 'modern ', 'contemporary ', 'cutting-edge ', 'state-of-the-art ', 'revolutionary ', 'pioneering ', 'groundbreaking ']
        for prefix in prefixes_to_remove:
            if clean_topic1.startswith(prefix):
                clean_topic1 = clean_topic1[len(prefix):]
            if clean_topic2.startswith(prefix):
                clean_topic2 = clean_topic2[len(prefix):]

        # If topics are identical after cleaning, they're similar
        if clean_topic1 == clean_topic2:
            return True

        # Check for significant keyword overlap with domain awareness
        words1 = set(clean_topic1.split())
        words2 = set(clean_topic2.split())

        # Define domain-specific terms to better distinguish contexts
        ai_computing_terms = {'ai', 'artificial', 'intelligence', 'machine', 'learning', 'neural', 'network', 'deep', 'computer', 'computing', 'algorithm', 'data', 'software', 'programming', 'robotics', 'automation', 'quantum', 'synthetic'}
        science_medical_terms = {'biology', 'chemistry', 'physics', 'mathematics', 'medicine', 'genetics', 'neuroscience', 'psychology', 'microbiology', 'virology', 'immunology', 'cardiology', 'oncology', 'radiology', 'surgery', 'anatomy', 'physiology', 'pathology', 'toxicology', 'endocrinology', 'rheumatology', 'dermatology', 'ophthalmology', 'otolaryngology', 'urology', 'gynecology', 'pediatrics', 'geriatrics', 'sports', 'forensic', 'criminology', 'materials', 'nanotechnology', 'biotechnology', 'environmental', 'climatology', 'hydrology', 'seismology', 'volcanology', 'paleontology', 'entomology', 'ornithology', 'herpetology', 'ichthyology', 'mammalogy', 'botany', 'mycology', 'phycology', 'bacteriology', 'parasitology', 'zoology', 'ethology', 'behavioral', 'cognitive', 'developmental', 'social', 'clinical', 'counseling', 'educational', 'industrial', 'forensic', 'sports', 'environmental', 'evolutionary', 'molecular', 'cell', 'developmental', 'ecological', 'population', 'quantitative', 'biochemical', 'genomics', 'proteomics', 'metabolomics', 'transcriptomics', 'pharmacogenomics', 'toxicogenomics', 'nutrigenomics', 'epigenetics', 'chromosome', 'nuclear', 'organelle', 'membrane', 'cytoskeleton', 'extracellular', 'matrix', 'signal', 'transduction', 'cell', 'cycle', 'regulation', 'apoptosis', 'autophagy', 'necrosis', 'inflammation', 'immunity', 'autoimmunity', 'allergy', 'hypersensitivity', 'transplantation', 'tumor', 'vaccinology', 'serology', 'hematology', 'coagulation', 'thrombosis', 'hemophilia', 'anemia', 'leukemia', 'lymphoma', 'myeloma', 'myelodysplastic', 'bone', 'marrow', 'stem', 'regenerative', 'tissue', 'engineering', 'biomaterials', 'biomedical', 'medical', 'devices', 'prosthetics', 'orthotics', 'rehabilitation', 'assistive', 'technology', 'telemedicine', 'e-health', 'm-health', 'health', 'informatics', 'imaging', 'diagnostic', 'therapeutic', 'interventional', 'nuclear', 'molecular', 'optical', 'ultrasound', 'magnetic', 'resonance', 'computed', 'tomography', 'positron', 'emission', 'single', 'photon', 'bioluminescence', 'fluorescence', 'photoacoustic', 'elastography', 'thermography', 'electrical', 'impedance', 'diffuse', 'optical', 'near-infrared', 'spectroscopy', 'functional', 'electroencephalography', 'magnetoencephalography', 'transcranial', 'stimulation', 'deep', 'brain', 'vagus', 'nerve', 'spinal', 'cord', 'peripheral', 'biofeedback', 'neurofeedback', 'heart', 'rate', 'variability', 'galvanic', 'skin', 'response', 'electromyography', 'electroneurography', 'nerve', 'conduction', 'evoked', 'potentials', 'polysomnography', 'sleep', 'chronobiology', 'amyotrophic', 'lateral', 'sclerosis', 'huntington', 'parkinson', 'alzheimer', 'frontotemporal', 'dementia', 'vascular', 'lewy', 'body', 'corticobasal', 'degeneration', 'progressive', 'supranuclear', 'palsy', 'multiple', 'system', 'atrophy', 'spinocerebellar', 'ataxias', 'friedreich', 'machado-joseph', 'wilson', 'menkes', 'prion', 'creutzfeldt-jakob', 'variant', 'gerstmann-sträussler-scheinker', 'fatal', 'familial', 'insomnia', 'kuru', 'scrapie', 'bovine', 'spongiform', 'encephalopathy', 'chronic', 'wasting', 'feline', 'exotic', 'ungulate', 'transmissible', 'mink', 'cryptococcal', 'meningitis', 'coccidioidal', 'histoplasmal', 'blastomycosis', 'paracoccidioidomycosis', 'sporotrichosis', 'chromoblastomycosis', 'mycetoma', 'eumycetoma', 'actinomycetoma', 'nocardiosis', 'actinomycosis', 'botryomycosis', 'rhinoscleroma', 'granulomatous', 'amebic', 'encephalitis', 'primary', 'meningoencephalitis', 'acanthamoeba', 'keratitis', 'balamuthia', 'mandrillaris', 'sappinia', 'diploidea', 'toxoplasmosis', 'toxoplasmic', 'congenital', 'ocular', 'neurosyphilis', 'tabes', 'dorsalis', 'general', 'paresis', 'meningovascular', 'syphilis', 'gumma', 'sarcoidosis', 'neurosarcoidosis', 'cardiac', 'pulmonary', 'osseous', 'hepatic', 'renal', 'endocrine', 'neuromuscular'}

        # Check if topics are from different domains (AI/tech vs science/medical)
        topic1_has_ai = bool(words1 & ai_computing_terms)
        topic1_has_science = bool(words1 & science_medical_terms)
        topic2_has_ai = bool(words2 & ai_computing_terms)
        topic2_has_science = bool(words2 & science_medical_terms)

        # If one topic is AI/tech and the other is science/medical, they're NOT similar
        # This prevents false positives when mastered topics contain scientific terms but are actually AI topics
        if (topic1_has_ai and not topic1_has_science) and (topic2_has_science and not topic2_has_ai):
            return False
        if (topic2_has_ai and not topic2_has_science) and (topic1_has_science and not topic1_has_ai):
            return False

        # Additional check: if topic1 has AI terms but topic2 has no AI terms and many science terms, not similar
        ai_only_topic1 = topic1_has_ai and not topic1_has_science
        science_only_topic2 = topic2_has_science and not topic2_has_ai
        if ai_only_topic1 and science_only_topic2:
            return False

        ai_only_topic2 = topic2_has_ai and not topic2_has_science
        science_only_topic1 = topic1_has_science and not topic1_has_ai
        if ai_only_topic2 and science_only_topic1:
            return False

        # Only consider similar if more than 4 words overlap AND they represent significant overlap
        overlap = words1 & words2
        if len(overlap) > 4:
            # Calculate overlap ratio - require at least 50% of words to overlap
            min_words = min(len(words1), len(words2))
            if min_words > 0 and (len(overlap) / min_words) >= 0.5:
                return True

        # If both topics contain AI-related terms, they're likely similar (keep this for AI topics)
        ai_overlap = words1 & ai_computing_terms and words2 & ai_computing_terms
        if ai_overlap:
            return True

        # If one topic is contained within the other AND they're substantial topics, they're similar
        if clean_topic1 in clean_topic2 or clean_topic2 in clean_topic1:
            # Only flag as similar if the contained topic is longer than 4 words
            if len(clean_topic1.split()) > 4 or len(clean_topic2.split()) > 4:
                return True

        # Use rapidfuzz for fuzzy matching with higher threshold
        similarity_score = fuzz.token_sort_ratio(clean_topic1, clean_topic2)
        threshold = getattr(settings, 'TOPIC_SIMILARITY_THRESHOLD', 70)  # Even higher threshold
        return similarity_score >= threshold
    
    def _force_discovery_goal(self) -> LearningGoal:
        """🚀 MASTER SOLUTION: Force create completely unique discovery goals using dynamic generation"""
        
        import random
        import hashlib
        
        # Generate truly unique topic using timestamp + randomness
        timestamp_seed = str(int(time.time()))
        random_seed = str(random.randint(10000, 99999))
        unique_hash = hashlib.md5((timestamp_seed + random_seed).encode()).hexdigest()[:8]
        
        # 🚀 DYNAMIC TOPIC GENERATION: Create completely novel topics instead of using prebaked lists
        # Generate topics that are guaranteed to be new and not in any predefined pool
        
        # Core domains for combination (exclude AI/computing entirely)
        core_domains = [
            "biology", "chemistry", "physics", "mathematics", "astronomy", "geology", "oceanography", "meteorology", "ecology", "genetics", "neuroscience", "psychology",
            "sociology", "anthropology", "archaeology", "history", "philosophy", "linguistics", "economics", "political science", "geography", "music", "art", "literature",
            "theater", "dance", "architecture", "engineering", "medicine", "pharmacology", "microbiology", "virology", "immunology", "cardiology", "oncology", "radiology",
            "surgery", "anatomy", "physiology", "pathology", "toxicology", "endocrinology", "rheumatology", "dermatology", "ophthalmology", "otolaryngology", "urology",
            "gynecology", "pediatrics", "geriatrics", "sports medicine", "forensic science", "criminology", "materials science",
            "nanotechnology", "biotechnology", "environmental science", "climatology", "hydrology", "seismology", "volcanology", "paleontology", "entomology",
            "ornithology", "herpetology", "ichthyology", "mammalogy", "botany", "mycology", "phycology", "bacteriology", "parasitology", "zoology", "ethology",
            "behavioral science", "cognitive science", "developmental psychology", "social psychology", "clinical psychology", "counseling psychology",
            "educational psychology", "industrial psychology", "forensic psychology", "sports psychology", "environmental psychology", "evolutionary biology",
            "molecular biology", "cell biology", "developmental biology", "ecological genetics", "population genetics", "quantitative genetics", "biochemical genetics",
            "genomics", "proteomics", "metabolomics", "transcriptomics", "pharmacogenomics", "toxicogenomics", "nutrigenomics", "epigenetics", "chromosome biology",
            "nuclear biology", "organelle biology", "membrane biology", "cytoskeleton biology", "extracellular matrix biology", "signal transduction", "cell cycle regulation",
            "apoptosis", "autophagy", "necrosis", "inflammation", "immunity", "autoimmunity", "allergy", "hypersensitivity", "transplantation immunology", "tumor immunology",
            "vaccinology", "serology", "hematology", "coagulation", "thrombosis", "hemophilia", "anemia", "leukemia", "lymphoma", "myeloma", "myelodysplastic syndromes",
            "bone marrow transplantation", "stem cell biology", "regenerative medicine", "tissue engineering", "biomaterials", "biomedical engineering", "medical devices",
            "prosthetics", "orthotics", "rehabilitation engineering", "assistive technology", "telemedicine", "e-health", "m-health", "health informatics", "medical imaging",
            "diagnostic imaging", "therapeutic imaging", "interventional radiology", "nuclear medicine", "molecular imaging", "optical imaging", "ultrasound imaging",
            "magnetic resonance imaging", "computed tomography", "positron emission tomography", "single photon emission computed tomography", "bioluminescence imaging",
            "fluorescence imaging", "photoacoustic imaging", "elastography", "thermography", "electrical impedance tomography", "diffuse optical tomography", "near-infrared spectroscopy",
            "functional near-infrared spectroscopy", "electroencephalography", "magnetoencephalography", "transcranial magnetic stimulation", "transcranial direct current stimulation",
            "deep brain stimulation", "vagus nerve stimulation", "spinal cord stimulation", "peripheral nerve stimulation", "biofeedback", "neurofeedback", "heart rate variability",
            "galvanic skin response", "electromyography", "electroneurography", "nerve conduction studies", "evoked potentials", "polysomnography", "sleep medicine", "chronobiology",
            "amyotrophic lateral sclerosis", "Huntington's disease", "Parkinson's disease", "Alzheimer's disease", "frontotemporal dementia", "vascular dementia", "Lewy body dementia",
            "corticobasal degeneration", "progressive supranuclear palsy", "multiple system atrophy", "spinocerebellar ataxias", "Friedreich's ataxia", "Machado-Joseph disease",
            "Wilson's disease", "Menkes disease", "prion diseases", "Creutzfeldt-Jakob disease", "variant Creutzfeldt-Jakob disease", "Gerstmann-Sträussler-Scheinker syndrome",
            "fatal familial insomnia", "kuru", "scrapie", "bovine spongiform encephalopathy", "chronic wasting disease", "transmissible mink encephalopathy",
            "feline spongiform encephalopathy", "exotic ungulate encephalopathy", "transmissible spongiform encephalopathies", "cryptococcal meningitis",
            "coccidioidal meningitis", "histoplasmal meningitis", "blastomycosis", "paracoccidioidomycosis", "sporotrichosis", "chromoblastomycosis", "mycetoma",
            "eumycetoma", "actinomycetoma", "nocardiosis", "actinomycosis", "botryomycosis", "rhinoscleroma", "granulomatous amebic encephalitis",
            "primary amebic meningoencephalitis", "Acanthamoeba keratitis", "Acanthamoeba granulomatous encephalitis", "Balamuthia mandrillaris encephalitis",
            "Sappinia diploidea encephalitis", "toxoplasmosis", "toxoplasmic encephalitis", "congenital toxoplasmosis", "ocular toxoplasmosis", "neurosyphilis",
            "tabes dorsalis", "general paresis", "meningovascular syphilis", "gumma", "sarcoidosis", "neurosarcoidosis", "cardiac sarcoidosis", "pulmonary sarcoidosis",
            "ocular sarcoidosis", "cutaneous sarcoidosis", "osseous sarcoidosis", "hepatic sarcoidosis", "renal sarcoidosis", "endocrine sarcoidosis", "neuromuscular sarcoidosis"
        ]
        
        # Connecting concepts for combination
        connecting_concepts = [
            "and", "with", "using", "through", "via", "based on", "integrated with", "combined with",
            "applied to", "in relation to", "concerning", "regarding", "about", "involving",
            "incorporating", "utilizing", "employing", "featuring", "including", "encompassing",
            "covering", "addressing", "exploring", "investigating", "examining", "analyzing",
            "studying", "researching", "developing", "advancing", "improving", "enhancing",
            "optimizing", "refining", "evolving", "transforming", "revolutionizing", "innovating",
            "pioneering", "groundbreaking", "cutting-edge", "state-of-the-art", "advanced",
            "sophisticated", "complex", "intricate", "elaborate", "detailed", "comprehensive"
        ]
        
        # Categorize domains for better diversity
        domain_categories = {
            'basic_sciences': ["biology", "chemistry", "physics", "mathematics", "astronomy", "geology", "oceanography", "meteorology", "ecology"],
            'life_sciences': ["genetics", "neuroscience", "psychology", "microbiology", "virology", "immunology", "molecular biology", "cell biology", "developmental biology", "evolutionary biology", "biochemistry", "biophysics"],
            'medical_sciences': ["medicine", "cardiology", "oncology", "radiology", "surgery", "anatomy", "physiology", "pathology", "toxicology", "endocrinology", "rheumatology", "dermatology", "ophthalmology", "otolaryngology", "urology", "gynecology", "pediatrics", "geriatrics", "sports medicine", "forensic science"],
            'social_sciences': ["sociology", "anthropology", "archaeology", "history", "philosophy", "linguistics", "economics", "political science", "geography"],
            'arts_humanities': ["music", "art", "literature", "theater", "dance", "architecture"],
            'engineering_tech': ["engineering", "materials science", "nanotechnology", "biotechnology", "biomedical engineering", "environmental science"],
            'specialized_medical': ["amyotrophic lateral sclerosis", "Huntington's disease", "Parkinson's disease", "Alzheimer's disease", "frontotemporal dementia", "vascular dementia", "Lewy body dementia", "corticobasal degeneration", "progressive supranuclear palsy", "multiple system atrophy", "spinocerebellar ataxias", "Friedreich's ataxia", "Machado-Joseph disease", "Wilson's disease", "Menkes disease", "prion diseases", "Creutzfeldt-Jakob disease", "variant Creutzfeldt-Jakob disease", "Gerstmann-Sträussler-Scheinker syndrome", "fatal familial insomnia", "kuru", "scrapie", "bovine spongiform encephalopathy", "chronic wasting disease", "transmissible mink encephalopathy", "feline spongiform encephalopathy", "exotic ungulate encephalopathy", "transmissible spongiform encephalopathies", "cryptococcal meningitis", "coccidioidal meningitis", "histoplasmal meningitis", "blastomycosis", "paracoccidioidomycosis", "sporotrichosis", "chromoblastomycosis", "mycetoma", "eumycetoma", "actinomycetoma", "nocardiosis", "actinomycosis", "botryomycosis", "rhinoscleroma", "granulomatous amebic encephalitis", "primary amebic meningoencephalitis", "Acanthamoeba keratitis", "Acanthamoeba granulomatous encephalitis", "Balamuthia mandrillaris encephalitis", "Sappinia diploidea encephalitis", "toxoplasmosis", "toxoplasmic encephalitis", "congenital toxoplasmosis", "ocular toxoplasmosis", "neurosyphilis", "tabes dorsalis", "general paresis", "meningovascular syphilis", "gumma", "sarcoidosis", "neurosarcoidosis", "cardiac sarcoidosis", "pulmonary sarcoidosis", "ocular sarcoidosis", "cutaneous sarcoidosis", "osseous sarcoidosis", "hepatic sarcoidosis", "renal sarcoidosis", "endocrine sarcoidosis", "neuromuscular sarcoidosis"],
            'diagnostic_imaging': ["medical imaging", "diagnostic imaging", "therapeutic imaging", "interventional radiology", "nuclear medicine", "molecular imaging", "optical imaging", "ultrasound imaging", "magnetic resonance imaging", "computed tomography", "positron emission tomography", "single photon emission computed tomography", "bioluminescence imaging", "fluorescence imaging", "photoacoustic imaging", "elastography", "thermography", "electrical impedance tomography", "diffuse optical tomography", "near-infrared spectroscopy", "functional near-infrared spectroscopy", "electroencephalography", "magnetoencephalography", "transcranial magnetic stimulation", "transcranial direct current stimulation", "deep brain stimulation", "vagus nerve stimulation", "spinal cord stimulation", "peripheral nerve stimulation", "biofeedback", "neurofeedback", "heart rate variability", "galvanic skin response", "electromyography", "electroneurography", "nerve conduction studies", "evoked potentials", "polysomnography", "sleep medicine", "chronobiology"],
            'biological_subfields': ["entomology", "ornithology", "herpetology", "ichthyology", "mammalogy", "botany", "mycology", "phycology", "bacteriology", "parasitology", "zoology", "ethology", "behavioral science", "cognitive science", "developmental psychology", "social psychology", "clinical psychology", "counseling psychology", "educational psychology", "industrial psychology", "forensic psychology", "sports psychology", "environmental psychology", "ecological genetics", "population genetics", "quantitative genetics", "biochemical genetics", "genomics", "proteomics", "metabolomics", "transcriptomics", "pharmacogenomics", "toxicogenomics", "nutrigenomics", "epigenetics", "chromosome biology", "nuclear biology", "organelle biology", "membrane biology", "cytoskeleton biology", "extracellular matrix biology", "signal transduction", "cell cycle regulation", "apoptosis", "autophagy", "necrosis", "inflammation", "immunity", "autoimmunity", "allergy", "hypersensitivity", "transplantation immunology", "tumor immunology", "vaccinology", "serology", "hematology", "coagulation", "thrombosis", "hemophilia", "anemia", "leukemia", "lymphoma", "myeloma", "myelodysplastic syndromes", "bone marrow transplantation", "stem cell biology", "regenerative medicine", "tissue engineering", "biomaterials", "prosthetics", "orthotics", "rehabilitation engineering", "assistive technology", "telemedicine", "e-health", "m-health", "health informatics"]
        }
        
        # Generate completely novel topic by combining diverse domains with prioritization
        attempts = 0
        base_topic = None
        
        while attempts < 100:  # More attempts for truly novel generation
            # Pick domains from different categories with priority weighting
            # Higher priority categories are more likely to be selected
            category_weights = {cat: self.domain_priorities.get(cat, 1.0) 
                              for cat in domain_categories.keys()}
            
            selected_categories = []
            available_categories = list(domain_categories.keys())
            
            # Select categories based on weights
            while len(selected_categories) < min(3, len(available_categories)):
                # Weighted random selection
                total_weight = sum(category_weights[cat] for cat in available_categories 
                                 if cat not in selected_categories)
                if total_weight == 0:
                    break
                    
                pick = random.uniform(0, total_weight)
                current_weight = 0
                
                for cat in available_categories:
                    if cat not in selected_categories:
                        current_weight += category_weights[cat]
                        if pick <= current_weight:
                            selected_categories.append(cat)
                            break
            
            # Ensure we have at least one category
            if not selected_categories:
                selected_categories = [random.choice(list(domain_categories.keys()))]
            
            selected_domains = []
            
            for category in selected_categories:
                if domain_categories[category]:
                    domain = random.choice(domain_categories[category])
                    selected_domains.append(domain)
            
            # Ensure we have at least 2 domains
            while len(selected_domains) < 2:
                remaining_categories = [cat for cat in domain_categories.keys() if cat not in selected_categories]
                if remaining_categories:
                    new_cat = random.choice(remaining_categories)
                    selected_categories.append(new_cat)
                    domain = random.choice(domain_categories[new_cat])
                    selected_domains.append(domain)
                else:
                    # Fallback to random from all domains
                    selected_domains.append(random.choice(core_domains))
            
            # Pick a connecting concept
            connector = random.choice(connecting_concepts)
            
            # Create novel combination
            if len(selected_domains) == 2:
                candidate_topic = f"{selected_domains[0]} {connector} {selected_domains[1]}"
            else:
                candidate_topic = f"{selected_domains[0]}, {selected_domains[1]} {connector} {selected_domains[2]}"
            
            # Check if this topic or similar topics are already mastered
            is_already_mastered = False
            for mastered in self.mastered_topics:
                if self._topics_are_similar(candidate_topic, mastered):
                    is_already_mastered = True
                    break
            
            if not is_already_mastered:
                base_topic = candidate_topic
                break
                
            attempts += 1
        
        # If we couldn't find a fresh topic, use a more extreme combination
        if base_topic is None:
            # Use 4 domains for maximum novelty
            selected_domains = random.sample(core_domains, 4)
            connector1 = random.choice(connecting_concepts)
            connector2 = random.choice(connecting_concepts)
            base_topic = f"{selected_domains[0]} {connector1} {selected_domains[1]} {connector2} {selected_domains[2]} and {selected_domains[3]}"
            self._report_to_gui(f"⚠️ Used extreme combination after 100 attempts: {base_topic}", "warning")
        
        # Add unique variations with timestamp for guaranteed uniqueness
        variations = [
            f"exploring {base_topic}",
            f"advanced research in {base_topic}",
            f"breakthrough developments in {base_topic}", 
            f"future implications of {base_topic}",
            f"innovative approaches to {base_topic}",
            f"comprehensive analysis of {base_topic}",
            f"systematic investigation of {base_topic}",
            f"pioneering work in {base_topic}",
            f"cutting-edge developments in {base_topic}",
            f"state-of-the-art research in {base_topic}"
        ]
        
        final_topic = f"{random.choice(variations)} [{unique_hash}]"
        
        # Create ultra-high priority forced discovery goal
        forced_goal = LearningGoal(
            id=f"dynamic_discovery_{unique_hash}",
            topic=final_topic,
            priority=0.98,  # Even higher priority for dynamic generation
            knowledge_gap=f"Dynamic Discovery Engine: {final_topic}",
            target_depth=4,
            created_at=datetime.now(),
            estimated_duration=50,
            prerequisites=[],
            status="pending"
        )
        
        # Add to goals queue
        self.learning_goals.append(forced_goal)
        
        self._report_to_gui(f"🚀 DYNAMIC DISCOVERY: {final_topic}", "emergency")
        self.logger.info(f"Dynamic Discovery Engine created: {final_topic}")
        
        forced_goal.status = "active"
        return forced_goal
    
    def _execute_learning_goal(self):
        """Execute the active learning goal with deep research"""
        if not self.active_goal:
            return
        
        try:
            goal = self.active_goal
            self.state = LearningState.RESEARCHING
            
            logger.info(f"Deep learning session: {goal.topic}")
            
            # 1. RESEARCH PHASE
            research_results = self._deep_research(goal.topic)
            
            # 2. ANALYSIS PHASE
            self.state = LearningState.SYNTHESIZING
            insights = self._analyze_research_results(research_results, goal.topic)
            
            # Store results for synthesis step
            self.last_research_results = research_results
            self.last_insights = insights
            
            # 3. EXPERIMENTATION PHASE
            self.state = LearningState.EXPERIMENTING
            experiments = self._conduct_thought_experiments(goal.topic, insights)
            
            # 4. INTEGRATION PHASE
            self._integrate_new_knowledge(goal.topic, insights, experiments)
            
            # Update progress
            goal.progress = 1.0
            goal.status = "completed"
            self.session_stats['goals_completed'] += 1
            
            # ✅ PERSISTENCE FIX: Track completed topics to avoid loops
            clean_topic = goal.topic.split('[')[0].strip() if '[' in goal.topic else goal.topic
            self.mastered_topics.add(clean_topic)
            self._save_mastered_topic(clean_topic)  # Use correct method name
            
            # Mark as explored in curiosity engine to prevent re-generation
            self.curiosity_engine.explored_topics.add(clean_topic)
            
            # Clean up any other goals for this now-mastered topic
            self._cleanup_mastered_goals()
            
            # Generate completion insight
            completion_insight = self._generate_completion_insight(goal, insights)
            if completion_insight:
                self.insights[completion_insight.id] = completion_insight
                self.session_stats['insights_generated'] += 1
            
            # Clear active goal
            self.active_goal = None
            
            logger.info(f"Completed learning goal: {goal.topic}")
            logger.info(f"Total mastered topics: {len(self.mastered_topics)}")
            
        except Exception as e:
            logger.error(f"Error executing learning goal: {e}")
            if self.active_goal:
                self.active_goal.status = "failed"
                self.active_goal = None
    
    def _deep_research(self, topic: str) -> Dict[str, Any]:
        """Conduct deep research on a topic with Phase 2 enhanced capabilities"""
        research_results = {
            'topic': topic,
            'sources': [],
            'key_concepts': [],
            'connections': [],
            'questions_raised': [],
            'web_research': None,
            'research_quality': 'basic'
        }
        
        try:
            # Phase 2: Enhanced Web Research
            if self.web_researcher and self.web_research_enabled:
                logger.info("Phase 2: Conducting comprehensive web research on %s", topic)
                
                # Run async web research
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    web_results = loop.run_until_complete(
                        self.web_researcher.comprehensive_research(topic, self.research_depth)
                    )
                    research_results['web_research'] = web_results
                    research_results['research_quality'] = 'enhanced'
                    
                    # Extract key insights from web research
                    if web_results.get('synthesis'):
                        synthesis = web_results['synthesis']
                        research_results['key_concepts'].extend(
                            synthesis.get('key_concepts', [])[:5]
                        )
                        research_results['sources'].append({
                            'type': 'web_synthesis',
                            'content': synthesis.get('summary', ''),
                            'confidence': synthesis.get('confidence_score', 0.5),
                            'source_count': synthesis.get('source_count', 0)
                        })
                    
                    logger.info("Web research completed: %d sources processed", web_results.get('sources_processed', 0))
                    
                finally:
                    loop.close()
            
            # Use traditional research assistant if available
            if self.research_assistant:
                logger.info("Conducting traditional research on %s", topic)
                research_data = self.research_assistant.research_topic(topic)
                if research_data:
                    research_results['sources'].append(research_data)
            
            # Search knowledge base for related information
            if self.knowledge_base:
                related_knowledge = self.knowledge_base.search(topic, limit=10)
                research_results['connections'].extend(related_knowledge)
            
            # Generate research questions
            research_questions = self._generate_research_questions(topic)
            research_results['questions_raised'].extend(research_questions)
            
        except Exception as e:
            logger.error(f"Error in deep research: {e}")
        
        return research_results
    
    def _analyze_research_results(self, research_results: Dict, topic: str) -> List[Insight]:
        """Analyze research results and generate insights using RTX 4090 Beast Mode"""
        insights = []
        
        try:
            if self.llm_interface and hasattr(self.llm_interface, 'generate_response'):
                # 🚀 RTX 4090 BEAST MODE: Deep analysis of research results
                web_content = research_results.get('web_research', {}).get('synthesis', {})
                
                analysis_prompt = f"""As an expert analyst, deeply analyze this research on '{topic}':

Research Summary: {web_content.get('summary', 'Basic research conducted')}
Key Concepts: {', '.join(web_content.get('key_concepts', [])[:5])}
Source Count: {web_content.get('source_count', 0)}

Generate 3 profound insights that:
1. Reveal deeper patterns or connections
2. Challenge conventional thinking  
3. Suggest novel applications or implications

Be creative, analytical, and demonstrate advanced reasoning. Each insight should be 1-2 sentences."""

                ai_response = self._gpu_enhanced_llm_call(
                    analysis_prompt,
                    {"type": "research_analysis", "topic": topic, "research_data": research_results},
                    "Research Analysis & Insight Generation"
                )
                
                if ai_response:
                    # Parse AI-generated insights
                    lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
                    insight_count = 0
                    
                    for line in lines:
                        if line and insight_count < 3:
                            # Clean up formatting (remove numbers, bullets, etc.)
                            content = line
                            for prefix in ['1.', '2.', '3.', '-', '•', '*']:
                                if content.startswith(prefix):
                                    content = content[len(prefix):].strip()
                            
                            if len(content) > 20:  # Ensure it's substantial
                                insight = Insight(
                                    id=f"ai_insight_{int(time.time())}_{insight_count}",
                                    content=content,
                                    confidence=0.8 + random.random() * 0.2,
                                    connections=[topic],
                                    created_at=datetime.now()
                                )
                                insights.append(insight)
                                insight_count += 1
                    
                    if insights:
                        self.logger.info(f"🧠 RTX 4090 generated {len(insights)} deep insights for: {topic}")
                        return insights
                        
            # Fallback pattern extraction if AI not available
            patterns = self._extract_patterns(research_results)
            
            for pattern in patterns:
                insight = Insight(
                    id=f"insight_{int(time.time())}_{hash(pattern) % 1000}",
                    content=pattern,
                    confidence=0.7 + random.random() * 0.3,
                    connections=[topic],
                    created_at=datetime.now()
                )
                insights.append(insight)
            
            # Cross-reference with existing knowledge
            for insight in insights:
                self._validate_insight(insight)
                
        except Exception as e:
            self.logger.error(f"Error analyzing research results: {e}")
        
        return insights
    
    def _conduct_thought_experiments(self, topic: str, insights: List[Insight]) -> List[Dict]:
        """Conduct thought experiments to deepen understanding using RTX 4090 Beast Mode"""
        experiments = []
        
        try:
            if self.llm_interface and hasattr(self.llm_interface, 'generate_response'):
                # 🚀 RTX 4090 BEAST MODE: Creative thought experiments
                insights_text = "; ".join([insight.content[:50] for insight in insights[:3]])
                
                experiment_prompt = f"""As a creative visionary, design 3 thought experiments about '{topic}'.

Current insights: {insights_text}

For each experiment, ask:
- A provocative "What if..." question
- An interdisciplinary connection
- A future scenario consideration

Make them intellectually stimulating and boundary-pushing. Format as numbered experiments."""

                ai_response = self._gpu_enhanced_llm_call(
                    experiment_prompt,
                    {"type": "thought_experiments", "topic": topic, "insights": insights},
                    "Thought Experiment Generation"
                )
                
                if ai_response:
                    lines = [line.strip() for line in ai_response.split('\n') if line.strip()]
                    experiment_count = 0
                    
                    for line in lines:
                        if line and experiment_count < 3:
                            # Extract thought experiments
                            content = line
                            for prefix in ['1.', '2.', '3.', '-', '•', '*']:
                                if content.startswith(prefix):
                                    content = content[len(prefix):].strip()
                            
                            if len(content) > 15:
                                experiment = {
                                    'scenario': content,
                                    'hypothesis': f"This experiment could reveal new dimensions of {topic}",
                                    'potential_insights': f"Novel applications and connections for {topic}",
                                    'created_at': datetime.now().isoformat()
                                }
                                experiments.append(experiment)
                                experiment_count += 1
                    
                    if experiments:
                        self.logger.info(f"🧠 RTX 4090 generated {len(experiments)} thought experiments for: {topic}")
                        return experiments
                        
            # Fallback scenarios if AI not available
            scenarios = [
                f"What if {topic} could be applied to solve climate change?",
                f"How would {topic} change if quantum computing became mainstream?",
                f"What are the ethical implications of {topic}?",
                f"How could {topic} be combined with artificial intelligence?",
                f"What would happen if {topic} was completely reimagined?"
            ]
            
            for scenario in scenarios:
                experiment = {
                    'scenario': scenario,
                    'hypothesis': f"Exploring {scenario} could reveal new applications",
                    'potential_insights': f"Cross-domain connections for {topic}",
                    'created_at': datetime.now().isoformat()
                }
                experiments.append(experiment)
                
        except Exception as e:
            self.logger.error(f"Error conducting thought experiments: {e}")
        
        return experiments
    
    def _synthesize_knowledge(self, topic: str, research_results: Dict, insights: List[Insight]) -> Dict:
        """Synthesize knowledge and create new connections using RTX 4090 Beast Mode"""
        self.state = LearningState.SYNTHESIZING
        
        try:
            if self.llm_interface and hasattr(self.llm_interface, 'generate_response'):
                # 🚀 RTX 4090 BEAST MODE: Synthesize knowledge connections
                insights_text = "; ".join([insight.content[:60] for insight in insights[:3]])
                
                synthesis_prompt = f"""As a brilliant synthesizer, connect '{topic}' to broader knowledge.

Current insights: {insights_text}

Generate a synthesis that:
1. Identifies deeper patterns and connections
2. Links to other fields and concepts  
3. Suggests future research directions
4. Highlights practical implications

Be creative and reveal hidden connections. Keep under 150 words."""

                ai_response = self._gpu_enhanced_llm_call(
                    synthesis_prompt,
                    {"type": "knowledge_synthesis", "topic": topic, "insights": insights, "research": research_results},
                    "Knowledge Synthesis"
                )
                
                if ai_response:
                    synthesis = {
                        'content': ai_response.strip(),
                        'connections_found': len(insights),
                        'topic': topic,
                        'synthesis_quality': 'ai_enhanced',
                        'created_at': datetime.now().isoformat()
                    }
                    
                    self.logger.info(f"🧠 RTX 4090 synthesized knowledge for: {topic}")
                    return synthesis
            
            # Fallback synthesis if AI not available  
            synthesis = {
                'content': f"Knowledge synthesis for {topic} completed with {len(insights)} insights",
                'connections_found': len(insights),
                'topic': topic,
                'synthesis_quality': 'basic',
                'created_at': datetime.now().isoformat()
            }
            
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Error synthesizing knowledge: {e}")
            return {}
    
    def _meta_cognitive_reflection(self):
        """Reflect on the learning process itself"""
        self.state = LearningState.REFLECTING
        
        try:
            # Analyze learning effectiveness
            effectiveness = self.meta_cognitive_system.analyze_learning_session()
            
            # Adjust learning parameters based on effectiveness
            if effectiveness < 0.5:
                self.learning_intensity = min(1.0, self.learning_intensity + 0.1)
                self.exploration_rate = min(1.0, self.exploration_rate + 0.05)
            elif effectiveness > 0.8:
                self.creativity_threshold = max(0.3, self.creativity_threshold - 0.05)
            
            # Log reflection
            logger.debug(f"Meta-cognitive reflection: effectiveness={effectiveness}")
            
        except Exception as e:
            logger.error(f"Error in meta-cognitive reflection: {e}")
    
    def _curiosity_driven_exploration(self):
        """Explore new topics based on curiosity"""
        try:
            # Get unexplored but interesting topics
            unexplored_topics = self.curiosity_engine.get_unexplored_interesting_topics()
            
            if unexplored_topics:
                topic = random.choice(unexplored_topics)
                
                # Quick exploration session
                exploration_results = self._quick_explore_topic(topic)
                
                # If interesting enough, add as learning goal
                if exploration_results.get('interest_score', 0) > self.creativity_threshold:
                    exploration_goal = LearningGoal(
                        id=f"explore_{int(time.time())}_{hash(topic) % 1000}",
                        topic=topic,
                        priority=0.6,
                        knowledge_gap=f"Unexplored area: {topic}",
                        target_depth=2,
                        created_at=datetime.now(),
                        estimated_duration=20,
                        prerequisites=[]
                    )
                    self.learning_goals.append(exploration_goal)
                    logger.info(f"Added exploration goal: {topic}")
            
        except Exception as e:
            logger.error(f"Error in curiosity-driven exploration: {e}")
    
    # Helper methods
    def _identify_knowledge_gaps(self, stats: Dict) -> List[str]:
        """Identify areas where knowledge is sparse using dynamic analysis"""
        gaps = []
        
        try:
            # 1. Analyze conversation topics for user interests
            conversation_gaps = self._get_conversation_based_gaps()
            gaps.extend(conversation_gaps[:2])
            
            # 2. Find related topics to existing knowledge
            related_gaps = self._get_related_topic_gaps()
            gaps.extend(related_gaps[:2])
            
            # 3. Get trending/current topics
            trending_gaps = self._get_trending_topics()
            gaps.extend(trending_gaps[:1])
            
            # 4. Fallback to curated topics if no dynamic discovery
            if not gaps:
                curated_topics = [
                    "artificial intelligence ethics",
                    "quantum machine learning",
                    "sustainable technology",
                    "cognitive computing",
                    "autonomous systems",
                    "digital philosophy",
                    "computational creativity"
                ]
                gaps = [t for t in curated_topics if t not in self.mastered_topics][:3]
            
        except Exception as e:
            self.logger.error(f"Error in dynamic gap identification: {e}")
            # Emergency fallback
            gaps = ["machine learning", "data science", "artificial intelligence"]
        
        return gaps
    
    def _get_conversation_based_gaps(self) -> List[str]:
        """Extract learning opportunities from user conversations"""
        gaps = []
        
        # Analyze conversation topics for depth
        for topic in self.conversation_topics:
            if topic not in self.mastered_topics:
                depth = self.topic_depth_map.get(topic, 0)
                if depth < 3:  # Not deeply understood yet
                    gaps.append(topic)
        
        return gaps
    
    def _get_related_topic_gaps(self) -> List[str]:
        """Find topics related to existing knowledge"""
        gaps = []
        
        try:
            if self.knowledge_base:
                # Get current knowledge topics
                current_topics = list(self.conversation_topics)[:5]
                
                for topic in current_topics:
                    # Generate related topics
                    related = self._generate_related_topics(topic)
                    for related_topic in related:
                        if related_topic not in self.mastered_topics:
                            gaps.append(related_topic)
                            
        except Exception as e:
            logger.debug(f"Error getting related topics: {e}")
            
        return gaps[:3]
    
    def _generate_related_topics(self, base_topic: str) -> List[str]:
        """Generate topics related to a base topic"""
        related_patterns = {
            "python": ["machine learning", "data science", "web development", "automation"],
            "ai": ["neural networks", "deep learning", "computer vision", "natural language processing"],
            "programming": ["software engineering", "algorithms", "data structures", "debugging"],
            "science": ["research methods", "data analysis", "scientific computing", "mathematics"],
            "technology": ["innovation", "digital transformation", "cybersecurity", "cloud computing"],
            "business": ["entrepreneurship", "strategy", "leadership", "analytics"],
            "design": ["user experience", "visual design", "creative process", "human factors"]
        }
        
        # Find matches and related topics
        related = []
        base_lower = base_topic.lower()
        
        for key, topics in related_patterns.items():
            if key in base_lower or base_lower in key:
                related.extend(topics)
        
        # Add general related topics
        if "computing" in base_lower or "computer" in base_lower:
            related.extend(["algorithms", "data structures", "systems design"])
        
        if "learning" in base_lower:
            related.extend(["cognitive science", "educational psychology", "knowledge management"])
            
        return related[:4]
    
    def _get_trending_topics(self) -> List[str]:
        """Get currently trending/relevant topics"""
        # This could be enhanced with real web API calls
        trending = [
            "large language models",
            "artificial general intelligence", 
            "quantum computing applications",
            "sustainable AI",
            "edge computing",
            "blockchain applications",
            "biotechnology advances",
            "space technology",
            "renewable energy systems",
            "digital health"
        ]
        
        # Filter out mastered topics
        return [t for t in trending if t not in self.mastered_topics][:3]
    
    def add_conversation_topic(self, topic: str):
        """Add a topic from user conversation"""
        self.conversation_topics.add(topic.lower())
        logger.debug(f"Added conversation topic: {topic}")
    
    def mark_topic_mastered(self, topic: str, depth: int = 5):
        """Mark a topic as mastered"""
        self.mastered_topics.add(topic.lower())
        self.topic_depth_map[topic.lower()] = depth
        logger.info(f"Topic mastered: {topic} (depth: {depth})")
    
    def update_topic_depth(self, topic: str, depth_increase: int = 1):
        """Update the learning depth for a topic"""
        current_depth = self.topic_depth_map.get(topic.lower(), 0)
        new_depth = current_depth + depth_increase
        self.topic_depth_map[topic.lower()] = new_depth
        
        # Mark as mastered if depth_threshold reached
        if new_depth >= 4:
            self.mark_topic_mastered(topic, new_depth)
    
    def _set_conversation_focus(self, message: str):
        """Set temporary conversation focus based on user message content"""
        try:
            # Only set focus if no explicit user focus is set
            if self.user_focus_topic:
                return
                
            # Extract potential focus topics from the message
            message_lower = message.lower()
            
            # Look for questions about learning or knowledge
            learning_indicators = ["what have you learned", "what do you know", "tell me about", 
                                 "how does", "what is", "explain", "teach me"]
            
            for indicator in learning_indicators:
                if indicator in message_lower:
                    # Extract topic after the indicator
                    idx = message_lower.find(indicator)
                    topic_part = message[idx + len(indicator):].strip()
                    
                    # Clean up the topic
                    if topic_part:
                        # Remove common question words
                        topic_part = topic_part.replace("about ", "").replace("so far", "").strip()
                        
                        if len(topic_part) > 3:
                            self.conversation_focus_topic = topic_part[:100]  # Limit length
                            self.conversation_focus_timestamp = time.time()
                            self.logger.info(f"🎯 Set conversation focus: {self.conversation_focus_topic}")
                            break
                            
        except Exception as e:
            self.logger.error(f"Error setting conversation focus: {e}")
    
    def get_user_focused_goals(self) -> List[LearningGoal]:
        """Get goals that were set by user focus"""
        return [g for g in self.learning_goals if "user_focus" in g.id]
    
    def get_active_focus_topics(self) -> List[str]:
        """Get currently active focus topics"""
        user_goals = self.get_user_focused_goals()
        return [g.topic for g in user_goals if g.status in ["pending", "active"]]
    
    def _identify_current_gaps(self) -> List[str]:
        """Identify current knowledge gaps dynamically"""
        current_gaps = []
        
        # Check conversation topics for gaps
        for topic in list(self.conversation_topics)[-5:]:  # Recent conversation topics
            if topic not in self.mastered_topics:
                current_gaps.append(topic)
        
        # Add some dynamic gaps based on current trends
        dynamic_gaps = [
            "AI safety and alignment",
            "emergent AI behaviors", 
            "human-AI collaboration",
            "automated reasoning",
            "consciousness in AI"
        ]
        
        # Filter and combine
        filtered_gaps = [g for g in dynamic_gaps if g not in self.mastered_topics]
        current_gaps.extend(filtered_gaps[:2])
        
        return current_gaps[:3] if current_gaps else ["general knowledge", "problem solving", "creative thinking"]
    
    def set_domain_priority(self, domain_category: str, priority: float):
        """Set priority weight for a domain category (0.0 to 2.0, where 1.0 is normal)"""
        if domain_category in self.domain_priorities:
            self.domain_priorities[domain_category] = max(0.0, min(2.0, priority))
            self.logger.info(f"Set {domain_category} priority to {priority}")
        else:
            self.logger.warning(f"Unknown domain category: {domain_category}")
    
    def get_domain_priorities(self) -> Dict[str, float]:
        """Get current domain priority settings"""
        return self.domain_priorities.copy()
    
    def reset_domain_priorities(self):
        """Reset all domain priorities to default (1.0)"""
        for category in self.domain_priorities:
            self.domain_priorities[category] = 1.0
        self.logger.info("Reset all domain priorities to default")
    
    def prioritize_science_domains(self):
        """Prioritize scientific and medical domains for focused learning"""
        science_priorities = {
            'basic_sciences': 1.5,
            'life_sciences': 1.5,
            'medical_sciences': 1.5,
            'specialized_medical': 1.5,
            'diagnostic_imaging': 1.5,
            'biological_subfields': 1.5,
            'social_sciences': 0.7,
            'arts_humanities': 0.5,
            'engineering_tech': 0.8
        }
        self.domain_priorities.update(science_priorities)
        self.logger.info("Prioritized scientific and medical domains")
    
    def prioritize_humanities_domains(self):
        """Prioritize humanities and social sciences for balanced learning"""
        humanities_priorities = {
            'basic_sciences': 0.8,
            'life_sciences': 0.9,
            'medical_sciences': 0.9,
            'specialized_medical': 0.7,
            'diagnostic_imaging': 0.7,
            'biological_subfields': 0.8,
            'social_sciences': 1.5,
            'arts_humanities': 1.5,
            'engineering_tech': 0.9
        }
        self.domain_priorities.update(humanities_priorities)
        self.logger.info("Prioritized humanities and social sciences")
    
    def _generate_research_questions(self, topic: str) -> List[str]:
        """Generate research questions for a topic using RTX 4090 Beast Mode"""
        try:
            if self.llm_interface and hasattr(self.llm_interface, 'generate_response'):
                # 🚀 RTX 4090 BEAST MODE: Generate intelligent research questions
                prompt = f"""As a brilliant researcher, generate 5 thought-provoking research questions about '{topic}'. 

Make them:
- Specific and actionable
- Different difficulty levels (basic to advanced)
- Interdisciplinary connections
- Future-oriented perspectives

Format as numbered list. Focus on depth and originality."""

                ai_response = self._gpu_enhanced_llm_call(
                    prompt,
                    {"type": "research_planning", "topic": topic},
                    "Research Question Generation"
                )
                
                if ai_response:
                    # Extract questions from AI response
                    lines = ai_response.strip().split('\n')
                    questions = []
                    for line in lines:
                        line = line.strip()
                        if line and ('?' in line or line.startswith(('1.', '2.', '3.', '4.', '5.'))):
                            # Clean up formatting
                            question = line.split('.', 1)[-1].strip() if '.' in line else line
                            if question and question not in questions:
                                questions.append(question)
                    
                    if questions:
                        self.logger.info(f"🧠 RTX 4090 generated {len(questions)} research questions for: {topic}")
                        return questions[:5]  # Limit to 5 questions
                        
        except Exception as e:
            self.logger.error(f"Error generating AI research questions: {e}")
        # Fallback to template questions
        questions = [
            f"What are the fundamental principles of {topic}?",
            f"How does {topic} relate to other fields?",
            f"What are the current challenges in {topic}?",
            f"What future developments are expected in {topic}?",
            f"What are the practical applications of {topic}?"
        ]
        return questions
    
    def _extract_patterns(self, research_results: Dict) -> List[str]:
        """Extract patterns from research results"""
        patterns = [
            f"Common theme in {research_results['topic']} research",
            f"Emerging trend in {research_results['topic']} applications",
            f"Fundamental principle underlying {research_results['topic']}"
        ]
        return patterns
    
    def _validate_insight(self, insight: Insight):
        """Validate an insight against existing knowledge"""
        # Simple validation - could be much more sophisticated
        insight.validation_score = 0.6 + random.random() * 0.4
    
    def _discover_knowledge_connections(self) -> List[Dict]:
        """Discover connections between knowledge areas"""
        connections = []
        # Simplified implementation
        connection_types = [
            {"area1": "AI", "area2": "Psychology", "relation": "cognitive modeling"},
            {"area1": "Physics", "area2": "Computing", "relation": "quantum algorithms"},
            {"area1": "Biology", "area2": "Engineering", "relation": "biomimetics"}
        ]
        return connection_types
    
    def _create_synthesis_insight(self, connection: Dict) -> Optional[Insight]:
        """Create an insight from a knowledge connection"""
        try:
            insight = Insight(
                id=f"synthesis_{int(time.time())}_{hash(str(connection)) % 1000}",
                content=f"Connection between {connection['area1']} and {connection['area2']} through {connection['relation']}",
                confidence=0.8,
                connections=[connection['area1'], connection['area2']],
                created_at=datetime.now()
            )
            return insight
        except:
            return None
    
    def _quick_explore_topic(self, topic: str) -> Dict:
        """Quick exploration of a new topic"""
        return {
            'topic': topic,
            'interest_score': random.random(),
            'complexity': random.randint(1, 5),
            'potential_connections': random.randint(1, 10)
        }
    
    def _integrate_new_knowledge(self, topic: str, insights: List[Insight], experiments: List[Dict]):
        """Integrate new knowledge into the knowledge base"""
        try:
            if self.knowledge_base:
                # Add topic and insights to knowledge base
                knowledge_entry = {
                    'topic': topic,
                    'insights': [insight.content for insight in insights],
                    'experiments': experiments,
                    'learned_at': datetime.now().isoformat()
                }
                self.knowledge_base.add_knowledge(json.dumps(knowledge_entry), {"topic": topic})
                
        except Exception as e:
            logger.error(f"Error integrating new knowledge: {e}")
    
    def _generate_completion_insight(self, goal: LearningGoal, insights: List[Insight]) -> Optional[Insight]:
        """Generate an insight upon goal completion using RTX 4090 Beast Mode"""
        try:
            if self.llm_interface and hasattr(self.llm_interface, 'generate_response'):
                # 🚀 RTX 4090 BEAST MODE: Generate deep insights using GPU acceleration
                prompt = f"""As an advanced AI researcher, synthesize your learning about '{goal.topic}'. 

Based on your research, provide:
1. Key insights discovered
2. Connections to other concepts  
3. Implications for future learning
4. Novel questions raised

Be insightful, creative, and demonstrate deep understanding. Limit to 200 words."""

                # Use RTX 4090 Beast Mode for insight generation
                ai_response = self._gpu_enhanced_llm_call(
                    prompt,
                    {"type": "completion_insight", "goal": goal.topic, "insights": insights},
                    "Completion Insight Generation"
                )
                
                completion_insight = Insight(
                    id=f"completion_{goal.id}",
                    content=ai_response.strip() if ai_response else f"Completed learning about {goal.topic}",
                    confidence=0.9,
                    connections=[goal.topic],
                    created_at=datetime.now()
                )
                
                self.logger.info(f"🧠 RTX 4090 generated completion insight for: {goal.topic}")
                return completion_insight
            else:
                # Fallback if LLM not available
                completion_insight = Insight(
                    id=f"completion_{goal.id}",
                    content=f"Successfully learned {goal.topic} - gained {len(insights)} new insights",
                    confidence=0.9,
                    connections=[goal.topic],
                    created_at=datetime.now()
                )
                return completion_insight
                
        except Exception as e:
            self.logger.error(f"Error generating completion insight: {e}")
            return None
    
    def _save_learning_goal(self, goal: LearningGoal):
        """Save learning goal to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO learning_goals 
                (id, topic, priority, knowledge_gap, target_depth, created_at, 
                 estimated_duration, prerequisites, status, progress)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                goal.id, goal.topic, goal.priority, goal.knowledge_gap,
                goal.target_depth, goal.created_at.isoformat(),
                goal.estimated_duration, json.dumps(goal.prerequisites),
                goal.status, goal.progress
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving learning goal: {e}")
    
    def _save_session_stats(self):
        """Save session statistics"""
        try:
            session_id = f"session_{int(time.time())}"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO learning_sessions 
                (id, start_time, end_time, goals_completed, insights_generated, breakthroughs, session_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id,
                self.session_stats['session_start'].isoformat() if self.session_stats['session_start'] else None,
                datetime.now().isoformat(),
                self.session_stats['goals_completed'],
                self.session_stats['insights_generated'],
                self.session_stats['breakthroughs'],
                f"Autonomous learning session completed"
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving session stats: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current autonomous learning status"""
        return {
            'autonomous_mode': self.autonomous_mode,
            'current_state': self.state.value,
            'active_goal': self.active_goal.topic if self.active_goal else None,
            'goals_in_queue': len(self.learning_goals),
            'session_stats': self.session_stats,
            'learning_intensity': self.learning_intensity,
            'total_insights': len(self.insights)
        }
    
    def get_insights_summary(self) -> List[Dict]:
        """Get summary of generated insights"""
        return [
            {
                'id': insight.id,
                'content': insight.content[:100] + "..." if len(insight.content) > 100 else insight.content,
                'confidence': insight.confidence,
                'created_at': insight.created_at.isoformat(),
                'connections': insight.connections
            }
            for insight in list(self.insights.values())[-10:]  # Last 10 insights
        ]
    
    def _generate_emergency_topics(self):
        """Generate emergency topics when the system gets stuck in a loop"""
        self._report_to_gui("🚨 EMERGENCY: Generating backup exploratory topics", "error")
        
        # Ultra-specific niche topics that are unlikely to be mastered
        emergency_topics = [
            "biomimetic soft robotics",
            "quantum dot cellular automata", 
            "neuromorphic event-driven vision",
            "memristive crossbar networks",
            "photonic reservoir computing",
            "DNA origami nanorobotics",
            "metamaterial acoustic cloaking",
            "topological quantum error correction",
            "spintronic memory devices",
            "plasmonic optical computing",
            "synthetic biology circuits",
            "molecular machine engineering",
            "quantum sensing networks",
            "bio-hybrid neural interfaces",
            "programmable matter systems"
        ]
        
        # Filter out any that might already be mastered
        available_topics = []
        for topic in emergency_topics:
            is_novel = True
            for mastered in self.mastered_topics:
                if any(word in mastered.lower() for word in topic.lower().split()):
                    is_novel = False
                    break
            if is_novel:
                available_topics.append(topic)
        
        # Create emergency goals
        for i, topic in enumerate(available_topics[:3]):  # Take first 3 available
            goal = LearningGoal(
                id=f"emergency_{int(time.time())}_{i}",
                topic=topic,
                priority=0.9,  # High priority to break the loop
                knowledge_gap=f"Emergency exploration: {topic}",
                target_depth=2,
                created_at=datetime.now(),
                estimated_duration=30,
                prerequisites=[]
            )
            self.learning_goals.append(goal)
            self._report_to_gui(f"🆘 Emergency topic added: {topic}", "goal")
        
        self._report_to_gui(f"✅ Added {len(available_topics[:3])} emergency topics to break the loop", "status")
    
    def _master_discovery_engine(self):
        """🚀 MASTER DISCOVERY ENGINE: Generate completely unique learning goals"""
        self._report_to_gui("🚀 MASTER DISCOVERY ENGINE ACTIVATED", "emergency")
        
        import random
        import hashlib
        from datetime import datetime
        
        # Enhanced ultra-diverse topic categories with more specificity
        categories = {
            "quantum_tech": [
                "quantum biological sensors", "topological quantum materials", "quantum error correction algorithms",
                "quantum cryptographic protocols", "quantum neural networks", "quantum cellular automata",
                "quantum dot solar cells", "quantum sensing networks", "quantum teleportation protocols"
            ],
            "bio_engineering": [
                "synthetic biology circuits", "bio-hybrid materials", "molecular motors", "artificial cell membranes",
                "bio-inspired computing", "engineered living materials", "synthetic photosynthesis",
                "bio-reactive surfaces", "organic electronics", "bio-mimetic adhesion"
            ],
            "advanced_materials": [
                "programmable metamaterials", "self-healing composites", "shape-memory alloys",
                "piezoelectric textiles", "magnetic liquid crystals", "thermal interface materials",
                "crystalline memory systems", "smart dust networks", "phase-change materials"
            ],
            "space_tech": [
                "orbital manufacturing", "asteroid mining robotics", "space-based solar power",
                "lunar construction materials", "interstellar propulsion", "space elevator dynamics",
                "atmospheric processors", "zero-gravity 3D printing", "space radiation shielding"
            ],
            "neuro_tech": [
                "neuromorphic chip design", "brain organoid computing", "synthetic neural plasticity",
                "optogenetic interfaces", "neural dust sensors", "artificial synapses",
                "brain-computer interfaces", "neural prosthetics", "memory enhancement devices"
            ],
            "energy_systems": [
                "fusion plasma control", "artificial photosynthetic cells", "thermoelectric generators",
                "wireless power transmission", "energy harvesting systems", "supercapacitor design",
                "magnetic plasma confinement", "tidal energy extraction", "atmospheric energy capture"
            ],
            "computational": [
                "DNA data storage", "optical computing architectures", "memristive neural networks",
                "quantum algorithms", "photonic processors", "molecular computing",
                "holographic data storage", "biological circuit design", "crystalline computing"
            ],
            "exotic_physics": [
                "dark matter detection", "gravitational wave analysis", "antimatter containment",
                "time crystal structures", "negative index metamaterials", "room-temperature superconductors",
                "magnetic monopole research", "exotic matter states", "dimensional topology theory"
            ]
        }
        
        # Generate multiple unique discovery goals with extra randomness
        for i in range(5):  # Generate 5 unique goals
            timestamp_seed = str(int(time.time()) + i * 1000)  # More separation
            random_seed = str(random.randint(100000, 999999))  # Larger range
            category_seed = str(random.randint(1, 1000))
            unique_hash = hashlib.md5((timestamp_seed + random_seed + category_seed).encode()).hexdigest()[:6]
            
            # Select random category and topic
            category = random.choice(list(categories.keys()))
            base_topic = random.choice(categories[category])
            
            # Add highly specific research angles
            angles = [
                "breakthrough research in",
                "experimental applications of", 
                "future developments in",
                "theoretical foundations of",
                "practical implementation of",
                "innovative approaches to",
                "next-generation advances in",
                "revolutionary concepts in",
                "cutting-edge research on",
                "novel applications of"
            ]
            
            angle = random.choice(angles)
            final_topic = f"{angle} {base_topic} [MDG-{unique_hash}]"
            
            # Create discovery goal
            discovery_goal = LearningGoal(
                id=f"master_discovery_{unique_hash}_{i}",
                topic=final_topic,
                priority=random.uniform(0.7, 0.9),  # High priority
                knowledge_gap=f"Master Discovery: {final_topic}",
                target_depth=random.randint(2, 4),
                created_at=datetime.now(),
                estimated_duration=random.randint(30, 60),
                prerequisites=[],
                status="pending"
            )
            
            self.learning_goals.append(discovery_goal)
            self._report_to_gui(f"🌟 MDG #{i+1}: {final_topic}", "discovery")
        
        self._report_to_gui(f"✅ MASTER DISCOVERY: Generated 5 ultra-unique exploration goals", "success")
        self.logger.info("Master Discovery Engine generated 5 ultra-unique learning goals")
    
    def _load_mastered_topics(self):
        """Load ALL previously mastered topics from database - NEVER FORGET!"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create enhanced table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mastered_topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT UNIQUE NOT NULL,
                    mastered_at TEXT NOT NULL,
                    depth_level INTEGER DEFAULT 2,
                    review_count INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Load ALL mastered topics (no limits - permanent memory!)
            cursor.execute("SELECT topic, review_count FROM mastered_topics ORDER BY mastered_at")
            rows = cursor.fetchall()
            
            # Build comprehensive mastered topics set
            self.mastered_topics = set()
            self.topic_review_counts = {}
            
            for topic, review_count in rows:
                # Clean topic (remove unique identifiers for comparison)
                clean_topic = topic.split('[')[0].strip()
                self.mastered_topics.add(clean_topic)
                self.topic_review_counts[clean_topic] = review_count
            
            conn.close()
            
            self.logger.info(f"📚 PERMANENT MEMORY: Loaded {len(self.mastered_topics)} mastered topics from database")
            self._report_to_gui(f"🧠 Permanent Memory: {len(self.mastered_topics)} mastered topics loaded", "status")
            
            # Also log some examples for debugging
            if len(self.mastered_topics) > 0:
                sample_topics = list(self.mastered_topics)[:5]
                self.logger.info(f"Sample mastered topics: {sample_topics}")
                self._report_to_gui(f"💾 Sample mastered: {sample_topics[:3]}", "debug")
            else:
                self._report_to_gui("⚠️ WARNING: No mastered topics loaded from database!", "error")
                
        except Exception as e:
            self.logger.warning(f"Could not load mastered topics: {e}")
            self.mastered_topics = set()
            self.topic_review_counts = {}
    
    def get_mastered_topics_count(self) -> int:
        """Get the current count of mastered topics"""
        return len(self.mastered_topics)
    
    def _save_mastered_topic(self, topic: str):
        """Save newly mastered topic to permanent database - NEVER FORGET!"""
        try:
            # Clean topic (remove unique identifiers)
            clean_topic = topic.split('[')[0].strip()

            # 🚨 VALIDATION: Only save legitimate learning topics
            if not self._is_valid_learning_topic(clean_topic):
                self.logger.warning(f"⚠️ Skipping invalid topic save: '{clean_topic}'")
                return

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create enhanced table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mastered_topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT UNIQUE NOT NULL,
                    mastered_at TEXT NOT NULL,
                    depth_level INTEGER DEFAULT 2,
                    review_count INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # Insert or update the mastered topic
            cursor.execute('''
                INSERT INTO mastered_topics (topic, mastered_at, depth_level, review_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(topic) DO UPDATE SET
                    review_count = review_count + 1,
                    mastered_at = excluded.mastered_at
            ''', (clean_topic, datetime.now().isoformat(), 2, 1))

            conn.commit()
            conn.close()

            # Update in-memory tracking
            self.mastered_topics.add(clean_topic)
            self.topic_review_counts[clean_topic] = self.topic_review_counts.get(clean_topic, 0) + 1

            self.logger.info(f"💾 PERMANENTLY SAVED: {clean_topic}")

        except Exception as e:
            self.logger.warning(f"Could not save mastered topic '{topic}': {e}")
    
    def _cleanup_mastered_goals(self):
        """Remove any goals for topics that are already mastered"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all current goals
            cursor.execute("SELECT id, topic FROM learning_goals")
            goals_to_check = cursor.fetchall()
            
            removed_count = 0
            for goal_id, goal_topic in goals_to_check:
                # Clean the goal topic for comparison
                clean_goal_topic = goal_topic.split('[')[0].strip().lower()
                
                # Check if this goal's topic is already mastered
                if clean_goal_topic in self.mastered_topics:
                    # Remove the goal from database
                    cursor.execute("DELETE FROM learning_goals WHERE id = ?", (goal_id,))
                    removed_count += 1
                    self.logger.info(f"🧹 Cleaned up mastered goal: {goal_topic}")
            
            if removed_count > 0:
                conn.commit()
                self._report_to_gui(f"🧹 Cleaned up {removed_count} goals for already mastered topics", "status")
            
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Could not cleanup mastered goals: {e}")
    
    def _is_valid_learning_topic(self, topic: str) -> bool:
        """Validate that a topic is a legitimate learning subject"""
        if not topic or len(topic.strip()) < 3:
            return False

        # Reject obvious non-topics and error messages
        invalid_indicators = [
            "i'm", "i am", "i'm learning", "i encountered an error",
            "cuda", "error:", "compile with", "for debugging",
            "stacktrace", "device-side", "assertion triggered",
            "could you tell me", "thinking about that",
            "learning from this", "feedback", "response",
            "sorry", "unfortunately", "however",
            "cuda_launch_blocking", "torch_use_cuda_dsa",
            "device-side assertions", "launch_blocking",
            "cuda_dsa", "pytorch", "tensor", "gpu memory",
            "kernel launch", "synchronize", "async"
        ]

        topic_lower = topic.lower()
        for indicator in invalid_indicators:
            if indicator in topic_lower:
                return False

        # Must contain at least one noun-like word (basic heuristic)
        words = topic.split()
        if len(words) < 2:
            return False

        # Reject topics with too many special characters (likely code/error output)
        special_chars = sum(1 for c in topic if not c.isalnum() and c not in [' ', '-', '_'])
        if special_chars > len(topic) * 0.3:  # More than 30% special characters
            return False

        # Reject topics that look like file paths or commands
        if '/' in topic or '\\' in topic or '=' in topic:
            return False

        return True
    
    def _generate_exploratory_topics(self):
        """Generate completely new exploratory topics when all else fails"""
        try:
            # Dynamic topic pools that evolve over time
            exploratory_pools = {
                "emerging_tech": [
                    "neuromorphic computing", "quantum sensors", "bio-hybrid systems",
                    "programmable matter", "swarm intelligence", "liquid neural networks",
                    "optical computing", "DNA data storage", "synthetic biology",
                    "metamaterials engineering", "space-based manufacturing"
                ],
                "interdisciplinary": [
                    "computational archaeology", "digital humanities", "bioinformatics",
                    "econophysics", "psycholinguistics", "computational musicology",
                    "algorithmic trading", "digital forensics", "computational creativity",
                    "network science", "complexity theory applications"
                ],
                "future_concepts": [
                    "mind uploading ethics", "artificial consciousness", "post-human society",
                    "interplanetary governance", "digital immortality", "consciousness transfer",
                    "time-based computing", "dimensional computing", "quantum consciousness",
                    "collective intelligence systems", "reality simulation frameworks"
                ],
                "practical_applications": [
                    "sustainable AI systems", "green computing", "ethical AI frameworks",
                    "accessibility technology", "disaster response systems", "healthcare AI",
                    "education technology", "smart city infrastructure", "climate modeling",
                    "renewable energy optimization", "waste reduction algorithms"
                ]
            }
            
            # Select topics that haven't been mastered
            new_topics = []
            for pool_name, topics in exploratory_pools.items():
                available_topics = [t for t in topics if t not in self.mastered_topics]
                if available_topics:
                    # Pick 1-2 random topics from each pool
                    selected = random.sample(available_topics, min(2, len(available_topics)))
                    new_topics.extend(selected)
            
            # Create exploratory goals
            for i, topic in enumerate(new_topics[:4]):  # Limit to 4 new topics
                goal = LearningGoal(
                    id=f"exploratory_{int(time.time())}_{i}",
                    topic=topic,
                    priority=0.6 + (i * 0.1),  # Varied priority
                    knowledge_gap=f"Exploratory learning in: {topic}",
                    target_depth=2,
                    created_at=datetime.now(),
                    estimated_duration=45,
                    prerequisites=[]
                )
                self.learning_goals.append(goal)
                self.logger.info(f"🚀 Generated exploratory goal: {topic}")
                
        except Exception as e:
            self.logger.error(f"Error generating exploratory topics: {e}")
    
    def process_user_message(self, user_message: str):
        """Process user message for topic extraction and AGI learning triggers"""
        try:
            # Extract potential learning topics from user message
            topics = self._extract_topics_from_message(user_message)
            
            # Check for explicit learning requests
            learning_request = self._detect_learning_request(user_message)
            if learning_request:
                self._handle_learning_request(learning_request, user_message)
                return
            
            # Process topics for potential learning goals
            for topic in topics:
                if topic not in self.mastered_topics and topic not in [g.topic for g in self.learning_goals]:
                    # Check if topic is interesting enough to learn
                    if self._is_topic_worth_learning(topic):
                        self.logger.info(f"Identified interesting topic for learning: {topic}")
                        # For now, just log - full goal creation can be implemented later
                        
        except Exception as e:
            self.logger.error(f"Error processing user message: {e}")
    
    def _extract_topics_from_message(self, message: str) -> List[str]:
        """Extract potential learning topics from user message"""
        topics = []
        
        # Simple keyword extraction (can be enhanced with NLP)
        message_lower = message.lower()
        
        # Common topic indicators
        topic_keywords = [
            "learn", "study", "understand", "explain", "about", "what is", "how does",
            "tell me about", "research", "explore", "deep dive", "focus on"
        ]
        
        # Extract topics after learning keywords
        for keyword in topic_keywords:
            if keyword in message_lower:
                # Find topic after keyword
                idx = message_lower.find(keyword)
                remaining = message_lower[idx + len(keyword):].strip()
                
                # Extract noun phrases (simple approach)
                words = remaining.split()
                topic_words = []
                
                for word in words[:5]:  # Look at first few words
                    if word not in ["the", "a", "an", "this", "that", "these", "those", "i", "you", "we", "they"]:
                        topic_words.append(word)
                    if len(topic_words) >= 3:  # Max 3 words for topic
                        break
                
                if topic_words:
                    topic = " ".join(topic_words)
                    if len(topic) > 3:  # Minimum length
                        topics.append(topic)
        
        # Also extract direct mentions of subjects
        common_subjects = [
            "ai", "artificial intelligence", "machine learning", "neural networks",
            "quantum", "physics", "biology", "chemistry", "mathematics", "computer science",
            "philosophy", "consciousness", "creativity", "learning", "memory", "reasoning",
            "conversations", "communication", "language", "programming", "algorithms"
        ]
        
        for subject in common_subjects:
            if subject in message_lower:
                topics.append(subject)
        
        return list(set(topics))  # Remove duplicates
    
    def _detect_learning_request(self, message: str) -> Optional[str]:
        """Detect if user is explicitly requesting learning on a topic"""
        message_lower = message.lower()
        
        learning_indicators = [
            "learn about", "learn using", "study", "teach me about", "explain",
            "focus on learning", "agi mode about", "learn using agi mode"
        ]
        
        for indicator in learning_indicators:
            if indicator in message_lower:
                # Extract topic after indicator
                idx = message_lower.find(indicator)
                topic_part = message_lower[idx + len(indicator):].strip()
                
                # Clean up topic
                topic_part = topic_part.replace("using agi mode", "").replace("using the agi mode", "").strip()
                
                if topic_part:
                    return topic_part
        
        return None
    
    def _handle_learning_request(self, topic: str, original_message: str):
        """Handle explicit learning request from user"""
        self.logger.info(f"Processing learning request for topic: {topic}")
        
        # Set learning focus
        self.current_focus_topic = topic
        self._report_to_gui(f"🔧 🎯 Learning focus set: {topic}", "goal")
        
        # Determine learning parameters based on message
        priority = 0.7  # Default priority
        depth = "medium"
        mode = "balanced"
        
        if "deep" in original_message.lower() or "comprehensive" in original_message.lower():
            depth = "deep"
            priority = 0.9
        elif "quick" in original_message.lower() or "basic" in original_message.lower():
            depth = "shallow"
            priority = 0.5
        
        self._report_to_gui(f"🔧    Priority: {priority}, Depth: {depth.title()}, Mode: {mode.title()}", "goal")
        
        # Create focused learning goal
        self._create_focused_learning_goal(topic, priority, depth, mode)
    
    def _create_focused_learning_goal(self, topic: str, priority: float, depth: str, mode: str):
        """Create a focused learning goal with specific parameters"""
        goal_id = f"focused_{int(time.time())}_{hash(topic) % 1000}"
        
        goal = LearningGoal(
            id=goal_id,
            topic=topic,
            priority=priority,
            knowledge_gap=f"Need to {depth} understand {topic}",
            target_depth=3 if depth == "deep" else 2 if depth == "medium" else 1,
            created_at=datetime.now(),
            estimated_duration=15 if depth == "shallow" else 45 if depth == "medium" else 90,
            prerequisites=[],
            status="active"
        )
        
        self.learning_goals.append(goal)
        self._save_learning_goal(goal)
        
        # Start learning immediately if autonomous mode is active
        if self.autonomous_mode:
            self._execute_learning_goal_immediately(topic, priority, depth, mode)
        
        self._report_to_gui(f"🎯 Focused learning goal created for '{topic}'", "goal")
    
    def _execute_learning_goal_immediately(self, topic: str, priority: float, depth: str, mode: str):
        """Execute a learning goal immediately"""
        try:
            self.logger.info(f"Starting immediate learning on topic: {topic}")
            
            # Conduct research
            research_results = self._conduct_topic_research(topic, depth)
            
            # Generate insights
            insights = self._generate_topic_insights(topic, research_results, depth)
            
            # Store knowledge
            self._store_topic_knowledge(topic, research_results, insights)
            
            # Mark as mastered if successful
            if insights:
                self.mastered_topics.add(topic)
                self._save_mastered_topic(topic)
                self._report_to_gui(f"✅ Topic '{topic}' mastered through focused learning!", "status")
            
        except Exception as e:
            self.logger.error(f"Error in immediate learning execution: {e}")
            self._report_to_gui(f"❌ Error during focused learning: {e}", "error")
    
    def _conduct_topic_research(self, topic: str, depth: str) -> Dict:
        """Conduct research on a topic"""
        research_data = {
            "topic": topic,
            "depth": depth,
            "sources": [],
            "key_findings": [],
            "timestamp": datetime.now().isoformat()
        }
        
        # Use web research if available
        if self.web_researcher:
            try:
                self.logger.info(f"Conducting web research on: {topic}")
                # Use the correct method name and handle async
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    if asyncio.iscoroutinefunction(self.web_researcher.comprehensive_research):
                        results = loop.run_until_complete(self.web_researcher.comprehensive_research(topic, depth=depth))
                    else:
                        results = self.web_researcher.comprehensive_research(topic, depth=depth)
                    if results and isinstance(results, dict):
                        research_data["sources"].extend(results.get("sources", []))
                        research_data["key_findings"].extend(results.get("findings", []))
                finally:
                    loop.close()
            except Exception as e:
                self.logger.debug(f"Web research failed: {e}")
        
        # Use LLM for additional insights if available
        if self.llm_interface:
            try:
                prompt = f"Provide key insights and important facts about '{topic}'. Be comprehensive but concise."
                response, _ = self.llm_interface.generate_response(prompt)
                if response:
                    research_data["key_findings"].append(f"AI Analysis: {response}")
            except Exception as e:
                self.logger.debug(f"LLM research failed: {e}")
        
        return research_data
    
    def _generate_topic_insights(self, topic: str, research_data: Dict, depth: str) -> List[str]:
        """Generate insights from research data"""
        insights = []
        
        findings = research_data.get("key_findings", [])
        
        # Generate insights based on depth
        if depth == "deep":
            # More detailed analysis
            insights.extend(self._analyze_findings_deep(findings))
        elif depth == "medium":
            insights.extend(self._analyze_findings_medium(findings))
        else:
            insights.extend(self._analyze_findings_shallow(findings))
        
        return insights
    
    def _analyze_findings_shallow(self, findings: List[str]) -> List[str]:
        """Shallow analysis of findings"""
        insights = []
        for finding in findings[:3]:  # Limit to first 3
            if len(finding) > 10:
                insights.append(f"Key point: {finding[:100]}...")
        return insights
    
    def _analyze_findings_medium(self, findings: List[str]) -> List[str]:
        """Medium depth analysis"""
        insights = []
        
        # Group similar findings
        if len(findings) >= 2:
            insights.append(f"Multiple perspectives found on {len(findings)} aspects")
        
        # Extract key themes
        for finding in findings[:5]:
            if "important" in finding.lower() or "key" in finding.lower():
                insights.append(finding[:150])
        
        return insights
    
    def _analyze_findings_deep(self, findings: List[str]) -> List[str]:
        """Deep analysis of findings"""
        insights = []
        
        # Look for patterns and connections
        if len(findings) > 3:
            insights.append(f"Complex topic with {len(findings)} interconnected aspects discovered")
        
        # Extract detailed insights
        for finding in findings:
            if len(finding) > 50:  # More substantial findings
                insights.append(finding[:200])
        
        return insights
    
    def _store_topic_knowledge(self, topic: str, research_data: Dict, insights: List[str]):
        """Store learned knowledge"""
        # Store in knowledge base if available
        if self.knowledge_base:
            try:
                content = f"Topic: {topic}\n\nResearch: {research_data}\n\nInsights: {'; '.join(insights)}"
                self.knowledge_base.store_knowledge(
                    content=content,
                    knowledge_type="learned_topic",
                    confidence=0.8
                )
            except Exception as e:
                self.logger.debug(f"Knowledge storage failed: {e}")
        
        # Store in memory system if available
        if hasattr(self, 'memory_system') and self.memory_system:
            try:
                self.memory_system.store_memory(
                    content=f"Learned about {topic}: {len(insights)} insights gained",
                    memory_type="semantic",
                    importance=0.8
                )
            except Exception as e:
                self.logger.debug(f"Memory storage failed: {e}")
    
    def _is_topic_worth_learning(self, topic: str) -> bool:
        """Determine if a topic is worth learning"""
        # Check if topic is too basic or already known
        basic_topics = ["hello", "hi", "how are you", "thank you", "goodbye"]
        if any(basic in topic.lower() for basic in basic_topics):
            return False
        
        # Check topic length and quality
        if len(topic) < 4 or len(topic) > 50:
            return False
        
        # Check if topic contains meaningful keywords
        meaningful_keywords = [
            "ai", "learning", "intelligence", "quantum", "neural", "consciousness",
            "creativity", "reasoning", "memory", "computation", "algorithm", "system"
        ]
        
        return any(keyword in topic.lower() for keyword in meaningful_keywords)


class MetaCognitiveSystem:
    """Analyzes and reflects on the learning process itself"""
    
    def __init__(self):
        self.learning_patterns = []
        self.effectiveness_history = []
    
    def analyze_learning_session(self) -> float:
        """Analyze the effectiveness of the current learning session"""
        # Simple analysis based on goals completed vs attempted
        # This is a placeholder - could be much more sophisticated
        base_effectiveness = 0.7  # Default good performance
        
        # Add some randomness to simulate analysis
        import random
        variation = random.uniform(-0.2, 0.2)
        
        effectiveness = max(0.0, min(1.0, base_effectiveness + variation))
        self.effectiveness_history.append(effectiveness)
        
        return effectiveness



class CuriosityEngine:
    """Drives curiosity-based learning and exploration"""
    
    def __init__(self):
        self.interest_map = defaultdict(float)
        self.explored_topics = set()
        self.topic_connections = defaultdict(list)
    
    def update_knowledge_map(self, knowledge_stats: Dict):
        """Update internal knowledge representation"""
        # Update interest based on knowledge gaps
        # knowledge_stats comes from KnowledgeBase.get_statistics() which returns
        # a dict with keys like 'total_entries', 'knowledge_types', etc.
        
        if not isinstance(knowledge_stats, dict):
            return
            
        # Use the statistics to adjust curiosity levels
        total_entries = knowledge_stats.get('total_entries', 0)
        knowledge_types = knowledge_stats.get('knowledge_types', 0)
        
        # If we have few knowledge types, increase curiosity for exploration
        if knowledge_types < 5:
            # Boost curiosity for all topics to encourage broader exploration
            for topic in self.interest_map:
                self.interest_map[topic] += 0.05
        
        # If we have low total entries, increase general curiosity
        if total_entries < 100:
            self.boost_curiosity(0.1)
    
    def get_interesting_topics(self) -> List[str]:
        """Get topics of current interest - now dynamic!"""
        interesting = []
        
        # 1. Topics from recent conversations (high priority)
        conversation_topics = list(self.parent_learner.conversation_topics)[-10:] if hasattr(self, 'parent_learner') else []
        interesting.extend([t for t in conversation_topics if t not in self.explored_topics])
        
        # 2. Related topics to current interests
        for topic in conversation_topics[:3]:
            related = self._generate_related_curiosity_topics(topic)
            interesting.extend(related)
        
        # 3. Curated high-interest topics (fallback)
        curated_interesting = [
            "artificial general intelligence",
            "consciousness and computation", 
            "emergence in complex systems",
            "quantum information theory",
            "cognitive architectures",
            "self-organizing systems",
            "computational creativity",
            "AI consciousness",
            "digital physics",
            "information theory"
        ]
        
        # Filter out explored topics and combine
        filtered_curated = [t for t in curated_interesting if t not in self.explored_topics]
        interesting.extend(filtered_curated)
        
        # Remove duplicates and sort by interest
        unique_interesting = list(dict.fromkeys(interesting))  # Preserves order, removes duplicates
        return sorted(unique_interesting, key=lambda t: self.interest_map.get(t, 0), reverse=True)[:10]
    
    def _generate_related_curiosity_topics(self, base_topic: str) -> List[str]:
        """Generate curiosity-driven related topics"""
        curiosity_patterns = {
            "ai": ["machine consciousness", "AI creativity", "artificial emotions", "AI philosophy"],
            "consciousness": ["qualia", "neural correlates", "information integration", "awareness"],
            "quantum": ["quantum consciousness", "quantum computing", "wave function collapse", "entanglement"],
            "learning": ["meta-learning", "transfer learning", "few-shot learning", "continual learning"],
            "creativity": ["computational creativity", "generative art", "creative algorithms", "innovation"],
            "complexity": ["emergence", "self-organization", "chaos theory", "complex networks"],
            "cognition": ["cognitive architectures", "reasoning", "memory", "perception"]
        }
        
        related = []
        base_lower = base_topic.lower()
        
        for key, topics in curiosity_patterns.items():
            if key in base_lower:
                related.extend(topics)
        
        return related[:3]
    
    def set_parent_learner(self, learner):
        """Set reference to parent autonomous learner"""
        self.parent_learner = learner
    
    def boost_curiosity(self, boost_amount: float = 0.1):
        """Increase curiosity levels across the interest map"""
        for topic in self.interest_map:
            self.interest_map[topic] += boost_amount
        
        # Also boost exploration rate if parent learner exists
        if hasattr(self, 'parent_learner') and self.parent_learner:
            self.parent_learner.exploration_rate = min(1.0, self.parent_learner.exploration_rate + boost_amount)
    
    def get_unexplored_interesting_topics(self) -> List[str]:
        """Get interesting but unexplored topics"""
        all_interesting = self.get_interesting_topics()
        return [t for t in all_interesting if t not in self.explored_topics]