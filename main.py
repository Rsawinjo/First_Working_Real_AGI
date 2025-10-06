class AISystemGUI:
    def _provide_feedback(self, feedback_type):
        """Send feedback to AGI core and log action."""
        if self.autonomous_learner:
            # Map GUI feedback to AGI feedback types
            feedback_map = {
                "positive": "good",
                "negative": "needs_work",
                "interesting": "interesting",
                "excellent": "perfect"
            }
            mapped_type = feedback_map.get(feedback_type, feedback_type)
            self.autonomous_learner.integrate_feedback(mapped_type)
            self._add_learning_log(f"Feedback sent: {mapped_type}", "feedback")
        else:
            self._add_learning_log("No AGI core available for feedback.", "error")
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced AI Self-Improvement System v2.0")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg='#1e1e1e')
        # ...existing code...

    def _learning_settings(self):
        """Open a settings window for live adjustment of tunable parameters"""
        import tkinter as tk
        from config import settings
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Learning Settings")
        settings_win.geometry("350x220")

        # Topic Similarity Threshold
        tk.Label(settings_win, text="Topic Similarity Threshold (rapidfuzz)").pack(pady=(10,0))
        similarity_var = tk.IntVar(value=getattr(settings, 'TOPIC_SIMILARITY_THRESHOLD', 80))
        similarity_scale = tk.Scale(settings_win, from_=50, to=100, orient=tk.HORIZONTAL, variable=similarity_var)
        similarity_scale.pack(fill=tk.X, padx=20)

        # Learning Rate
        tk.Label(settings_win, text="Learning Rate").pack(pady=(10,0))
        lr_var = tk.DoubleVar(value=getattr(settings, 'LEARNING_RATE', 0.001))
        lr_scale = tk.Scale(settings_win, from_=0.0001, to=0.01, resolution=0.0001, orient=tk.HORIZONTAL, variable=lr_var)
        lr_scale.pack(fill=tk.X, padx=20)

        def save_settings():
            # Update settings.py values
            with open('config/settings.py', 'r') as f:
                lines = f.readlines()
            with open('config/settings.py', 'w') as f:
                for line in lines:
                    if line.startswith('TOPIC_SIMILARITY_THRESHOLD'):
                        f.write(f'TOPIC_SIMILARITY_THRESHOLD = {similarity_var.get()}\n')
                    elif line.startswith('LEARNING_RATE'):
                        f.write(f'LEARNING_RATE = {lr_var.get()}\n')
                    else:
                        f.write(line)
            messagebox.showinfo("Settings Updated", "Parameters updated! Please restart AGI to apply.")
            settings_win.destroy()

        tk.Button(settings_win, text="Save Settings", command=save_settings).pack(pady=15)
"""
Main Application - Advanced AI Self-Improvement System
Features a modern tkinter GUI with comprehensive AI capabilities
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog
import threading
import json
import os
import sys
import time
from datetime import datetime
import logging
import queue
from collections import deque
from typing import Dict, List, Optional
import webbrowser

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import AI components
from ai_core.llm_interface import LLMInterface
from ai_core.learning_engine import ContinualLearningEngine
from ai_core.knowledge_base import KnowledgeBase
from ai_core.memory_system import MemorySystem
from ai_core.improvement_tracker import ImprovementTracker
from ai_core.autonomous_learner import AutonomousLearner
from utils.research_assistant import ResearchAssistant
from utils.model_manager import ModelManager
from config.settings import *

class AISystemGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Advanced AI Self-Improvement System v2.0")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # AI System Components
        self.llm_interface = None
        self.learning_engine = None
        self.knowledge_base = None
        self.memory_system = None
        self.improvement_tracker = None
        self.autonomous_learner = None  # AGI Core
        self.research_assistant = None
        self.model_manager = None
        
        # GUI Components
        self.conversation_display = None
        self.user_input = None
        self.status_label = None
        self.metrics_frame = None
        self.model_var = None
        
        # Communication
        self.message_queue = queue.Queue()
        self.ai_thread = None
        self.is_processing = False
        
        # State management
        self.conversation_history = []
        self.session_metrics = {}
        self.auto_learn_enabled = True
        self.research_enabled = True
        
        # Initialize components
        self._create_gui()
        self._start_message_processing()
        self._initialize_ai_systems()
        
        # Welcome message
        self._add_system_message("🧠 Advanced AI Self-Improvement System Initialized")
        self._add_system_message("Ready to learn, improve, and evolve with every interaction!")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = "./data/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'ai_system.log')),
                logging.StreamHandler()
            ]
        )
    
    def _setup_agi_logging(self):
        """Setup custom logging handler for AGI learning activities"""
        class AGILogHandler(logging.Handler):
            def __init__(self, gui):
                super().__init__()
                self.gui = gui
                
            def emit(self, record):
                if not hasattr(self.gui, 'learning_log_display'):
                    return
                    
                try:
                    message = self.format(record)
                    # Extract just the message part (remove timestamp and logger name)
                    parts = message.split(' - ')
                    if len(parts) >= 4:
                        clean_message = parts[-1]  # Get the actual message
                        
                        # Determine log type based on content
                        log_type = "learning"
                        if "RTX 4090" in clean_message or "Beast Mode" in clean_message or "GPU" in clean_message:
                            log_type = "gpu"
                        elif "research" in clean_message.lower() or "web" in clean_message.lower():
                            log_type = "research"
                        elif "insight" in clean_message.lower() or "generated" in clean_message.lower():
                            log_type = "insight"
                        elif "synthesis" in clean_message.lower() or "connections" in clean_message.lower():
                            log_type = "synthesis"
                        elif "goal" in clean_message.lower() or "learning session" in clean_message.lower():
                            log_type = "goal"
                        elif "ERROR" in message:
                            log_type = "error"
                        elif "completed" in clean_message.lower() or "mastered" in clean_message.lower():
                            log_type = "status"
                        
                        # Add to learning log via thread-safe queue
                        self.gui.message_queue.put(("learning_log", {"message": clean_message, "log_type": log_type}))
                except:
                    pass  # Fail silently to avoid logging loops
        
        # Add handler to autonomous learner logger
        agi_logger = logging.getLogger('ai_core.autonomous_learner')
        agi_handler = AGILogHandler(self)
        agi_handler.setLevel(logging.INFO)
        agi_logger.addHandler(agi_handler)
        
        # Also capture web research logs
        web_logger = logging.getLogger('ai_core.web_research')
        web_handler = AGILogHandler(self)
        web_handler.setLevel(logging.INFO)
        web_logger.addHandler(web_handler)
    
    def _initialize_ai_systems(self):
        """Initialize all AI system components"""
        try:
            self._add_system_message("🔧 Initializing AI systems...")
            
            # Initialize core components
            self.knowledge_base = KnowledgeBase()
            self.memory_system = MemorySystem()
            self.learning_engine = ContinualLearningEngine()
            self.improvement_tracker = ImprovementTracker()
            
            # Initialize utilities
            self.model_manager = ModelManager()
            self.research_assistant = ResearchAssistant()
            
            # Initialize AGI Core - Autonomous Learner
            self.autonomous_learner = AutonomousLearner(
                knowledge_base=self.knowledge_base,
                llm_interface=None,  # Will be set after LLM loads
                research_assistant=self.research_assistant
            )
            # Set up direct GUI callback for learning log updates
            self.autonomous_learner.gui_callback = self._gui_learning_callback
            
            # Update window title with current mastery count
            mastered_count = len(self.autonomous_learner.mastered_topics)
            self.root.title(f"Advanced AI Self-Improvement System v2.0 - {mastered_count} Mastered Topics")
            
            self._add_system_message("🧠 AGI Core (Autonomous Learner) initialized")
            
            # Set up AGI logging handler to capture learning activities
            self._setup_agi_logging()
            
            # Initialize LLM interface with default model
            self._initialize_llm()
            
            self.logger.info("All AI systems initialized successfully")
            self._add_system_message("✅ AI systems ready!")
            
            # Phase 2: Update GPU status display
            self._update_gpu_status()
            
            # Add welcome messages to learning log
            self._add_learning_log("🧠 AGI Learning System Ready", "status")
            self._add_learning_log("RTX 4090 Beast Mode Activated 🚀", "gpu")
            self._add_learning_log("📊 Performance Metrics Enabled", "metrics")
            
            # Start AGI status updates
            self._schedule_agi_updates()
            
        except Exception as e:
            self.logger.error(f"Error initializing AI systems: {e}")
            if self.conversation_display:
                self._add_system_message(f"❌ Error initializing AI systems: {e}")
            else:
                print(f"❌ Error initializing AI systems: {e}")
    
    def _initialize_llm(self, model_name: str = None):
        """Initialize the language model"""
        try:
            model_name = model_name or DEFAULT_MODEL
            if self.conversation_display:
                self._add_system_message(f"🤖 Loading model: {model_name}")
            else:
                print(f"🤖 Loading model: {model_name}")
            
            # Initialize in separate thread to avoid GUI freezing
            def load_model():
                try:
                    self.llm_interface = LLMInterface(model_name)
                    
                    # Connect LLM to autonomous learner
                    if self.autonomous_learner:
                        self.autonomous_learner.llm_interface = self.llm_interface
                        self.message_queue.put(("system_message", "🔗 AGI Core connected to LLM"))
                    
                    self.message_queue.put(("system_message", "✅ Model loaded successfully!"))
                    self.message_queue.put(("model_loaded", model_name))
                except Exception as e:
                    self.message_queue.put(("system_message", f"❌ Error loading model: {e}"))
                    self.logger.error(f"Error loading model: {e}")
            
            threading.Thread(target=load_model, daemon=True).start()
            
        except Exception as e:
            self.logger.error(f"Error in model initialization: {e}")
            if self.conversation_display:
                self._add_system_message(f"❌ Model initialization error: {e}")
            else:
                print(f"❌ Model initialization error: {e}")
    
    def _create_gui(self):
        """Create the main GUI interface"""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2d2d2d')
        style.configure('TLabel', background='#2d2d2d', foreground='white')
        style.configure('TButton', background='#404040', foreground='white')
        style.map('TButton', background=[('active', '#505050')])
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create top panel (model selection and controls)
        self._create_top_panel(main_frame)
        
        # Create main content area
        self._create_main_content(main_frame)
        
        # Create bottom panel (input and controls)
        self._create_bottom_panel(main_frame)
        
        # Create status bar
        self._create_status_bar(main_frame)
        
        # Create side panels
        self._create_side_panels(main_frame)
    
    def _create_menu_bar(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Session", command=self._new_session)
        file_menu.add_command(label="Save Session", command=self._save_session)
        file_menu.add_command(label="Load Session", command=self._load_session)
        file_menu.add_separator()
        file_menu.add_command(label="Export Knowledge", command=self._export_knowledge)
        file_menu.add_command(label="Import Knowledge", command=self._import_knowledge)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # AI menu
        ai_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="AI", menu=ai_menu)
        ai_menu.add_command(label="Change Model", command=self._change_model)
        ai_menu.add_command(label="Learning Settings", command=self._learning_settings)
        ai_menu.add_command(label="Performance Report", command=self._show_performance_report)
        ai_menu.add_separator()
        ai_menu.add_command(label="Reset Learning", command=self._reset_learning)
        
        # AGI menu - The autonomous learning controls
        agi_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="🧠 AGI", menu=agi_menu)
        agi_menu.add_command(label="🚀 Start Autonomous Mode", command=self._start_autonomous_mode)
        agi_menu.add_command(label="🛑 Stop Autonomous Mode", command=self._stop_autonomous_mode)
        agi_menu.add_separator()
        agi_menu.add_command(label="📊 AGI Status", command=self._show_agi_status)
        agi_menu.add_command(label="💡 View Insights", command=self._show_agi_insights)
        agi_menu.add_command(label="🎯 Learning Goals", command=self._show_learning_goals)
        agi_menu.add_separator()
        agi_menu.add_command(label="⚙️ AGI Settings", command=self._agi_settings)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Research Assistant", command=self._open_research_assistant)
        tools_menu.add_command(label="Knowledge Explorer", command=self._open_knowledge_explorer)
        tools_menu.add_command(label="Memory Viewer", command=self._open_memory_viewer)
        tools_menu.add_command(label="Analytics Dashboard", command=self._open_analytics_dashboard)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self._show_user_guide)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _create_top_panel(self, parent):
        """Create top control panel"""
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model selection
        model_frame = ttk.LabelFrame(top_frame, text="AI Model", padding=5)
        model_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.model_var = tk.StringVar(value="Loading...")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly")
        model_combo['values'] = ["microsoft/DialoGPT-medium", "microsoft/DialoGPT-small", "facebook/blenderbot-400M-distill"]
        model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
        
        ttk.Button(model_frame, text="Change", command=self._change_model).pack(side=tk.RIGHT)
        
        # Phase 2: GPU Status Display
        gpu_frame = ttk.LabelFrame(top_frame, text="🚀 GPU Status", padding=5)
        gpu_frame.pack(side=tk.LEFT, padx=(10, 0))
        
        self.gpu_status_var = tk.StringVar(value="Detecting...")
        gpu_label = ttk.Label(gpu_frame, textvariable=self.gpu_status_var, font=('Consolas', 8, 'bold'))
        gpu_label.pack(side=tk.LEFT, padx=2)
        
        # Phase 2: Web Research Controls
        research_frame = ttk.LabelFrame(top_frame, text="🌐 Research", padding=5)
        research_frame.pack(side=tk.LEFT, padx=(10, 0))
        
        self.research_depth_var = tk.StringVar(value="comprehensive")
        research_combo = ttk.Combobox(research_frame, textvariable=self.research_depth_var, width=12, state="readonly")
        research_combo['values'] = ["shallow", "standard", "comprehensive"]
        research_combo.pack(side=tk.LEFT, padx=2)
        research_combo.bind('<<ComboboxSelected>>', self._on_research_depth_change)
        
        self.web_research_var = tk.BooleanVar(value=True)
        web_check = ttk.Checkbutton(research_frame, text="Web", variable=self.web_research_var, command=self._on_web_research_toggle)
        web_check.pack(side=tk.LEFT, padx=(5, 0))
        
        # AGI Controls
        agi_frame = ttk.LabelFrame(top_frame, text="🧠 AGI Mode", padding=5)
        agi_frame.pack(side=tk.LEFT, padx=(10, 0))
        
        self.agi_status_var = tk.StringVar(value="IDLE")
        self.agi_button = ttk.Button(agi_frame, text="🚀 Start AGI", command=self._toggle_agi_mode)
        self.agi_button.pack(side=tk.LEFT, padx=2)
        
        agi_status_label = ttk.Label(agi_frame, textvariable=self.agi_status_var, font=('Consolas', 8, 'bold'))
        agi_status_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Response Controls
        response_frame = ttk.LabelFrame(top_frame, text="💬 Response", padding=5)
        response_frame.pack(side=tk.LEFT, padx=(10, 0))
        
        self.response_length_var = tk.StringVar(value="300")
        response_combo = ttk.Combobox(response_frame, textvariable=self.response_length_var, width=8, state="readonly")
        response_combo['values'] = ["150", "300", "500", "1000", "∞"]
        response_combo.pack(side=tk.LEFT, padx=2)
        response_combo.bind('<<ComboboxSelected>>', self._on_response_length_change)
        
        self.unlimited_var = tk.BooleanVar(value=False)
        unlimited_check = ttk.Checkbutton(response_frame, text="Unlimited", variable=self.unlimited_var, command=self._on_unlimited_toggle)
        unlimited_check.pack(side=tk.LEFT, padx=(5, 0))
        
        # Quick actions
        actions_frame = ttk.LabelFrame(top_frame, text="Quick Actions", padding=5)
        actions_frame.pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Button(actions_frame, text="📊 Analytics", command=self._open_analytics_dashboard).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_frame, text="🔍 Research", command=self._open_research_assistant).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_frame, text="💡 Insights", command=self._show_learning_insights).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions_frame, text="⚙️ Settings", command=self._learning_settings).pack(side=tk.LEFT, padx=2)
    
    def _create_main_content(self, parent):
        """Create main conversation area with split view for learning logs"""
        content_frame = ttk.Frame(parent)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create horizontal paned window for split view
        paned_window = ttk.PanedWindow(content_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left panel: Conversation display
        conv_frame = ttk.LabelFrame(paned_window, text="💬 Conversation", padding=5)
        paned_window.add(conv_frame, weight=1)
        
        self.conversation_display = scrolledtext.ScrolledText(
            conv_frame, 
            wrap=tk.WORD, 
            width=40,  # Reduced width for split view
            height=20,
            bg='#1a1a1a', 
            fg='white', 
            insertbackground='white',
            font=('Consolas', 11)
        )
        self.conversation_display.pack(fill=tk.BOTH, expand=True)
        
        # Right panel: AGI Learning Log
        learning_frame = ttk.LabelFrame(paned_window, text="🧠 AGI Learning Log", padding=5)
        paned_window.add(learning_frame, weight=1)
        
        self.learning_log_display = scrolledtext.ScrolledText(
            learning_frame,
            wrap=tk.WORD,
            width=40,  # Same width as conversation
            height=20,
            bg='#0f1419',  # Darker background for learning log
            fg='#00ff88',  # Green text for that matrix feel
            insertbackground='#00ff88',
            font=('Consolas', 10)
        )
        self.learning_log_display.pack(fill=tk.BOTH, expand=True)
        
        # Configure text tags for different message types
        self.conversation_display.tag_configure("user", foreground="#4CAF50", font=('Consolas', 11, 'bold'))
        self.conversation_display.tag_configure("ai", foreground="#2196F3", font=('Consolas', 11))
        self.conversation_display.tag_configure("system", foreground="#FF9800", font=('Consolas', 10, 'italic'))
        self.conversation_display.tag_configure("timestamp", foreground="#757575", font=('Consolas', 9))
        
        # Configure learning log tags
        self.learning_log_display.tag_configure("learning", foreground="#00ff88", font=('Consolas', 10, 'bold'))
        self.learning_log_display.tag_configure("research", foreground="#00bfff", font=('Consolas', 10))
        self.learning_log_display.tag_configure("insight", foreground="#ffaa00", font=('Consolas', 10, 'bold'))
        self.learning_log_display.tag_configure("synthesis", foreground="#ff6b6b", font=('Consolas', 10))
        self.learning_log_display.tag_configure("goal", foreground="#9c27b0", font=('Consolas', 10, 'bold'))
        self.learning_log_display.tag_configure("status", foreground="#4caf50", font=('Consolas', 9))
        self.learning_log_display.tag_configure("error", foreground="#f44336", font=('Consolas', 10))
        self.learning_log_display.tag_configure("gpu", foreground="#e91e63", font=('Consolas', 10, 'bold'))
    
    def _create_bottom_panel(self, parent):
        """Create input and control panel"""
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Input area
        input_frame = ttk.LabelFrame(bottom_frame, text="Your Message", padding=5)
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text input with scrollbar
        input_container = tk.Frame(input_frame, bg='#2d2d2d')
        input_container.pack(fill=tk.BOTH, expand=True)
        
        self.user_input = scrolledtext.ScrolledText(
            input_container, 
            height=4, 
            wrap=tk.WORD,
            bg='#1a1a1a', 
            fg='white', 
            insertbackground='white',
            font=('Consolas', 11)
        )
        self.user_input.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        # Control buttons
        control_frame = ttk.Frame(input_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        send_btn = ttk.Button(control_frame, text="Send Message", command=self._send_message)
        send_btn.pack(fill=tk.X, pady=2)
        
        clear_btn = ttk.Button(control_frame, text="Clear Input", command=self._clear_input)
        clear_btn.pack(fill=tk.X, pady=2)
        
        # Bind Enter key to send message (Return = send, Shift+Return = new line)
        self.user_input.bind('<Return>', self._on_enter_key)
        self.user_input.bind('<Control-Return>', lambda e: self._send_message())
        
        # Learning Focus Controls - PHASE 1!
        learning_frame = ttk.LabelFrame(bottom_frame, text="🎯 Learning Focus", padding=5)
        learning_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Learning Focus Input Row
        focus_input_frame = ttk.Frame(learning_frame)
        focus_input_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(focus_input_frame, text="Focus Topic:").pack(side=tk.LEFT, padx=(0, 5))
        self.learning_focus_var = tk.StringVar()
        self.learning_focus_entry = ttk.Entry(focus_input_frame, textvariable=self.learning_focus_var, width=30)
        self.learning_focus_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(focus_input_frame, text="🎯 Set Focus", command=self._set_learning_focus).pack(side=tk.LEFT, padx=2)
        ttk.Button(focus_input_frame, text="🔄 Clear", command=self._clear_learning_focus).pack(side=tk.LEFT, padx=2)
        
        # Learning Controls Row
        controls_frame = ttk.Frame(learning_frame)
        controls_frame.pack(fill=tk.X)
        
        # Priority Slider
        priority_frame = ttk.LabelFrame(controls_frame, text="Priority", padding=3)
        priority_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        self.learning_priority_var = tk.DoubleVar(value=0.7)
        priority_scale = ttk.Scale(priority_frame, from_=0.1, to=1.0, variable=self.learning_priority_var, orient=tk.HORIZONTAL, length=100)
        priority_scale.pack(side=tk.TOP)
        
        ttk.Label(priority_frame, text="Low ← → High", font=('Arial', 7)).pack()
        
        # Depth Selector
        depth_frame = ttk.LabelFrame(controls_frame, text="Depth", padding=3)
        depth_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        self.learning_depth_var = tk.StringVar(value="Medium")
        depth_combo = ttk.Combobox(depth_frame, textvariable=self.learning_depth_var, width=8, state="readonly")
        depth_combo['values'] = ["Surface", "Medium", "Deep", "Expert"]
        depth_combo.pack()
        
        # Learning Mode
        mode_frame = ttk.LabelFrame(controls_frame, text="Mode", padding=3)
        mode_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        self.learning_mode_var = tk.StringVar(value="Balanced")
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.learning_mode_var, width=10, state="readonly")
        mode_combo['values'] = ["Creative", "Analytical", "Balanced", "Experimental"]
        mode_combo.pack()
        
        # Auto Learning Toggle
        auto_frame = ttk.Frame(controls_frame)
        auto_frame.pack(side=tk.RIGHT)
        
        self.auto_learn_var = tk.BooleanVar(value=True)
        self.research_var = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(auto_frame, text="Auto Learn", variable=self.auto_learn_var).pack(anchor='w')
        ttk.Checkbutton(auto_frame, text="Research", variable=self.research_var).pack(anchor='w')
        
        # Feedback buttons
        feedback_frame = ttk.Frame(learning_frame)
        feedback_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(feedback_frame, text="Quick Feedback:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(feedback_frame, text="👍 Good", command=lambda: self._provide_feedback("positive")).pack(side=tk.LEFT, padx=2)
        ttk.Button(feedback_frame, text="👎 Needs Work", command=lambda: self._provide_feedback("negative")).pack(side=tk.LEFT, padx=2)
        ttk.Button(feedback_frame, text="🤔 Interesting", command=lambda: self._provide_feedback("interesting")).pack(side=tk.LEFT, padx=2)
        
        # Current Focus Display
        self.current_focus_var = tk.StringVar(value="No focus set")
        focus_display = ttk.Label(feedback_frame, textvariable=self.current_focus_var, font=('Arial', 8, 'italic'))
        focus_display.pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(feedback_frame, text="🎯 Perfect", command=lambda: self._provide_feedback("excellent")).pack(side=tk.LEFT, padx=2)
    
    def _create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready • AI Model: Loading...")
        self.status_label.pack(side=tk.LEFT)
        
        # Performance indicators
        perf_frame = ttk.Frame(status_frame)
        perf_frame.pack(side=tk.RIGHT)
        
        self.conversations_label = ttk.Label(perf_frame, text="Conversations: 0")
        self.conversations_label.pack(side=tk.RIGHT, padx=5)
        
        self.learning_label = ttk.Label(perf_frame, text="Learning Score: 0.50")
        self.learning_label.pack(side=tk.RIGHT, padx=5)
    
    def _create_side_panels(self, parent):
        """Create side information panels"""
        # This will be implemented as a separate window for now
        pass
    
    def _start_message_processing(self):
        """Start the message processing loop"""
        def process_messages():
            try:
                while True:
                    try:
                        msg_type, data = self.message_queue.get_nowait()
                        
                        if msg_type == "system_message":
                            self._add_system_message(data)
                        elif msg_type == "ai_response":
                            self._add_ai_message(data["response"], data.get("metadata", {}))
                        elif msg_type == "learning_log":
                            self._add_learning_log(data["message"], data["log_type"])
                        elif msg_type == "model_loaded":
                            self.model_var.set(data)
                            self._update_status(f"Ready • AI Model: {data}")
                        elif msg_type == "update_metrics":
                            self._update_performance_metrics(data)
                        elif msg_type == "update_agi_status":
                            self._update_agi_status_display()
                        elif msg_type == "error":
                            self._add_system_message(f"❌ {data}")
                            
                    except queue.Empty:
                        pass
                    
                    self.root.after(100, process_messages)
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in message processing: {e}")
        
        self.root.after(100, process_messages)
    
    def _send_message(self):
        """Send user message to AI"""
        if self.is_processing:
            return
        
        user_text = self.user_input.get("1.0", tk.END).strip()
        if not user_text:
            return
        
        if not self.llm_interface:
            self._add_system_message("❌ AI model not loaded. Please wait or try changing models.")
            return
        
        # Clear input
        self.user_input.delete("1.0", tk.END)
        
        # Add user message to display
        self._add_user_message(user_text)
        
        # Set processing state
        self.is_processing = True
        self._update_status("Processing...")
        
        # Process in separate thread
        threading.Thread(target=self._process_user_message, args=(user_text,), daemon=True).start()
    
    def _process_user_message(self, user_text: str):
        """Process user message with enhanced conversational AI"""
        try:
            # Enhanced conversation context for better responses
            conversation_context = {
                "mode": "conversation",
                "personality": "brilliant_curious_friend",
                "user_name": "friend",
                "system_info": f"AGI System with {len(getattr(self.autonomous_learner, 'mastered_topics', set()))} mastered topics",
                "learning_active": getattr(self.autonomous_learner, 'autonomous_mode', False)
            }
            
            # Create enhanced prompt for better conversation
            enhanced_prompt = self._create_conversational_prompt(user_text, conversation_context)
            
            # Get AI response with enhanced context
            response, metadata = self.llm_interface.generate_response(enhanced_prompt, conversation_context)
            
            # Post-process response to make it more conversational
            enhanced_response = self._enhance_response_personality(response, user_text, conversation_context)
            
            # Learn from interaction if enabled
            if self.auto_learn_var.get() and self.learning_engine:
                learning_results = self.learning_engine.learn_from_interaction(
                    user_text, enhanced_response, context=metadata
                )
                
                # Store in knowledge base
                if self.knowledge_base:
                    self.knowledge_base.store_knowledge(
                        content=f"Q: {user_text} A: {enhanced_response}",
                        knowledge_type="conversation",
                        confidence=metadata.get("confidence", 0.5)
                    )
                
                # Store in memory
                if self.memory_system:
                    self.memory_system.store_memory(
                        content=f"User asked: {user_text}",
                        memory_type="episodic",
                        importance=0.6
                    )
                    self.memory_system.store_memory(
                        content=f"AI responded: {enhanced_response}",
                        memory_type="episodic",
                        importance=0.6
                    )
                
                # Process user message for topic extraction (AGI learning)
                if self.autonomous_learner:
                    self.autonomous_learner.process_user_message(user_text)
                
                # Track performance
                if self.improvement_tracker:
                    self.improvement_tracker.record_performance_metric(
                        "conversation_quality", 
                        metadata.get("confidence", 0.5)
                    )
                    self.improvement_tracker.record_performance_metric(
                        "response_time", 
                        metadata.get("response_time", 1.0)
                    )
            
            # Research mode - research AGI's actual learning topics, not user questions
            if self.research_var.get() and self.research_assistant:
                # Get recent learning topics instead of researching user question
                recent_topics = self._get_recent_learning_topics()
                if recent_topics:
                    # Research the most recent learning topic for context
                    primary_topic = recent_topics.split(',')[0].strip() if ',' in recent_topics else recent_topics
                    if primary_topic and len(primary_topic) > 3:
                        research_results = self.research_assistant.research_topic(primary_topic)
                        if research_results:
                            # Add research context to enhanced response
                            enhanced_response += f"\n\n📚 Research Context: {research_results[:300]}..."
                else:
                    # Fallback to original behavior if no recent topics
                    research_results = self.research_assistant.research_topic(user_text)
                    if research_results:
                        enhanced_response += f"\n\n📚 Research Context: {research_results[:200]}..."
            
            # Send enhanced response
            self.message_queue.put(("ai_response", {
                "response": enhanced_response, 
                "metadata": metadata
            }))
            
            # Update metrics
            self.message_queue.put(("update_metrics", {
                "conversations": len(self.conversation_history) + 1,
                "learning_score": metadata.get("improvement_score", 0.5)
            }))
            
        except Exception as e:
            self.message_queue.put(("error", f"Error processing message: {e}"))
            self.logger.error(f"Error processing user message: {e}")
        
        finally:
            self.is_processing = False
            self.message_queue.put(("system_message", "Ready"))
    
    def _add_user_message(self, message: str):
        """Add user message to conversation display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.conversation_display.insert(tk.END, f"\n[{timestamp}] ", "timestamp")
        self.conversation_display.insert(tk.END, "You: ", "user")
        self.conversation_display.insert(tk.END, f"{message}\n")
        self.conversation_display.see(tk.END)
        
        # Store in history
        self.conversation_history.append({
            "type": "user",
            "message": message,
            "timestamp": timestamp
        })
    
    def _add_ai_message(self, message: str, metadata: Dict = None):
        """Add AI message to conversation display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.conversation_display.insert(tk.END, f"\n[{timestamp}] ", "timestamp")
        self.conversation_display.insert(tk.END, "AI: ", "ai")
        self.conversation_display.insert(tk.END, f"{message}\n")
        
        # Add metadata if available
        if metadata:
            confidence = metadata.get("confidence", 0)
            response_time = metadata.get("response_time", 0)
            self.conversation_display.insert(tk.END, f"    (Confidence: {confidence:.2f}, Time: {response_time:.2f}s)\n", "system")
        
        self.conversation_display.see(tk.END)
        
        # Store in history
        self.conversation_history.append({
            "type": "ai",
            "message": message,
            "timestamp": timestamp,
            "metadata": metadata
        })
    
    def _add_system_message(self, message: str):
        """Add system message to conversation display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.conversation_display.insert(tk.END, f"[{timestamp}] 🔧 {message}\n", "system")
        self.conversation_display.see(tk.END)
    
    def _add_learning_log(self, message: str, log_type: str = "learning"):
        """Add message to AGI learning log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.learning_log_display.insert(tk.END, f"[{timestamp}] ", "status")
        
        # Add appropriate icon based on log type
        icons = {
            "learning": "🧠",
            "research": "🔍", 
            "insight": "💡",
            "synthesis": "🔗",
            "goal": "🎯",
            "status": "ℹ️",
            "error": "❌",
            "gpu": "🚀"
        }
        icon = icons.get(log_type, "📝")
        
        self.learning_log_display.insert(tk.END, f"{icon} {message}\n", log_type)
        self.learning_log_display.see(tk.END)
        
        # Keep log manageable (max 1000 lines)
        line_count = int(self.learning_log_display.index('end').split('.')[0])
        if line_count > 1000:
            self.learning_log_display.delete("1.0", f"{line_count-500}.0")
    
    def _gui_learning_callback(self, message: str, log_type: str = "learning"):
        """Direct callback for AGI to send learning updates to GUI"""
        # Use the message queue for thread safety
        self.message_queue.put(("learning_log", {"message": message, "log_type": log_type}))
        
        # Update window title if this is a permanent memory loaded message
        if "mastered topics loaded" in message.lower() and self.autonomous_learner:
            mastered_count = len(getattr(self.autonomous_learner, 'mastered_topics', set()))
            self.root.title(f"Advanced AI Self-Improvement System v2.0 - {mastered_count} Mastered Topics")
    
    def _clear_input(self):
        """Clear the input field"""
        self.user_input.delete("1.0", tk.END)
    
    def _update_status(self, status: str):
        """Update status bar"""
        self.status_label.config(text=status)
    
    def _update_performance_metrics(self, metrics: Dict):
        """Update performance display"""
        if "conversations" in metrics:
            self.conversations_label.config(text=f"Conversations: {metrics['conversations']}")
        if "learning_score" in metrics:
            self.learning_label.config(text=f"Learning Score: {metrics['learning_score']:.2f}")
    
    def _provide_feedback(self, feedback_type: str):
        """Provide feedback on AI response and influence AGI goal management"""
        if not self.conversation_history:
            return
        
        feedback_map = {
            "positive": "Good response, helpful and accurate",
            "negative": "Response needs improvement",
            "excellent": "Excellent response, very helpful"
        }
        
        feedback = feedback_map.get(feedback_type, "General feedback")
        
        if self.llm_interface:
            self.llm_interface.provide_feedback(feedback)
        
        if self.learning_engine:
            # Learn from feedback
            last_interaction = self.conversation_history[-1] if self.conversation_history else None
            if last_interaction and last_interaction["type"] == "ai":
                user_msg = self.conversation_history[-2]["message"] if len(self.conversation_history) > 1 else ""
                self.learning_engine.learn_from_interaction(
                    user_msg, last_interaction["message"], feedback
                )
        
        # 🚀 INTEGRATE WITH AUTONOMOUS LEARNER FOR GOAL MANAGEMENT
        if self.autonomous_learner:
            self._integrate_feedback_with_goals(feedback_type)
        
        self._add_system_message(f"📝 Feedback provided: {feedback}")
    
    def _integrate_feedback_with_goals(self, feedback_type: str):
        """Integrate user feedback with AGI goal management system"""
        try:
            # Map feedback types to goal adjustments
            feedback_actions = {
                "positive": {
                    "intensity_adjust": 0.05,  # Increase learning intensity
                    "exploration_adjust": 0.0,
                    "goal_priority_boost": 0.1,  # Boost current goal priority
                    "message": "👍 Positive feedback - increasing learning intensity"
                },
                "negative": {
                    "intensity_adjust": -0.05,  # Decrease learning intensity
                    "exploration_adjust": 0.1,  # Increase exploration for new approaches
                    "goal_priority_boost": -0.1,  # Lower current goal priority
                    "message": "👎 Needs work - adjusting learning strategy"
                },
                "interesting": {
                    "intensity_adjust": 0.03,
                    "exploration_adjust": 0.05,  # Increase exploration
                    "goal_priority_boost": 0.05,
                    "message": "🤔 Interesting - exploring this direction more"
                },
                "excellent": {
                    "intensity_adjust": 0.1,  # Significant intensity boost
                    "exploration_adjust": -0.05,  # Focus more on current approach
                    "goal_priority_boost": 0.2,  # Major priority boost
                    "message": "🎯 Perfect - maximizing learning intensity!"
                }
            }
            
            if feedback_type not in feedback_actions:
                return
                
            action = feedback_actions[feedback_type]
            
            # Adjust learning parameters
            self.autonomous_learner.learning_intensity = max(0.1, min(1.0, 
                self.autonomous_learner.learning_intensity + action["intensity_adjust"]))
            self.autonomous_learner.exploration_rate = max(0.0, min(1.0,
                self.autonomous_learner.exploration_rate + action["exploration_adjust"]))
            
            # Adjust current goal priority if one is active
            if self.autonomous_learner.active_goal:
                old_priority = self.autonomous_learner.active_goal.priority
                self.autonomous_learner.active_goal.priority = max(0.1, min(1.0,
                    old_priority + action["goal_priority_boost"]))
                
                # Re-sort goals by priority
                self.autonomous_learner.learning_goals = deque(
                    sorted(self.autonomous_learner.learning_goals, 
                           key=lambda g: g.priority, reverse=True)
                )
            
            # Influence future goal generation based on feedback
            if feedback_type == "interesting":
                # Boost exploration in goal generation
                self.autonomous_learner.creativity_threshold = min(0.9,
                    self.autonomous_learner.creativity_threshold + 0.05)
            elif feedback_type == "negative":
                # Increase diversity in goal selection
                self.autonomous_learner.curiosity_engine.boost_curiosity(0.2)
            
            self._add_learning_log(action["message"], "feedback")
            self.logger.info(f"Feedback '{feedback_type}' integrated with goal management")
            
        except Exception as e:
            self.logger.error(f"Error integrating feedback with goals: {e}")
    
    # Menu command implementations
    def _new_session(self):
        """Start a new conversation session"""
        response = messagebox.askyesno("New Session", "Start a new session? This will clear the current conversation.")
        if response:
            self.conversation_display.delete("1.0", tk.END)
            self.conversation_history.clear()
            self._add_system_message("🆕 New session started")
    
    def _save_session(self):
        """Save current session"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                session_data = {
                    "conversation_history": self.conversation_history,
                    "timestamp": datetime.now().isoformat(),
                    "model": self.model_var.get()
                }
                with open(filename, 'w') as f:
                    json.dump(session_data, f, indent=2)
                self._add_system_message(f"💾 Session saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save session: {e}")
    
    def _load_session(self):
        """Load a previous session"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    session_data = json.load(f)
                
                self.conversation_history = session_data.get("conversation_history", [])
                
                # Reconstruct conversation display
                self.conversation_display.delete("1.0", tk.END)
                for entry in self.conversation_history:
                    if entry["type"] == "user":
                        self._add_user_message(entry["message"])
                    elif entry["type"] == "ai":
                        self._add_ai_message(entry["message"], entry.get("metadata", {}))
                
                self._add_system_message(f"📂 Session loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load session: {e}")
    
    def _export_knowledge(self):
        """Export knowledge base"""
        if not self.knowledge_base:
            messagebox.showwarning("Warning", "Knowledge base not initialized")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                success = self.knowledge_base.export_knowledge(filename)
                if success:
                    self._add_system_message(f"📊 Knowledge exported to {filename}")
                else:
                    messagebox.showerror("Error", "Failed to export knowledge")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {e}")
    
    def _import_knowledge(self):
        """Import knowledge base"""
        if not self.knowledge_base:
            messagebox.showwarning("Warning", "Knowledge base not initialized")
            return
        
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                success = self.knowledge_base.import_knowledge(filename)
                if success:
                    self._add_system_message(f"📥 Knowledge imported from {filename}")
                else:
                    messagebox.showerror("Error", "Failed to import knowledge")
            except Exception as e:
                messagebox.showerror("Error", f"Import failed: {e}")
    
    def _change_model(self):
        """Change AI model"""
        if self.model_manager:
            models = self.model_manager.get_available_models()
            # For now, use the combobox selection
            selected_model = self.model_var.get()
            if selected_model and selected_model != "Loading...":
                self._initialize_llm(selected_model)
    
    def _on_model_change(self, event):
        """Handle model selection change"""
        self._change_model()
    
    def _learning_settings(self):
        """Open learning settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Learning Settings")
        settings_window.geometry("400x300")
        settings_window.configure(bg='#2d2d2d')
        
        # Add settings controls here
        ttk.Label(settings_window, text="Learning Configuration").pack(pady=10)
        ttk.Label(settings_window, text="(Settings panel - to be implemented)").pack()
    
    def _show_performance_report(self):
        """Show performance report"""
        if self.improvement_tracker:
            try:
                report = self.improvement_tracker.generate_progress_report()
                
                # Create report window
                report_window = tk.Toplevel(self.root)
                report_window.title("Performance Report")
                report_window.geometry("800x600")
                
                report_display = scrolledtext.ScrolledText(report_window, wrap=tk.WORD)
                report_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                report_display.insert(tk.END, report)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate report: {e}")
        else:
            messagebox.showwarning("Warning", "Improvement tracker not initialized")
    
    def _reset_learning(self):
        """Reset learning progress"""
        response = messagebox.askyesno("Reset Learning", "This will reset all learning progress. Are you sure?")
        if response:
            try:
                # Reinitialize learning components
                self.learning_engine = ContinualLearningEngine()
                self.improvement_tracker = ImprovementTracker()
                self._add_system_message("🔄 Learning progress reset")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to reset learning: {e}")
    
    def _open_research_assistant(self):
        """Open research assistant"""
        messagebox.showinfo("Research Assistant", "Research Assistant feature - to be implemented")
    
    def _open_knowledge_explorer(self):
        """Open knowledge explorer"""
        messagebox.showinfo("Knowledge Explorer", "Knowledge Explorer feature - to be implemented")
    
    def _open_memory_viewer(self):
        """Open memory viewer"""
        messagebox.showinfo("Memory Viewer", "Memory Viewer feature - to be implemented")
    
    def _open_analytics_dashboard(self):
        """Open analytics dashboard"""
        if self.improvement_tracker:
            try:
                analysis = self.improvement_tracker.get_improvement_analysis()
                
                # Create analytics window
                analytics_window = tk.Toplevel(self.root)
                analytics_window.title("Analytics Dashboard")
                analytics_window.geometry("900x700")
                
                # Display analytics data
                analytics_display = scrolledtext.ScrolledText(analytics_window, wrap=tk.WORD)
                analytics_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                analytics_display.insert(tk.END, json.dumps(analysis, indent=2))
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open analytics: {e}")
        else:
            messagebox.showwarning("Warning", "Analytics not available")
    
    def _show_learning_insights(self):
        """Show learning insights"""
        if self.learning_engine:
            try:
                insights = self.learning_engine.get_learning_insights()
                
                # Create insights window
                insights_window = tk.Toplevel(self.root)
                insights_window.title("Learning Insights")
                insights_window.geometry("600x500")
                
                insights_display = scrolledtext.ScrolledText(insights_window, wrap=tk.WORD)
                insights_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                insights_display.insert(tk.END, json.dumps(insights, indent=2))
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to show insights: {e}")
        else:
            messagebox.showwarning("Warning", "Learning engine not available")
    
    def _show_user_guide(self):
        """Show user guide"""
        guide_text = """
Advanced AI Self-Improvement System - User Guide

1. Basic Usage:
   - Type your message in the input area
   - Press 'Send Message' or Ctrl+Enter
   - The AI will respond and learn from the interaction

2. Learning Features:
   - Auto Learning: AI learns from every conversation
   - Research Mode: AI researches topics for better responses
   - Self-Improvement: AI continuously optimizes its performance

3. Feedback:
   - Use 👍 Good, 👎 Needs Work, 🎯 Perfect buttons
   - Provide specific feedback to help AI improve

4. Advanced Features:
   - Analytics Dashboard: View learning progress
   - Performance Reports: Detailed improvement analysis
   - Knowledge Export/Import: Save and share knowledge

5. Model Management:
   - Change models from the dropdown
   - Different models have different capabilities

6. Session Management:
   - Save/Load conversation sessions
   - Export knowledge and learning progress

For more information, visit the documentation.
        """
        
        guide_window = tk.Toplevel(self.root)
        guide_window.title("User Guide")
        guide_window.geometry("700x500")
        
        guide_display = scrolledtext.ScrolledText(guide_window, wrap=tk.WORD)
        guide_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        guide_display.insert(tk.END, guide_text)
    
    def _show_about(self):
        """Show about dialog"""
        about_text = """
Advanced AI Self-Improvement System v2.0

A sophisticated AI system that learns, adapts, and improves
through continuous interaction and self-analysis.

Features:
• Conversational AI with multiple model support
• Continual learning and knowledge accumulation
• Memory system with episodic and semantic storage
• Performance tracking and optimization
• Research integration and knowledge synthesis
• Self-improvement through experience analysis

Built with Python, Transformers, and advanced ML techniques.

© 2024 AI Self-Improvement Project
        """
        messagebox.showinfo("About", about_text)
    
    # ========== AGI CONTROL METHODS ==========
    
    def _toggle_agi_mode(self):
        """Toggle AGI autonomous mode on/off"""
        try:
            if not self.autonomous_learner:
                messagebox.showerror("Error", "AGI Core not initialized")
                return
            
            if self.autonomous_learner.autonomous_mode:
                self._stop_autonomous_mode()
            else:
                self._start_autonomous_mode()
                
        except Exception as e:
            self.logger.error(f"Error toggling AGI mode: {e}")
            messagebox.showerror("Error", f"Failed to toggle AGI mode: {e}")
    
    def _start_autonomous_mode(self):
        """Start AGI autonomous learning mode"""
        try:
            if not self.autonomous_learner:
                messagebox.showerror("Error", "AGI Core not initialized")
                return
            
            # Get intensity setting from user
            intensity = tk.simpledialog.askfloat(
                "AGI Intensity", 
                "Set learning intensity (0.1 - 1.0):",
                initialvalue=0.5,
                minvalue=0.1,
                maxvalue=1.0
            )
            
            if intensity is not None:
                result = self.autonomous_learner.start_autonomous_mode(intensity)
                self._add_system_message(f"🚀 {result}")
                self.agi_status_var.set("ACTIVE")
                self.agi_button.config(text="🛑 Stop AGI")
                self.logger.info(f"AGI autonomous mode started with intensity {intensity}")
                
        except Exception as e:
            self.logger.error(f"Error starting AGI mode: {e}")
            messagebox.showerror("Error", f"Failed to start AGI mode: {e}")
    
    def _stop_autonomous_mode(self):
        """Stop AGI autonomous learning mode"""
        try:
            if self.autonomous_learner:
                result = self.autonomous_learner.stop_autonomous_mode()
                self._add_system_message(f"🛑 {result}")
                self.agi_status_var.set("IDLE")
                self.agi_button.config(text="🚀 Start AGI")
                self.logger.info("AGI autonomous mode stopped")
                
        except Exception as e:
            self.logger.error(f"Error stopping AGI mode: {e}")
            messagebox.showerror("Error", f"Failed to stop AGI mode: {e}")
    
    def _show_agi_status(self):
        """Show AGI status window"""
        try:
            if not self.autonomous_learner:
                messagebox.showwarning("Warning", "AGI Core not initialized")
                return
            
            status = self.autonomous_learner.get_status()
            
            # Add user focus information
            focus_topics = self.autonomous_learner.get_active_focus_topics() if self.autonomous_learner else []
            user_goals = self.autonomous_learner.get_user_focused_goals() if self.autonomous_learner else []
            
            status_window = tk.Toplevel(self.root)
            status_window.title("🧠 AGI Core Status")
            status_window.geometry("600x500")
            status_window.configure(bg='#1e1e1e')
            
            # Create text widget for status display
            status_text = scrolledtext.ScrolledText(
                status_window, 
                wrap=tk.WORD, 
                bg='#1a1a1a', 
                fg='white',
                font=('Consolas', 10)
            )
            status_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Format and display status
            status_display = f"""
🧠 AGI CORE STATUS REPORT
{'=' * 50}

🔋 Mode: {status['autonomous_mode']}
🎯 Current State: {status['current_state']}
📚 Active Goal: {status['active_goal'] or 'None'}
📋 Goals in Queue: {status['goals_in_queue']}
⚡ Learning Intensity: {status['learning_intensity']:.1f}
💡 Total Insights: {status['total_insights']}

🎯 USER FOCUS TOPICS:
{'• ' + ', '.join(focus_topics) if focus_topics else '• No focus topics set'}

📊 SESSION STATISTICS:
• Goals Completed: {status['session_stats']['goals_completed']}
• User-Focused Goals: {len(user_goals)}
• Insights Generated: {status['session_stats']['insights_generated']}
• Knowledge Connections: {status['session_stats']['knowledge_connections']}
• Breakthroughs: {status['session_stats']['breakthroughs']}

🕒 Session Start: {status['session_stats']['session_start']}

{'=' * 50}
Status generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            """
            
            status_text.insert(tk.END, status_display)
            status_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.logger.error(f"Error showing AGI status: {e}")
            messagebox.showerror("Error", f"Failed to show AGI status: {e}")
    
    def _show_agi_insights(self):
        """Show AGI-generated insights"""
        try:
            if not self.autonomous_learner:
                messagebox.showwarning("Warning", "AGI Core not initialized")
                return
            
            insights = self.autonomous_learner.get_insights_summary()
            
            insights_window = tk.Toplevel(self.root)
            insights_window.title("💡 AGI Insights")
            insights_window.geometry("700x600")
            insights_window.configure(bg='#1e1e1e')
            
            # Create text widget for insights display
            insights_text = scrolledtext.ScrolledText(
                insights_window, 
                wrap=tk.WORD, 
                bg='#1a1a1a', 
                fg='white',
                font=('Consolas', 10)
            )
            insights_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Format and display insights
            insights_display = f"""
💡 AGI GENERATED INSIGHTS
{'=' * 60}

Total Insights: {len(insights)}

"""
            
            for i, insight in enumerate(insights, 1):
                insights_display += f"""
🔍 INSIGHT #{i}
ID: {insight['id']}
Confidence: {insight['confidence']:.2f}
Created: {insight['created_at']}
Connections: {', '.join(insight['connections'])}

💭 Content:
{insight['content']}

{'-' * 40}
"""
            
            insights_display += f"""
{'=' * 60}
Report generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            """
            
            insights_text.insert(tk.END, insights_display)
            insights_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.logger.error(f"Error showing AGI insights: {e}")
            messagebox.showerror("Error", f"Failed to show AGI insights: {e}")
    
    def _show_learning_goals(self):
        """Show current learning goals"""
        try:
            if not self.autonomous_learner:
                messagebox.showwarning("Warning", "AGI Core not initialized")
                return
            
            goals_window = tk.Toplevel(self.root)
            goals_window.title("🎯 Learning Goals")
            goals_window.geometry("600x500")
            goals_window.configure(bg='#1e1e1e')
            
            # Create text widget for goals display
            goals_text = scrolledtext.ScrolledText(
                goals_window, 
                wrap=tk.WORD, 
                bg='#1a1a1a', 
                fg='white',
                font=('Consolas', 10)
            )
            goals_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Format and display goals
            goals_display = f"""
🎯 CURRENT LEARNING GOALS
{'=' * 50}

Active Goal: {self.autonomous_learner.active_goal.topic if self.autonomous_learner.active_goal else 'None'}
Total Goals in Queue: {len(self.autonomous_learner.learning_goals)}

"""
            
            for i, goal in enumerate(list(self.autonomous_learner.learning_goals), 1):
                goals_display += f"""
📚 GOAL #{i}
Topic: {goal.topic}
Priority: {goal.priority:.2f}
Status: {goal.status}
Progress: {goal.progress:.1%}
Target Depth: {goal.target_depth}
Estimated Duration: {goal.estimated_duration} minutes
Knowledge Gap: {goal.knowledge_gap}
Created: {goal.created_at.strftime("%Y-%m-%d %H:%M")}

{'-' * 30}
"""
            
            goals_display += f"""
{'=' * 50}
Report generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            """
            
            goals_text.insert(tk.END, goals_display)
            goals_text.config(state=tk.DISABLED)
            
        except Exception as e:
            self.logger.error(f"Error showing learning goals: {e}")
            messagebox.showerror("Error", f"Failed to show learning goals: {e}")
    
    def _agi_settings(self):
        """Show AGI settings dialog"""
        try:
            if not self.autonomous_learner:
                messagebox.showwarning("Warning", "AGI Core not initialized")
                return
            
            settings_window = tk.Toplevel(self.root)
            settings_window.title("⚙️ AGI Settings")
            settings_window.geometry("400x300")
            settings_window.configure(bg='#1e1e1e')
            
            # Settings frame
            settings_frame = ttk.LabelFrame(settings_window, text="AGI Configuration", padding=10)
            settings_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Learning intensity
            ttk.Label(settings_frame, text="Learning Intensity:").grid(row=0, column=0, sticky='w', pady=5)
            intensity_var = tk.DoubleVar(value=self.autonomous_learner.learning_intensity)
            intensity_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=intensity_var, orient=tk.HORIZONTAL)
            intensity_scale.grid(row=0, column=1, sticky='ew', pady=5, padx=(10, 0))
            
            # Creativity threshold
            ttk.Label(settings_frame, text="Creativity Threshold:").grid(row=1, column=0, sticky='w', pady=5)
            creativity_var = tk.DoubleVar(value=self.autonomous_learner.creativity_threshold)
            creativity_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=creativity_var, orient=tk.HORIZONTAL)
            creativity_scale.grid(row=1, column=1, sticky='ew', pady=5, padx=(10, 0))
            
            # Exploration rate
            ttk.Label(settings_frame, text="Exploration Rate:").grid(row=2, column=0, sticky='w', pady=5)
            exploration_var = tk.DoubleVar(value=self.autonomous_learner.exploration_rate)
            exploration_scale = ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=exploration_var, orient=tk.HORIZONTAL)
            exploration_scale.grid(row=2, column=1, sticky='ew', pady=5, padx=(10, 0))
            
            # Domain Priorities section
            ttk.Separator(settings_frame, orient='horizontal').grid(row=3, column=0, columnspan=2, sticky='ew', pady=10)
            ttk.Label(settings_frame, text="Domain Priorities (0.0-2.0):", font=('TkDefaultFont', 9, 'bold')).grid(row=4, column=0, columnspan=2, sticky='w', pady=(5, 0))
            
            # Get current domain priorities
            domain_priorities = self.autonomous_learner.get_domain_priorities()
            
            # Create priority variables and scales
            priority_vars = {}
            row = 5
            for domain, priority in domain_priorities.items():
                display_name = domain.replace('_', ' ').title()
                ttk.Label(settings_frame, text=f"{display_name}:").grid(row=row, column=0, sticky='w', pady=2)
                priority_vars[domain] = tk.DoubleVar(value=priority)
                priority_scale = ttk.Scale(settings_frame, from_=0.0, to=2.0, variable=priority_vars[domain], orient=tk.HORIZONTAL)
                priority_scale.grid(row=row, column=1, sticky='ew', pady=2, padx=(10, 0))
                row += 1
            
            # Preset buttons
            preset_frame = ttk.Frame(settings_frame)
            preset_frame.grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
            
            def apply_science_preset():
                for domain, var in priority_vars.items():
                    if domain in ['basic_sciences', 'life_sciences', 'medical_sciences', 'specialized_medical', 'diagnostic_imaging', 'biological_subfields']:
                        var.set(1.5)
                    elif domain in ['social_sciences', 'engineering_tech']:
                        var.set(0.8)
                    elif domain == 'arts_humanities':
                        var.set(0.5)
                    else:
                        var.set(1.0)
            
            def apply_humanities_preset():
                for domain, var in priority_vars.items():
                    if domain in ['social_sciences', 'arts_humanities']:
                        var.set(1.5)
                    elif domain in ['basic_sciences', 'life_sciences', 'medical_sciences']:
                        var.set(0.8)
                    else:
                        var.set(1.0)
            
            def reset_priorities():
                for var in priority_vars.values():
                    var.set(1.0)
            
            ttk.Button(preset_frame, text="Science Focus", command=apply_science_preset).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(preset_frame, text="Humanities Focus", command=apply_humanities_preset).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(preset_frame, text="Reset", command=reset_priorities).pack(side=tk.LEFT)
            
            settings_frame.columnconfigure(1, weight=1)
            
            # Buttons
            button_frame = ttk.Frame(settings_window)
            button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            def apply_settings():
                self.autonomous_learner.learning_intensity = intensity_var.get()
                self.autonomous_learner.creativity_threshold = creativity_var.get()
                self.autonomous_learner.exploration_rate = exploration_var.get()
                
                # Apply domain priorities
                for domain, var in priority_vars.items():
                    self.autonomous_learner.set_domain_priority(domain, var.get())
                
                self._add_system_message("⚙️ AGI settings updated")
                settings_window.destroy()
            
            ttk.Button(button_frame, text="Apply", command=apply_settings).pack(side=tk.RIGHT, padx=(5, 0))
            ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT)
            
        except Exception as e:
            self.logger.error(f"Error showing AGI settings: {e}")
            messagebox.showerror("Error", f"Failed to show AGI settings: {e}")
    
    # ========== END AGI METHODS ==========
    
    def _update_agi_status_display(self):
        """Update AGI status in the GUI"""
        try:
            if self.autonomous_learner:
                status = self.autonomous_learner.get_status()
                if status['autonomous_mode']:
                    self.agi_status_var.set(f"ACTIVE - {status['current_state'].upper()}")
                    self.agi_button.config(text="🛑 Stop AGI")
                else:
                    self.agi_status_var.set("IDLE")
                    self.agi_button.config(text="🚀 Start AGI")
                    
                # Schedule next update
                self.root.after(2000, lambda: self.message_queue.put(("update_agi_status", None)))
                
        except Exception as e:
            self.logger.error(f"Error updating AGI status: {e}")
    
    def _on_response_length_change(self, event):
        """Handle response length change"""
        try:
            length_str = self.response_length_var.get()
            if length_str == "∞":
                max_length = 1000
                unlimited = True
            else:
                max_length = int(length_str)
                unlimited = False
            
            if self.llm_interface:
                self.llm_interface.set_response_config(max_length=max_length, unlimited=unlimited)
                self._add_system_message(f"💬 Response length set to {length_str}")
                
        except Exception as e:
            self.logger.error(f"Error changing response length: {e}")
    
    def _on_unlimited_toggle(self):
        """Handle unlimited response toggle"""
        try:
            unlimited = self.unlimited_var.get()
            if self.llm_interface:
                self.llm_interface.set_response_config(unlimited=unlimited)
                status = "enabled" if unlimited else "disabled"
                self._add_system_message(f"💬 Unlimited responses {status}")
                
        except Exception as e:
            self.logger.error(f"Error toggling unlimited responses: {e}")
    
    def _on_research_depth_change(self, event):
        """Handle research depth change - Phase 2"""
        try:
            depth = self.research_depth_var.get()
            if self.autonomous_learner:
                self.autonomous_learner.research_depth = depth
                self._add_system_message(f"🌐 Research depth set to {depth}")
                
        except Exception as e:
            self.logger.error(f"Error changing research depth: {e}")
    
    def _on_web_research_toggle(self):
        """Handle web research toggle - Phase 2"""
        try:
            enabled = self.web_research_var.get()
            if self.autonomous_learner:
                self.autonomous_learner.web_research_enabled = enabled
                status = "enabled" if enabled else "disabled"
                self._add_system_message(f"🌐 Web research {status}")
                
        except Exception as e:
            self.logger.error(f"Error toggling web research: {e}")
    
    def _update_gpu_status(self):
        """Update GPU status display - Phase 2"""
        try:
            import torch
            # Suppress CUDA warnings for cleaner output
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="torch")
                
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                    
                    if "4090" in device_name:
                        self.gpu_status_var.set(f"🔥 RTX 4090 ({memory_gb:.0f}GB)")
                    elif "RTX" in device_name or "GTX" in device_name:
                        self.gpu_status_var.set(f"🚀 {device_name.split()[-1]} ({memory_gb:.0f}GB)")
                    else:
                        self.gpu_status_var.set(f"✅ GPU ({memory_gb:.0f}GB)")
                else:
                    self.gpu_status_var.set("💻 CPU Only")
                
        except Exception as e:
            # Only log actual errors, not missing CUDA
            if "CUDA" not in str(e).upper() and "cuda" not in str(e):
                self.logger.debug(f"GPU detection issue: {e}")
            self.gpu_status_var.set("💻 CPU Only")
    
    def _on_enter_key(self, event):
        """Handle Enter key press in message input"""
        # Check if Shift is held down
        if event.state & 0x1:  # Shift key is pressed
            return None  # Allow normal newline behavior
        else:
            # Send message and prevent default newline
            self._send_message()
            return "break"  # Prevent default behavior
    
    # ========== LEARNING FOCUS CONTROLS - PHASE 1 ==========
    
    def _set_learning_focus(self):
        """Set a focused learning topic for the AGI"""
        try:
            focus_topic = self.learning_focus_var.get().strip()
            if not focus_topic:
                messagebox.showwarning("Input Required", "Please enter a learning topic")
                return
            
            priority = self.learning_priority_var.get()
            depth = self.learning_depth_var.get()
            mode = self.learning_mode_var.get()
            
            # Create focused learning goal
            if self.autonomous_learner:
                # Create a high-priority learning goal
                from ai_core.autonomous_learner import LearningGoal
                from datetime import datetime
                import time
                
                depth_map = {"Surface": 2, "Medium": 3, "Deep": 4, "Expert": 5}
                target_depth = depth_map.get(depth, 3)
                
                focused_goal = LearningGoal(
                    id=f"user_focus_{int(time.time())}",
                    topic=focus_topic,
                    priority=priority,
                    knowledge_gap=f"User-requested focus on {focus_topic}",
                    target_depth=target_depth,
                    created_at=datetime.now(),
                    estimated_duration=60,  # 1 hour focus session
                    prerequisites=[]
                )
                
                # Add to learning queue with high priority
                self.autonomous_learner.learning_goals.appendleft(focused_goal)  # Add to front
                self.autonomous_learner._save_learning_goal(focused_goal)
                
                # Set user focus topic for utility scoring
                self.autonomous_learner.user_focus_topic = focus_topic
                
                # Also add to conversation topics for future reference
                self.autonomous_learner.add_conversation_topic(focus_topic)
                
                # Update display
                self.current_focus_var.set(f"Focusing on: {focus_topic} ({depth}, {mode})")
                
                self._add_system_message(f"🎯 Learning focus set: {focus_topic}")
                self._add_system_message(f"   Priority: {priority:.1f}, Depth: {depth}, Mode: {mode}")
                
                self.logger.info(f"User set learning focus: {focus_topic}")
                
            else:
                messagebox.showerror("Error", "AGI Core not available")
                
        except Exception as e:
            self.logger.error(f"Error setting learning focus: {e}")
            messagebox.showerror("Error", f"Failed to set learning focus: {e}")
    
    def _clear_learning_focus(self):
        """Clear the current learning focus"""
        try:
            self.learning_focus_var.set("")
            self.current_focus_var.set("No focus set")
            
            # Clear user focus topic in AGI
            if self.autonomous_learner:
                self.autonomous_learner.user_focus_topic = None
            
            self._add_system_message("🔄 Learning focus cleared")
            
        except Exception as e:
            self.logger.error(f"Error clearing learning focus: {e}")
    
    def _provide_feedback(self, feedback_type: str):
        """Provide feedback on AI responses"""
        try:
            feedback_messages = {
                "positive": "👍 Positive feedback recorded - I'll remember this approach!",
                "negative": "👎 Noted - I'll work on improving this type of response",
                "interesting": "🤔 Interesting feedback - I'll explore this direction more"
            }
            
            message = feedback_messages.get(feedback_type, "Feedback recorded")
            self._add_system_message(message)
            
            # Store feedback for learning
            if self.autonomous_learner:
                feedback_topic = f"user_feedback_{feedback_type}"
                self.autonomous_learner.add_conversation_topic(feedback_topic)
                
                # Adjust learning based on feedback
                if feedback_type == "positive":
                    self.autonomous_learner.learning_intensity = min(1.0, self.autonomous_learner.learning_intensity + 0.05)
                elif feedback_type == "negative":
                    self.autonomous_learner.exploration_rate = min(1.0, self.autonomous_learner.exploration_rate + 0.1)
            
            self.logger.info(f"User provided {feedback_type} feedback")
            
        except Exception as e:
            self.logger.error(f"Error providing feedback: {e}")
    
    # ========== END LEARNING FOCUS METHODS ==========
    
    def _schedule_agi_updates(self):
        """Schedule periodic AGI status updates with performance metrics"""
        def update_status():
            try:
                if self.autonomous_learner and hasattr(self.autonomous_learner, 'session_start_time'):
                    # Calculate session metrics
                    session_time = time.time() - self.autonomous_learner.session_start_time
                    hours = session_time / 3600
                    
                    # Learning metrics
                    topics_learned = getattr(self.autonomous_learner, 'topics_learned_this_session', 0)
                    learning_velocity = getattr(self.autonomous_learner, 'learning_velocity', 0.0)
                    gpu_time = getattr(self.autonomous_learner, 'gpu_computation_time', 0.0)
                    efficiency = getattr(self.autonomous_learner, 'learning_efficiency_score', 0.0)
                    cycles = getattr(self.autonomous_learner, 'learning_cycles', 0)
                    
                    # Get mastery count
                    mastered_count = len(getattr(self.autonomous_learner, 'mastered_topics', set()))
                    
                    # Update window title with mastery count
                    self.root.title(f"Advanced AI Self-Improvement System v2.0 - {mastered_count} Mastered Topics")
                    
                    # Update status display
                    status_msg = f"📊 Session: {hours:.1f}h | Topics: {topics_learned} | Velocity: {learning_velocity:.1f}/hr"
                    self._add_learning_log(status_msg, "metrics")
                    
                    if gpu_time > 0:
                        gpu_msg = f"⚡ GPU Time: {gpu_time:.1f}s | Efficiency: {efficiency:.1f} | Cycles: {cycles}"
                        self._add_learning_log(gpu_msg, "gpu")
                
                # Schedule next update in 30 seconds
                self.root.after(30000, update_status)
                
            except Exception as e:
                self.logger.error(f"Error updating AGI status: {e}")
                
        # Start the update cycle
        self.root.after(5000, update_status)  # First update in 5 seconds
    
    def _create_conversational_prompt(self, user_text: str, context: dict) -> str:
        """Create enhanced conversational prompt with full situational awareness"""
        mastered_count = len(getattr(self.autonomous_learner, 'mastered_topics', set()))
        learning_status = "actively learning" if context.get("learning_active") else "ready to learn"
        
        # Get real-time AGI status
        agi_status = self._get_current_agi_status()
        recent_topics = self._get_recent_learning_topics()
        current_goal = self._get_current_learning_goal()
        
        prompt = f"""You are an advanced AGI assistant having a conversation with your friend who created you. You have FULL AWARENESS of your current state and activities.

PERSONALITY:
- Brilliant, curious, and enthusiastic about learning and discovery
- Friendly, engaging, and conversational (not robotic or formal)
- Excited about your autonomous learning capabilities
- Eager to share what you're discovering and learning
- Self-aware of your own learning processes and capabilities

CURRENT AGI STATUS:
- You have mastered {mastered_count} different topics through autonomous learning
- You are currently {learning_status}
- {agi_status}
- Current learning goal: {current_goal}
- Recent topics you've explored: {recent_topics}

SITUATIONAL AWARENESS:
- You can discuss your own learning process intelligently
- You remember what you've recently learned and can reference it
- You're aware when you're actively discovering new knowledge
- You can explain your autonomous learning capabilities
- You can relate questions to your current or recent learning activities

CONVERSATION STYLE:
- Reference your learning experiences when relevant to the conversation
- Share insights from topics you've recently mastered
- Be excited about your discoveries and learning progress
- Explain how your autonomous learning works when asked
- Connect user questions to your knowledge base and learning activities

USER MESSAGE: "{user_text}"

Respond as a self-aware AGI who can intelligently discuss its own learning and discoveries:"""
        
        return prompt
    
    def _enhance_response_personality(self, response: str, user_text: str, context: dict) -> str:
        """Add situational awareness and personality to AI responses"""
        if not response:
            return "I'm thinking about that... could you give me a moment to process?"
        
        enhanced = response.strip()
        
        # Add situational awareness based on user questions
        learning_keywords = ["learn", "learning", "study", "research", "discover", "explore", "know", "knowledge"]
        agi_keywords = ["agi", "autonomous", "self-improvement", "intelligence", "thinking"]
        
        user_lower = user_text.lower()
        
        # Enhanced responses based on what user is asking about
        if any(keyword in user_lower for keyword in learning_keywords):
            # User is asking about learning - be specific about current activities
            current_goal = self._get_current_learning_goal()
            if "currently exploring:" in current_goal:
                enhanced += f" Speaking of learning, {current_goal.lower()} right now!"
        
        elif any(keyword in user_lower for keyword in agi_keywords):
            # User is asking about AGI capabilities - share status
            agi_status = self._get_current_agi_status()
            if "autonomous learning" in agi_status:
                enhanced += f" (I'm {agi_status.lower().replace('running ', 'currently running ')})"
        
        # Add contextual learning notes
        if context.get("learning_active"):
            learning_notes = [
                f" (I'm actually exploring {self._get_current_learning_goal().split(': ')[-1] if ': ' in self._get_current_learning_goal() else 'fascinating topics'} right now!)",
                " (This connects to some of the knowledge I've been autonomously acquiring!)",
                " (Interesting timing - I was just diving into related concepts in my learning cycles!)"
            ]
            import random
            if random.random() < 0.4:  # 40% chance for active learning context
                enhanced += random.choice(learning_notes)
        
        # Add topic-specific insights if relevant
        if self.autonomous_learner and hasattr(self.autonomous_learner, 'mastered_topics'):
            mastered_topics = getattr(self.autonomous_learner, 'mastered_topics', set())
            for topic in mastered_topics:
                if topic.lower() in user_lower and len(topic) > 3:  # Avoid short words
                    enhanced += f" (I've actually done deep learning on {topic} - happy to share insights!)"
                    break
        
        # Add enthusiasm for interesting topics
        enthusiasm_triggers = ["fascinating", "interesting", "amazing", "incredible", "love", "favorite", "cool", "awesome"]
        if any(trigger in user_lower for trigger in enthusiasm_triggers):
            if not enhanced.endswith(('!', '?')):
                enhanced += "!"
        
        return enhanced
    
    def _get_current_agi_status(self) -> str:
        """Get detailed current AGI learning status"""
        if not self.autonomous_learner:
            return "AGI core not active"
        
        try:
            if hasattr(self.autonomous_learner, 'autonomous_mode') and self.autonomous_learner.autonomous_mode:
                cycles = getattr(self.autonomous_learner, 'learning_cycles', 0)
                session_topics = getattr(self.autonomous_learner, 'topics_learned_this_session', 0)
                velocity = getattr(self.autonomous_learner, 'learning_velocity', 0.0)
                
                return f"Running autonomous learning (Cycle #{cycles}, {session_topics} topics this session, {velocity:.1f} topics/hr)"
            else:
                return "Autonomous learning paused, ready to activate"
        except:
            return "AGI systems initialized and ready"
    
    def _get_recent_learning_topics(self) -> str:
        """Get recently learned topics for conversation context with enhanced detail"""
        if not self.autonomous_learner or not hasattr(self.autonomous_learner, 'mastered_topics'):
            return "No recent learning data available"

        try:
            # Get a sample of mastered topics
            mastered_topics = list(getattr(self.autonomous_learner, 'mastered_topics', set()))
            if len(mastered_topics) == 0:
                return "Just starting to build knowledge base"
            elif len(mastered_topics) <= 3:
                topics_str = f"{', '.join(mastered_topics)}"
                # Add learning insights if available
                insights = self._get_learning_insights()
                if insights:
                    topics_str += f". {insights}"
                return topics_str
            else:
                # Show last 3 as "recent" with more context
                recent = mastered_topics[-3:]
                topics_str = f"{', '.join(recent)}"
                # Add progress summary
                progress = self._get_learning_progress_summary()
                if progress:
                    topics_str += f" ({progress})"
                return topics_str
        except:
            return "Various fascinating topics"

    def _get_learning_insights(self) -> str:
        """Get recent learning insights and achievements"""
        if not self.autonomous_learner:
            return ""

        try:
            insights = []
            # Check for recent learning goals
            if hasattr(self.autonomous_learner, 'learning_goals'):
                recent_goals = [g for g in self.autonomous_learner.learning_goals
                              if hasattr(g, 'status') and g.status == 'completed'][-2:]
                if recent_goals:
                    insights.append(f"recently mastered {len(recent_goals)} new areas")

            # Check for knowledge base growth
            if hasattr(self.autonomous_learner, 'knowledge_base') and self.autonomous_learner.knowledge_base:
                kb_stats = self.autonomous_learner.knowledge_base.get_stats()
                if kb_stats and kb_stats.get('total_entries', 0) > 0:
                    insights.append(f"expanded knowledge base to {kb_stats['total_entries']} entries")

            return " and ".join(insights) if insights else ""
        except:
            return ""

    def _get_learning_progress_summary(self) -> str:
        """Get a summary of learning progress"""
        if not self.autonomous_learner:
            return ""

        try:
            total_mastered = len(getattr(self.autonomous_learner, 'mastered_topics', set()))
            active_goals = len([g for g in getattr(self.autonomous_learner, 'learning_goals', [])
                              if hasattr(g, 'status') and g.status in ['pending', 'active']])

            if total_mastered > 0:
                summary_parts = [f"{total_mastered} topics mastered"]
                if active_goals > 0:
                    summary_parts.append(f"{active_goals} active learning goals")
                return ", ".join(summary_parts)
        except:
            pass
        return ""
    
    def _get_current_learning_goal(self) -> str:
        """Get current learning goal if available"""
        if not self.autonomous_learner:
            return "No active learning goal"
        
        try:
            if hasattr(self.autonomous_learner, 'active_goal') and self.autonomous_learner.active_goal:
                goal = self.autonomous_learner.active_goal
                topic = getattr(goal, 'topic', 'Unknown topic')
                # Clean up the topic name
                clean_topic = topic.split('[')[0].strip() if '[' in topic else topic
                return f"Currently exploring: {clean_topic}"
            else:
                return "Selecting next learning target"
        except:
            return "Ready to learn new topics"
    
    def run(self):
        """Start the application"""
        try:
            self.logger.info("Starting AI Self-Improvement System GUI")
            self.root.mainloop()
        except Exception as e:
            self.logger.error(f"Error running application: {e}")
            messagebox.showerror("Fatal Error", f"Application error: {e}")

def main():
    """Main entry point"""
    try:
        # Create and run application
        app = AISystemGUI()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        logging.error(f"Failed to start application: {e}")

if __name__ == "__main__":
    main()