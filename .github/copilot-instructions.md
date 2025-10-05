<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Advanced AI Self-Improvement System

Python-based AGI-like system implementing continual learning through LLM interactions, featuring tkinter GUI, vector knowledge base, and autonomous self-improvement.

### Architecture Overview
- **Entry Point**: `main.py` - Tkinter GUI application initializing all AI components
- **Core Components** (`ai_core/`):
  - `autonomous_learner.py` - AGI core with goal-driven learning and web research
  - `learning_engine.py` - Continual learning with experience buffer and pattern recognition
  - `knowledge_base.py` - ChromaDB vector storage with semantic search and knowledge graphs
  - `memory_system.py` - Long/short-term memory with SQLite persistence
  - `llm_interface.py` - Hugging Face model management and inference
  - `improvement_tracker.py` - Performance metrics and self-improvement analytics
- **Data Layer**: SQLite databases in `data/` + ChromaDB vector store
- **Configuration**: Runtime-modifiable settings in `config/settings.py`

### Development Workflow
- **Environment**: Use `.venv/` virtual environment (activated via `.\.venv\Scripts\Activate.ps1`)
- **Dependencies**: Install via `pip install -r requirements.txt`
- **Models**: Cached in `data/models/` (DialoGPT-medium/small by default)
- **Testing**: Run `pytest` in `tests/` directory (minimal unit tests for core components)
- **Logging**: All logs to `data/logs/ai_system.log` with custom AGI activity handlers

### Key Patterns & Conventions
- **Settings Modification**: Update `config/settings.py` constants, restart required for changes
- **Learning Parameters**: `TOPIC_SIMILARITY_THRESHOLD` (80), `LEARNING_RATE` (0.001) adjustable via GUI
- **Embedding Model**: Fixed to `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Feedback Integration**: Use `autonomous_learner.integrate_feedback()` with types: "good", "needs_work", "interesting", "perfect"
- **Thread Safety**: Use `message_queue.put()` for GUI updates from background threads
- **GPU Optimization**: RTX 4090 detection with automatic acceleration when available
- **Web Research**: Async capabilities via `AdvancedWebResearcher` for knowledge expansion

### Integration Points
- **Vector DB**: ChromaDB collection "ai_knowledge" for semantic storage/retrieval
- **LLM Interface**: Hugging Face transformers with model switching capability
- **External Research**: aiohttp-based web scraping with parallel processing
- **Analytics**: Matplotlib/Plotly for learning metrics visualization

### Common Development Tasks
- **Add New Learning Mechanism**: Extend `ContinualLearningEngine` with new pattern recognition
- **Enhance Knowledge Storage**: Modify `KnowledgeBase` schema and vector indexing
- **GUI Extension**: Add controls to `AISystemGUI` class with thread-safe message queuing
- **Model Integration**: Update `LLMInterface` for new Hugging Face architectures
- **Research Enhancement**: Extend `AdvancedWebResearcher` for new data sources

### Performance Considerations
- Experience buffer limited to 10,000 interactions
- Vector similarity searches use cosine similarity with configurable thresholds
- GPU acceleration automatically enabled for RTX series cards
- Memory consolidation runs in background threads to avoid blocking GUI