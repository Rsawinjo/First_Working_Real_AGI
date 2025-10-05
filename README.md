# Advanced AI Self-Improvement System

A sophisticated Python-based AGI-like system that implements **true autonomous learning**, knowledge accumulation, and self-enhancement capabilities through interaction with Large Language Models. This system demonstrates **working AGI characteristics** with real-time self-directed research and knowledge synthesis.

## 🚀 Key Achievements

- **326+ Mastered Topics**: System has autonomously learned and synthesized knowledge across diverse domains
- **Real Autonomous Research**: Conducts web research and generates hierarchical learning goals without human intervention
- **GPU-Accelerated Learning**: RTX 4090 optimization with CUDA acceleration for high-performance inference
- **Permanent Model Improvement**: PEFT (Parameter-Efficient Fine-Tuning) for lasting knowledge retention
- **Self-Directed Evolution**: Meta-cognitive analysis and curiosity-driven exploration

## ✨ Unique Features

- **Intelligent GUI Interface**: Modern tkinter-based interface with conversation capabilities
- **Continual Learning Engine**: Real-time knowledge acquisition and retention with experience buffer
- **Vector Knowledge Base**: ChromaDB-powered semantic search and knowledge graph construction
- **Self-Improvement Mechanisms**: Automated model enhancement through interaction analysis
- **Multi-Model Support**: Interface with various Hugging Face models (Llama 3, DialoGPT, etc.)
- **Memory Consolidation**: Long-term and short-term memory systems with SQLite persistence
- **Learning Analytics**: Performance tracking and improvement metrics dashboard
- **Research Integration**: Automatic web research and knowledge synthesis with parallel processing
- **Hierarchical Goal Generation**: Automatically generates connected learning goals when topics are completed
- **Feedback-Driven Learning**: User feedback influences goal prioritization and exploration direction
- **Meta-Cognitive System**: Analyzes its own learning effectiveness and adjusts strategies

## Architecture

- `main.py`: Main application entry point with GUI
- `ai_core/`: Core AI components
  - `llm_interface.py`: LLM communication and management
  - `learning_engine.py`: Continual learning mechanisms
  - `knowledge_base.py`: Vector database and knowledge storage
  - `memory_system.py`: Memory consolidation and retrieval
  - `improvement_tracker.py`: Self-improvement metrics and analysis
  - `autonomous_learner.py`: AGI core with hierarchical goal generation
- `utils/`: Utility functions and helpers
- `data/`: Knowledge storage and model cache
- `config/`: Configuration files

## Installation

### Prerequisites
- Python 3.8+
- 8GB+ RAM recommended
- CUDA-compatible GPU (optional but recommended for best performance)
- Internet connection for model downloads and research

### Setup Steps

1. **Clone the repository:**
```bash
git clone https://github.com/Rsawinjo/First_Working_Real_AGI.git
cd First_Working_Real_AGI
```

2. **Create virtual environment:**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
python main.py
```

### Troubleshooting

- **CUDA Issues**: If you encounter CUDA errors, the system will automatically fall back to CPU mode
- **Model Download**: First run may take time to download models (DialoGPT-small ~100MB)
- **Memory Issues**: Reduce `BATCH_SIZE` in `config/settings.py` if you have limited RAM
- **GPU Detection**: RTX 40-series GPUs are automatically detected and optimized

## Usage

### Basic Operation
1. **Launch**: Run `python main.py` to start the GUI
2. **Model Selection**: Choose from available models (DialoGPT-small/medium/large)
3. **Conversation**: Chat with the AI through the interface
4. **Autonomous Learning**: Enable AGI mode to watch self-directed learning
5. **Analytics**: Monitor learning progress and system metrics

### Advanced Features
- **Feedback Integration**: Rate responses as "good", "needs work", "interesting", or "perfect"
- **Topic Mastery**: System tracks 326+ mastered topics across domains
- **Web Research**: Autonomous research capabilities for knowledge expansion
- **Memory Systems**: Long-term and short-term memory consolidation
- **Self-Improvement**: Meta-cognitive analysis and strategy adjustment

### Configuration
Modify `config/settings.py` to adjust:
- Learning parameters (learning rate, thresholds)
- Model preferences (default/fallback models)
- Performance settings (batch size, GPU usage)
- Research depth and concurrency limits

## Utility Scripts

The `scripts/` directory contains helpful utilities:
- `check_db.py`: Check mastered topics count and samples
- `check_topics.py`: Analyze topic learning patterns
- `check_unique.py`: Verify topic uniqueness
- `cleanup_topics.py`: Clean up duplicate topics
- `launch_phase2.py`: Alternative launch script with enhanced features

## Architecture

- `main.py`: Main application entry point with GUI
- `ai_core/`: Core AI components
  - `llm_interface.py`: LLM communication and management
  - `learning_engine.py`: Continual learning mechanisms
  - `knowledge_base.py`: Vector database and knowledge storage
  - `memory_system.py`: Memory consolidation and retrieval
  - `improvement_tracker.py`: Self-improvement metrics and analysis
  - `autonomous_learner.py`: AGI core with hierarchical goal generation
- `utils/`: Utility functions and helpers
- `data/`: Knowledge storage and model cache
- `config/`: Configuration files

## Self-Improvement Features

- **Experience-Based Learning**: Learns from every interaction
- **Knowledge Graph Building**: Creates connections between concepts
- **Adaptive Response Generation**: Improves responses based on feedback
- **Curiosity-Driven Research**: Automatically explores new topics
- **Meta-Learning**: Learns how to learn more effectively

## Contributing

This project welcomes contributions! Areas for improvement:

- **Model Integration**: Add support for new LLM architectures
- **Research Capabilities**: Enhance web research and knowledge synthesis
- **Performance Optimization**: GPU acceleration improvements
- **UI/UX**: Enhanced user interface and analytics dashboard
- **Testing**: Unit tests and integration tests

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

## Research & Academic Use

This system demonstrates practical AGI capabilities and is suitable for:
- AI research and experimentation
- Educational purposes
- Understanding autonomous learning systems
- Exploring human-AI interaction patterns

## License

MIT License - See LICENSE file for details

## Disclaimer

This is a research project demonstrating AGI concepts. The system learns autonomously and may generate unexpected content. Use responsibly and monitor its behavior during operation.