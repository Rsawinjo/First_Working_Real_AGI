# Configuration for the AI Self-Improvement System

# Model Settings
DEFAULT_MODEL = "microsoft/DialoGPT-medium"
FALLBACK_MODEL = "microsoft/DialoGPT-small"
MODEL_CACHE_DIR = "./data/models"

# Learning Parameters
LEARNING_RATE = 0.001
MEMORY_RETENTION_DAYS = 30
KNOWLEDGE_THRESHOLD = 0.75
IMPROVEMENT_INTERVAL = 100  # conversations

# Topic Filtering
TOPIC_SIMILARITY_THRESHOLD = 40  # rapidfuzz token_sort_ratio threshold for topic similarity

# Vector Database Settings
VECTOR_DB_PATH = "./data/knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384

# GUI Settings
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
CHAT_HISTORY_LIMIT = 1000

# Research Settings
RESEARCH_ENABLED = True
MAX_RESEARCH_RESULTS = 5
RESEARCH_DEPTH = 3

# Performance Settings
MAX_CONCURRENT_REQUESTS = 3
CACHE_SIZE = 1000
BATCH_SIZE = 8

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "./data/ai_system.log"

# Auto-improvement settings
AUTO_IMPROVE = True
IMPROVEMENT_THRESHOLD = 0.8
CURIOSITY_FACTOR = 0.3