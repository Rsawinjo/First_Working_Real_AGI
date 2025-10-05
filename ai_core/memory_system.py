"""
Memory System - Long-term and short-term memory management
Implements memory consolidation, retrieval, and context management
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import deque, defaultdict
import threading
import time
import pickle
import sqlite3
from dataclasses import dataclass, asdict
import uuid

@dataclass
class MemoryItem:
    """Represents a single memory item"""
    id: str
    content: str
    memory_type: str  # episodic, semantic, procedural, emotional
    timestamp: str
    importance: float
    confidence: float
    access_count: int
    last_accessed: str
    decay_rate: float
    associations: List[str]
    emotional_valence: float  # -1 (negative) to 1 (positive)
    context: Dict[str, Any]
    metadata: Dict[str, Any]

class MemorySystem:
    def __init__(self, data_dir: str = "./data", config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.config = config or {}
        
        # Memory storage
        self.short_term_memory = deque(maxlen=1000)  # Recent memories
        self.working_memory = deque(maxlen=100)      # Current context
        self.episodic_buffer = deque(maxlen=5000)    # Personal experiences
        
        # Memory databases
        self.memory_db_path = os.path.join(data_dir, "memory.db")
        self.associations_db_path = os.path.join(data_dir, "associations.db")
        
        # Memory management parameters
        self.consolidation_threshold = 0.7
        self.forgetting_curve_factor = 0.9
        self.emotional_amplification = 1.5
        self.importance_decay_rate = 0.01
        
        # Memory metrics
        self.memory_stats = {
            'total_memories': 0,
            'consolidated_memories': 0,
            'forgotten_memories': 0,
            'retrieval_successes': 0,
            'retrieval_attempts': 0,
            'last_consolidation': datetime.now().isoformat()
        }
        
        # Association network
        self.memory_associations = defaultdict(list)
        self.association_strengths = defaultdict(float)
        
        # Initialize system
        self._initialize_memory_databases()
        self._start_memory_consolidation_loop()
    
    def _initialize_memory_databases(self):
        """Initialize SQLite databases for persistent memory storage"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        try:
            # Memory database
            with sqlite3.connect(self.memory_db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        memory_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        importance REAL NOT NULL,
                        confidence REAL NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        last_accessed TEXT,
                        decay_rate REAL DEFAULT 0.01,
                        emotional_valence REAL DEFAULT 0.0,
                        context TEXT,
                        metadata TEXT,
                        is_consolidated BOOLEAN DEFAULT 0
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)
                ''')
                
                conn.commit()
            
            # Associations database
            with sqlite3.connect(self.associations_db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS memory_associations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_memory_id TEXT NOT NULL,
                        target_memory_id TEXT NOT NULL,
                        association_type TEXT NOT NULL,
                        strength REAL NOT NULL,
                        created_at TEXT NOT NULL,
                        last_reinforced TEXT,
                        FOREIGN KEY (source_memory_id) REFERENCES memories (id),
                        FOREIGN KEY (target_memory_id) REFERENCES memories (id)
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_source_memory ON memory_associations(source_memory_id)
                ''')
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_target_memory ON memory_associations(target_memory_id)
                ''')
                
                conn.commit()
            
            self.logger.info("Memory databases initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing memory databases: {e}")
    
    def store_memory(self, content: str, memory_type: str = "episodic", 
                    importance: float = 0.5, confidence: float = 0.8,
                    emotional_valence: float = 0.0, context: Dict = None,
                    metadata: Dict = None) -> str:
        """Store a new memory"""
        try:
            # Create memory item
            memory_id = str(uuid.uuid4())
            memory = MemoryItem(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                timestamp=datetime.now().isoformat(),
                importance=importance,
                confidence=confidence,
                access_count=0,
                last_accessed=datetime.now().isoformat(),
                decay_rate=self._calculate_decay_rate(memory_type, importance),
                associations=[],
                emotional_valence=emotional_valence,
                context=context or {},
                metadata=metadata or {}
            )
            
            # Store in appropriate buffer
            if memory_type == "episodic":
                self.episodic_buffer.append(memory)
            
            # Always add to short-term memory first
            self.short_term_memory.append(memory)
            
            # Update working memory if relevant to current context
            if self._is_relevant_to_working_memory(memory):
                self.working_memory.append(memory)
            
            # Create associations with existing memories
            self._create_memory_associations(memory)
            
            # Update statistics
            self.memory_stats['total_memories'] += 1
            
            self.logger.debug(f"Memory stored: {memory_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")
            return None
    
    def _calculate_decay_rate(self, memory_type: str, importance: float) -> float:
        """Calculate decay rate based on memory type and importance"""
        base_rates = {
            "episodic": 0.02,      # Personal experiences
            "semantic": 0.005,     # Facts and knowledge
            "procedural": 0.001,   # Skills and procedures
            "emotional": 0.008     # Emotional memories
        }
        
        base_rate = base_rates.get(memory_type, 0.01)
        
        # Important memories decay slower
        importance_factor = max(0.1, 1.0 - importance)
        
        return base_rate * importance_factor
    
    def _is_relevant_to_working_memory(self, memory: MemoryItem) -> bool:
        """Determine if memory is relevant to current working memory"""
        if not self.working_memory:
            return True  # First memory is always relevant
        
        # Check for content similarity with recent working memory
        recent_contents = [m.content.lower() for m in list(self.working_memory)[-5:]]
        memory_content = memory.content.lower()
        
        # Simple keyword overlap check
        memory_words = set(memory_content.split())
        for content in recent_contents:
            content_words = set(content.split())
            overlap = len(memory_words.intersection(content_words))
            if overlap > 2:  # At least 2 word overlap
                return True
        
        # Check emotional relevance
        if abs(memory.emotional_valence) > 0.5:
            return True
        
        # Check importance
        if memory.importance > 0.7:
            return True
        
        return False
    
    def _create_memory_associations(self, new_memory: MemoryItem):
        """Create associations between new memory and existing memories"""
        try:
            # Find similar memories for association
            recent_memories = list(self.short_term_memory)[-50:]  # Last 50 memories
            
            for existing_memory in recent_memories:
                if existing_memory.id == new_memory.id:
                    continue
                
                # Calculate association strength
                association_strength = self._calculate_association_strength(
                    new_memory, existing_memory
                )
                
                if association_strength > 0.3:  # Threshold for creating association
                    # Store association
                    self._store_association(
                        new_memory.id, 
                        existing_memory.id, 
                        "content_similarity",
                        association_strength
                    )
                    
                    # Update memory objects
                    new_memory.associations.append(existing_memory.id)
                    existing_memory.associations.append(new_memory.id)
            
        except Exception as e:
            self.logger.error(f"Error creating memory associations: {e}")
    
    def _calculate_association_strength(self, memory1: MemoryItem, memory2: MemoryItem) -> float:
        """Calculate the strength of association between two memories"""
        try:
            strength = 0.0
            
            # Content similarity
            content_similarity = self._calculate_content_similarity(
                memory1.content, memory2.content
            )
            strength += content_similarity * 0.4
            
            # Temporal proximity
            time1 = datetime.fromisoformat(memory1.timestamp)
            time2 = datetime.fromisoformat(memory2.timestamp)
            time_diff = abs((time1 - time2).total_seconds())
            temporal_factor = max(0, 1.0 - (time_diff / 3600))  # 1 hour window
            strength += temporal_factor * 0.2
            
            # Emotional similarity
            emotional_similarity = 1.0 - abs(memory1.emotional_valence - memory2.emotional_valence) / 2.0
            strength += emotional_similarity * 0.2
            
            # Context similarity
            context_similarity = self._calculate_context_similarity(
                memory1.context, memory2.context
            )
            strength += context_similarity * 0.2
            
            return min(strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating association strength: {e}")
            return 0.0
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two content strings"""
        try:
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
            
        except Exception as e:
            self.logger.error(f"Error calculating content similarity: {e}")
            return 0.0
    
    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Calculate similarity between two context dictionaries"""
        try:
            if not context1 or not context2:
                return 0.0
            
            # Simple key overlap
            keys1 = set(context1.keys())
            keys2 = set(context2.keys())
            
            if not keys1 or not keys2:
                return 0.0
            
            common_keys = keys1.intersection(keys2)
            all_keys = keys1.union(keys2)
            
            return len(common_keys) / len(all_keys)
            
        except Exception as e:
            self.logger.error(f"Error calculating context similarity: {e}")
            return 0.0
    
    def _store_association(self, source_id: str, target_id: str, 
                          association_type: str, strength: float):
        """Store memory association in database"""
        try:
            with sqlite3.connect(self.associations_db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO memory_associations
                    (source_memory_id, target_memory_id, association_type, strength, 
                     created_at, last_reinforced)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (source_id, target_id, association_type, strength,
                      datetime.now().isoformat(), datetime.now().isoformat()))
                
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error storing association: {e}")
    
    def retrieve_memories(self, query: str, memory_type: str = None, 
                         limit: int = 10, min_confidence: float = 0.3) -> List[MemoryItem]:
        """Retrieve memories based on query"""
        try:
            self.memory_stats['retrieval_attempts'] += 1
            
            # Search in different memory stores
            candidates = []
            
            # Search working memory first (most relevant)
            candidates.extend(self._search_memory_buffer(
                list(self.working_memory), query, memory_type
            ))
            
            # Search short-term memory
            candidates.extend(self._search_memory_buffer(
                list(self.short_term_memory), query, memory_type
            ))
            
            # Search episodic buffer
            candidates.extend(self._search_memory_buffer(
                list(self.episodic_buffer), query, memory_type
            ))
            
            # Search consolidated memories in database
            consolidated_memories = self._search_consolidated_memories(
                query, memory_type, limit * 2
            )
            candidates.extend(consolidated_memories)
            
            # Remove duplicates
            seen_ids = set()
            unique_candidates = []
            for memory in candidates:
                if memory.id not in seen_ids:
                    seen_ids.add(memory.id)
                    unique_candidates.append(memory)
            
            # Filter by confidence
            filtered_candidates = [
                m for m in unique_candidates if m.confidence >= min_confidence
            ]
            
            # Calculate retrieval scores
            for memory in filtered_candidates:
                memory.retrieval_score = self._calculate_retrieval_score(memory, query)
            
            # Sort by retrieval score
            filtered_candidates.sort(key=lambda m: m.retrieval_score, reverse=True)
            
            # Update access statistics
            result_memories = filtered_candidates[:limit]
            self._update_memory_access(result_memories)
            
            if result_memories:
                self.memory_stats['retrieval_successes'] += 1
            
            return result_memories
            
        except Exception as e:
            self.logger.error(f"Error retrieving memories: {e}")
            return []
    
    def _search_memory_buffer(self, memory_buffer: List[MemoryItem], 
                             query: str, memory_type: str = None) -> List[MemoryItem]:
        """Search memories in a buffer"""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for memory in memory_buffer:
            # Type filter
            if memory_type and memory.memory_type != memory_type:
                continue
            
            # Content matching
            memory_content = memory.content.lower()
            memory_words = set(memory_content.split())
            
            # Calculate word overlap
            overlap = len(query_words.intersection(memory_words))
            if overlap > 0 or query_lower in memory_content:
                results.append(memory)
        
        return results
    
    def _search_consolidated_memories(self, query: str, memory_type: str = None, 
                                    limit: int = 20) -> List[MemoryItem]:
        """Search consolidated memories in database"""
        try:
            with sqlite3.connect(self.memory_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Build search query
                where_conditions = ["is_consolidated = 1"]
                params = []
                
                # Content search
                query_terms = query.lower().split()
                for term in query_terms:
                    where_conditions.append("LOWER(content) LIKE ?")
                    params.append(f"%{term}%")
                
                # Type filter
                if memory_type:
                    where_conditions.append("memory_type = ?")
                    params.append(memory_type)
                
                where_clause = " AND ".join(where_conditions)
                sql = f'''
                    SELECT * FROM memories 
                    WHERE {where_clause}
                    ORDER BY importance DESC, confidence DESC, last_accessed DESC
                    LIMIT ?
                '''
                params.append(limit)
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                # Convert to MemoryItem objects
                memories = []
                for row in rows:
                    memory = MemoryItem(
                        id=row['id'],
                        content=row['content'],
                        memory_type=row['memory_type'],
                        timestamp=row['timestamp'],
                        importance=row['importance'],
                        confidence=row['confidence'],
                        access_count=row['access_count'],
                        last_accessed=row['last_accessed'],
                        decay_rate=row['decay_rate'],
                        associations=[],  # Will be loaded separately if needed
                        emotional_valence=row['emotional_valence'],
                        context=json.loads(row['context'] or '{}'),
                        metadata=json.loads(row['metadata'] or '{}')
                    )
                    memories.append(memory)
                
                return memories
                
        except Exception as e:
            self.logger.error(f"Error searching consolidated memories: {e}")
            return []
    
    def _calculate_retrieval_score(self, memory: MemoryItem, query: str) -> float:
        """Calculate retrieval score for memory relevance"""
        try:
            score = 0.0
            
            # Content relevance
            content_score = self._calculate_content_similarity(memory.content, query)
            score += content_score * 0.4
            
            # Importance factor
            score += memory.importance * 0.2
            
            # Confidence factor
            score += memory.confidence * 0.1
            
            # Recency factor
            memory_time = datetime.fromisoformat(memory.timestamp)
            time_diff = (datetime.now() - memory_time).total_seconds()
            recency_score = max(0, 1.0 - (time_diff / (24 * 3600)))  # 24 hour window
            score += recency_score * 0.1
            
            # Access frequency factor
            access_score = min(memory.access_count / 100, 1.0)  # Normalize to 100 accesses
            score += access_score * 0.1
            
            # Emotional relevance
            if abs(memory.emotional_valence) > 0.3:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating retrieval score: {e}")
            return 0.0
    
    def _update_memory_access(self, memories: List[MemoryItem]):
        """Update access statistics for retrieved memories"""
        try:
            for memory in memories:
                memory.access_count += 1
                memory.last_accessed = datetime.now().isoformat()
                
                # Boost importance slightly on access
                memory.importance = min(memory.importance + 0.01, 1.0)
            
            # Update database for consolidated memories
            memory_ids = [m.id for m in memories]
            if memory_ids:
                with sqlite3.connect(self.memory_db_path) as conn:
                    cursor = conn.cursor()
                    
                    placeholders = ','.join('?' * len(memory_ids))
                    sql = f'''
                        UPDATE memories 
                        SET access_count = access_count + 1,
                            last_accessed = ?,
                            importance = MIN(importance + 0.01, 1.0)
                        WHERE id IN ({placeholders})
                    '''
                    params = [datetime.now().isoformat()] + memory_ids
                    cursor.execute(sql, params)
                    conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error updating memory access: {e}")
    
    def consolidate_memories(self) -> int:
        """Consolidate short-term memories to long-term storage"""
        try:
            consolidated_count = 0
            current_time = datetime.now()
            
            # Identify memories for consolidation
            memories_to_consolidate = []
            
            for memory in list(self.short_term_memory):
                memory_time = datetime.fromisoformat(memory.timestamp)
                age_hours = (current_time - memory_time).total_seconds() / 3600
                
                # Consolidation criteria
                should_consolidate = (
                    age_hours > 24 or  # Older than 24 hours
                    memory.importance > self.consolidation_threshold or  # High importance
                    memory.access_count > 10 or  # Frequently accessed
                    abs(memory.emotional_valence) > 0.7  # Emotionally significant
                )
                
                if should_consolidate:
                    memories_to_consolidate.append(memory)
            
            # Store in database
            with sqlite3.connect(self.memory_db_path) as conn:
                for memory in memories_to_consolidate:
                    conn.execute('''
                        INSERT OR REPLACE INTO memories
                        (id, content, memory_type, timestamp, importance, confidence,
                         access_count, last_accessed, decay_rate, emotional_valence,
                         context, metadata, is_consolidated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ''', (
                        memory.id, memory.content, memory.memory_type,
                        memory.timestamp, memory.importance, memory.confidence,
                        memory.access_count, memory.last_accessed, memory.decay_rate,
                        memory.emotional_valence, json.dumps(memory.context),
                        json.dumps(memory.metadata)
                    ))
                    consolidated_count += 1
                
                conn.commit()
            
            # Update statistics
            self.memory_stats['consolidated_memories'] += consolidated_count
            self.memory_stats['last_consolidation'] = current_time.isoformat()
            
            self.logger.info(f"Consolidated {consolidated_count} memories")
            return consolidated_count
            
        except Exception as e:
            self.logger.error(f"Error consolidating memories: {e}")
            return 0
    
    def forget_old_memories(self) -> int:
        """Apply forgetting curve to old, unimportant memories"""
        try:
            forgotten_count = 0
            current_time = datetime.now()
            
            # Process short-term memory
            memories_to_remove = []
            for memory in list(self.short_term_memory):
                # Apply decay
                memory_time = datetime.fromisoformat(memory.timestamp)
                age_days = (current_time - memory_time).days
                
                # Calculate current strength
                decay_factor = (self.forgetting_curve_factor ** (age_days * memory.decay_rate))
                current_strength = memory.importance * decay_factor
                
                # Forget if strength is too low
                if current_strength < 0.1 and memory.access_count < 2:
                    memories_to_remove.append(memory)
                    forgotten_count += 1
                else:
                    # Update importance with decay
                    memory.importance = max(current_strength, 0.1)
            
            # Remove forgotten memories
            for memory in memories_to_remove:
                try:
                    self.short_term_memory.remove(memory)
                except ValueError:
                    pass  # Memory already removed
            
            # Process database memories
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                # Find old, low-importance memories
                cursor.execute('''
                    DELETE FROM memories 
                    WHERE importance < 0.1 
                    AND access_count < 2 
                    AND julianday('now') - julianday(timestamp) > 30
                ''')
                
                db_forgotten = cursor.rowcount
                forgotten_count += db_forgotten
                conn.commit()
            
            # Update statistics
            self.memory_stats['forgotten_memories'] += forgotten_count
            
            self.logger.debug(f"Forgot {forgotten_count} memories")
            return forgotten_count
            
        except Exception as e:
            self.logger.error(f"Error forgetting memories: {e}")
            return 0
    
    def get_memory_context(self, max_memories: int = 10) -> List[Dict]:
        """Get current memory context for conversation"""
        try:
            # Get most relevant working memories
            working_context = list(self.working_memory)[-max_memories:]
            
            # Convert to context format
            context = []
            for memory in working_context:
                context.append({
                    'content': memory.content,
                    'type': memory.memory_type,
                    'importance': memory.importance,
                    'emotional_valence': memory.emotional_valence,
                    'timestamp': memory.timestamp
                })
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting memory context: {e}")
            return []
    
    def get_memory_stats(self) -> Dict:
        """Get memory system statistics"""
        try:
            stats = self.memory_stats.copy()
            
            # Current buffer sizes
            stats['short_term_size'] = len(self.short_term_memory)
            stats['working_memory_size'] = len(self.working_memory)
            stats['episodic_buffer_size'] = len(self.episodic_buffer)
            
            # Database statistics
            with sqlite3.connect(self.memory_db_path) as conn:
                cursor = conn.cursor()
                
                # Total memories in database
                cursor.execute("SELECT COUNT(*) FROM memories")
                stats['database_memories'] = cursor.fetchone()[0]
                
                # Memory type distribution
                cursor.execute('''
                    SELECT memory_type, COUNT(*) 
                    FROM memories 
                    GROUP BY memory_type
                ''')
                stats['memory_type_distribution'] = dict(cursor.fetchall())
                
                # Average importance and confidence
                cursor.execute('''
                    SELECT AVG(importance), AVG(confidence), AVG(access_count)
                    FROM memories
                ''')
                avg_stats = cursor.fetchone()
                stats['avg_importance'] = avg_stats[0] or 0
                stats['avg_confidence'] = avg_stats[1] or 0
                stats['avg_access_count'] = avg_stats[2] or 0
            
            # Association statistics
            with sqlite3.connect(self.associations_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memory_associations")
                stats['total_associations'] = cursor.fetchone()[0]
            
            # Retrieval success rate
            if stats['retrieval_attempts'] > 0:
                stats['retrieval_success_rate'] = (
                    stats['retrieval_successes'] / stats['retrieval_attempts']
                )
            else:
                stats['retrieval_success_rate'] = 0.0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {e}")
            return self.memory_stats
    
    def _start_memory_consolidation_loop(self):
        """Start background memory consolidation process"""
        def consolidation_loop():
            while True:
                try:
                    time.sleep(1800)  # Run every 30 minutes
                    
                    # Consolidate memories
                    self.consolidate_memories()
                    
                    # Apply forgetting
                    self.forget_old_memories()
                    
                    # Clean up old associations
                    self._cleanup_old_associations()
                    
                except Exception as e:
                    self.logger.error(f"Error in memory consolidation loop: {e}")
        
        consolidation_thread = threading.Thread(target=consolidation_loop, daemon=True)
        consolidation_thread.start()
    
    def _cleanup_old_associations(self):
        """Clean up old or weak associations"""
        try:
            with sqlite3.connect(self.associations_db_path) as conn:
                cursor = conn.cursor()
                
                # Remove weak associations older than 30 days
                cursor.execute('''
                    DELETE FROM memory_associations 
                    WHERE strength < 0.2 
                    AND julianday('now') - julianday(created_at) > 30
                ''')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error cleaning up associations: {e}")
    
    def export_memories(self, filepath: str):
        """Export memories to file"""
        try:
            export_data = {
                'short_term_memory': [asdict(m) for m in self.short_term_memory],
                'working_memory': [asdict(m) for m in self.working_memory],
                'episodic_buffer': [asdict(m) for m in self.episodic_buffer],
                'memory_stats': self.memory_stats,
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Add database memories
            with sqlite3.connect(self.memory_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM memories")
                export_data['database_memories'] = [dict(row) for row in cursor.fetchall()]
            
            # Add associations
            with sqlite3.connect(self.associations_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM memory_associations")
                export_data['associations'] = [dict(row) for row in cursor.fetchall()]
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Memories exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting memories: {e}")
            return False