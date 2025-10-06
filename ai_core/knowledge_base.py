"""
Knowledge Base System - Vector database for storing and retrieving learned information
Implements semantic search and knowledge graph capabilities
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
import sqlite3
import threading
from collections import defaultdict
import hashlib
import pickle
import time

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class KnowledgeBase:
    def __init__(self, data_dir: str = "./data", config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.config = config or {}
        
        # Database paths
        self.db_path = os.path.join(data_dir, "knowledge.db")
        self.vector_db_path = os.path.join(data_dir, "vector_db")
        
        # Knowledge storage
        self.knowledge_graph = defaultdict(list)
        self.semantic_memories = {}
        self.factual_knowledge = {}
        self.procedural_knowledge = {}
        
        # Vector database components
        self.vector_db = None
        self.embedding_model = None
        self.collection_name = "ai_knowledge"
        
        # Caching
        self.query_cache = {}
        self.embedding_cache = {}
        
        # Knowledge metrics
        self.knowledge_stats = {
            'total_entries': 0,
            'last_updated': datetime.now().isoformat(),
            'retrieval_count': 0,
            'successful_retrievals': 0
        }
        
        # Initialize components
        self._initialize_database()
        self._initialize_vector_store()
        self._start_maintenance_loop()
    
    def _initialize_database(self):
        """Initialize SQLite database for structured knowledge"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS knowledge_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content_hash TEXT UNIQUE,
                        content TEXT NOT NULL,
                        knowledge_type TEXT NOT NULL,
                        source TEXT,
                        confidence REAL DEFAULT 0.5,
                        importance REAL DEFAULT 0.5,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 0,
                        tags TEXT,
                        metadata TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS knowledge_relations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_id INTEGER,
                        target_id INTEGER,
                        relation_type TEXT,
                        strength REAL DEFAULT 0.5,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (source_id) REFERENCES knowledge_entries (id),
                        FOREIGN KEY (target_id) REFERENCES knowledge_entries (id)
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_content_hash ON knowledge_entries(content_hash)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge_entries(knowledge_type)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_created_at ON knowledge_entries(created_at)
                ''')
                
                # Create compatibility view/table for legacy code
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS knowledge (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        knowledge_type TEXT NOT NULL,
                        confidence REAL DEFAULT 0.5,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
            
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def _initialize_vector_store(self):
        """Initialize vector database for semantic search"""
        try:
            # Initialize embedding model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                model_name = self.config.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
                self.embedding_model = SentenceTransformer(model_name)
                self.logger.info(f"Embedding model loaded: {model_name}")
            else:
                self.logger.warning("SentenceTransformers not available, using simple embeddings")
                self.embedding_model = None
            
            # Initialize ChromaDB if available
            if CHROMADB_AVAILABLE:
                os.makedirs(self.vector_db_path, exist_ok=True)
                client = chromadb.PersistentClient(path=self.vector_db_path)
                
                # Get or create collection
                try:
                    self.vector_db = client.get_collection(name=self.collection_name)
                    self.logger.info(f"Connected to existing collection: {self.collection_name}")
                except ValueError:
                    # Collection doesn't exist, create it
                    self.vector_db = client.create_collection(name=self.collection_name)
                    self.logger.info(f"Created new collection: {self.collection_name}")
                except Exception as e:
                    # Try to create collection anyway
                    self.logger.warning(f"Collection access error: {e}, creating new one...")
                    try:
                        self.vector_db = client.create_collection(name=self.collection_name)
                        self.logger.info(f"Created new collection: {self.collection_name}")
                    except Exception as e2:
                        self.logger.error(f"Failed to create collection: {e2}")
                        self.vector_db = None
                
                self.logger.info("Vector database initialized successfully")
            else:
                self.logger.warning("ChromaDB not available, using in-memory storage")
                self.vector_db = None
        
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {e}")
            self.vector_db = None
            self.embedding_model = None
    
    def store_knowledge(self, content: str, knowledge_type: str = "general", 
                       source: str = None, confidence: float = 0.5, 
                       importance: float = 0.5, tags: List[str] = None,
                       metadata: Dict = None) -> str:
        """Store knowledge in the database"""
        try:
            # Create content hash for deduplication
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Prepare data
            tags_str = json.dumps(tags or [])
            metadata_str = json.dumps(metadata or {})
            
            # Store in SQLite
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if knowledge already exists
                cursor.execute(
                    "SELECT id FROM knowledge_entries WHERE content_hash = ?",
                    (content_hash,)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing entry
                    cursor.execute('''
                        UPDATE knowledge_entries 
                        SET confidence = MAX(confidence, ?), 
                            importance = MAX(importance, ?),
                            last_accessed = CURRENT_TIMESTAMP,
                            access_count = access_count + 1
                        WHERE content_hash = ?
                    ''', (confidence, importance, content_hash))
                    entry_id = existing[0]
                else:
                    # Insert new entry
                    cursor.execute('''
                        INSERT INTO knowledge_entries 
                        (content_hash, content, knowledge_type, source, confidence, 
                         importance, tags, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (content_hash, content, knowledge_type, source, 
                          confidence, importance, tags_str, metadata_str))
                    entry_id = cursor.lastrowid
                    
                    # Also store in legacy knowledge table for compatibility
                    cursor.execute('''
                        INSERT INTO knowledge 
                        (content, knowledge_type, confidence, metadata)
                        VALUES (?, ?, ?, ?)
                    ''', (content, knowledge_type, confidence, metadata_str))
                
                conn.commit()
            
            # Store in vector database for semantic search
            self._store_vector_embedding(content, str(entry_id), metadata or {})
            
            # Update knowledge graph
            self._update_knowledge_graph(content, entry_id, knowledge_type)
            
            # Update statistics
            self.knowledge_stats['total_entries'] += 1 if not existing else 0
            self.knowledge_stats['last_updated'] = datetime.now().isoformat()
            
            self.logger.debug(f"Knowledge stored: {content[:50]}...")
            return content_hash
            
        except Exception as e:
            self.logger.error(f"Error storing knowledge: {e}")
            return None
    
    def _store_vector_embedding(self, content: str, doc_id: str, metadata: Dict):
        """Store content embedding in vector database"""
        try:
            if self.vector_db and self.embedding_model:
                # Generate embedding
                embedding = self.embedding_model.encode(content).tolist()
                
                # Store in ChromaDB
                self.vector_db.add(
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
            
            elif self.embedding_model:
                # Store in memory if ChromaDB not available
                embedding = self.embedding_model.encode(content)
                self.embedding_cache[doc_id] = {
                    'embedding': embedding,
                    'content': content,
                    'metadata': metadata
                }
        
        except Exception as e:
            self.logger.error(f"Error storing vector embedding: {e}")
    
    def _update_knowledge_graph(self, content: str, entry_id: int, knowledge_type: str):
        """Update knowledge graph with new connections"""
        try:
            # Extract key concepts from content
            concepts = self._extract_concepts(content)
            
            # Create nodes and relationships
            for concept in concepts:
                if concept not in self.knowledge_graph:
                    self.knowledge_graph[concept] = []
                
                # Add connection to this knowledge entry
                self.knowledge_graph[concept].append({
                    'entry_id': entry_id,
                    'knowledge_type': knowledge_type,
                    'strength': 1.0,
                    'created_at': datetime.now().isoformat()
                })
                
                # Limit connections per concept
                if len(self.knowledge_graph[concept]) > 100:
                    # Keep most recent and strongest connections
                    self.knowledge_graph[concept] = sorted(
                        self.knowledge_graph[concept],
                        key=lambda x: (x['strength'], x['created_at']),
                        reverse=True
                    )[:80]
        
        except Exception as e:
            self.logger.error(f"Error updating knowledge graph: {e}")
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        # Simple concept extraction (can be enhanced with NLP)
        words = text.lower().split()
        
        # Filter meaningful words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'i', 
            'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 
            'those', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would'
        }
        
        concepts = []
        for word in words:
            if (len(word) > 3 and 
                word not in stop_words and 
                word.isalpha()):
                concepts.append(word)
        
        return list(set(concepts))
    
    def retrieve_knowledge(self, query: str, knowledge_type: str = None, 
                          limit: int = 10, min_confidence: float = 0.3) -> List[Dict]:
        """Retrieve relevant knowledge based on query"""
        try:
            self.knowledge_stats['retrieval_count'] += 1
            
            # Check cache first
            cache_key = f"{query}_{knowledge_type}_{limit}_{min_confidence}"
            if cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                if (datetime.now() - cached_result['timestamp']).seconds < 300:  # 5 minutes
                    return cached_result['results']
            
            results = []
            
            # Semantic search using vector database
            if self.vector_db and self.embedding_model:
                semantic_results = self._semantic_search(query, limit * 2)
                results.extend(semantic_results)
            
            # Keyword-based search in SQLite
            keyword_results = self._keyword_search(query, knowledge_type, limit, min_confidence)
            results.extend(keyword_results)
            
            # Knowledge graph search
            graph_results = self._graph_search(query, limit)
            results.extend(graph_results)
            
            # Deduplicate and rank results
            results = self._deduplicate_and_rank_results(results, query)
            
            # Limit results
            results = results[:limit]
            
            # Update access statistics
            self._update_access_stats(results)
            
            # Cache results
            self.query_cache[cache_key] = {
                'results': results,
                'timestamp': datetime.now()
            }
            
            if results:
                self.knowledge_stats['successful_retrievals'] += 1
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving knowledge: {e}")
            return []
    
    def _semantic_search(self, query: str, limit: int) -> List[Dict]:
        """Perform semantic search using vector embeddings"""
        try:
            if self.vector_db and self.embedding_model:
                # Query ChromaDB
                results = self.vector_db.query(
                    query_texts=[query],
                    n_results=min(limit, 100)
                )
                
                semantic_results = []
                if results['documents'] and results['documents'][0]:
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                    )):
                        semantic_results.append({
                            'content': doc,
                            'knowledge_type': metadata.get('knowledge_type', 'general'),
                            'confidence': 1.0 - min(distance, 1.0),  # Convert distance to confidence
                            'source': 'semantic_search',
                            'metadata': metadata,
                            'relevance_score': 1.0 - min(distance, 1.0)
                        })
                
                return semantic_results
            
            elif self.embedding_model and self.embedding_cache:
                # Use in-memory embeddings
                query_embedding = self.embedding_model.encode(query)
                similarities = []
                
                for doc_id, cached_data in self.embedding_cache.items():
                    similarity = self._cosine_similarity(
                        query_embedding, cached_data['embedding']
                    )
                    similarities.append((doc_id, similarity, cached_data))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                results = []
                for doc_id, similarity, data in similarities[:limit]:
                    results.append({
                        'content': data['content'],
                        'confidence': similarity,
                        'source': 'semantic_search',
                        'metadata': data['metadata'],
                        'relevance_score': similarity
                    })
                
                return results
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except:
            return 0.0
    
    def _keyword_search(self, query: str, knowledge_type: str, 
                       limit: int, min_confidence: float) -> List[Dict]:
        """Perform keyword-based search in SQLite database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Build search query
                search_terms = query.lower().split()
                where_conditions = []
                params = []
                
                # Content search
                for term in search_terms:
                    where_conditions.append("LOWER(content) LIKE ?")
                    params.append(f"%{term}%")
                
                # Knowledge type filter
                if knowledge_type:
                    where_conditions.append("knowledge_type = ?")
                    params.append(knowledge_type)
                
                # Confidence filter
                where_conditions.append("confidence >= ?")
                params.append(min_confidence)
                
                # Execute query
                where_clause = " AND ".join(where_conditions)
                sql = f'''
                    SELECT * FROM knowledge_entries 
                    WHERE {where_clause}
                    ORDER BY importance DESC, confidence DESC, last_accessed DESC
                    LIMIT ?
                '''
                params.append(limit)
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                # Convert to result format
                results = []
                for row in rows:
                    # Calculate relevance score based on term matches
                    content_lower = row['content'].lower()
                    matches = sum(1 for term in search_terms if term in content_lower)
                    relevance_score = matches / len(search_terms)
                    
                    results.append({
                        'id': row['id'],
                        'content': row['content'],
                        'knowledge_type': row['knowledge_type'],
                        'source': row['source'] or 'database',
                        'confidence': row['confidence'],
                        'importance': row['importance'],
                        'created_at': row['created_at'],
                        'last_accessed': row['last_accessed'],
                        'access_count': row['access_count'],
                        'tags': json.loads(row['tags'] or '[]'),
                        'metadata': json.loads(row['metadata'] or '{}'),
                        'relevance_score': relevance_score
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error in keyword search: {e}")
            return []
    
    def _graph_search(self, query: str, limit: int) -> List[Dict]:
        """Search using knowledge graph connections"""
        try:
            # Extract concepts from query
            query_concepts = self._extract_concepts(query)
            
            # Find connected knowledge entries
            connected_entries = set()
            concept_scores = {}
            
            for concept in query_concepts:
                if concept in self.knowledge_graph:
                    for connection in self.knowledge_graph[concept]:
                        entry_id = connection['entry_id']
                        connected_entries.add(entry_id)
                        
                        # Calculate concept relevance score
                        if entry_id not in concept_scores:
                            concept_scores[entry_id] = 0
                        concept_scores[entry_id] += connection['strength']
            
            # Retrieve the connected entries
            if not connected_entries:
                return []
            
            results = []
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get entries by IDs
                placeholders = ','.join('?' * len(connected_entries))
                sql = f'''
                    SELECT * FROM knowledge_entries 
                    WHERE id IN ({placeholders})
                    ORDER BY importance DESC, confidence DESC
                    LIMIT ?
                '''
                params = list(connected_entries) + [limit]
                
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                
                for row in rows:
                    relevance_score = concept_scores.get(row['id'], 0.5)
                    
                    results.append({
                        'id': row['id'],
                        'content': row['content'],
                        'knowledge_type': row['knowledge_type'],
                        'source': row['source'] or 'knowledge_graph',
                        'confidence': row['confidence'],
                        'importance': row['importance'],
                        'relevance_score': relevance_score,
                        'metadata': json.loads(row['metadata'] or '{}')
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in graph search: {e}")
            return []
    
    def _deduplicate_and_rank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Remove duplicates and rank results by relevance"""
        try:
            # Deduplicate by content hash
            seen_content = set()
            unique_results = []
            
            for result in results:
                content = result['content']
                content_hash = hashlib.sha256(content.encode()).hexdigest()
                
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_results.append(result)
            
            # Calculate final relevance scores
            for result in unique_results:
                relevance = result.get('relevance_score', 0.5)
                confidence = result.get('confidence', 0.5)
                importance = result.get('importance', 0.5)
                
                # Combined score
                final_score = (
                    relevance * 0.5 +
                    confidence * 0.3 +
                    importance * 0.2
                )
                result['final_score'] = final_score
            
            # Sort by final score
            unique_results.sort(key=lambda x: x['final_score'], reverse=True)
            
            return unique_results
            
        except Exception as e:
            self.logger.error(f"Error in deduplication and ranking: {e}")
            return results
    
    def _update_access_stats(self, results: List[Dict]):
        """Update access statistics for retrieved knowledge"""
        try:
            if not results:
                return
            
            entry_ids = [r['id'] for r in results if 'id' in r]
            if not entry_ids:
                return
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update access count and timestamp
                placeholders = ','.join('?' * len(entry_ids))
                sql = f'''
                    UPDATE knowledge_entries 
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE id IN ({placeholders})
                '''
                cursor.execute(sql, entry_ids)
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating access stats: {e}")
    
    def add_knowledge_relation(self, source_content: str, target_content: str, 
                              relation_type: str, strength: float = 0.5):
        """Add a relationship between two pieces of knowledge"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find source and target IDs
                source_hash = hashlib.sha256(source_content.encode()).hexdigest()
                target_hash = hashlib.sha256(target_content.encode()).hexdigest()
                
                cursor.execute(
                    "SELECT id FROM knowledge_entries WHERE content_hash = ?",
                    (source_hash,)
                )
                source_row = cursor.fetchone()
                
                cursor.execute(
                    "SELECT id FROM knowledge_entries WHERE content_hash = ?",
                    (target_hash,)
                )
                target_row = cursor.fetchone()
                
                if source_row and target_row:
                    source_id, target_id = source_row[0], target_row[0]
                    
                    # Insert relationship
                    cursor.execute('''
                        INSERT OR REPLACE INTO knowledge_relations
                        (source_id, target_id, relation_type, strength)
                        VALUES (?, ?, ?, ?)
                    ''', (source_id, target_id, relation_type, strength))
                    
                    conn.commit()
                    self.logger.debug(f"Added relation: {relation_type}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error adding knowledge relation: {e}")
            return False
    
    def get_knowledge_stats(self) -> Dict:
        """Get knowledge base statistics"""
        try:
            stats = self.knowledge_stats.copy()
            
            # Get detailed stats from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total entries
                cursor.execute("SELECT COUNT(*) FROM knowledge_entries")
                stats['total_entries'] = cursor.fetchone()[0]
                
                # Knowledge types distribution
                cursor.execute('''
                    SELECT knowledge_type, COUNT(*) 
                    FROM knowledge_entries 
                    GROUP BY knowledge_type
                ''')
                stats['knowledge_types'] = dict(cursor.fetchall())
                
                # Average confidence and importance
                cursor.execute('''
                    SELECT AVG(confidence), AVG(importance), AVG(access_count)
                    FROM knowledge_entries
                ''')
                avg_stats = cursor.fetchone()
                stats['avg_confidence'] = avg_stats[0] or 0
                stats['avg_importance'] = avg_stats[1] or 0
                stats['avg_access_count'] = avg_stats[2] or 0
                
                # Most accessed knowledge
                cursor.execute('''
                    SELECT content, access_count 
                    FROM knowledge_entries 
                    ORDER BY access_count DESC 
                    LIMIT 5
                ''')
                stats['most_accessed'] = [
                    {'content': row[0][:100], 'access_count': row[1]}
                    for row in cursor.fetchall()
                ]
            
            # Vector database stats
            if self.vector_db:
                try:
                    collection_info = self.vector_db.count()
                    stats['vector_entries'] = collection_info
                except:
                    stats['vector_entries'] = len(self.embedding_cache)
            else:
                stats['vector_entries'] = len(self.embedding_cache)
            
            # Knowledge graph stats
            stats['concept_count'] = len(self.knowledge_graph)
            stats['total_connections'] = sum(
                len(connections) for connections in self.knowledge_graph.values()
            )
            
            # Cache stats
            stats['query_cache_size'] = len(self.query_cache)
            stats['embedding_cache_size'] = len(self.embedding_cache)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge stats: {e}")
            return self.knowledge_stats
    
    def _start_maintenance_loop(self):
        """Start background maintenance tasks"""
        def maintenance_loop():
            while True:
                time.sleep(3600)  # Run every hour
                try:
                    self._cleanup_cache()
                    self._optimize_database()
                    self._update_knowledge_importance()
                except Exception as e:
                    self.logger.error(f"Error in maintenance loop: {e}")
        
        maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
        maintenance_thread.start()
    
    def _cleanup_cache(self):
        """Clean up old cache entries"""
        current_time = datetime.now()
        
        # Clean query cache (remove entries older than 1 hour)
        expired_keys = []
        for key, data in self.query_cache.items():
            if (current_time - data['timestamp']).seconds > 3600:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.query_cache[key]
        
        # Limit embedding cache size
        if len(self.embedding_cache) > 10000:
            # Keep most recent 8000 entries
            sorted_items = sorted(
                self.embedding_cache.items(),
                key=lambda x: x[1].get('timestamp', ''),
                reverse=True
            )
            self.embedding_cache = dict(sorted_items[:8000])
    
    def _optimize_database(self):
        """Optimize database performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error optimizing database: {e}")
    
    def _update_knowledge_importance(self):
        """Update importance scores based on access patterns"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update importance based on access count and recency
                cursor.execute('''
                    UPDATE knowledge_entries 
                    SET importance = MIN(1.0, 
                        importance + (access_count * 0.01) + 
                        CASE 
                            WHEN julianday('now') - julianday(last_accessed) < 7 
                            THEN 0.05 
                            ELSE 0 
                        END
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating knowledge importance: {e}")
    
    def export_knowledge(self, filepath: str, format: str = 'json'):
        """Export knowledge base to file"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM knowledge_entries ORDER BY created_at")
                entries = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT * FROM knowledge_relations ORDER BY created_at")
                relations = [dict(row) for row in cursor.fetchall()]
            
            export_data = {
                'knowledge_entries': entries,
                'knowledge_relations': relations,
                'knowledge_graph': dict(self.knowledge_graph),
                'export_timestamp': datetime.now().isoformat(),
                'stats': self.get_knowledge_stats()
            }
            
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format.lower() == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(export_data, f)
            
            self.logger.info(f"Knowledge exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting knowledge: {e}")
            return False
    
    def import_knowledge(self, filepath: str, format: str = 'json'):
        """Import knowledge from file"""
        try:
            if format.lower() == 'json':
                with open(filepath, 'r') as f:
                    import_data = json.load(f)
            elif format.lower() == 'pickle':
                with open(filepath, 'rb') as f:
                    import_data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Import knowledge entries
            entries = import_data.get('knowledge_entries', [])
            for entry in entries:
                self.store_knowledge(
                    content=entry['content'],
                    knowledge_type=entry['knowledge_type'],
                    source=entry.get('source'),
                    confidence=entry.get('confidence', 0.5),
                    importance=entry.get('importance', 0.5),
                    tags=json.loads(entry.get('tags', '[]')),
                    metadata=json.loads(entry.get('metadata', '{}'))
                )
            
            # Import knowledge graph
            knowledge_graph = import_data.get('knowledge_graph', {})
            for concept, connections in knowledge_graph.items():
                self.knowledge_graph[concept].extend(connections)
            
            self.logger.info(f"Knowledge imported from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing knowledge: {e}")
            return False
    
    # ========== MISSING METHODS FOR AUTONOMOUS LEARNER ==========
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search knowledge base for relevant information"""
        try:
            if self.vector_db:
                # Semantic search using ChromaDB
                results = self.vector_db.query(
                    query_texts=[query],
                    n_results=limit
                )
                
                # Format results
                search_results = []
                if results['documents'] and results['documents'][0]:
                    for i, doc in enumerate(results['documents'][0]):
                        result = {
                            'content': doc,
                            'score': 1.0 - results['distances'][0][i] if results['distances'] else 0.5,
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                        }
                        search_results.append(result)
                
                return search_results
            else:
                # Fallback: simple text search in database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT content, confidence, metadata FROM knowledge_entries 
                    WHERE content LIKE ? OR tags LIKE ?
                    ORDER BY confidence DESC LIMIT ?
                ''', (f"%{query}%", f"%{query}%", limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'content': row[0],
                        'score': row[1],
                        'metadata': json.loads(row[2]) if row[2] else {}
                    })
                
                conn.close()
                return results
                
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get knowledge base statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basic stats from knowledge_entries table
            cursor.execute('SELECT COUNT(*) FROM knowledge_entries')
            total_entries = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT knowledge_type) FROM knowledge_entries')
            knowledge_types = cursor.fetchone()[0]
            
            cursor.execute('SELECT knowledge_type, COUNT(*) FROM knowledge_entries GROUP BY knowledge_type')
            type_distribution = dict(cursor.fetchall())
            
            cursor.execute('SELECT AVG(confidence) FROM knowledge_entries')
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            conn.close()
            
            stats = {
                'total_entries': total_entries,
                'knowledge_types': knowledge_types,
                'type_distribution': type_distribution,
                'average_confidence': avg_confidence,
                'knowledge_graph_size': len(self.knowledge_graph),
                'vector_db_available': self.vector_db is not None
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting knowledge statistics: {e}")
            return {'total_entries': 0, 'knowledge_types': 0}
    
    def add_knowledge(self, content: str, metadata: Dict = None):
        """Add knowledge to the knowledge base (alias for store_knowledge)"""
        return self.store_knowledge(
            content=content,
            knowledge_type="autonomous_learning",
            metadata=metadata or {},
            confidence=0.7
        )