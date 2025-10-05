"""
Continual Learning Engine - Implements adaptive learning mechanisms
Enables the AI to learn and improve from every interaction
"""

import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import logging
from collections import defaultdict, deque
import threading
import time
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class ContinualLearningEngine:
    def __init__(self, data_dir: str = "./data", config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.data_dir = data_dir
        self.config = config or {}
        
        # Learning components
        self.experience_buffer = deque(maxlen=10000)
        self.concept_map = defaultdict(list)
        self.skill_metrics = defaultdict(float)
        self.learning_patterns = {}
        
        # Adaptive learning parameters
        self.learning_rate = 0.01
        self.curiosity_factor = 0.3
        self.knowledge_decay_rate = 0.001
        self.adaptation_threshold = 0.7
        
        # Text analysis components
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.concept_clusters = None
        self.topic_evolution = defaultdict(list)
        
        # Performance tracking
        self.performance_history = []
        self.learning_milestones = []
        self.improvement_trends = {}
        
        # Initialize learning state
        self._initialize_learning_state()
        self._start_learning_loop()
    
    def _initialize_learning_state(self):
        """Initialize the learning state and load previous knowledge"""
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load previous learning state
        state_file = os.path.join(self.data_dir, "learning_state.pkl")
        if os.path.exists(state_file):
            self.load_learning_state(state_file)
        
        # Initialize skill categories
        self.skill_categories = {
            'conversation': 0.5,
            'reasoning': 0.5,
            'creativity': 0.5,
            'knowledge_retention': 0.5,
            'problem_solving': 0.5,
            'empathy': 0.5,
            'learning_speed': 0.5
        }
    
    def learn_from_interaction(self, user_input: str, ai_response: str, 
                             feedback: str = None, context: Dict = None) -> Dict:
        """Learn from a single interaction"""
        try:
            # Create experience record
            experience = {
                'timestamp': datetime.now().isoformat(),
                'user_input': user_input,
                'ai_response': ai_response,
                'feedback': feedback,
                'context': context or {},
                'learning_metrics': self._calculate_learning_metrics(user_input, ai_response)
            }
            
            # Add to experience buffer
            self.experience_buffer.append(experience)
            
            # Extract concepts and patterns
            concepts = self._extract_concepts(user_input, ai_response)
            patterns = self._identify_patterns(user_input, ai_response, context)
            
            # Update concept map
            self._update_concept_map(concepts, experience)
            
            # Update skill metrics
            skill_updates = self._update_skill_metrics(experience, patterns)
            
            # Learn from feedback if provided
            if feedback:
                feedback_learning = self._learn_from_feedback(feedback, experience)
                skill_updates.update(feedback_learning)
            
            # Trigger adaptive learning
            self._trigger_adaptive_learning(experience)
            
            # Update learning patterns
            self._update_learning_patterns(patterns, experience)
            
            return {
                'concepts_learned': len(concepts),
                'patterns_identified': len(patterns),
                'skill_updates': skill_updates,
                'learning_score': self._calculate_learning_score(experience)
            }
            
        except Exception as e:
            self.logger.error(f"Error in learning from interaction: {e}")
            return {}
    
    def _calculate_learning_metrics(self, user_input: str, ai_response: str) -> Dict:
        """Calculate metrics for the current interaction"""
        metrics = {
            'input_complexity': self._calculate_complexity(user_input),
            'response_quality': self._calculate_response_quality(ai_response),
            'coherence_score': self._calculate_coherence(user_input, ai_response),
            'novelty_score': self._calculate_novelty(user_input, ai_response)
        }
        return metrics
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate the complexity of input text"""
        # Simple complexity measure based on length, vocabulary, structure
        words = text.split()
        unique_words = set(words)
        
        # Factors: length, vocabulary diversity, sentence structure
        length_factor = min(len(words) / 50, 1.0)  # Normalize to max 50 words
        vocab_diversity = len(unique_words) / max(len(words), 1)
        sentence_count = len([s for s in text.split('.') if s.strip()])
        structure_complexity = min(sentence_count / 5, 1.0)  # Normalize to max 5 sentences
        
        complexity = (length_factor + vocab_diversity + structure_complexity) / 3
        return min(complexity, 1.0)
    
    def _calculate_response_quality(self, response: str) -> float:
        """Calculate the quality of AI response"""
        if not response:
            return 0.0
        
        # Quality factors
        length_score = min(len(response) / 100, 1.0)  # Optimal around 100 chars
        completeness_score = 1.0 if response.strip().endswith(('.', '!', '?')) else 0.5
        coherence_score = self._text_coherence_score(response)
        
        quality = (length_score + completeness_score + coherence_score) / 3
        return min(quality, 1.0)
    
    def _text_coherence_score(self, text: str) -> float:
        """Calculate text coherence using simple heuristics"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.7  # Default for single sentence
        
        # Check for repetition (lower coherence)
        words = text.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        
        # Check for logical flow (simplified)
        transition_words = ['however', 'therefore', 'moreover', 'furthermore', 'additionally']
        has_transitions = any(word in text.lower() for word in transition_words)
        
        coherence = unique_ratio * 0.7 + (0.3 if has_transitions else 0.0)
        return min(coherence, 1.0)
    
    def _calculate_coherence(self, user_input: str, ai_response: str) -> float:
        """Calculate coherence between input and response"""
        try:
            # Simple keyword overlap method
            user_words = set(user_input.lower().split())
            response_words = set(ai_response.lower().split())
            
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
            user_words -= stop_words
            response_words -= stop_words
            
            if not user_words:
                return 0.5
            
            overlap = len(user_words.intersection(response_words))
            coherence = overlap / len(user_words)
            
            return min(coherence, 1.0)
        except:
            return 0.5
    
    def _calculate_novelty(self, user_input: str, ai_response: str) -> float:
        """Calculate novelty of the interaction"""
        # Check against recent experiences
        recent_experiences = list(self.experience_buffer)[-50:]  # Last 50 interactions
        
        if not recent_experiences:
            return 1.0  # First interaction is novel
        
        # Calculate similarity with recent interactions
        similarities = []
        current_text = user_input + " " + ai_response
        
        for exp in recent_experiences:
            exp_text = exp['user_input'] + " " + exp['ai_response']
            similarity = self._text_similarity(current_text, exp_text)
            similarities.append(similarity)
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0
        novelty = 1.0 - max_similarity
        
        return max(novelty, 0.1)  # Minimum novelty
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union)
        except:
            return 0.0
    
    def _extract_concepts(self, user_input: str, ai_response: str) -> List[str]:
        """Extract key concepts from the interaction"""
        # Combine text for concept extraction
        combined_text = user_input + " " + ai_response
        
        # Simple keyword extraction (can be enhanced with NLP)
        words = combined_text.lower().split()
        
        # Filter meaningful words (nouns, adjectives, etc.)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        concepts = []
        for word in words:
            if (len(word) > 3 and 
                word not in stop_words and 
                word.isalpha()):
                concepts.append(word)
        
        # Return unique concepts
        return list(set(concepts))
    
    def _identify_patterns(self, user_input: str, ai_response: str, context: Dict) -> List[str]:
        """Identify patterns in the interaction"""
        patterns = []
        
        # Input patterns
        if len(user_input.split()) > 20:
            patterns.append('long_input')
        if '?' in user_input:
            patterns.append('question_input')
        if any(word in user_input.lower() for word in ['explain', 'how', 'why', 'what']):
            patterns.append('explanation_request')
        
        # Response patterns
        if len(ai_response.split()) > 30:
            patterns.append('detailed_response')
        if ai_response.count('.') > 2:
            patterns.append('multi_sentence_response')
        
        # Context patterns
        if context and context.get('history'):
            patterns.append('contextual_conversation')
        
        return patterns
    
    def _update_concept_map(self, concepts: List[str], experience: Dict):
        """Update the concept map with new learning"""
        for concept in concepts:
            self.concept_map[concept].append({
                'timestamp': experience['timestamp'],
                'context': experience['user_input'][:100],  # First 100 chars
                'strength': experience['learning_metrics'].get('novelty_score', 0.5)
            })
            
            # Limit concept entries
            if len(self.concept_map[concept]) > 50:
                self.concept_map[concept] = self.concept_map[concept][-30:]
    
    def _update_skill_metrics(self, experience: Dict, patterns: List[str]) -> Dict:
        """Update skill metrics based on the interaction"""
        updates = {}
        learning_metrics = experience['learning_metrics']
        
        # Update conversation skill
        conversation_score = (
            learning_metrics.get('response_quality', 0.5) * 0.6 +
            learning_metrics.get('coherence_score', 0.5) * 0.4
        )
        self.skill_categories['conversation'] = self._update_skill_value(
            self.skill_categories['conversation'], conversation_score
        )
        updates['conversation'] = conversation_score
        
        # Update reasoning skill (based on complexity handling)
        reasoning_score = min(
            learning_metrics.get('input_complexity', 0.5) * 
            learning_metrics.get('response_quality', 0.5) * 2, 1.0
        )
        self.skill_categories['reasoning'] = self._update_skill_value(
            self.skill_categories['reasoning'], reasoning_score
        )
        updates['reasoning'] = reasoning_score
        
        # Update creativity (based on novelty)
        creativity_score = learning_metrics.get('novelty_score', 0.5)
        self.skill_categories['creativity'] = self._update_skill_value(
            self.skill_categories['creativity'], creativity_score
        )
        updates['creativity'] = creativity_score
        
        # Update learning speed (based on pattern recognition)
        learning_speed_score = min(len(patterns) / 5, 1.0)  # Normalize to max 5 patterns
        self.skill_categories['learning_speed'] = self._update_skill_value(
            self.skill_categories['learning_speed'], learning_speed_score
        )
        updates['learning_speed'] = learning_speed_score
        
        return updates
    
    def _update_skill_value(self, current_value: float, new_score: float) -> float:
        """Update skill value with exponential moving average"""
        alpha = self.learning_rate
        updated_value = alpha * new_score + (1 - alpha) * current_value
        return max(0.0, min(1.0, updated_value))
    
    def _learn_from_feedback(self, feedback: str, experience: Dict) -> Dict:
        """Learn from explicit feedback"""
        feedback_updates = {}
        
        # Analyze feedback sentiment and content
        feedback_score = self._analyze_feedback_quality(feedback)
        
        # Update relevant skills based on feedback
        if 'helpful' in feedback.lower() or 'good' in feedback.lower():
            # Positive feedback - boost relevant skills
            boost_factor = 0.1
            for skill in self.skill_categories:
                old_value = self.skill_categories[skill]
                self.skill_categories[skill] = min(old_value + boost_factor, 1.0)
                feedback_updates[skill] = boost_factor
        
        elif 'wrong' in feedback.lower() or 'bad' in feedback.lower():
            # Negative feedback - identify areas for improvement
            penalty_factor = 0.05
            self.skill_categories['reasoning'] = max(
                self.skill_categories['reasoning'] - penalty_factor, 0.1
            )
            feedback_updates['reasoning'] = -penalty_factor
        
        # Store feedback for pattern analysis
        self.feedback_patterns = getattr(self, 'feedback_patterns', [])
        self.feedback_patterns.append({
            'feedback': feedback,
            'score': feedback_score,
            'timestamp': experience['timestamp']
        })
        
        return feedback_updates
    
    def _analyze_feedback_quality(self, feedback: str) -> float:
        """Analyze the quality and sentiment of feedback"""
        positive_indicators = ['good', 'great', 'helpful', 'correct', 'excellent', 'amazing']
        negative_indicators = ['bad', 'wrong', 'unhelpful', 'incorrect', 'poor', 'terrible']
        
        feedback_lower = feedback.lower()
        positive_count = sum(1 for word in positive_indicators if word in feedback_lower)
        negative_count = sum(1 for word in negative_indicators if word in feedback_lower)
        
        if positive_count > negative_count:
            return 0.8 + (positive_count * 0.1)
        elif negative_count > positive_count:
            return 0.2 - (negative_count * 0.1)
        else:
            return 0.5
    
    def _trigger_adaptive_learning(self, experience: Dict):
        """Trigger adaptive learning mechanisms"""
        # Check if learning should be intensified
        novelty_score = experience['learning_metrics'].get('novelty_score', 0.5)
        
        if novelty_score > 0.8:  # High novelty - increase learning rate
            self.learning_rate = min(self.learning_rate * 1.1, 0.1)
        elif novelty_score < 0.3:  # Low novelty - decrease learning rate
            self.learning_rate = max(self.learning_rate * 0.95, 0.001)
        
        # Update curiosity factor based on performance
        avg_skill = np.mean(list(self.skill_categories.values()))
        if avg_skill > 0.8:
            self.curiosity_factor = min(self.curiosity_factor * 1.05, 0.5)
        elif avg_skill < 0.4:
            self.curiosity_factor = max(self.curiosity_factor * 0.95, 0.1)
    
    def _update_learning_patterns(self, patterns: List[str], experience: Dict):
        """Update learning patterns for future reference"""
        for pattern in patterns:
            if pattern not in self.learning_patterns:
                self.learning_patterns[pattern] = {
                    'count': 0,
                    'success_rate': 0.5,
                    'last_seen': None
                }
            
            self.learning_patterns[pattern]['count'] += 1
            self.learning_patterns[pattern]['last_seen'] = experience['timestamp']
            
            # Update success rate based on response quality
            quality = experience['learning_metrics'].get('response_quality', 0.5)
            current_rate = self.learning_patterns[pattern]['success_rate']
            alpha = 0.1
            self.learning_patterns[pattern]['success_rate'] = (
                alpha * quality + (1 - alpha) * current_rate
            )
    
    def _calculate_learning_score(self, experience: Dict) -> float:
        """Calculate overall learning score for the interaction"""
        metrics = experience['learning_metrics']
        
        # Weighted combination of learning factors
        learning_score = (
            metrics.get('novelty_score', 0.5) * 0.3 +
            metrics.get('input_complexity', 0.5) * 0.2 +
            metrics.get('response_quality', 0.5) * 0.3 +
            metrics.get('coherence_score', 0.5) * 0.2
        )
        
        return min(learning_score, 1.0)
    
    def get_curiosity_driven_questions(self) -> List[str]:
        """Generate questions to drive learning and exploration"""
        questions = []
        
        # Analyze knowledge gaps
        concept_strengths = {}
        for concept, entries in self.concept_map.items():
            avg_strength = np.mean([entry['strength'] for entry in entries])
            concept_strengths[concept] = avg_strength
        
        # Identify weak concepts for exploration
        weak_concepts = [concept for concept, strength in concept_strengths.items() 
                        if strength < 0.5]
        
        # Generate questions about weak concepts
        for concept in weak_concepts[:3]:  # Top 3 weak concepts
            questions.append(f"Can you tell me more about {concept}?")
            questions.append(f"How does {concept} relate to other topics we've discussed?")
        
        # General exploration questions
        if self.skill_categories['creativity'] < 0.6:
            questions.extend([
                "What's something creative I could help you with?",
                "Can you share an interesting problem that requires creative thinking?"
            ])
        
        if self.skill_categories['reasoning'] < 0.6:
            questions.extend([
                "Can you give me a logical puzzle to solve?",
                "What's a complex problem you'd like me to analyze?"
            ])
        
        return questions[:5]  # Return top 5 questions
    
    def get_learning_insights(self) -> Dict:
        """Get insights about the learning progress"""
        insights = {
            'skill_levels': self.skill_categories.copy(),
            'learning_rate': self.learning_rate,
            'curiosity_factor': self.curiosity_factor,
            'total_experiences': len(self.experience_buffer),
            'concept_count': len(self.concept_map),
            'pattern_count': len(self.learning_patterns),
            'strongest_skills': [],
            'weakest_skills': [],
            'learning_trends': {}
        }
        
        # Identify strongest and weakest skills
        sorted_skills = sorted(self.skill_categories.items(), key=lambda x: x[1], reverse=True)
        insights['strongest_skills'] = sorted_skills[:3]
        insights['weakest_skills'] = sorted_skills[-3:]
        
        # Calculate learning trends
        if len(self.performance_history) > 10:
            recent_performance = self.performance_history[-10:]
            older_performance = self.performance_history[-20:-10] if len(self.performance_history) > 20 else []
            
            if older_performance:
                recent_avg = np.mean(recent_performance)
                older_avg = np.mean(older_performance)
                insights['learning_trends']['improvement'] = recent_avg - older_avg
            else:
                insights['learning_trends']['improvement'] = 0.0
        
        return insights
    
    def _start_learning_loop(self):
        """Start the background learning optimization loop"""
        def learning_loop():
            while True:
                time.sleep(60)  # Check every minute
                self._background_learning_optimization()
        
        self.learning_thread = threading.Thread(target=learning_loop, daemon=True)
        self.learning_thread.start()
    
    def _background_learning_optimization(self):
        """Perform background learning optimization"""
        try:
            # Knowledge decay simulation
            self._apply_knowledge_decay()
            
            # Pattern reinforcement
            self._reinforce_successful_patterns()
            
            # Concept clustering and organization
            self._organize_concept_knowledge()
            
            # Performance trend analysis
            self._analyze_performance_trends()
            
        except Exception as e:
            self.logger.error(f"Error in background learning optimization: {e}")
    
    def _apply_knowledge_decay(self):
        """Apply knowledge decay to simulate forgetting"""
        current_time = datetime.now()
        
        # Decay concept strengths over time
        for concept, entries in self.concept_map.items():
            for entry in entries:
                entry_time = datetime.fromisoformat(entry['timestamp'])
                days_old = (current_time - entry_time).days
                
                # Apply exponential decay
                decay_factor = np.exp(-self.knowledge_decay_rate * days_old)
                entry['strength'] *= decay_factor
    
    def _reinforce_successful_patterns(self):
        """Reinforce patterns that lead to successful interactions"""
        for pattern, data in self.learning_patterns.items():
            if data['success_rate'] > 0.7:  # Successful pattern
                # Increase its influence on learning
                data['reinforcement'] = data.get('reinforcement', 1.0) * 1.05
            elif data['success_rate'] < 0.3:  # Unsuccessful pattern
                # Decrease its influence
                data['reinforcement'] = data.get('reinforcement', 1.0) * 0.95
    
    def _organize_concept_knowledge(self):
        """Organize concepts into clusters for better understanding"""
        if len(self.concept_map) < 10:
            return
        
        try:
            # Create concept vectors for clustering
            concepts = list(self.concept_map.keys())
            concept_texts = []
            
            for concept in concepts:
                # Combine all contexts for this concept
                contexts = [entry['context'] for entry in self.concept_map[concept]]
                concept_texts.append(' '.join(contexts))
            
            # Vectorize and cluster
            if hasattr(self, 'vectorizer') and concept_texts:
                vectors = self.vectorizer.fit_transform(concept_texts)
                
                # Perform clustering
                n_clusters = min(5, len(concepts) // 3)  # Adaptive cluster count
                if n_clusters > 1:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(vectors)
                    
                    # Store cluster information
                    self.concept_clusters = {
                        concept: int(cluster) for concept, cluster in zip(concepts, clusters)
                    }
            
        except Exception as e:
            self.logger.error(f"Error in concept organization: {e}")
    
    def _analyze_performance_trends(self):
        """Analyze performance trends for learning insights"""
        if len(self.experience_buffer) < 20:
            return
        
        # Calculate recent performance metrics
        recent_experiences = list(self.experience_buffer)[-20:]
        performance_scores = []
        
        for exp in recent_experiences:
            metrics = exp['learning_metrics']
            # Combine metrics into single performance score
            performance = (
                metrics.get('response_quality', 0.5) * 0.4 +
                metrics.get('coherence_score', 0.5) * 0.3 +
                metrics.get('novelty_score', 0.5) * 0.3
            )
            performance_scores.append(performance)
        
        # Store in performance history
        self.performance_history.extend(performance_scores)
        
        # Limit history size
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-800:]
    
    def save_learning_state(self, filepath: str):
        """Save the current learning state"""
        state = {
            'skill_categories': self.skill_categories,
            'concept_map': dict(self.concept_map),
            'learning_patterns': self.learning_patterns,
            'performance_history': self.performance_history,
            'learning_rate': self.learning_rate,
            'curiosity_factor': self.curiosity_factor,
            'concept_clusters': getattr(self, 'concept_clusters', {}),
            'feedback_patterns': getattr(self, 'feedback_patterns', [])
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info("Learning state saved successfully")
    
    def load_learning_state(self, filepath: str):
        """Load previously saved learning state"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.skill_categories = state.get('skill_categories', self.skill_categories)
            self.concept_map = defaultdict(list, state.get('concept_map', {}))
            self.learning_patterns = state.get('learning_patterns', {})
            self.performance_history = state.get('performance_history', [])
            self.learning_rate = state.get('learning_rate', 0.01)
            self.curiosity_factor = state.get('curiosity_factor', 0.3)
            self.concept_clusters = state.get('concept_clusters', {})
            self.feedback_patterns = state.get('feedback_patterns', [])
            
            self.logger.info("Learning state loaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error loading learning state: {e}")
            return False