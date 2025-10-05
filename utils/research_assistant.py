"""
Research Assistant - Automated web research and knowledge synthesis
Enables the AI to research topics and gather external knowledge
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
import threading
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
import wikipedia
import re
from functools import partial

class ResearchAssistant:
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Research parameters
        self.max_results = self.config.get('MAX_RESEARCH_RESULTS', 5)
        self.research_depth = self.config.get('RESEARCH_DEPTH', 3)
        self.timeout = 30  # Increased timeout for Wikipedia API calls
        
        # Knowledge sources
        self.search_engines = {
            'wikipedia': self._search_wikipedia,
            'web_search': self._web_search_fallback
        }
        
        # Cache for research results
        self.research_cache = {}
        self.cache_duration = 3600  # 1 hour
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # 1 second between requests
    
    def research_topic(self, topic: str, depth: Optional[int] = None) -> Optional[str]:
        """Research a topic and return synthesized information with timeout protection"""
        
        def _research_with_timeout():
            """Execute research with potential timeout"""
            try:
                research_depth = depth or self.research_depth
                
                # Check cache first
                cache_key = f"{topic.lower()}_{research_depth}"
                if cache_key in self.research_cache:
                    cached_result = self.research_cache[cache_key]
                    if (datetime.now() - cached_result['timestamp']).seconds < self.cache_duration:
                        return cached_result['content']
                
                # Extract key terms for research
                research_terms = self._extract_research_terms(topic)
                
                # Gather information from multiple sources
                research_results = []
                
                for term in research_terms[:3]:  # Limit to 3 main terms
                    # Wikipedia search
                    wiki_result = self._search_wikipedia(term)
                    if wiki_result:
                        research_results.append({
                            'source': 'wikipedia',
                            'term': term,
                            'content': wiki_result
                        })
                    
                    # Rate limiting
                    self._rate_limit()
                
                # Synthesize findings
                synthesized_content = self._synthesize_research(research_results, topic)
                
                # Cache result
                self.research_cache[cache_key] = {
                    'content': synthesized_content,
                    'timestamp': datetime.now()
                }
                
                return synthesized_content
                
            except Exception as e:
                self.logger.error(f"Error in research_topic: {e}")
                return None
        
        # Execute with timeout protection (longer timeout for overall research)
        result: List[Optional[str]] = [None]
        
        def target():
            result[0] = _research_with_timeout()
        
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=60)  # Allow up to 60 seconds for overall research
        
        if thread.is_alive():
            self.logger.warning(f"Research for topic '{topic}' timed out after 60s")
            return None
        
        return result[0]
    
    def _extract_research_terms(self, topic: str) -> List[str]:
        """Extract key research terms from the topic"""
        try:
            # Remove common words and extract meaningful terms
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'must',
                'can', 'what', 'how', 'why', 'when', 'where', 'who'
            }
            
            # Clean and tokenize
            words = re.findall(r'\b\w+\b', topic.lower())
            
            # Filter meaningful terms
            research_terms = []
            for word in words:
                if (len(word) > 2 and 
                    word not in stop_words and 
                    word.isalpha()):
                    research_terms.append(word)
            
            # Add original topic as primary term
            if topic.strip():
                research_terms.insert(0, topic.strip())
            
            return research_terms[:5]  # Limit to 5 terms
            
        except Exception as e:
            self.logger.error(f"Error extracting research terms: {e}")
            return [topic]
    
    def _search_wikipedia(self, term: str) -> Optional[str]:
        """Search Wikipedia for information on a term with timeout protection"""
        
        def _wikipedia_search_with_timeout():
            """Execute Wikipedia search with potential timeout"""
            try:
                # Search for the term
                search_results = wikipedia.search(term, results=3)
                
                if not search_results:
                    return None
                
                # Get summary of the most relevant article
                try:
                    page = wikipedia.page(search_results[0])
                    summary = wikipedia.summary(search_results[0], sentences=3)
                    
                    # Clean and format the summary
                    clean_summary = self._clean_text(summary)
                    
                    return f"Wikipedia: {clean_summary}"
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    # Handle disambiguation - try the first option
                    try:
                        if e.options:
                            page = wikipedia.page(e.options[0])
                            summary = wikipedia.summary(e.options[0], sentences=2)
                            clean_summary = self._clean_text(summary)
                            return f"Wikipedia: {clean_summary}"
                    except:
                        pass
                
                except wikipedia.exceptions.PageError:
                    # Page doesn't exist, try next result
                    if len(search_results) > 1:
                        try:
                            summary = wikipedia.summary(search_results[1], sentences=2)
                            clean_summary = self._clean_text(summary)
                            return f"Wikipedia: {clean_summary}"
                        except:
                            pass
                
                return None
                
            except Exception as e:
                self.logger.error(f"Error searching Wikipedia for '{term}': {e}")
                return None
        
        # Execute with timeout protection
        result: List[Optional[str]] = [None]  # Use list to modify from inner function
        
        def target():
            result[0] = _wikipedia_search_with_timeout()
        
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=self.timeout)  # Wait for up to self.timeout seconds
        
        if thread.is_alive():
            self.logger.warning(f"Wikipedia search for '{term}' timed out after {self.timeout}s")
            return None
        
        return result[0]
    
    def _web_search_fallback(self, term: str) -> Optional[str]:
        """Fallback web search method (placeholder)"""
        # This is a placeholder for web search functionality
        # In a production system, you might integrate with search APIs
        # like Google Custom Search, Bing Search API, etc.
        
        try:
            # Simple demonstration - this would be replaced with actual search API
            search_term = quote_plus(term)
            
            # For now, return a formatted placeholder response
            return f"Web Search: Information about '{term}' (Search API integration needed)"
            
        except Exception as e:
            self.logger.error(f"Error in web search fallback for '{term}': {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and format text content"""
        try:
            # Remove extra whitespace
            cleaned = ' '.join(text.split())
            
            # Remove common Wikipedia artifacts
            artifacts = [
                '(disambiguation)',
                '(redirected from',
                'For other uses, see',
                'This article is about'
            ]
            
            for artifact in artifacts:
                if artifact in cleaned:
                    # Remove the sentence containing the artifact
                    sentences = cleaned.split('.')
                    cleaned_sentences = []
                    for sentence in sentences:
                        if artifact not in sentence:
                            cleaned_sentences.append(sentence)
                    cleaned = '. '.join(cleaned_sentences)
            
            # Limit length
            if len(cleaned) > 500:
                # Truncate at sentence boundary
                sentences = cleaned.split('.')
                truncated = []
                current_length = 0
                
                for sentence in sentences:
                    if current_length + len(sentence) < 450:
                        truncated.append(sentence)
                        current_length += len(sentence)
                    else:
                        break
                
                cleaned = '. '.join(truncated)
                if cleaned and not cleaned.endswith('.'):
                    cleaned += '.'
            
            return cleaned
            
        except Exception as e:
            self.logger.error(f"Error cleaning text: {e}")
            return text
    
    def _synthesize_research(self, research_results: List[Dict], original_topic: str) -> str:
        """Synthesize research results into coherent information"""
        try:
            if not research_results:
                return f"No research results found for '{original_topic}'"
            
            # Group results by source
            source_content = {}
            for result in research_results:
                source = result['source']
                if source not in source_content:
                    source_content[source] = []
                source_content[source].append(result['content'])
            
            # Build synthesized response
            synthesis_parts = []
            
            # Add introduction
            synthesis_parts.append(f"Research on '{original_topic}':")
            
            # Add content from each source
            for source, contents in source_content.items():
                if contents:
                    # Combine content from same source
                    combined_content = ' '.join(contents)
                    
                    # Remove redundancy
                    unique_content = self._remove_redundant_content(combined_content)
                    
                    if unique_content:
                        synthesis_parts.append(f"\n{unique_content}")
            
            # Combine all parts
            synthesized = '\n'.join(synthesis_parts)
            
            # Final cleanup and validation
            if len(synthesized) > 1000:
                synthesized = synthesized[:950] + "..."
            
            return synthesized
            
        except Exception as e:
            self.logger.error(f"Error synthesizing research: {e}")
            return f"Research completed for '{original_topic}' (synthesis error)"
    
    def _remove_redundant_content(self, content: str) -> str:
        """Remove redundant information from content"""
        try:
            sentences = content.split('.')
            unique_sentences = []
            seen_content = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(sentence) > 10:
                    # Create a simplified version for comparison
                    simplified = ' '.join(sorted(sentence.lower().split()))
                    
                    if simplified not in seen_content:
                        seen_content.add(simplified)
                        unique_sentences.append(sentence)
            
            return '. '.join(unique_sentences)
            
        except Exception as e:
            self.logger.error(f"Error removing redundant content: {e}")
            return content
    
    def _rate_limit(self):
        """Apply rate limiting between requests"""
        try:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Error in rate limiting: {e}")
    
    def get_research_suggestions(self, topic: str) -> List[str]:
        """Get suggestions for further research on a topic"""
        try:
            suggestions = []
            
            # Extract base terms
            terms = self._extract_research_terms(topic)
            
            # Generate related research suggestions
            for term in terms[:3]:
                suggestions.extend([
                    f"How does {term} work?",
                    f"What are the applications of {term}?",
                    f"History and development of {term}",
                    f"Current research in {term}",
                    f"Future prospects for {term}"
                ])
            
            # Remove duplicates and limit
            unique_suggestions = list(set(suggestions))
            return unique_suggestions[:8]
            
        except Exception as e:
            self.logger.error(f"Error generating research suggestions: {e}")
            return []
    
    def search_specific_source(self, term: str, source: str) -> Optional[str]:
        """Search a specific source for information"""
        try:
            if source in self.search_engines:
                return self.search_engines[source](term)
            else:
                self.logger.warning(f"Unknown source: {source}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error searching source '{source}' for '{term}': {e}")
            return None
    
    def get_research_stats(self) -> Dict:
        """Get statistics about research activity"""
        try:
            stats = {
                'cache_size': len(self.research_cache),
                'available_sources': list(self.search_engines.keys()),
                'max_results': self.max_results,
                'research_depth': self.research_depth,
                'cache_hit_ratio': 0.0  # Would be calculated with usage tracking
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting research stats: {e}")
            return {}
    
    def clear_cache(self):
        """Clear the research cache"""
        try:
            self.research_cache.clear()
            self.logger.info("Research cache cleared")
            
        except Exception as e:
            self.logger.error(f"Error clearing research cache: {e}")
    
    def export_research_data(self, filepath: str) -> bool:
        """Export research cache and settings"""
        try:
            export_data = {
                'cache': self.research_cache,
                'config': self.config,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Research data exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting research data: {e}")
            return False
    
    def import_research_data(self, filepath: str) -> bool:
        """Import research cache and settings"""
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)
            
            # Import cache
            if 'cache' in import_data:
                self.research_cache.update(import_data['cache'])
            
            # Import config
            if 'config' in import_data:
                self.config.update(import_data['config'])
                self.max_results = self.config.get('MAX_RESEARCH_RESULTS', 5)
                self.research_depth = self.config.get('RESEARCH_DEPTH', 3)
            
            self.logger.info(f"Research data imported from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing research data: {e}")
            return False