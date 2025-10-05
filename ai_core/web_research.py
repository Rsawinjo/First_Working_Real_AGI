"""
Enhanced Web Research Module - Phase 2
Advanced web search, content analysis, and knowledge synthesis
"""

import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import logging
import json
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class AdvancedWebResearcher:
    """
    Advanced web research system with parallel processing,
    intelligent content extraction, and knowledge synthesis
    """
    
    def __init__(self, knowledge_base=None):
        self.logger = logging.getLogger(__name__)
        self.knowledge_base = knowledge_base
        
        # Research configuration
        self.max_sources = 10
        self.max_depth = 3
        self.timeout = 30
        self.parallel_requests = 5
        
        # Content processing
        self.content_cache = {}
        self.processed_urls = set()
        self.research_session = requests.Session()
        
        # Enhanced search engines and APIs
        self.search_engines = {
            'duckduckgo': 'https://html.duckduckgo.com/html/',
            'bing': 'https://www.bing.com/search',
            'startpage': 'https://www.startpage.com/sp/search'
        }
        
        # Content quality filters
        self.quality_indicators = [
            'github.com', 'stackoverflow.com', 'arxiv.org', 'wikipedia.org',
            'medium.com', 'towardsdatascience.com', 'papers.with.code',
            'openai.com', 'deepmind.com', 'research.google', 'microsoft.com'
        ]
        
        self.logger.info("Advanced Web Researcher initialized - Phase 2 ready!")
    
    async def comprehensive_research(self, topic: str, depth: str = "comprehensive") -> Dict:
        """
        Perform comprehensive research on a topic with parallel processing
        """
        self.logger.info("Starting comprehensive research: %s", topic)
        start_time = time.time()
        
        try:
            # Multi-engine search
            search_tasks = []
            for engine_name, engine_url in self.search_engines.items():
                search_tasks.append(self._search_engine(topic, engine_name))
            
            # Execute searches in parallel
            all_results = []
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
                
                for results in search_results:
                    if isinstance(results, list):
                        all_results.extend(results)
            
            # Filter and rank results
            filtered_results = self._filter_and_rank_results(all_results, topic)
            
            # Extract content in parallel
            content_tasks = []
            for result in filtered_results[:self.max_sources]:
                content_tasks.append(self._extract_content(result['url']))
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                contents = await asyncio.gather(*content_tasks, return_exceptions=True)
            
            # Process and synthesize information
            processed_content = []
            for i, content in enumerate(contents):
                if isinstance(content, dict) and content.get('text'):
                    content['source'] = filtered_results[i]
                    processed_content.append(content)
            
            # Generate comprehensive summary
            synthesis = self._synthesize_knowledge(topic, processed_content)
            
            research_time = time.time() - start_time
            
            result = {
                'topic': topic,
                'research_depth': depth,
                'sources_found': len(all_results),
                'sources_processed': len(processed_content),
                'research_time': research_time,
                'synthesis': synthesis,
                'detailed_sources': processed_content,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in knowledge base if available
            if self.knowledge_base:
                await self._store_research_results(result)
            
            self.logger.info("Research completed in %.2fs - %d sources processed", research_time, len(processed_content))
            return result
            
        except Exception as e:
            self.logger.error(f"Research error: {e}")
            return {
                'topic': topic,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _search_engine(self, query: str, engine: str) -> List[Dict]:
        """Search a specific engine for results"""
        try:
            if engine == 'duckduckgo':
                return await self._search_duckduckgo(query)
            elif engine == 'bing':
                return await self._search_bing(query)
            else:
                return []
        except Exception as e:
            self.logger.warning(f"Search engine {engine} failed: {e}")
            return []
    
    async def _search_duckduckgo(self, query: str) -> List[Dict]:
        """Search DuckDuckGo"""
        try:
            params = {'q': query}
            async with aiohttp.ClientSession() as session:
                async with session.get(self.search_engines['duckduckgo'], params=params) as response:
                    html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            for result in soup.find_all('a', class_='result__a')[:10]:
                title = result.get_text().strip()
                url = result.get('href')
                if url and title:
                    results.append({
                        'title': title,
                        'url': url,
                        'engine': 'duckduckgo',
                        'relevance_score': self._calculate_relevance(title, query)
                    })
            
            return results
            
        except Exception as e:
            self.logger.warning(f"DuckDuckGo search failed: {e}")
            return []
    
    async def _search_bing(self, query: str) -> List[Dict]:
        """Search Bing (basic implementation)"""
        try:
            # Note: For production use, consider using Bing Search API
            params = {'q': query}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.search_engines['bing'], params=params, headers=headers) as response:
                    html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            for result in soup.find_all('h2')[:10]:
                link = result.find('a')
                if link:
                    title = link.get_text().strip()
                    url = link.get('href')
                    if url and title:
                        results.append({
                            'title': title,
                            'url': url,
                            'engine': 'bing',
                            'relevance_score': self._calculate_relevance(title, query)
                        })
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Bing search failed: {e}")
            return []
    
    def _filter_and_rank_results(self, results: List[Dict], topic: str) -> List[Dict]:
        """Filter and rank search results by quality and relevance"""
        # Remove duplicates
        seen_urls = set()
        filtered = []
        
        for result in results:
            url = result.get('url', '')
            if url not in seen_urls:
                seen_urls.add(url)
                
                # Calculate quality score
                quality_score = 0
                for indicator in self.quality_indicators:
                    if indicator in url.lower():
                        quality_score += 2
                
                # Add topic relevance
                title_relevance = self._calculate_relevance(result.get('title', ''), topic)
                
                result['quality_score'] = quality_score
                result['total_score'] = quality_score + title_relevance
                filtered.append(result)
        
        # Sort by total score
        return sorted(filtered, key=lambda x: x.get('total_score', 0), reverse=True)
    
    def _calculate_relevance(self, text: str, query: str) -> float:
        """Calculate relevance score between text and query"""
        text_lower = text.lower()
        query_words = query.lower().split()
        
        score = 0
        for word in query_words:
            if word in text_lower:
                score += 1
                
        return score / len(query_words) if query_words else 0
    
    async def _extract_content(self, url: str) -> Dict:
        """Extract and process content from a URL"""
        try:
            if url in self.content_cache:
                return self.content_cache[url]
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    html = await response.text()
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract main content
            content_selectors = [
                'article', 'main', '.content', '#content', '.post-content',
                '.entry-content', '.article-content', 'p'
            ]
            
            text_content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    text_content = ' '.join([elem.get_text().strip() for elem in elements])
                    break
            
            if not text_content:
                text_content = soup.get_text()
            
            # Clean and process text
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            text_content = text_content[:5000]  # Limit length
            
            result = {
                'url': url,
                'text': text_content,
                'word_count': len(text_content.split()),
                'extracted_at': datetime.now().isoformat()
            }
            
            self.content_cache[url] = result
            return result
            
        except Exception as e:
            self.logger.warning(f"Content extraction failed for {url}: {e}")
            return {'url': url, 'error': str(e)}
    
    def _synthesize_knowledge(self, topic: str, content_list: List[Dict]) -> Dict:
        """Synthesize knowledge from multiple sources"""
        if not content_list:
            return {'summary': f"No content found for topic: {topic}"}
        
        # Combine all text
        all_text = []
        source_count = 0
        total_words = 0
        
        for content in content_list:
            if content.get('text'):
                all_text.append(content['text'])
                source_count += 1
                total_words += content.get('word_count', 0)
        
        combined_text = ' '.join(all_text)
        
        # Extract key concepts (simple keyword extraction)
        words = combined_text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        synthesis = {
            'summary': f"Research synthesis for '{topic}' from {source_count} sources",
            'total_words_processed': total_words,
            'key_concepts': [kw[0] for kw in top_keywords],
            'source_count': source_count,
            'research_depth': 'comprehensive' if source_count >= 5 else 'standard',
            'confidence_score': min(source_count * 0.2, 1.0)
        }
        
        return synthesis
    
    async def _store_research_results(self, results: Dict):
        """Store research results in knowledge base"""
        try:
            if self.knowledge_base:
                # Store the research synthesis
                content_text = json.dumps(results['synthesis'])
                metadata = {
                    'topic': results['topic'],
                    'source_type': 'web_research',
                    'confidence': results['synthesis'].get('confidence_score', 0.5),
                    'sources_count': results['sources_processed'],
                    'research_time': results['research_time']
                }
                
                self.knowledge_base.add_knowledge(
                    content=content_text,
                    metadata=metadata
                )
                
                self.logger.info("Research results stored in knowledge base")
                
        except Exception as e:
            self.logger.error(f"Failed to store research results: {e}")
    
    def get_research_stats(self) -> Dict:
        """Get research system statistics"""
        return {
            'cached_content': len(self.content_cache),
            'processed_urls': len(self.processed_urls),
            'search_engines': len(self.search_engines),
            'quality_indicators': len(self.quality_indicators)
        }