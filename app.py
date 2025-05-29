import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import time
import requests
from io import StringIO
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Handle optional imports gracefully
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.error("sentence-transformers not installed. Some features will be disabled.")

try:
    import nltk
    NLTK_AVAILABLE = True
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt')
        except:
            st.warning("Could not download NLTK punkt tokenizer. Text processing may be limited.")
except ImportError:
    NLTK_AVAILABLE = False
    st.warning("NLTK not available. Using basic text processing.")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not available. Semantic analysis will be disabled.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available. Using basic charts.")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'llm_responses' not in st.session_state:
    st.session_state.llm_responses = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

class LLMCrawler:
    def __init__(self):
        self.responses = {}
        
    def query_openai(self, query: str, api_key: str) -> Dict:
        """Query OpenAI GPT"""
        if not OPENAI_AVAILABLE:
            return {
                "provider": "OpenAI GPT-4",
                "query": query,
                "response": "",
                "success": False,
                "error": "OpenAI library not available",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant providing accurate, comprehensive answers."},
                    {"role": "user", "content": query}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            return {
                "provider": "OpenAI GPT-4",
                "query": query,
                "response": response.choices[0].message.content,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "provider": "OpenAI GPT-4",
                "query": query,
                "response": "",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def query_gemini(self, query: str, api_key: str) -> Dict:
        """Query Google Gemini"""
        if not GEMINI_AVAILABLE:
            return {
                "provider": "Google Gemini",
                "query": query,
                "response": "",
                "success": False,
                "error": "Google Generative AI library not available",
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(query)
            return {
                "provider": "Google Gemini",
                "query": query,
                "response": response.text,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "provider": "Google Gemini",
                "query": query,
                "response": "",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def query_claude(self, query: str, api_key: str) -> Dict:
        """Query Anthropic Claude"""
        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1500,
                "messages": [
                    {"role": "user", "content": query}
                ]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "provider": "Anthropic Claude",
                    "query": query,
                    "response": result['content'][0]['text'],
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "provider": "Anthropic Claude",
                    "query": query,
                    "response": "",
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "provider": "Anthropic Claude",
                "query": query,
                "response": "",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def crawl_multiple_llms(self, queries: List[str], api_keys: Dict[str, str], 
                           delay_between_requests: float = 1.0) -> List[Dict]:
        """Crawl multiple LLMs with rate limiting"""
        all_responses = []
        
        for query in queries:
            query_responses = []
            
            # Query each LLM
            if api_keys.get('openai') and OPENAI_AVAILABLE:
                st.write(f"Querying OpenAI for: {query[:50]}...")
                response = self.query_openai(query, api_keys['openai'])
                query_responses.append(response)
                time.sleep(delay_between_requests)
            elif api_keys.get('openai'):
                st.warning("OpenAI API key provided but library not available")
            
            if api_keys.get('gemini') and GEMINI_AVAILABLE:
                st.write(f"Querying Gemini for: {query[:50]}...")
                response = self.query_gemini(query, api_keys['gemini'])
                query_responses.append(response)
                time.sleep(delay_between_requests)
            elif api_keys.get('gemini'):
                st.warning("Gemini API key provided but library not available")
            
            if api_keys.get('claude'):
                st.write(f"Querying Claude for: {query[:50]}...")
                response = self.query_claude(query, api_keys['claude'])
                query_responses.append(response)
                time.sleep(delay_between_requests)
            
            all_responses.extend(query_responses)
        
        return all_responses

class SEOAnalyzer:
    def __init__(self):
        self.model = None
        self.embeddings = None
        
    @st.cache_resource
    def load_model(_self):
        """Load sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return None
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Failed to load sentence transformer model: {e}")
            return None
    
    def extract_passages(self, text: str, max_length: int = 200) -> List[str]:
        """Break content into passages"""
        if not text or pd.isna(text):
            return []
        
        text_str = str(text)
        
        if NLTK_AVAILABLE:
            try:
                # Split by sentences using NLTK
                sentences = nltk.sent_tokenize(text_str)
            except:
                # Fallback to basic sentence splitting
                sentences = self._basic_sentence_split(text_str)
        else:
            sentences = self._basic_sentence_split(text_str)
        
        passages = []
        current_passage = ""
        
        for sentence in sentences:
            if len(current_passage + sentence)  List[str]:
        """Basic sentence splitting when NLTK is not available"""
        # Simple regex-based sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def generate_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for text passages"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.warning("Sentence transformers not available. Skipping embedding generation.")
            return None
        
        if not self.model:
            self.model = self.load_model()
        
        if self.model is None:
            return None
            
        try:
            return self.model.encode(texts)
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return None
    
    def analyze_user_intent(self, query: str, intent: str, reasoning: str) -> Dict:
        """Analyze user intent and provide insights"""
        intent_analysis = {
            'query': query,
            'declared_intent': intent,
            'reasoning': reasoning,
            'intent_category': self.categorize_intent(intent),
            'optimization_strategy': self.get_optimization_strategy(intent)
        }
        return intent_analysis
    
    def categorize_intent(self, intent: str) -> str:
        """Categorize user intent"""
        intent_lower = intent.lower()
        
        if any(word in intent_lower for word in ['buy', 'purchase', 'price', 'cost', 'deal']):
            return 'Transactional'
        elif any(word in intent_lower for word in ['how', 'what', 'why', 'when', 'where', 'guide', 'tutorial']):
            return 'Informational'
        elif any(word in intent_lower for word in ['best', 'vs', 'compare', 'review', 'top']):
            return 'Commercial Investigation'
        elif any(word in intent_lower for word in ['near', 'location', 'address', 'directions']):
            return 'Local'
        else:
            return 'Informational'
    
    def get_optimization_strategy(self, intent: str) -> List[str]:
        """Get optimization strategies based on intent"""
        intent_lower = intent.lower()
        strategies = []
        
        if 'informational' in intent_lower or any(word in intent_lower for word in ['how', 'what', 'guide']):
            strategies.extend([
                "Create comprehensive, step-by-step content",
                "Use FAQ format for common questions",
                "Include relevant examples and case studies",
                "Structure with clear headings and subheadings"
            ])
        
        if 'transactional' in intent_lower or any(word in intent_lower for word in ['buy', 'purchase']):
            strategies.extend([
                "Include product specifications and comparisons",
                "Add pricing information and availability",
                "Include customer reviews and testimonials",
                "Optimize for commercial keywords"
            ])
        
        if 'comparison' in intent_lower or 'vs' in intent_lower:
            strategies.extend([
                "Create detailed comparison tables",
                "Highlight pros and cons",
                "Include objective evaluation criteria",
                "Add visual comparisons where possible"
            ])
        
        return strategies if strategies else ["Focus on comprehensive, accurate information"]
    
    def find_semantic_gaps(self, target_embeddings: np.ndarray, 
                          content_embeddings: np.ndarray, 
                          threshold: float = 0.7) -> List[int]:
        """Find content gaps using cosine similarity"""
        if not SKLEARN_AVAILABLE or target_embeddings is None or content_embeddings is None:
            return []
        
        try:
            similarities = cosine_similarity(target_embeddings, content_embeddings)
            max_similarities = np.max(similarities, axis=1)
            gaps = np.where(max_similarities  Dict:
        """Analyze LLM responses for patterns and insights"""
        analysis = {
            'total_responses': len(responses),
            'successful_responses': len([r for r in responses if r.get('success', False)]),
            'providers': list(set([r['provider'] for r in responses])),
            'average_response_length': 0,
            'common_topics': [],
            'response_quality_score': 0
        }
        
        successful_responses = [r for r in responses if r.get('success', False)]
        
        if successful_responses:
            # Calculate average response length
            lengths = [len(r['response']) for r in successful_responses]
            analysis['average_response_length'] = sum(lengths) / len(lengths)
            
            # Extract common themes (simplified)
            all_text = ' '.join([r['response'] for r in successful_responses])
            words = re.findall(r'\b\w+\b', all_text.lower())
            word_freq = {}
            for word in words:
                if len(word) > 4:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top 10 most common words
            analysis['common_topics'] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Simple quality score based on response length and success rate
            success_rate = len(successful_responses) / len(responses) if len(responses) > 0 else 0
            avg_length_score = min(analysis['average_response_length'] / 1000, 1)  # Normalize to max 1000 chars
            analysis['response_quality_score'] = (success_rate + avg_length_score) / 2
        
        return analysis
