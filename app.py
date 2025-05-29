import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import re
from typing import List, Dict, Tuple
import asyncio
import aiohttp
import json
from datetime import datetime
import time
import google.generativeai as genai
import openai
from concurrent.futures import ThreadPoolExecutor
import requests

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'llm_responses' not in st.session_state:
    st.session_state.llm_responses = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def load_queries_from_csv(file) -> List[Dict]:
    """Load queries from CSV with 'query' and 'type' columns"""
    df = pd.read_csv(file)
    required_columns = {'query', 'type'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")
    return df[['query', 'type']].to_dict('records')

class LLMCrawler:
    def __init__(self):
        self.responses = {}
        
    def query_openai(self, query: str, api_key: str) -> Dict:
        """Query OpenAI GPT"""
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

    def crawl_multiple_llms(self, queries: List[Dict], api_keys: Dict[str, str], 
                           delay_between_requests: float = 1.0) -> List[Dict]:
        """Crawl multiple LLMs with rate limiting and query/type pairs"""
        all_responses = []
        for q in queries:
            query_text = q['query']
            query_type = q['type']
            query_responses = []
            if api_keys.get('openai'):
                st.write(f"Querying OpenAI ({query_type}): {query_text[:50]}...")
                response = self.query_openai(query_text, api_keys['openai'])
                response['type'] = query_type
                response['query'] = query_text
                query_responses.append(response)
                time.sleep(delay_between_requests)
            if api_keys.get('gemini'):
                st.write(f"Querying Gemini ({query_type}): {query_text[:50]}...")
                response = self.query_gemini(query_text, api_keys['gemini'])
                response['type'] = query_type
                response['query'] = query_text
                query_responses.append(response)
                time.sleep(delay_between_requests)
            if api_keys.get('claude'):
                st.write(f"Querying Claude ({query_type}): {query_text[:50]}...")
                response = self.query_claude(query_text, api_keys['claude'])
                response['type'] = query_type
                response['query'] = query_text
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
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_passages(self, text: str, max_length: int = 200) -> List[str]:
        """Break content into passages"""
        if not text or pd.isna(text):
            return []
        sentences = nltk.sent_tokenize(str(text))
        passages = []
        current_passage = ""
        for sentence in sentences:
            if len(current_passage + sentence) <= max_length:
                current_passage += sentence + " "
            else:
                if current_passage:
                    passages.append(current_passage.strip())
                current_passage = sentence + " "
        if current_passage:
            passages.append(current_passage.strip())
        return passages
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for text passages"""
        if not self.model:
            self.model = self.load_model()
        return self.model.encode(texts)
    
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
        similarities = cosine_similarity(target_embeddings, content_embeddings)
        max_similarities = np.max(similarities, axis=1)
        gaps = np.where(max_similarities < threshold)[0]
        return gaps.tolist()
    
    def analyze_llm_responses(self, responses: List[Dict]) -> Dict:
        """Analyze LLM responses for patterns and insights, grouped by type"""
        analysis = {
            'total_responses': len(responses),
            'successful_responses': len([r for r in responses if r.get('success', False)]),
            'providers': list(set([r['provider'] for r in responses])),
            'types': list(set([r.get('type', 'unknown') for r in responses])),
            'type_metrics': {},
            'average_response_length': 0,
            'common_topics': [],
            'response_quality_score': 0
        }
        successful_responses = [r for r in responses if r.get('success', False)]
        if successful_responses:
            # Type-specific metrics
            type_groups = {}
            for resp in successful_responses:
                q_type = resp.get('type', 'unknown')
                if q_type not in type_groups:
                    type_groups[q_type] = []
                type_groups[q_type].append(resp)
            analysis['type_metrics'] = {
                q_type: {
                    'count': len(type_resps),
                    'avg_length': np.mean([len(r['response']) for r in type_resps]),
                    'success_rate': len([r for r in type_resps if r['success']])/len(type_resps)
                }
                for q_type, type_resps in type_groups.items()
            }
            lengths = [len(r['response']) for r in successful_responses]
            analysis['average_response_length'] = sum(lengths) / len(lengths)
            all_text = ' '.join([r['response'] for r in successful_responses])
            words = re.findall(r'\b\w+\b', all_text.lower())
            word_freq = {}
            for word in words:
                if len(word) > 4:
                    word_freq[word] = word_freq.get(word, 0) + 1
            analysis['common_topics'] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            success_rate = len(successful_responses) / len(responses) if len(responses) > 0 else 0
            avg_length_score = min(analysis['average_response_length'] / 1000, 1)
            analysis['response_quality_score'] = (success_rate + avg_length_score) / 2
        return analysis

# Example usage in Streamlit app:
def main():
    st.set_page_config(
        page_title="Advanced SEO AI Citations Analyzer",
        page_icon="🤖",
        layout="wide"
    )
    st.title("🤖 Advanced SEO AI Citations Analyzer")
    st.markdown("Upload CSV with queries and types, then crawl multiple LLMs for comprehensive analysis")
    analyzer = SEOAnalyzer()
    crawler = LLMCrawler()
    st.sidebar.header("Configuration")
    api_keys = {}
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if openai_key:
        api_keys['openai'] = openai_key
    gemini_key = st.sidebar.text_input("Google Gemini API Key", type="password")
    if gemini_key:
        api_keys['gemini'] = gemini_key
    claude_key = st.sidebar.text_input("Anthropic Claude API Key", type="password")
    if claude_key:
        api_keys['claude'] = claude_key
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.05
    )
    passage_length = st.sidebar.slider(
        "Max Passage Length", 
        min_value=50, 
        max_value=500, 
        value=200, 
        step=50
    )
    request_delay = st.sidebar.slider(
        "Delay Between API Requests (seconds)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5
    )
    uploaded_file = st.file_uploader(
        "Choose CSV file with query data",
        type="csv",
        help="CSV should contain columns: query, type"
    )
    if uploaded_file is not None:
        try:
            queries = load_queries_from_csv(uploaded_file)
            st.success(f"Loaded {len(queries)} queries")
            with st.expander("Data Preview", expanded=True):
                st.dataframe(pd.DataFrame(queries).head())
                type_dist = pd.DataFrame(queries)['type'].value_counts()
                fig = px.pie(values=type_dist.values, names=type_dist.index, 
                            title="Query Type Distribution")
                st.plotly_chart(fig, use_container_width=True)
            if st.button("Crawl LLMs"):
                responses = crawler.crawl_multiple_llms(
                    queries, api_keys, delay_between_requests=request_delay
                )
                st.session_state.llm_responses = responses
                st.success(f"Received {len(responses)} LLM responses")
                analysis = analyzer.analyze_llm_responses(responses)
                st.session_state.analysis_results = analysis
                st.json(analysis)
        except Exception as e:
            st.error(f"Error loading or processing CSV: {str(e)}")

if __name__ == "__main__":
    main()
