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

# ... (keep all existing imports and try/except blocks the same) ...

# ==============================================================================
# CSV HANDLING
# ==============================================================================
def load_queries_from_csv(file) -> List[Dict]:
    """Load queries from CSV with 'query' and 'type' columns"""
    df = pd.read_csv(file)
    
    # Validate CSV structure
    required_columns = {'query', 'type'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")
    
    return df[['query', 'type']].to_dict('records')

# ==============================================================================
# LLM CRAWLER UPDATES
# ==============================================================================
class LLMCrawler:
    def __init__(self):
        self.responses = {}
    
    # ... (keep existing query methods for OpenAI/Gemini/Claude the same) ...
    
    def crawl_multiple_llms(self, queries: List[Dict], api_keys: Dict[str, str], 
                           delay_between_requests: float = 1.0) -> List[Dict]:
        """Crawl multiple LLMs with rate limiting
        Args:
            queries: List of dicts with 'query' and 'type' keys
            api_keys: Dictionary of API keys
        """
        all_responses = []
        
        for q in queries:
            query_text = q['query']
            query_type = q['type']
            
            query_responses = []
            
            # Query each LLM and add type metadata
            if api_keys.get('openai') and OPENAI_AVAILABLE:
                st.write(f"Querying OpenAI ({query_type}): {query_text[:50]}...")
                response = self.query_openai(query_text, api_keys['openai'])
                response.update({
                    'query_type': query_type,
                    'query': query_text  # Include original query for reference
                })
                query_responses.append(response)
                time.sleep(delay_between_requests)
            
            if api_keys.get('gemini') and GEMINI_AVAILABLE:
                st.write(f"Querying Gemini ({query_type}): {query_text[:50]}...")
                response = self.query_gemini(query_text, api_keys['gemini'])
                response.update({
                    'query_type': query_type,
                    'query': query_text
                })
                query_responses.append(response)
                time.sleep(delay_between_requests)
            
            if api_keys.get('claude'):
                st.write(f"Querying Claude ({query_type}): {query_text[:50]}...")
                response = self.query_claude(query_text, api_keys['claude'])
                response.update({
                    'query_type': query_type,
                    'query': query_text
                })
                query_responses.append(response)
                time.sleep(delay_between_requests)
            
            all_responses.extend(query_responses)
        
        return all_responses

# ==============================================================================
# UPDATED ANALYSIS METHODS
# ==============================================================================
class SEOAnalyzer:
    # ... (keep existing init and other methods the same) ...
    
    def analyze_llm_responses(self, responses: List[Dict]) -> Dict:
        """Analyze LLM responses with type-aware metrics"""
        analysis = {
            'total_responses': len(responses),
            'successful_responses': len([r for r in responses if r.get('success', False)]),
            'providers': list(set([r['provider'] for r in responses])),
            'query_types': list(set([r.get('query_type', 'unknown') for r in responses])),
            'type_metrics': {},
            'average_response_length': 0,
            'common_topics': [],
            'response_quality_score': 0
        }
        
        successful_responses = [r for r in responses if r.get('success', False)]
        
        if successful_responses:
            # Calculate type-specific metrics
            type_groups = {}
            for resp in successful_responses:
                q_type = resp.get('query_type', 'unknown')
                if q_type not in type_groups:
                    type_groups[q_type] = []
                type_groups[q_type].append(resp)
            
            # Populate type-specific stats
            analysis['type_metrics'] = {
                q_type: {
                    'count': len(responses),
                    'avg_length': np.mean([len(r['response']) for r in responses]),
                    'success_rate': len([r for r in responses if r['success']])/len(responses)
                }
                for q_type, responses in type_groups.items()
            }
            
            # Global averages
            lengths = [len(r['response']) for r in successful_responses]
            analysis['average_response_length'] = sum(lengths) / len(lengths)
            
            # Common topics (unchanged)
            all_text = ' '.join([r['response'] for r in successful_responses])
            words = re.findall(r'\b\w+\b', all_text.lower())
            word_freq = {}
            for word in words:
                if len(word) > 4:
                    word_freq[word] = word_freq.get(word, 0) + 1
            analysis['common_topics'] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Quality score calculation (unchanged)
            success_rate = len(successful_responses) / len(responses) if len(responses) > 0 else 0
            avg_length_score = min(analysis['average_response_length'] / 1000, 1)
            analysis['response_quality_score'] = (success_rate + avg_length_score) / 2
        
        return analysis

# ==============================================================================
# STREAMLIT UI INTEGRATION
# ==============================================================================
def main():
    st.title("LLM SEO Analyzer")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload query CSV", type=['csv'], 
                                    help="CSV must contain 'query' and 'type' columns")
    
    if uploaded_file:
        try:
            queries = load_queries_from_csv(uploaded_file)
            st.success(f"Loaded {len(queries)} queries with types: {', '.join(set(q['type'] for q in queries))}")
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return
    
    # ... (rest of Streamlit UI code remains the same) ...

# Keep all other existing code below this line unchanged
