import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import re
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import time
import google.generativeai as genai
import openai
import requests

# Safe import for BeautifulSoup
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    st.warning("beautifulsoup4 not installed. Passage extraction from URLs will be disabled.")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'llm_responses' not in st.session_state:
    st.session_state.llm_responses = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def load_queries_from_csv(file) -> List[Dict]:
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

    def crawl_multiple_llms(self, queries: List[Dict], api_keys: Dict[str, str], delay_between_requests: float = 1.0) -> List[Dict]:
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
        return SentenceTransformer('all-MiniLM-L6-v2')

    def extract_passages(self, text: str, max_length: int = 200) -> List[str]:
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

    def generate_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self.model:
            self.model = self.load_model()
        return self.model.encode(texts)

    def analyze_llm_responses(self, responses: List[Dict]) -> Dict:
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

# --- Passage Extraction from URLs ---
def extract_passages_from_url(url, max_length=200):
    if not BEAUTIFULSOUP_AVAILABLE:
        st.warning("beautifulsoup4 not installed. Cannot extract passages from URLs.")
        return []
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        text = soup.get_text(separator=' ')
        sentences = re.split(r'(?<=[.!?]) +', text)
        passages = []
        current = ''
        for sent in sentences:
            if len(current) + len(sent) < max_length:
                current += ' ' + sent
            else:
                if current.strip():
                    passages.append(current.strip())
                current = sent
        if current.strip():
            passages.append(current.strip())
        passages = [p for p in passages if len(p) > 40]
        return passages
    except Exception as e:
        st.warning(f"Could not extract from {url}: {e}")
        return []

def score_passages(query, passages, model):
    query_vec = model.encode([query])
    passage_vecs = model.encode(passages)
    scores = cosine_similarity(query_vec, passage_vecs)[0]
    return list(zip(passages, scores))

def suggest_optimized_passage(query, top_competitor_passage, your_passage, openai_key):
    if not openai_key:
        return "OpenAI key required for optimization."
    prompt = (
        f"Query: {query}\n\n"
        f"Competitor's top passage:\n{top_competitor_passage}\n\n"
        f"Your current passage:\n{your_passage}\n\n"
        "Rewrite your passage to be more relevant to the query and competitive with the top passage, "
        "while maintaining your unique voice and accuracy."
    )
    client = openai.OpenAI(api_key=openai_key)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert SEO content editor."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

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

    st.header("Passage Relevance Engineering")
    query = st.text_input("Enter your target query:")
    your_url = st.text_input("Enter your URL (your page):")
    competitor_urls = st.text_area("Enter competitor URLs (one per line):")
    openai_key2 = st.text_input("OpenAI API Key (for optimization)", type="password", key="openai_key2")

    if st.button("Extract and Score Passages"):
        if not query or not your_url or not competitor_urls:
            st.warning("Please provide a query, your URL, and at least one competitor URL.")
        else:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            your_passages = extract_passages_from_url(your_url)
            st.subheader("Your Passages")
            st.write(your_passages)
            comp_urls = [u.strip() for u in competitor_urls.splitlines() if u.strip()]
            comp_passages = []
            for url in comp_urls:
                passages = extract_passages_from_url(url)
                comp_passages.extend(passages)
            st.subheader("Competitor Passages")
            st.write(comp_passages[:10])
            your_scores = score_passages(query, your_passages, model)
            comp_scores = score_passages(query, comp_passages, model)
            your_top = sorted(your_scores, key=lambda x: -x[1])[0] if your_scores else ("", 0)
            comp_top = sorted(comp_scores, key=lambda x: -x[1])[0] if comp_scores else ("", 0)
            st.markdown(f"**Your top passage (score {your_top[1]:.2f}):**\n\n{your_top[0]}")
            st.markdown(f"**Competitor's top passage (score {comp_top[1]:.2f}):**\n\n{comp_top[0]}")
            if st.button("Suggest Optimized Passage"):
                optimized = suggest_optimized_passage(query, comp_top[0], your_top[0], openai_key2)
                st.markdown("**Optimized Passage Suggestion:**")
                st.write(optimized)
                if st.button("Re-score Optimized Passage"):
                    opt_score = score_passages(query, [optimized], model)[0][1]
                    st.markdown(f"**Optimized Passage Score:** {opt_score:.2f}")

if __name__ == "__main__":
    main()

