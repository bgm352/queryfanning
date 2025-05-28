import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import re
from typing import List, Dict, Tuple
import json
from datetime import datetime
import time
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

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
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant providing accurate, comprehensive answers."},
                    {"role": "user", "content": query}
                ],
                "max_tokens": 1500,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "provider": "OpenAI GPT-3.5",
                    "query": query,
                    "response": result['choices'][0]['message']['content'],
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "provider": "OpenAI GPT-3.5",
                    "query": query,
                    "response": "",
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "provider": "OpenAI GPT-3.5",
                "query": query,
                "response": "",
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def query_gemini(self, query: str, api_key: str) -> Dict:
        """Query Google Gemini via REST API"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": query
                    }]
                }]
            }
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    return {
                        "provider": "Google Gemini",
                        "query": query,
                        "response": content,
                        "success": True,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "provider": "Google Gemini",
                        "query": query,
                        "response": "",
                        "success": False,
                        "error": "No content in response",
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                return {
                    "provider": "Google Gemini",
                    "query": query,
                    "response": "",
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
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
                "model": "claude-3-haiku-20240307",
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
        total_requests = len(queries) * len(api_keys)
        current_request = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, query in enumerate(queries):
            query_responses = []
            
            # Query each LLM
            if api_keys.get('openai'):
                current_request += 1
                status_text.text(f"Querying OpenAI for: {query[:50]}... ({current_request}/{total_requests})")
                progress_bar.progress(current_request / total_requests)
                
                response = self.query_openai(query, api_keys['openai'])
                query_responses.append(response)
                time.sleep(delay_between_requests)
            
            if api_keys.get('gemini'):
                current_request += 1
                status_text.text(f"Querying Gemini for: {query[:50]}... ({current_request}/{total_requests})")
                progress_bar.progress(current_request / total_requests)
                
                response = self.query_gemini(query, api_keys['gemini'])
                query_responses.append(response)
                time.sleep(delay_between_requests)
            
            if api_keys.get('claude'):
                current_request += 1
                status_text.text(f"Querying Claude for: {query[:50]}... ({current_request}/{total_requests})")
                progress_bar.progress(current_request / total_requests)
                
                response = self.query_claude(query, api_keys['claude'])
                query_responses.append(response)
                time.sleep(delay_between_requests)
            
            all_responses.extend(query_responses)
        
        status_text.empty()
        progress_bar.empty()
        return all_responses

class SEOAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def extract_passages(self, text: str, max_length: int = 200) -> List[str]:
        """Break content into passages using simple sentence splitting"""
        if not text or pd.isna(text):
            return []
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', str(text))
        passages = []
        current_passage = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_passage + sentence) <= max_length:
                current_passage += sentence + ". "
            else:
                if current_passage:
                    passages.append(current_passage.strip())
                current_passage = sentence + ". "
        
        if current_passage:
            passages.append(current_passage.strip())
            
        return passages
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF embeddings for text passages"""
        if not texts:
            return np.array([])
        
        # Clean texts
        clean_texts = [str(text) for text in texts if text and str(text).strip()]
        if not clean_texts:
            return np.array([])
        
        try:
            embeddings = self.vectorizer.fit_transform(clean_texts)
            return embeddings.toarray()
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return np.array([])
    
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
        
        if any(word in intent_lower for word in ['buy', 'purchase', 'price', 'cost', 'deal', 'shop']):
            return 'Transactional'
        elif any(word in intent_lower for word in ['how', 'what', 'why', 'when', 'where', 'guide', 'tutorial', 'learn']):
            return 'Informational'
        elif any(word in intent_lower for word in ['best', 'vs', 'compare', 'review', 'top', 'versus']):
            return 'Commercial Investigation'
        elif any(word in intent_lower for word in ['near', 'location', 'address', 'directions', 'local']):
            return 'Local'
        else:
            return 'Informational'
    
    def get_optimization_strategy(self, intent: str) -> List[str]:
        """Get optimization strategies based on intent"""
        intent_lower = intent.lower()
        strategies = []
        
        if any(word in intent_lower for word in ['how', 'what', 'guide', 'tutorial', 'learn']):
            strategies.extend([
                "Create comprehensive, step-by-step content",
                "Use FAQ format for common questions",
                "Include relevant examples and case studies",
                "Structure with clear headings and subheadings",
                "Add schema markup for HowTo or FAQ content"
            ])
        
        if any(word in intent_lower for word in ['buy', 'purchase', 'price', 'cost']):
            strategies.extend([
                "Include product specifications and comparisons",
                "Add pricing information and availability",
                "Include customer reviews and testimonials",
                "Optimize for commercial keywords",
                "Add product schema markup"
            ])
        
        if any(word in intent_lower for word in ['compare', 'vs', 'best', 'review']):
            strategies.extend([
                "Create detailed comparison tables",
                "Highlight pros and cons",
                "Include objective evaluation criteria",
                "Add visual comparisons where possible",
                "Use review schema markup"
            ])
        
        if any(word in intent_lower for word in ['near', 'location', 'local']):
            strategies.extend([
                "Optimize for local SEO",
                "Include location-specific information",
                "Add local business schema markup",
                "Include contact information and hours"
            ])
        
        return strategies if strategies else ["Focus on comprehensive, accurate information", "Use clear structure and headings", "Include relevant keywords naturally"]
    
    def find_semantic_gaps(self, target_embeddings: np.ndarray, 
                          content_embeddings: np.ndarray, 
                          threshold: float = 0.7) -> List[int]:
        """Find content gaps using cosine similarity"""
        if target_embeddings.size == 0 or content_embeddings.size == 0:
            return []
        
        try:
            similarities = cosine_similarity(target_embeddings, content_embeddings)
            max_similarities = np.max(similarities, axis=1)
            gaps = np.where(max_similarities < threshold)[0]
            return gaps.tolist()
        except Exception as e:
            st.error(f"Error finding semantic gaps: {str(e)}")
            return []
    
    def analyze_llm_responses(self, responses: List[Dict]) -> Dict:
        """Analyze LLM responses for patterns and insights"""
        analysis = {
            'total_responses': len(responses),
            'successful_responses': len([r for r in responses if r.get('success', False)]),
            'providers': list(set([r['provider'] for r in responses])),
            'average_response_length': 0,
            'common_topics': [],
            'response_quality_score': 0,
            'provider_success_rates': {}
        }
        
        successful_responses = [r for r in responses if r.get('success', False)]
        
        if successful_responses:
            # Calculate average response length
            lengths = [len(r['response']) for r in successful_responses]
            analysis['average_response_length'] = sum(lengths) / len(lengths)
            
            # Provider success rates
            for provider in analysis['providers']:
                provider_responses = [r for r in responses if r['provider'] == provider]
                provider_successful = [r for r in provider_responses if r.get('success', False)]
                if provider_responses:
                    analysis['provider_success_rates'][provider] = len(provider_successful) / len(provider_responses)
            
            # Extract common themes (simplified)
            all_text = ' '.join([r['response'] for r in successful_responses])
            words = re.findall(r'\b\w+\b', all_text.lower())
            word_freq = {}
            for word in words:
                if len(word) > 4 and word not in ['should', 'would', 'could', 'might', 'about', 'where', 'there', 'these', 'those']:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top 10 most common words
            analysis['common_topics'] = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Simple quality score based on response length and success rate
            success_rate = len(successful_responses) / len(responses)
            avg_length_score = min(analysis['average_response_length'] / 1000, 1)  # Normalize to 0-1
            analysis['response_quality_score'] = (success_rate + avg_length_score) / 2
        
        return analysis

def main():
    st.set_page_config(
        page_title="Advanced SEO AI Citations Analyzer",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Advanced SEO AI Citations Analyzer")
    st.markdown("Upload CSV with query types and user intents, then crawl multiple LLMs for comprehensive analysis")
    
    analyzer = SEOAnalyzer()
    crawler = LLMCrawler()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API Keys
    st.sidebar.subheader("API Keys")
    api_keys = {}
    
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Get from https://platform.openai.com/api-keys")
    if openai_key:
        api_keys['openai'] = openai_key
    
    gemini_key = st.sidebar.text_input("Google Gemini API Key", type="password", help="Get from https://ai.google.dev/")
    if gemini_key:
        api_keys['gemini'] = gemini_key
    
    claude_key = st.sidebar.text_input("Anthropic Claude API Key", type="password", help="Get from https://console.anthropic.com/")
    if claude_key:
        api_keys['claude'] = claude_key
    
    # Analysis parameters
    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.7, 
        step=0.05,
        help="Lower values find more gaps"
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
        step=0.5,
        help="Prevent rate limiting"
    )
    
    # Sample CSV template
    with st.sidebar.expander("📋 CSV Template"):
        st.write("Your CSV should have these columns:")
        st.code("""query,query_type,user_intent,reasoning
"how to optimize for AI search",how-to,Learn SEO techniques,User wants actionable steps
"best CRM software 2024",comparison,Compare CRM solutions,User researching before purchase
"what is machine learning",definition,Understand ML concepts,User needs foundational knowledge""")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose CSV file with query data",
        type="csv",
        help="CSV should contain columns: query, query_type, user_intent, reasoning"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} queries")
            
            # Validate required columns
            required_columns = ['query', 'query_type', 'user_intent', 'reasoning']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.write(f"Available columns: {', '.join(df.columns.tolist())}")
                st.write("Please ensure your CSV has the required columns as shown in the sidebar template.")
                return
            
            # Display data preview
            with st.expander("📊 Data Preview", expanded=True):
                st.dataframe(df.head(10))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show query type distribution
                    query_type_dist = df['query_type'].value_counts()
                    fig = px.pie(values=query_type_dist.values, names=query_type_dist.index, 
                               title="Query Type Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Show basic stats
                    st.metric("Total Queries", len(df))
                    st.metric("Unique Query Types", df['query_type'].nunique())
                    st.metric("Average Query Length", f"{df['query'].str.len().mean():.0f} chars")
            
            # Step 1: User Intent Analysis
            st.header("🎯 Step 1: User Intent Analysis")
            
            with st.spinner("Analyzing user intents..."):
                intent_analysis = []
                for _, row in df.iterrows():
                    analysis = analyzer.analyze_user_intent(
                        row['query'], 
                        row['user_intent'], 
                        row['reasoning']
                    )
                    intent_analysis.append(analysis)
            
            intent_df = pd.DataFrame(intent_analysis)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Intent category distribution
                intent_categories = intent_df['intent_category'].value_counts()
                fig = px.bar(x=intent_categories.index, y=intent_categories.values,
                           title="Categorized Intent Distribution",
                           color=intent_categories.values,
                           color_continuous_scale="viridis")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sample intent analysis
                st.subheader("Sample Intent Analysis")
                sample_idx = st.selectbox("Select query to analyze:", 
                                        range(len(intent_df)),
                                        format_func=lambda x: f"{x+1}. {intent_analysis[x]['query'][:40]}...")
                selected_analysis = intent_analysis[sample_idx]
                
                st.write(f"**Query:** {selected_analysis['query']}")
                st.write(f"**Declared Intent:** {selected_analysis['declared_intent']}")
                st.write(f"**Categorized as:** `{selected_analysis['intent_category']}`")
                st.write(f"**Reasoning:** {selected_analysis['reasoning']}")
                
                st.write("**Recommended Optimization Strategies:**")
                for strategy in selected_analysis['optimization_strategy']:
                    st.write(f"• {strategy}")
            
            # Step 2: LLM Crawling
            st.header("🕷️ Step 2: LLM Response Crawling")
            
            if not api_keys:
                st.warning("⚠️ Please add at least one API key in the sidebar to crawl LLMs")
                st.info("💡 You can get API keys from:\n- OpenAI: https://platform.openai.com/api-keys\n- Google Gemini: https://ai.google.dev/\n- Anthropic Claude: https://console.anthropic.com/")
            else:
                st.write(f"🔑 Available LLMs: {', '.join(api_keys.keys())}")
                
                # Select queries to crawl
                max_queries = st.number_input(
                    "Maximum queries to crawl (to manage API costs):",
                    min_value=1,
                    max_value=len(df),
                    value=min(5, len(df)),
                    help="Start with a small number to test, then increase"
                )
                
                queries_to_crawl = df['query'].head(max_queries).tolist()
                
                # Show cost estimation
                total_requests = max_queries * len(api_keys)
                st.info(f"💰 This will make approximately {total_requests} API requests")
                
                if st.session_state.llm_responses is None:
                    if st.button("🚀 Start LLM Crawling", type="primary"):
                        with st.spinner("Crawling LLMs... This may take a while"):
                            responses = crawler.crawl_multiple_llms(
                                queries_to_crawl, 
                                api_keys, 
                                request_delay
                            )
                            
                            st.session_state.llm_responses = responses
                            st.success(f"✅ Completed crawling {len(responses)} responses")
                else:
                    st.success(f"✅ Using cached responses ({len(st.session_state.llm_responses)} total)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Re-crawl LLMs"):
                            st.session_state.llm_responses = None
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️ Clear Cache"):
                            st.session_state.llm_responses = None
                            st.session_state.analysis_results = None
                            st.success("Cache cleared!")
            
            # Step 3: LLM Response Analysis
            if st.session_state.llm_responses:
                st.header("📊 Step 3: LLM Response Analysis")
                
                response_analysis = analyzer.analyze_llm_responses(st.session_state.llm_responses)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Responses", response_analysis['total_responses'])
                
                with col2:
                    st.metric("Successful Responses", response_analysis['successful_responses'])
                
                with col3:
                    success_rate = response_analysis['successful_responses'] / response_analysis['total_responses'] * 100
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                with col4:
                    st.metric("Quality Score", f"{response_analysis['response_quality_score']:.2f}")
                
                # Provider performance
                if response_analysis['provider_success_rates']:
                    st.subheader("Provider Performance")
                    provider_df = pd.DataFrame([
                        {"Provider": provider, "Success Rate": f"{rate:.1%}", "Success Rate Value": rate}
                        for provider, rate in response_analysis['provider_success_rates'].items()
                    ])
                    
                    fig = px.bar(provider_df, x="Provider", y="Success Rate Value", 
                               title="Success Rate by Provider",
                               color="Success Rate Value",
                               color_continuous_scale="RdYlGn")
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Common topics
                if response_analysis['common_topics']:
                    st.subheader("Most Common Topics in Responses")
                    topics_df = pd.DataFrame(response_analysis['common_topics'], columns=['Topic', 'Frequency'])
                    fig = px.bar(topics_df.head(10), x="Frequency", y="Topic", orientation="h",
                               title="Top Topics Mentioned",
                               color="Frequency",
                               color_continuous_scale="blues")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Response details
                with st.expander("🔍 Detailed Response Analysis"):
                    response_df = []
                    for response in st.session_state.llm_responses:
                        response_df.append({
                            'Provider': response['provider'],
                            'Query': response['query'][:50] + '...',
                            'Success': '✅' if response['success'] else '❌',
                            'Response Length': len(response.get('response', '')),
                            'Timestamp': response['timestamp'][:19]
                        })
                    
                    st.dataframe(pd.DataFrame(response_df))
                
                # Individual responses
                with st.expander("📝 Individual LLM Responses"):
                    for i, response in enumerate(st.session_state.llm_responses):
                        st.write(f"**{i+1}. {response['provider']} - {response['query'][:50]}...**")
                        if response['success']:
                            st.write(response['response'][:500] + "..." if len(response['response']) > 500 else response['response'])
                        else:
                            st.error(f"❌ Error: {response.get('error', 'Unknown error')}")
                        st.write("---")
                
                # Step 4: Semantic Analysis
                st.header("🧠 Step 4: Semantic Gap Analysis")
                
                if st.button("🔍 Perform Semantic Analysis"):
                    with st.spinner("Analyzing semantic patterns..."):
                        # Extract successful responses for analysis
                        successful_responses = [r for r in st.session_state.llm_responses if r.get('success', False)]
                        
                        if successful_responses:
                            # Generate embeddings for LLM responses
                            llm_texts = [r['response'] for r in successful_responses]
                            llm_passages = []
                            
                            for text in llm_texts:
                                passages = analyzer.extract_passages(text, passage_length)
                                llm_passages.extend(passages)
                            
                            if llm_passages:
                                llm_embeddings = analyzer.generate_embeddings(llm_passages)
                                
                                if llm_embeddings.size > 0:
                                    # Analyze patterns across different LLMs
                                    providers = list(set([r['provider'] for r in successful_responses]))
                                    
                                    provider_analysis = {}
                                    for provider in providers:
                                        provider_responses = [r for r in successful_responses if r['provider'] == provider]
                                        provider_texts = [r['response'] for r in provider_responses]
                                        provider_passages = []
                                        
                                        for text in provider_texts:
                                            passages = analyzer.extract_passages(text, passage_length)
                                            provider_passages.extend(passages)
                                        
                                        if provider_passages:
                                            provider_embeddings = analyzer.generate_embeddings(provider_passages)
                                            if provider_embeddings.size > 0:
                                                provider_analysis[provider] = {
                                                    'passages': provider_passages,
                                                    'embeddings': provider_embeddings,
                                                    'avg_response_length': sum(len(t) for t in provider_texts) / len(provider_texts)
                                                }
                                    
                                    st.session_state.analysis_results = {
                                        'llm_embeddings': llm_embeddings,
                                        'llm_passages': llm_passages,
                                        'provider_analysis': provider_analysis
                                    }
                                    
                                    # Display provider comparison
                                    if len(providers) > 1:
                                        st.subheader("Provider Comparison")
                                        
                                        comparison_data = []
                                        for provider, data in provider_analysis.items():
                                            comparison_data.append({
                                                'Provider': provider,
                                                'Avg Response Length': f"{data['avg_response_length']:.0f} chars",
                                                'Total Passages': len(data['passages'])
                                            })
