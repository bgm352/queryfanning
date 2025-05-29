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

    def crawl_multiple_llms(self, queries: List[str], api_keys: Dict[str, str], 
                           delay_between_requests: float = 1.0) -> List[Dict]:
        """Crawl multiple LLMs with rate limiting"""
        all_responses = []
        
        for query in queries:
            query_responses = []
            
            # Query each LLM
            if api_keys.get('openai'):
                st.write(f"Querying OpenAI for: {query[:50]}...")
                response = self.query_openai(query, api_keys['openai'])
                query_responses.append(response)
                time.sleep(delay_between_requests)
            
            if api_keys.get('gemini'):
                st.write(f"Querying Gemini for: {query[:50]}...")
                response = self.query_gemini(query, api_keys['gemini'])
                query_responses.append(response)
                time.sleep(delay_between_requests)
            
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
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_passages(self, text: str, max_length: int = 200) -> List[str]:
        """Break content into passages"""
        if not text or pd.isna(text):
            return []
        
        # Split by sentences
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
    
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if openai_key:
        api_keys['openai'] = openai_key
    
    gemini_key = st.sidebar.text_input("Google Gemini API Key", type="password")
    if gemini_key:
        api_keys['gemini'] = gemini_key
    
    claude_key = st.sidebar.text_input("Anthropic Claude API Key", type="password")
    if claude_key:
        api_keys['claude'] = claude_key
    
    # Analysis parameters
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
            st.success(f"Loaded {len(df)} queries")
            
            # Validate required columns
            required_columns = ['query', 'query_type', 'user_intent', 'reasoning']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.write(f"Available columns: {', '.join(df.columns.tolist())}")
                return
            
            # Display data preview
            with st.expander("Data Preview", expanded=True):
                st.dataframe(df.head())
                
                # Show query type distribution
                query_type_dist = df['query_type'].value_counts()
                fig = px.pie(values=query_type_dist.values, names=query_type_dist.index, 
                           title="Query Type Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Step 1: User Intent Analysis
            st.header("🎯 Step 1: User Intent Analysis")
            
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
                           title="Intent Categories")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Sample intent analysis
                st.subheader("Sample Intent Analysis")
                sample_idx = st.selectbox("Select query to analyze:", range(len(intent_df)))
                selected_analysis = intent_analysis[sample_idx]
                
                st.write(f"**Query:** {selected_analysis['query']}")
                st.write(f"**Declared Intent:** {selected_analysis['declared_intent']}")
                st.write(f"**Category:** {selected_analysis['intent_category']}")
                st.write(f"**Reasoning:** {selected_analysis['reasoning']}")
                
                st.write("**Optimization Strategies:**")
                for strategy in selected_analysis['optimization_strategy']:
                    st.write(f"• {strategy}")
            
            # Step 2: LLM Crawling
            st.header("🕷️ Step 2: LLM Response Crawling")
            
            if not api_keys:
                st.warning("Please add at least one API key in the sidebar to crawl LLMs")
            else:
                st.write(f"Available LLMs: {', '.join(api_keys.keys())}")
                
                # Select queries to crawl
                max_queries = st.number_input(
                    "Maximum queries to crawl (to manage API costs):",
                    min_value=1,
                    max_value=len(df),
                    value=min(5, len(df))
                )
                
                queries_to_crawl = df['query'].head(max_queries).tolist()
                
                if st.session_state.llm_responses is None:
                    if st.button("Start LLM Crawling"):
                        with st.spinner("Crawling LLMs... This may take a while"):
                            progress_bar = st.progress(0)
                            
                            responses = crawler.crawl_multiple_llms(
                                queries_to_crawl, 
                                api_keys, 
                                request_delay
                            )
                            
                            st.session_state.llm_responses = responses
                            progress_bar.progress(100)
                            st.success(f"Completed crawling {len(responses)} responses")
                else:
                    st.success(f"Using cached responses ({len(st.session_state.llm_responses)} total)")
                    
                    if st.button("Re-crawl LLMs"):
                        st.session_state.llm_responses = None
                        st.rerun()
            
            # Step 3: LLM Response Analysis
            if st.session_state.llm_responses:
                st.header("📊 Step 3: LLM Response Analysis")
                
                response_analysis = analyzer.analyze_llm_responses(st.session_state.llm_responses)
                
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
                
                # Response details
                with st.expander("Response Details"):
                    for i, response in enumerate(st.session_state.llm_responses):
                        st.write(f"**{response['provider']} - Query: {response['query'][:50]}...**")
                        if response['success']:
                            st.write(f"Response: {response['response'][:300]}...")
                        else:
                            st.error(f"Error: {response.get('error', 'Unknown error')}")
                        st.write("---")
                
                # Step 4: Semantic Analysis
                st.header("🧠 Step 4: Semantic Gap Analysis")
                
                if st.button("Perform Semantic Analysis"):
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
                                            'Avg Response Length': data['avg_response_length'],
                                            'Total Passages': len(data['passages'])
                                        })
                                    
                                    comparison_df = pd.DataFrame(comparison_data)
                                    st.dataframe(comparison_df)
                                    
                                    # Similarity between providers
                                    if len(providers) == 2:
                                        provider_names = list(provider_analysis.keys())
                                        emb1 = provider_analysis[provider_names[0]]['embeddings']
                                        emb2 = provider_analysis[provider_names[1]]['embeddings']
                                        
                                        similarities = cosine_similarity(emb1, emb2)
                                        avg_similarity = np.mean(similarities)
                                        
                                        st.metric(f"Average Similarity ({provider_names[0]} vs {provider_names[1]})", 
                                                f"{avg_similarity:.3f}")
                        else:
                            st.warning("No successful responses to analyze")
            
            # Step 5: Content Optimization Recommendations
            if st.session_state.analysis_results:
                st.header("✍️ Step 5: Content Optimization Recommendations")
                
                # Generate recommendations based on analysis
                recommendations = []
                
                for i, row in df.iterrows():
                    query = row['query']
                    intent_cat = intent_analysis[i]['intent_category']
                    strategies = intent_analysis[i]['optimization_strategy']
                    
                    # Find relevant LLM responses for this query
                    relevant_responses = [r for r in st.session_state.llm_responses 
                                        if r['query'] == query and r.get('success', False)]
                    
                    recommendation = {
                        'query': query,
                        'intent_category': intent_cat,
                        'strategies': strategies,
                        'llm_insights': len(relevant_responses),
                        'priority': 'High' if intent_cat in ['Transactional', 'Commercial Investigation'] else 'Medium'
                    }
                    recommendations.append(recommendation)
                
                # Display recommendations
                rec_df = pd.DataFrame(recommendations)
                st.dataframe(rec_df)
                
                # Detailed recommendations
                st.subheader("Detailed Optimization Plan")
                
                selected_query_idx = st.selectbox(
                    "Select query for detailed recommendations:",
                    range(len(recommendations)),
                    format_func=lambda x: recommendations[x]['query']
                )
                
                selected_rec = recommendations[selected_query_idx]
                
                st.write(f"**Query:** {selected_rec['query']}")
                st.write(f"**Intent Category:** {selected_rec['intent_category']}")
                st.write(f"**Priority:** {selected_rec['priority']}")
                
                st.write("**Optimization Strategies:**")
                for strategy in selected_rec['strategies']:
                    st.write(f"• {strategy}")
                
                # LLM-specific insights
                query_responses = [r for r in st.session_state.llm_responses 
                                 if r['query'] == selected_rec['query'] and r.get('success', False)]
                
                if query_responses:
                    st.write("**LLM Response Insights:**")
                    for response in query_responses:
                        with st.expander(f"{response['provider']} Response"):
                            st.write(response['response'][:500] + "..." if len(response['response']) > 500 else response['response'])
            
            # Step 6: Export Enhanced Results
            st.header("💾 Step 6: Export Enhanced Results")
            
            if st.button("Generate Comprehensive Report"):
                # Create detailed report
                report_data = {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_queries': len(df),
                    'intent_distribution': intent_df['intent_category'].value_counts().to_dict(),
                    'llm_responses': len(st.session_state.llm_responses) if st.session_state.llm_responses else 0,
                    'successful_responses': len([r for r in st.session_state.llm_responses if r.get('success', False)]) if st.session_state.llm_responses else 0,
                    'recommendations_generated': len(recommendations) if 'recommendations' in locals() else 0
                }
                
                # Enhanced CSV export
                enhanced_df = df.copy()
                
                # Add intent analysis
                for i, analysis in enumerate(intent_analysis):
                    enhanced_df.loc[i, 'intent_category'] = analysis['intent_category']
                    enhanced_df.loc[i, 'optimization_strategies'] = '; '.join(analysis['optimization_strategy'])
                
                # Add LLM response data
                if st.session_state.llm_responses:
                    for i, row in enhanced_df.iterrows():
                        query = row['query']
                        query_responses = [r for r in st.session_state.llm_responses if r['query'] == query]
                        
                        enhanced_df.loc[i, 'llm_responses_count'] = len(query_responses)
                        enhanced_df.loc[i, 'successful_llm_responses'] = len([r for r in query_responses if r.get('success', False)])
                        
                        # Add first successful response
                        successful_responses = [r for r in query_responses if r.get('success', False)]
                        if successful_responses:
                            enhanced_df.loc[i, 'sample_llm_response'] = successful_responses[0]['response'][:500]
                            enhanced_df.loc[i, 'responding_provider'] = successful_responses[0]['provider']
                
                # Download enhanced CSV
                csv_buffer = StringIO()
                enhanced_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="Download Enhanced Analysis CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"enhanced_seo_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Download JSON report
                json_report = json.dumps(report_data, indent=2)
                st.download_button(
                    label="Download JSON Report",
                    data=json_report,
                    file_name=f"seo_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your CSV has the required columns: query, query_type, user_intent, reasoning")

if __name__ == "__main__":
    main()


