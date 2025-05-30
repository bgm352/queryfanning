import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from typing import List, Dict, Tuple, Set
import json
from datetime import datetime
import time
from collections import Counter, defaultdict
import google.generativeai as genai
import openai
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textstat import flesch_reading_ease, flesch_kincaid_grade

# --- Robust NLTK Tokenizer Setup ---
try:
    nltk.data.find('tokenizers/punkt')
    punkt_available = True
except LookupError:
    punkt_available = False

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
try:
    from nltk.tokenize import word_tokenize
except ImportError:
    word_tokenize = None

def safe_word_tokenize(text):
    """Try to use nltk word_tokenize, else fallback to split."""
    if punkt_available and word_tokenize is not None:
        try:
            return word_tokenize(text)
        except LookupError:
            return text.split()
    else:
        return text.split()

# --- Streamlit session state ---
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'query_clusters' not in st.session_state:
    st.session_state.query_clusters = None
if 'content_gaps' not in st.session_state:
    st.session_state.content_gaps = None
if 'serp_analysis' not in st.session_state:
    st.session_state.serp_analysis = None

# --- QueryAnalyzer ---
class QueryAnalyzer:
    def __init__(self):
        self.model = None
        self.stop_words = set(stopwords.words('english'))

    @st.cache_resource
    def load_model(_self):
        return SentenceTransformer('all-MiniLM-L6-v2')

    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        if not text or pd.isna(text):
            return []
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = safe_word_tokenize(text)
        keywords = [word for word in tokens if word not in self.stop_words and len(word) >= min_length]
        return keywords

    def analyze_query_intent(self, query: str) -> Dict:
        query_lower = query.lower()
        informational_signals = ['how', 'what', 'why', 'when', 'where', 'who', 'guide', 'tutorial', 'learn', 'explain']
        commercial_signals = ['best', 'top', 'review', 'compare', 'vs', 'versus', 'alternative', 'option']
        transactional_signals = ['buy', 'purchase', 'price', 'cost', 'cheap', 'deal', 'discount', 'shop', 'order']
        navigational_signals = ['login', 'contact', 'about', 'home', 'official', 'website']
        local_signals = ['near', 'location', 'address', 'directions', 'local', 'nearby']

        intent_scores = {
            'informational': sum(1 for signal in informational_signals if signal in query_lower),
            'commercial': sum(1 for signal in commercial_signals if signal in query_lower),
            'transactional': sum(1 for signal in transactional_signals if signal in query_lower),
            'navigational': sum(1 for signal in navigational_signals if signal in query_lower),
            'local': sum(1 for signal in local_signals if signal in query_lower)
        }
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        word_count = len(query.split())
        has_question = any(q in query_lower for q in ['?', 'how', 'what', 'why', 'when', 'where', 'who'])
        has_brand = self.detect_brand_mentions(query)
        return {
            'query': query,
            'primary_intent': primary_intent[0] if primary_intent[1] > 0 else 'informational',
            'intent_confidence': primary_intent[1] / max(sum(intent_scores.values()), 1),
            'intent_scores': intent_scores,
            'word_count': word_count,
            'is_question': has_question,
            'has_brand': has_brand,
            'query_length_category': self.categorize_query_length(word_count),
            'keywords': self.extract_keywords(query)
        }

    def detect_brand_mentions(self, query: str) -> bool:
        brand_indicators = ['amazon', 'google', 'apple', 'microsoft', 'facebook', 'netflix', 'uber', 'airbnb']
        return any(brand in query.lower() for brand in brand_indicators)

    def categorize_query_length(self, word_count: int) -> str:
        if word_count <= 2:
            return 'Short-tail'
        elif word_count <= 4:
            return 'Medium-tail'
        else:
            return 'Long-tail'

    def cluster_queries(self, queries: List[str], n_clusters: int = 5) -> Dict:
        if not self.model:
            self.model = self.load_model()
        embeddings = self.model.encode(queries)
        kmeans = KMeans(n_clusters=min(n_clusters, len(queries)), random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append({
                'query': queries[i],
                'index': i
            })
        cluster_summaries = {}
        for cluster_id, queries_in_cluster in clusters.items():
            cluster_queries = [q['query'] for q in queries_in_cluster]
            all_keywords = []
            for query in cluster_queries:
                all_keywords.extend(self.extract_keywords(query))
            keyword_freq = Counter(all_keywords)
            top_keywords = keyword_freq.most_common(5)
            cluster_summaries[cluster_id] = {
                'size': len(cluster_queries),
                'queries': cluster_queries,
                'top_keywords': top_keywords,
                'representative_query': cluster_queries[0]
            }
        return {
            'embeddings': embeddings,
            'labels': cluster_labels,
            'clusters': dict(clusters),
            'summaries': cluster_summaries,
            'n_clusters': len(cluster_summaries)
        }

    def analyze_content_gaps(self, queries: List[str], existing_content: List[str] = None) -> Dict:
        if not self.model:
            self.model = self.load_model()
        all_keywords = []
        intent_distribution = defaultdict(int)
        length_distribution = defaultdict(int)
        for query in queries:
            analysis = self.analyze_query_intent(query)
            all_keywords.extend(analysis['keywords'])
            intent_distribution[analysis['primary_intent']] += 1
            length_distribution[analysis['query_length_category']] += 1
        keyword_freq = Counter(all_keywords)
        trending_keywords = keyword_freq.most_common(20)
        total_queries = len(queries)
        intent_percentages = {intent: (count/total_queries)*100 for intent, count in intent_distribution.items()}
        gaps = {
            'trending_keywords': trending_keywords,
            'intent_distribution': dict(intent_distribution),
            'intent_percentages': intent_percentages,
            'length_distribution': dict(length_distribution),
            'underserved_intents': [intent for intent, pct in intent_percentages.items() if pct < 10],
            'opportunity_keywords': [kw for kw, freq in trending_keywords if freq >= 2]
        }
        return gaps

# --- SERPAnalyzer ---
class SERPAnalyzer:
    def __init__(self):
        self.serp_features = [
            'featured_snippets', 'people_also_ask', 'local_pack',
            'knowledge_panel', 'image_pack', 'video_carousel',
            'shopping_results', 'news_results'
        ]

    def analyze_serp_features(self, query: str) -> Dict:
        query_lower = query.lower()
        features = {}
        features['featured_snippets'] = any(q in query_lower for q in ['how', 'what', 'why', 'when'])
        features['local_pack'] = any(loc in query_lower for loc in ['near', 'location', 'local', 'directions'])
        features['shopping_results'] = any(shop in query_lower for shop in ['buy', 'price', 'cheap', 'product'])
        features['image_pack'] = any(img in query_lower for img in ['photo', 'image', 'picture', 'design'])
        features['video_carousel'] = any(vid in query_lower for vid in ['how to', 'tutorial', 'guide'])
        features['people_also_ask'] = len(query.split()) <= 3
        return {
            'query': query,
            'predicted_features': features,
            'feature_count': sum(features.values()),
            'serp_complexity': 'High' if sum(features.values()) > 3 else 'Medium' if sum(features.values()) > 1 else 'Low'
        }

# --- ContentOptimizer ---
class ContentOptimizer:
    def __init__(self):
        pass

    def generate_content_suggestions(self, query_analysis: Dict, cluster_info: Dict = None) -> Dict:
        query = query_analysis['query']
        intent = query_analysis['primary_intent']
        keywords = query_analysis['keywords']
        suggestions = {
            'query': query,
            'primary_intent': intent,
            'content_type_recommendations': self.get_content_type_recommendations(intent),
            'keyword_targets': keywords[:5],
            'content_structure': self.get_content_structure(intent),
            'optimization_priorities': self.get_optimization_priorities(query_analysis)
        }
        return suggestions

    def get_content_type_recommendations(self, intent: str) -> List[str]:
        recommendations = {
            'informational': [
                'Comprehensive blog posts', 'How-to guides', 'FAQ sections', 'Educational videos', 'Infographics'
            ],
            'commercial': [
                'Comparison pages', 'Product review articles', 'Buying guides', 'Feature comparison tables', 'Expert recommendations'
            ],
            'transactional': [
                'Product pages', 'Landing pages', 'Price comparison tools', 'Customer testimonials', 'Call-to-action optimization'
            ],
            'navigational': [
                'Clear site navigation', 'Brand pages', 'Contact information', 'About pages', 'Site search optimization'
            ],
            'local': [
                'Location pages', 'Local business listings', 'Map integrations', 'Local testimonials', 'Contact information with address'
            ]
        }
        return recommendations.get(intent, recommendations['informational'])

    def get_content_structure(self, intent: str) -> List[str]:
        structures = {
            'informational': [
                'Clear headings and subheadings', 'Step-by-step instructions', 'Examples and case studies', 'Summary or conclusion', 'Related articles section'
            ],
            'commercial': [
                'Product comparison tables', 'Pros and cons lists', 'Feature highlights', 'User reviews and ratings', 'Clear pricing information'
            ],
            'transactional': [
                'Product specifications', 'Clear pricing and availability', 'Customer reviews', 'Easy purchase process', 'Trust signals and guarantees'
            ]
        }
        return structures.get(intent, structures['informational'])

    def get_optimization_priorities(self, query_analysis: Dict) -> List[str]:
        priorities = []
        if query_analysis['is_question']:
            priorities.append('Optimize for featured snippets')
        if query_analysis['query_length_category'] == 'Long-tail':
            priorities.append('Focus on long-tail keyword optimization')
        if query_analysis['intent_confidence'] > 0.7:
            priorities.append(f'Strong {query_analysis["primary_intent"]} intent - align content accordingly')
        if query_analysis['has_brand']:
            priorities.append('Include brand-specific information')
        priorities.append('Optimize for semantic search')
        priorities.append('Improve page loading speed')
        priorities.append('Ensure mobile responsiveness')
        return priorities

# --- Main Streamlit App ---
def main():
    st.set_page_config(
        page_title="QForia-Style SEO Query Analyzer",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 QForia-Style SEO Query Analyzer")
    st.markdown("Advanced query analysis tool for understanding search intent and identifying content opportunities")

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    # Gemini API Key Field
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        st.sidebar.success("Gemini API Key set!")

    analysis_mode = st.sidebar.selectbox(
        "Analysis Mode",
        ["Upload CSV", "Manual Entry", "Paste Text"]
    )

    n_clusters = st.sidebar.slider(
        "Number of Query Clusters",
        min_value=3,
        max_value=15,
        value=5
    )

    min_keyword_freq = st.sidebar.slider(
        "Minimum Keyword Frequency",
        min_value=1,
        max_value=10,
        value=2
    )

    # Data Input Section
    st.header("📊 Data Input")
    queries = []
    if analysis_mode == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'query' in df.columns:
                queries = df['query'].dropna().astype(str).tolist()
            else:
                st.error("CSV must contain a 'query' column.")
    elif analysis_mode == "Paste Text":
        text_input = st.text_area("Paste queries (one per line):")
        if text_input:
            queries = [line.strip() for line in text_input.split('\n') if line.strip()]
    elif analysis_mode == "Manual Entry":
        query_input = st.text_input("Enter a single query:")
        if query_input:
            queries = [query_input.strip()]

    if queries:
        st.success(f"{len(queries)} queries loaded.")
        query_analyzer = QueryAnalyzer()
        serp_analyzer = SERPAnalyzer()
        content_optimizer = ContentOptimizer()
        # Analyze queries
        cluster_result = query_analyzer.cluster_queries(queries, n_clusters=n_clusters)
        content_gaps = query_analyzer.analyze_content_gaps(queries)
        st.session_state['query_clusters'] = cluster_result
        st.session_state['content_gaps'] = content_gaps

        # Show cluster summaries
        st.subheader("🔗 Query Clusters")
        for cid, summary in cluster_result['summaries'].items():
            st.markdown(f"**Cluster {cid+1}** ({summary['size']} queries)")
            st.write("Top Keywords:", [kw for kw, _ in summary['top_keywords']])
            st.write("Representative Query:", summary['representative_query'])
            with st.expander("Show Queries"):
                st.write(summary['queries'])

        # Show content gaps
        st.subheader("🕳️ Content Gap Analysis")
        st.write("Trending Keywords:", [kw for kw, _ in content_gaps['trending_keywords']])
        st.write("Intent Distribution:", content_gaps['intent_distribution'])
        st.write("Underserved Intents:", content_gaps['underserved_intents'])

        # Word Cloud
        st.subheader("☁️ Keyword Word Cloud")
        word_freq = dict(content_gaps['trending_keywords'])
        if word_freq:
            wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        # SERP and Content Optimization for first query
        st.subheader("🔍 SERP & Content Optimization (First Query Example)")
        first_query = queries[0]
        query_analysis = query_analyzer.analyze_query_intent(first_query)
        serp_features = serp_analyzer.analyze_serp_features(first_query)
        content_suggestions = content_optimizer.generate_content_suggestions(query_analysis)
        st.write("Query Analysis:", query_analysis)
        st.write("SERP Features:", serp_features)
        st.write("Content Suggestions:", content_suggestions)

if __name__ == "__main__":
    main()


