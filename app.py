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
from io import StringIO
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

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'query_clusters' not in st.session_state:
    st.session_state.query_clusters = None
if 'content_gaps' not in st.session_state:
    st.session_state.content_gaps = None
if 'serp_analysis' not in st.session_state:
    st.session_state.serp_analysis = None

class QueryAnalyzer:
    def __init__(self):
        self.model = None
        self.stop_words = set(stopwords.words('english'))
        
    @st.cache_resource
    def load_model(_self):
        """Load sentence transformer model"""
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract keywords from text"""
        if not text or pd.isna(text):
            return []
        
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = word_tokenize(text)
        
        # Filter out stop words and short words
        keywords = [word for word in tokens 
                   if word not in self.stop_words and len(word) >= min_length]
        
        return keywords
    
    def analyze_query_intent(self, query: str) -> Dict:
        """Analyze query intent similar to QForia"""
        query_lower = query.lower()
        
        # Intent signals
        informational_signals = ['how', 'what', 'why', 'when', 'where', 'who', 'guide', 'tutorial', 'learn', 'explain']
        commercial_signals = ['best', 'top', 'review', 'compare', 'vs', 'versus', 'alternative', 'option']
        transactional_signals = ['buy', 'purchase', 'price', 'cost', 'cheap', 'deal', 'discount', 'shop', 'order']
        navigational_signals = ['login', 'contact', 'about', 'home', 'official', 'website']
        local_signals = ['near', 'location', 'address', 'directions', 'local', 'nearby']
        
        # Calculate intent scores
        intent_scores = {
            'informational': sum(1 for signal in informational_signals if signal in query_lower),
            'commercial': sum(1 for signal in commercial_signals if signal in query_lower),
            'transactional': sum(1 for signal in transactional_signals if signal in query_lower),
            'navigational': sum(1 for signal in navigational_signals if signal in query_lower),
            'local': sum(1 for signal in local_signals if signal in query_lower)
        }
        
        # Determine primary intent
        primary_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        # Additional query characteristics
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
        """Detect if query contains brand mentions"""
        # Common brand indicators (simplified)
        brand_indicators = ['amazon', 'google', 'apple', 'microsoft', 'facebook', 'netflix', 'uber', 'airbnb']
        return any(brand in query.lower() for brand in brand_indicators)
    
    def categorize_query_length(self, word_count: int) -> str:
        """Categorize query by length"""
        if word_count <= 2:
            return 'Short-tail'
        elif word_count <= 4:
            return 'Medium-tail'
        else:
            return 'Long-tail'
    
    def cluster_queries(self, queries: List[str], n_clusters: int = 5) -> Dict:
        """Cluster queries by semantic similarity"""
        if not self.model:
            self.model = self.load_model()
        
        # Generate embeddings
        embeddings = self.model.encode(queries)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(queries)), random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Organize results
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append({
                'query': queries[i],
                'index': i
            })
        
        # Generate cluster summaries
        cluster_summaries = {}
        for cluster_id, queries_in_cluster in clusters.items():
            cluster_queries = [q['query'] for q in queries_in_cluster]
            
            # Extract common keywords
            all_keywords = []
            for query in cluster_queries:
                all_keywords.extend(self.extract_keywords(query))
            
            keyword_freq = Counter(all_keywords)
            top_keywords = keyword_freq.most_common(5)
            
            cluster_summaries[cluster_id] = {
                'size': len(cluster_queries),
                'queries': cluster_queries,
                'top_keywords': top_keywords,
                'representative_query': cluster_queries[0]  # First query as representative
            }
        
        return {
            'embeddings': embeddings,
            'labels': cluster_labels,
            'clusters': dict(clusters),
            'summaries': cluster_summaries,
            'n_clusters': len(cluster_summaries)
        }
    
    def analyze_content_gaps(self, queries: List[str], existing_content: List[str] = None) -> Dict:
        """Identify content gaps similar to QForia's approach"""
        if not self.model:
            self.model = self.load_model()
        
        # Analyze query patterns
        all_keywords = []
        intent_distribution = defaultdict(int)
        length_distribution = defaultdict(int)
        
        for query in queries:
            analysis = self.analyze_query_intent(query)
            all_keywords.extend(analysis['keywords'])
            intent_distribution[analysis['primary_intent']] += 1
            length_distribution[analysis['query_length_category']] += 1
        
        # Identify trending keywords
        keyword_freq = Counter(all_keywords)
        trending_keywords = keyword_freq.most_common(20)
        
        # Identify underserved intents
        total_queries = len(queries)
        intent_percentages = {intent: (count/total_queries)*100 
                            for intent, count in intent_distribution.items()}
        
        # Gaps analysis
        gaps = {
            'trending_keywords': trending_keywords,
            'intent_distribution': dict(intent_distribution),
            'intent_percentages': intent_percentages,
            'length_distribution': dict(length_distribution),
            'underserved_intents': [intent for intent, pct in intent_percentages.items() if pct < 10],
            'opportunity_keywords': [kw for kw, freq in trending_keywords if freq >= 2]
        }
        
        return gaps

class SERPAnalyzer:
    def __init__(self):
        self.serp_features = [
            'featured_snippets', 'people_also_ask', 'local_pack', 
            'knowledge_panel', 'image_pack', 'video_carousel',
            'shopping_results', 'news_results'
        ]
    
    def analyze_serp_features(self, query: str) -> Dict:
        """Simulate SERP feature analysis (would integrate with real SERP API)"""
        query_lower = query.lower()
        
        # Predict likely SERP features based on query characteristics
        features = {}
        
        # Featured snippets more likely for question queries
        features['featured_snippets'] = any(q in query_lower for q in ['how', 'what', 'why', 'when'])
        
        # Local pack for location queries
        features['local_pack'] = any(loc in query_lower for loc in ['near', 'location', 'local', 'directions'])
        
        # Shopping results for product queries
        features['shopping_results'] = any(shop in query_lower for shop in ['buy', 'price', 'cheap', 'product'])
        
        # Image pack for visual queries
        features['image_pack'] = any(img in query_lower for img in ['photo', 'image', 'picture', 'design'])
        
        # Video carousel for tutorial queries
        features['video_carousel'] = any(vid in query_lower for vid in ['how to', 'tutorial', 'guide'])
        
        # People also ask for broad topics
        features['people_also_ask'] = len(query.split()) <= 3
        
        return {
            'query': query,
            'predicted_features': features,
            'feature_count': sum(features.values()),
            'serp_complexity': 'High' if sum(features.values()) > 3 else 'Medium' if sum(features.values()) > 1 else 'Low'
        }

class ContentOptimizer:
    def __init__(self):
        pass
    
    def generate_content_suggestions(self, query_analysis: Dict, cluster_info: Dict = None) -> Dict:
        """Generate content optimization suggestions"""
        query = query_analysis['query']
        intent = query_analysis['primary_intent']
        keywords = query_analysis['keywords']
        
        suggestions = {
            'query': query,
            'primary_intent': intent,
            'content_type_recommendations': self.get_content_type_recommendations(intent),
            'keyword_targets': keywords[:5],  # Top 5 keywords
            'content_structure': self.get_content_structure(intent),
            'optimization_priorities': self.get_optimization_priorities(query_analysis)
        }
        
        return suggestions
    
    def get_content_type_recommendations(self, intent: str) -> List[str]:
        """Get content type recommendations based on intent"""
        recommendations = {
            'informational': [
                'Comprehensive blog posts',
                'How-to guides',
                'FAQ sections',
                'Educational videos',
                'Infographics'
            ],
            'commercial': [
                'Comparison pages',
                'Product review articles',
                'Buying guides',
                'Feature comparison tables',
                'Expert recommendations'
            ],
            'transactional': [
                'Product pages',
                'Landing pages',
                'Price comparison tools',
                'Customer testimonials',
                'Call-to-action optimization'
            ],
            'navigational': [
                'Clear site navigation',
                'Brand pages',
                'Contact information',
                'About pages',
                'Site search optimization'
            ],
            'local': [
                'Location pages',
                'Local business listings',
                'Map integrations',
                'Local testimonials',
                'Contact information with address'
            ]
        }
        
        return recommendations.get(intent, recommendations['informational'])
    
    def get_content_structure(self, intent: str) -> List[str]:
        """Get content structure recommendations"""
        structures = {
            'informational': [
                'Clear headings and subheadings',
                'Step-by-step instructions',
                'Examples and case studies',
                'Summary or conclusion',
                'Related articles section'
            ],
            'commercial': [
                'Product comparison tables',
                'Pros and cons lists',
                'Feature highlights',
                'User reviews and ratings',
                'Clear pricing information'
            ],
            'transactional': [
                'Product specifications',
                'Clear pricing and availability',
                'Customer reviews',
                'Easy purchase process',
                'Trust signals and guarantees'
            ]
        }
        
        return structures.get(intent, structures['informational'])
    
    def get_optimization_priorities(self, query_analysis: Dict) -> List[str]:
        """Get optimization priorities based on query analysis"""
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

def main():
    st.set_page_config(
        page_title="QForia-Style SEO Query Analyzer",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 QForia-Style SEO Query Analyzer")
    st.markdown("Advanced query analysis tool for understanding search intent and identifying content opportunities")
    
    # Initialize analyzers
    query_analyzer = QueryAnalyzer()
    serp_analyzer = SERPAnalyzer()
    content_optimizer = ContentOptimizer()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
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
        uploaded_file = st.file_uploader(
            "Upload CSV with queries",
            type="csv",
            help="CSV should contain a 'query' column"
        )
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if 'query' in df.columns:
                queries = df['query'].dropna().tolist()
                st.success(f"Loaded {len(queries)} queries")
            else:
                st.error("CSV must contain a 'query' column")
    
    elif analysis_mode == "Manual Entry":
        query_input = st.text_area(
            "Enter queries (one per line)",
            height=200,
            placeholder="how to bake bread\nbest coffee makers 2024\nbuy organic flour online"
        )
        if query_input:
            queries = [q.strip() for q in query_input.split('\n') if q.strip()]
    
    elif analysis_mode == "Paste Text":
        text_input = st.text_area(
            "Paste text to extract queries from",
            height=200,
            placeholder="Paste your content here..."
        )
        if text_input:
            # Simple query extraction from text
            sentences = nltk.sent_tokenize(text_input)
            queries = [s for s in sentences if len(s.split()) <= 10 and '?' in s or any(word in s.lower() for word in ['how', 'what', 'why', 'best', 'top'])]
    
    if queries:
        st.write(f"**Analyzing {len(queries)} queries**")
        
        # Quick preview
        with st.expander("Query Preview"):
            for i, query in enumerate(queries[:10]):
                st.write(f"{i+1}. {query}")
            if len(queries) > 10:
                st.write(f"... and {len(queries) - 10} more")
        
        # Main Analysis
        if st.button("🚀 Start Analysis", type="primary"):
            with st.spinner("Analyzing queries..."):
                
                # Step 1: Individual Query Analysis
                st.header("🎯 Query Intent Analysis")
                
                query_analyses = []
                for query in queries:
                    analysis = query_analyzer.analyze_query_intent(query)
                    query_analyses.append(analysis)
                
                # Create analysis dataframe
                analysis_df = pd.DataFrame(query_analyses)
                
                # Intent Distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    intent_counts = analysis_df['primary_intent'].value_counts()
                    fig = px.pie(
                        values=intent_counts.values,
                        names=intent_counts.index,
                        title="Search Intent Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    length_counts = analysis_df['query_length_category'].value_counts()
                    fig = px.bar(
                        x=length_counts.index,
                        y=length_counts.values,
                        title="Query Length Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Analysis Table
                st.subheader("Detailed Query Analysis")
                display_df = analysis_df[['query', 'primary_intent', 'intent_confidence', 'word_count', 'is_question', 'has_brand']]
                st.dataframe(display_df, use_container_width=True)
                
                # Step 2: Query Clustering
                st.header("🎭 Query Clustering")
                
                clustering_results = query_analyzer.cluster_queries(queries, n_clusters)
                st.session_state.query_clusters = clustering_results
                
                # Cluster Visualization
                if len(queries) > 3:
                    # PCA for visualization
                    pca = PCA(n_components=2)
                    embeddings_2d = pca.fit_transform(clustering_results['embeddings'])
                    
                    fig = px.scatter(
                        x=embeddings_2d[:, 0],
                        y=embeddings_2d[:, 1],
                        color=clustering_results['labels'],
                        hover_data=[queries],
                        title="Query Clusters Visualization"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cluster Details
                st.subheader("Cluster Analysis")
                for cluster_id, summary in clustering_results['summaries'].items():
                    with st.expander(f"Cluster {cluster_id + 1} - {summary['size']} queries"):
                        st.write(f"**Top Keywords:** {', '.join([kw[0] for kw in summary['top_keywords']])}")
                        st.write(f"**Representative Query:** {summary['representative_query']}")
                        st.write("**All Queries in Cluster:**")
                        for query in summary['queries']:
                            st.write(f"• {query}")
                
                # Step 3: Content Gap Analysis
                st.header("🕳️ Content Gap Analysis")
                
                gaps_analysis = query_analyzer.analyze_content_gaps(queries)
                st.session_state.content_gaps = gaps_analysis
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Trending Keywords")
                    trending_kw_df = pd.DataFrame(gaps_analysis['trending_keywords'], columns=['Keyword', 'Frequency'])
                    st.dataframe(trending_kw_df)
                
                with col2:
                    st.subheader("Intent Opportunities")
                    intent_df = pd.DataFrame([
                        {'Intent': intent, 'Percentage': f"{pct:.1f}%", 'Count': gaps_analysis['intent_distribution'][intent]}
                        for intent, pct in gaps_analysis['intent_percentages'].items()
                    ])
                    st.dataframe(intent_df)
                
                # Word Cloud
                if gaps_analysis['trending_keywords']:
                    st.subheader("Keyword Cloud")
                    wordcloud_data = dict(gaps_analysis['trending_keywords'])
                    
                    # Create word cloud
                    fig, ax = plt.subplots(figsize=(10, 5))
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(wordcloud_data)
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                
                # Step 4: SERP Analysis
                st.header("🔍 SERP Feature Analysis")
                
                serp_analyses = []
                for query in queries[:20]:  # Limit to first 20 for performance
                    serp_analysis = serp_analyzer.analyze_serp_features(query)
                    serp_analyses.append(serp_analysis)
                
                st.session_state.serp_analysis = serp_analyses
                
                if serp_analyses:
                    # SERP Features Distribution
                    feature_counts = defaultdict(int)
                    for analysis in serp_analyses:
                        for feature, present in analysis['predicted_features'].items():
                            if present:
                                feature_counts[feature] += 1
                    
                    if feature_counts:
                        fig = px.bar(
                            x=list(feature_counts.values()),
                            y=list(feature_counts.keys()),
                            orientation='h',
                            title="Predicted SERP Features Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Step 5: Content Optimization Recommendations
                st.header("✍️ Content Optimization Recommendations")
                
                # Generate recommendations for top queries from each cluster
                recommendations = []
                for cluster_id, summary in clustering_results['summaries'].items():
                    representative_query = summary['representative_query']
                    query_analysis = next((qa for qa in query_analyses if qa['query'] == representative_query), None)
                    
                    if query_analysis:
                        recommendation = content_optimizer.generate_content_suggestions(query_analysis)
                        recommendation['cluster_id'] = cluster_id
                        recommendation['cluster_size'] = summary['size']
                        recommendations.append(recommendation)
                
                # Display recommendations
                for rec in recommendations:
                    with st.expander(f"Cluster {rec['cluster_id'] + 1}: {rec['query']} ({rec['cluster_size']} queries)"):
                        st.write(f"**Intent:** {rec['primary_intent']}")
                        
                        st.write("**Recommended Content Types:**")
                        for content_type in rec['content_type_recommendations']:
                            st.write(f"• {content_type}")
                        
                        st.write("**Content Structure:**")
                        for structure in rec['content_structure']:
                            st.write(f"• {structure}")
                        
                        st.write("**Optimization Priorities:**")
                        for priority in rec['optimization_priorities'][:5]:
                            st.write(f"• {priority}")
                        
                        st.write(f"**Target Keywords:** {', '.join(rec['keyword_targets'])}")
                
                # Step 6: Export Results
                st.header("💾 Export Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export detailed analysis
                    export_df = analysis_df.copy()
                    export_df['cluster_id'] = clustering_results['labels']
                    
                    csv_buffer = StringIO()
                    export_df.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="📄 Download Detailed Analysis",
                        data=csv_buffer.getvalue(),
                        file_name=f"query_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Export summary report
                    summary_report = {
                        'analysis_date': datetime.now().isoformat(),
                        'total_queries': len(queries),
                        'clusters_found': clustering_results['n_clusters'],
                        'intent_distribution': gaps_analysis['intent_distribution'],
                        'trending_keywords': gaps_analysis['trending_keywords'][:10],
                        'recommendations_count': len(recommendations)
                    }
                    
                    st.download_button(
                        label="📊 Download Summary Report",
                        data=json.dumps(summary_report, indent=2),
                        file_name=f"query_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    
    else:
        st.info("👆 Please enter your queries using one of the methods above to start the analysis.")
    
    # Help Section
    with st.expander("ℹ️ How to Use This Tool"):
        st.markdown("""
        ### QForia-Style Query Analysis
        
        This tool analyzes search queries to understand user intent and identify content opportunities:
        
        1. **Query Intent Analysis** - Categorizes queries by search intent (informational, commercial, etc.)
        2. **Query Clustering** - Groups similar queries to identify content themes
        3. **Content Gap Analysis** - Finds trending keywords and underserved intents
        4. **SERP Feature Prediction** - Predicts which SERP features may appear
        5. **Content Optimization** - Provides specific recommendations for each query cluster
        
        ### Input Methods:
        - **Upload CSV**: Upload a CSV file with a 'query' column
        - **Manual Entry**: Type or paste queries, one per line
        - **Paste Text**: Extract queries from existing content
        
        ### Analysis Features:
        - Semantic query clustering using AI
        - Intent classification and confidence scoring
        - Keyword trend identification
        - Content structure recommendations
        - Export capabilities for further analysis
        """)

if __name__ == "__main__":
    main()



