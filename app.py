import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import List, Dict
from collections import Counter, defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# NLTK safe setup
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    punkt_available = True
except LookupError:
    nltk.download('punkt')
    punkt_available = True
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
    if punkt_available and word_tokenize is not None:
        try:
            return word_tokenize(text)
        except LookupError:
            return text.split()
    else:
        return text.split()

# QueryAnalyzer class
class QueryAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        if not text or pd.isna(text): return []
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = safe_word_tokenize(text)
        return [word for word in tokens if word not in self.stop_words and len(word) >= min_length]

    def determine_type(self, query: str) -> str:
        q = query.lower()
        if any(x in q for x in [" vs ", "versus", "compare", "comparison"]):
            return "comparative"
        if any(x in q for x in ["near", "closest", "nearby"]):
            return "local"
        if any(x in q for x in ["how", "what", "why", "when", "where"]):
            return "question"
        if any(x in q for x in ["alternative", "related", "similar"]):
            return "related"
        if any(x in q for x in ["best", "top"]):
            return "reformulation"
        return "informational"

    def describe_user_intent(self, query: str, qtype: str) -> str:
        if qtype == "comparative":
            return "Understand differences or make a choice between options."
        elif qtype == "related":
            return "Explore related or similar topics or alternatives."
        elif qtype == "local":
            return "Find something close by or location-specific."
        elif qtype == "question":
            return "Get a direct answer or explanation."
        elif qtype == "reformulation":
            return "Find improved or more specific versions of a concept."
        else:
            return "Learn general information or context."

    def analyze_query_intent(self, query: str) -> Dict:
        signals = {
            'informational': ['how', 'what', 'why', 'when', 'where', 'who', 'guide', 'tutorial', 'learn', 'explain'],
            'commercial': ['best', 'top', 'review', 'compare', 'vs', 'versus', 'alternative', 'option'],
            'transactional': ['buy', 'purchase', 'price', 'cost', 'cheap', 'deal', 'discount', 'shop', 'order'],
            'navigational': ['login', 'contact', 'about', 'home', 'official', 'website'],
            'local': ['near', 'location', 'address', 'directions', 'local', 'nearby']
        }

        query_lower = query.lower()
        scores = {intent: sum(signal in query_lower for signal in sig_list) for intent, sig_list in signals.items()}
        total_score = sum(scores.values())
        primary = max(scores, key=scores.get) if total_score > 0 else 'informational'

        query_type = self.determine_type(query)
        user_intent_description = self.describe_user_intent(query, query_type)

        return {
            'query': query,
            'primary_intent': primary,
            'type': query_type,
            'user_intent': user_intent_description,
            'intent_confidence': scores[primary] / max(total_score, 1),
            'intent_scores': scores,
            'word_count': len(query.split()),
            'is_question': any(q in query_lower for q in ['?', 'how', 'what', 'why', 'when', 'where', 'who']),
            'has_brand': self.detect_brand_mentions(query),
            'query_length_category': self.categorize_query_length(len(query.split())),
            'keywords': self.extract_keywords(query)
        }

    def generate_reasoning(self, analysis: Dict) -> str:
        qtype = analysis['type']
        if qtype == "comparative":
            return "This query compares key features or entities to help users decide between them."
        elif qtype == "related":
            return "This query explores closely associated concepts or alternatives to the original idea."
        elif qtype == "local":
            return "This query targets geographically relevant information, suggesting a local search intent."
        elif qtype == "question":
            return "This query seeks a direct explanation or factual answer, often in natural question form."
        elif qtype == "reformulation":
            return "This query rephrases the original idea for clarity, specificity, or targeting variants."
        else:
            return "This is a general informational query aiming to understand or explore a topic."

    def detect_brand_mentions(self, query: str) -> bool:
        brand_indicators = ['amazon', 'google', 'apple', 'microsoft', 'facebook', 'netflix', 'uber', 'airbnb']
        return any(brand in query.lower() for brand in brand_indicators)

    def categorize_query_length(self, word_count: int) -> str:
        if word_count <= 2: return 'Short-tail'
        elif word_count <= 4: return 'Medium-tail'
        else: return 'Long-tail'

    def analyze_content_gaps(self, queries: List[str]) -> Dict:
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

        return {
            'trending_keywords': trending_keywords,
            'intent_distribution': dict(intent_distribution),
            'intent_percentages': intent_percentages,
            'length_distribution': dict(length_distribution),
            'underserved_intents': [intent for intent, pct in intent_percentages.items() if pct < 10],
            'opportunity_keywords': [kw for kw, freq in trending_keywords if freq >= 2]
        }

# ---- Streamlit App ----

st.set_page_config(page_title="Query Intent Analyzer", layout="wide")
st.title("🔎 Query Intent Analyzer")

analyzer = QueryAnalyzer()

# Single Query Analysis
st.header("Single Query Analysis")
query = st.text_input("Enter a query to analyze:", "")

if query:
    analysis = analyzer.analyze_query_intent(query)
    st.subheader("Analysis Result")
    st.json(analysis)
    st.markdown(f"**Reasoning:** {analyzer.generate_reasoning(analysis)}")

# Batch Query Analysis
st.header("Batch Query Analysis")
queries_text = st.text_area("Enter multiple queries (one per line):", "")

if queries_text:
    queries = [q.strip() for q in queries_text.splitlines() if q.strip()]
    if queries:
        gap_analysis = analyzer.analyze_content_gaps(queries)
        st.subheader("Content Gap Analysis")
        st.write(gap_analysis)

        # Intent Distribution Bar Chart
        if gap_analysis['intent_distribution']:
            st.subheader("Intent Distribution")
            intent_df = pd.DataFrame(
                list(gap_analysis['intent_distribution'].items()),
                columns=['Intent', 'Count']
            )
            st.bar_chart(intent_df.set_index('Intent'))

        # Word Cloud Visualization
        if gap_analysis['trending_keywords']:
            st.subheader("Trending Keywords Word Cloud")
            wordcloud = WordCloud(width=800, height=400, background_color='white')
            wc = wordcloud.generate_from_frequencies(dict(gap_analysis['trending_keywords']))
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        # Opportunity Keywords
        if gap_analysis['opportunity_keywords']:
            st.subheader("Opportunity Keywords")
            st.write(", ".join(gap_analysis['opportunity_keywords']))
