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
    if punkt_available and word_tokenize is not None:
        try:
            return word_tokenize(text)
        except LookupError:
            return text.split()
    else:
        return text.split()

# Gemini API
import google.generativeai as genai

def list_gemini_models(api_key):
    genai.configure(api_key=api_key)
    return [m.name for m in genai.list_models()]

class QueryAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        if not text or pd.isna(text): return []
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = safe_word_tokenize(text)
        return [word for word in tokens if word not in self.stop_words and len(word) >= min_length]
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
        gaps = {
            'trending_keywords': trending_keywords,
            'intent_distribution': dict(intent_distribution),
            'intent_percentages': intent_percentages,
            'length_distribution': dict(length_distribution),
            'underserved_intents': [intent for intent, pct in intent_percentages.items() if pct < 10],
            'opportunity_keywords': [kw for kw, freq in trending_keywords if freq >= 2]
        }
        return gaps

def gemini_generate_queries(api_key, model_name, seed_query, target_num=14):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    prompt = (
        f"You are an expert SEO strategist. Given the seed query: '{seed_query}', "
        f"generate {target_num} unique, high-quality search queries that cover reformulations, related queries, "
        f"comparisons, entity expansions, and personalized queries. "
        f"Output only the list of queries, one per line."
    )
    response = model.generate_content(prompt)
    queries = [line.strip('-•* \t') for line in response.text.split('\n') if line.strip()]
    queries = [q for q in queries if len(q) > 0]
    return queries[:target_num]

def main():
    st.set_page_config(
        page_title="QForia-style Fan Out Tool",
        page_icon="🔎",
        layout="wide"
    )
    st.title("🧠 QForia-style Fan Out Tool")
    st.markdown("Generate a comprehensive set of search queries from a seed query, just like QForia.")

    # Sidebar for API key and model
    st.sidebar.header("Gemini API Key")
    gemini_api_key = st.sidebar.text_input("Enter your Gemini API key", type="password")
    st.sidebar.markdown("[Get a Gemini API key](https://aistudio.google.com/app/apikey)")

    # List available models
    available_models = []
    if gemini_api_key:
        try:
            available_models = list_gemini_models(gemini_api_key)
            st.sidebar.success("Models loaded.")
        except Exception as e:
            st.sidebar.error(f"Model list error: {e}")

    model_name = None
    if available_models:
        model_name = st.sidebar.selectbox("Select Gemini Model", available_models)
    else:
        st.sidebar.info("Enter your API key to see available models.")

    # Input section
    st.header("Seed Query or Upload")
    col1, col2 = st.columns(2)
    with col1:
        seed_query = st.text_input("Enter a seed query (e.g. 'best electric SUV for mountains')", "")
    with col2:
        uploaded_file = st.file_uploader("Or upload a CSV of queries", type=["csv"])

    # Target number of queries
    target_num = st.slider("Target Number of Queries", min_value=5, max_value=30, value=14)

    # Fan Out button
    fan_out = st.button("🚀 Fan Out")

    queries = []
    plan_reasoning = ""
    if fan_out:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            if 'query' in df.columns:
                queries = df['query'].dropna().astype(str).tolist()
                plan_reasoning = "CSV mode: using uploaded queries for analysis."
            else:
                st.error("CSV must have a 'query' column.")
        elif not seed_query:
            st.warning("Please enter a seed query or upload a CSV.")
        elif not gemini_api_key or not model_name:
            st.warning("Please enter your Gemini API key and select a model.")
        else:
            try:
                with st.spinner("Generating queries with Gemini..."):
                    queries = gemini_generate_queries(gemini_api_key, model_name, seed_query, target_num)
                    plan_reasoning = (
                        f"The user query involves '{seed_query}'. To provide a comprehensive overview, "
                        f"{target_num} queries were generated to cover reformulations, related queries, "
                        f"comparisons, entity expansions, and personalized queries."
                    )
            except Exception as e:
                st.error(f"Gemini API error: {e}")
                queries = []

    if queries:
        st.markdown("### 🧠 Model's Query Generation Plan")
        st.markdown(f"🔹 **Target Number of Queries Decided by Model:** {target_num}")
        st.markdown(f"🔹 **Model's Reasoning for This Number:** {plan_reasoning}")
        st.markdown(f"🔹 **Actual Number of Queries Generated:** {len(queries)}")
        st.markdown("---")

        # Build DataFrame with extra columns
        analyzer = QueryAnalyzer()
        rows = []
        for q in queries:
            analysis = analyzer.analyze_query_intent(q)
            row = {
                "query": q,
                "type": analysis['primary_intent'],
                "user_inten": analysis['primary_intent'],
                "reasoning": f"Detected intent: {analysis['primary_intent']}, confidence: {analysis['intent_confidence']:.2f}"
            }
            rows.append(row)
        df_queries = pd.DataFrame(rows, columns=["query", "type", "user_inten", "reasoning"])

        st.markdown("#### Generated Queries")
        st.dataframe(df_queries, use_container_width=True)
        csv = df_queries.to_csv(index=False).encode('utf-8')
        st.download_button("Download as CSV", csv, "queries_fanned_out.csv", "text/csv")

        content_gaps = analyzer.analyze_content_gaps(queries)

        st.markdown("#### Trending Keywords")
        st.write([kw for kw, _ in content_gaps['trending_keywords']])

        st.markdown("#### Keyword Word Cloud")
        word_freq = dict(content_gaps['trending_keywords'])
        if word_freq:
            wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        st.markdown("#### Intent Distribution")
        st.bar_chart(pd.DataFrame.from_dict(content_gaps['intent_distribution'], orient='index', columns=['Count']))

        st.markdown("#### Underserved Intents")
        st.write(content_gaps['underserved_intents'])

if __name__ == "__main__":
    main()
