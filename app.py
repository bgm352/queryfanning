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

# AI APIs
import google.generativeai as genai
try:
    import openai
    openai_available = True
except ImportError:
    openai_available = False

def list_gemini_models(api_key):
    genai.configure(api_key=api_key)
    return [m.name for m in genai.list_models()]

def list_openai_models():
    return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]

class QueryAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        if not text or pd.isna(text): return []
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = safe_word_tokenize(text)
        return [word for word in tokens if word not in self.stop_words and len(word) >= min_length]

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

        return {
            'query': query,
            'primary_intent': primary,
            'intent_confidence': scores[primary] / max(total_score, 1),
            'intent_scores': scores,
            'word_count': len(query.split()),
            'is_question': any(q in query_lower for q in ['?', 'how', 'what', 'why', 'when', 'where', 'who']),
            'has_brand': self.detect_brand_mentions(query),
            'query_length_category': self.categorize_query_length(len(query.split())),
            'keywords': self.extract_keywords(query)
        }

    def generate_reasoning(self, analysis: Dict) -> str:
        reasoning = (
            f"Intent detected as **{analysis['primary_intent']}** "
            f"(confidence: {analysis['intent_confidence']:.2f}). "
        )
        reasoning += f"Keywords extracted: {', '.join(analysis['keywords']) or 'none'}. "
        reasoning += f"Query length classified as {analysis['query_length_category']}. "
        if analysis['has_brand']:
            reasoning += "Includes a potential brand mention. "
        if analysis['is_question']:
            reasoning += "Appears to be a direct question. "
        return reasoning.strip()

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
    return queries[:target_num]

def openai_generate_queries(api_key, model_name, seed_query, target_num=14):
    if not openai_available:
        raise ImportError("OpenAI library not available. Install with: pip install openai")

    client = openai.OpenAI(api_key=api_key)
    prompt = (
        f"You are an expert SEO strategist. Given the seed query: '{seed_query}', "
        f"generate {target_num} unique, high-quality search queries that cover reformulations, related queries, "
        f"comparisons, entity expansions, and personalized queries. "
        f"Output only the list of queries, one per line."
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert SEO strategist focused on generating comprehensive search query variations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    content = response.choices[0].message.content
    queries = [line.strip('-•* \t') for line in content.split('\n') if line.strip()]
    return queries[:target_num]

def main():
    st.set_page_config(page_title="QForia-style Fan Out Tool", page_icon="🔎", layout="wide")
    st.title("🧠 QForia-style Fan Out Tool")
    st.markdown("Generate a comprehensive set of search queries from a seed query using AI models.")

    st.sidebar.header("🤖 AI Model Configuration")
    st.sidebar.subheader("Choose AI Provider")
    ai_provider = st.sidebar.selectbox("Select AI Provider", options=["Gemini", "OpenAI (ChatGPT)"], index=0)
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"{ai_provider} Configuration")

    if ai_provider == "Gemini":
        api_key = st.sidebar.text_input("Enter your Gemini API key", type="password", key="gemini_key")
        st.sidebar.markdown("👉 [Get a Gemini API key](https://aistudio.google.com/app/apikey)")
        available_models = []
        if api_key:
            try:
                available_models = list_gemini_models(api_key)
                st.sidebar.success("✅ Gemini models loaded.")
            except Exception as e:
                st.sidebar.error(f"❌ Model list error: {e}")
        model_name = st.sidebar.selectbox("Select Gemini Model", available_models) if available_models else None

    else:
        api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password", key="openai_key")
        st.sidebar.markdown("👉 [Get an OpenAI API key](https://platform.openai.com/api-keys)")
        available_models = list_openai_models()
        model_name = st.sidebar.selectbox("Select OpenAI Model", available_models)
        if api_key:
            st.sidebar.success("✅ OpenAI configuration ready.")
        else:
            st.sidebar.info("ℹ️ Enter your OpenAI API key above.")
        if not openai_available:
            st.sidebar.error("❌ OpenAI library not installed. Run: `pip install openai`")

    st.header("Seed Query or Upload")
    col1, col2 = st.columns(2)
    with col1:
        seed_query = st.text_input("Enter a seed query (e.g. 'best electric SUV for mountains')", "")
    with col2:
        uploaded_file = st.file_uploader("Or upload a CSV of queries", type=["csv"])

    target_num = st.slider("Target Number of Queries", min_value=5, max_value=30, value=14)
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
        elif not api_key or not model_name:
            st.warning(f"Please enter your {ai_provider} API key and select a model.")
        else:
            try:
                with st.spinner(f"Generating queries with {ai_provider}..."):
                    if ai_provider == "Gemini":
                        queries = gemini_generate_queries(api_key, model_name, seed_query, target_num)
                    else:
                        queries = openai_generate_queries(api_key, model_name, seed_query, target_num)
                    plan_reasoning = (
                        f"The user query involves '{seed_query}'. To provide a comprehensive overview, "
                        f"{target_num} queries were generated using {ai_provider} {model_name} to cover reformulations, "
                        f"related queries, comparisons, entity expansions, and personalized queries."
                    )
            except Exception as e:
                st.error(f"{ai_provider} API error: {e}")

    if queries:
        st.markdown("### 🧠 Model's Query Generation Plan")
        st.markdown(f"🔹 **AI Provider Used:** {ai_provider}")
        st.markdown(f"🔹 **Model Used:** {model_name}")
        st.markdown(f"🔹 **Target Number of Queries Decided:** {target_num}")
        st.markdown(f"🔹 **Model's Reasoning for This Number:** {plan_reasoning}")
        st.markdown(f"🔹 **Actual Number of Queries Generated:** {len(queries)}")
        st.markdown("---")

        analyzer = QueryAnalyzer()
        rows = []
        for q in queries:
            analysis = analyzer.analyze_query_intent(q)
            ai_reasoning = analyzer.generate_reasoning(analysis)
            rows.append({
                "query": q,
                "type": analysis['primary_intent'],
                "user_intent": analysis['primary_intent'],
                "reasoning": ai_reasoning
            })

        df_queries = pd.DataFrame(rows)
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
