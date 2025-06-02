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
    """Return commonly available OpenAI models"""
    return [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ]

class QueryAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        if not text or pd.isna(text): return []
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = safe_word_tokenize(text)
        return [word for word in tokens if word not in self.stop_words and len(word) >= min_length]
    
    def classify_query_type(self, query: str, seed_query: str = "") -> str:
        """Classify query type based on relationship to seed query and content patterns"""
        query_lower = query.lower()
        seed_lower = seed_query.lower()
        
        # Comparative queries
        comparison_signals = ['vs', 'versus', 'compare', 'compared to', 'better than', 'difference between']
        if any(signal in query_lower for signal in comparison_signals):
            return 'comparative query'
        
        # Entity expansion (specific brands/models mentioned)
        entity_signals = ['tesla', 'ford', 'rivian', 'audi', 'bmw', 'subaru', 'toyota', 'honda', 'volkswagen', 'mercedes']
        if any(entity in query_lower for entity in entity_signals):
            # If it's comparing entities, already caught above
            return 'entity expansion'
        
        # Reformulation (similar core intent but different wording)
        if seed_query:
            seed_keywords = set(self.extract_keywords(seed_query))
            query_keywords = set(self.extract_keywords(query))
            overlap_ratio = len(seed_keywords.intersection(query_keywords)) / max(len(seed_keywords), 1)
            
            if overlap_ratio > 0.6:  # High keyword overlap suggests reformulation
                return 'reformulation'
        
        # Implicit queries (addressing unstated concerns)
        implicit_signals = ['range', 'charging', 'battery', 'anxiety', 'towing', 'capacity', 'safety', 'features']
        if any(signal in query_lower for signal in implicit_signals):
            return 'implicit query'
        
        # Personalized queries (specific user needs/constraints)
        personal_signals = ['affordable', 'budget', 'cheap', 'best for me', 'family', 'reviews', 'experiences', 'my']
        if any(signal in query_lower for signal in personal_signals):
            return 'personalized query'
        
        # Default to related query
        return 'related query'
    
    def generate_user_intent(self, query: str, query_type: str) -> str:
        """Generate concise user intent description"""
        query_lower = query.lower()
        
        if query_type == 'comparative query':
            return f"Compare options to determine the best choice for specific needs."
        elif query_type == 'entity expansion':
            return f"Get detailed information about a specific product or brand."
        elif query_type == 'reformulation':
            return f"Find the same information using different search terms."
        elif query_type == 'implicit query':
            return f"Address unstated concerns or requirements related to the main topic."
        elif query_type == 'personalized query':
            return f"Find solutions tailored to specific personal circumstances or preferences."
        else:  # related query
            return f"Explore related aspects or variations of the main topic."
    
    def generate_detailed_reasoning(self, query: str, query_type: str, seed_query: str = "") -> str:
        """Generate detailed reasoning for why this query was created"""
        query_lower = query.lower()
        
        reasoning_templates = {
            'reformulation': f"This reformulates the original query using different terminology while maintaining the core search intent.",
            'related query': f"This explores a related aspect that users commonly search for in connection with the main topic.",
            'comparative query': f"This directly compares specific options to help users make informed decisions between alternatives.",
            'entity expansion': f"This focuses on a specific brand/model to provide detailed information that users often seek.",
            'implicit query': f"This addresses common unstated concerns or requirements that users have but don't explicitly mention.",
            'personalized query': f"This tailors the search to specific user circumstances, constraints, or preferences for more targeted results."
        }
        
        base_reasoning = reasoning_templates.get(query_type, "This query explores relevant aspects of the main topic.")
        
        # Add specific details based on query content
        specific_details = []
        
        if 'winter' in query_lower or 'snow' in query_lower:
            specific_details.append("considering seasonal driving conditions")
        if 'charging' in query_lower:
            specific_details.append("addressing infrastructure concerns")
        if 'range' in query_lower:
            specific_details.append("focusing on battery performance")
        if 'safety' in query_lower:
            specific_details.append("emphasizing safety features")
        if 'towing' in query_lower:
            specific_details.append("considering utility requirements")
        if 'affordable' in query_lower or 'budget' in query_lower:
            specific_details.append("incorporating budget constraints")
        
        if specific_details:
            base_reasoning += f" It specifically focuses on {', '.join(specific_details)}."
        
        return base_reasoning
    
    def analyze_query_intent(self, query: str, seed_query: str = "") -> Dict:
        query_lower = query.lower()
        
        # Get query type using new classification
        query_type = self.classify_query_type(query, seed_query)
        user_intent = self.generate_user_intent(query, query_type)
        detailed_reasoning = self.generate_detailed_reasoning(query, query_type, seed_query)
        
        # Keep original intent scoring for analysis
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
            'type': query_type,
            'user_intent': user_intent,
            'reasoning': detailed_reasoning,
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
    queries = [q for q in queries if len(q) > 0]
    return queries[:target_num]

def main():
    st.set_page_config(
        page_title="QForia-style Fan Out Tool",
        page_icon="🔎",
        layout="wide"
    )
    
    st.title("🧠 QForia-style Fan Out Tool")
    st.markdown("Generate a comprehensive set of search queries from a seed query using AI models.")

    # Sidebar for API configuration
    st.sidebar.header("🤖 AI Model Configuration")
    
    # Model selection - make it more prominent
    st.sidebar.subheader("Choose AI Provider")
    ai_provider = st.sidebar.selectbox(
        "Select AI Provider",
        options=["Gemini", "OpenAI (ChatGPT)"],
        help="Choose between Google Gemini or OpenAI's ChatGPT models",
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # API key and model selection based on provider
    st.sidebar.subheader(f"{ai_provider} Configuration")
    
    if ai_provider == "Gemini":
        api_key = st.sidebar.text_input("Enter your Gemini API key", type="password", key="gemini_key")
        st.sidebar.markdown("👉 [Get a Gemini API key](https://aistudio.google.com/app/apikey)")
        
        # List available models
        available_models = []
        if api_key:
            try:
                available_models = list_gemini_models(api_key)
                st.sidebar.success("✅ Gemini models loaded.")
            except Exception as e:
                st.sidebar.error(f"❌ Model list error: {e}")
        
        model_name = None
        if available_models:
            model_name = st.sidebar.selectbox("Select Gemini Model", available_models)
        else:
            st.sidebar.info("ℹ️ Enter your API key to see available models.")
    
    elif ai_provider == "OpenAI (ChatGPT)":  # Changed this condition
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
        elif not api_key or not model_name:
            st.warning(f"Please enter your {ai_provider} API key and select a model.")
        else:
            try:
                with st.spinner(f"Generating queries with {ai_provider}..."):
                    if ai_provider == "Gemini":
                        queries = gemini_generate_queries(api_key, model_name, seed_query, target_num)
                    else:  # OpenAI (ChatGPT)
                        if not openai_available:
                            st.error("❌ OpenAI library not available. Install with: `pip install openai`")
                        else:
                            queries = openai_generate_queries(api_key, model_name, seed_query, target_num)
                    
                    plan_reasoning = (
                        f"The user query involves '{seed_query}'. To provide a comprehensive overview, "
                        f"{target_num} queries were generated using {ai_provider} {model_name} to cover reformulations, "
                        f"related queries, comparisons, entity expansions, and personalized queries."
                    )
            except Exception as e:
                st.error(f"{ai_provider} API error: {e}")
                queries = []

    if queries:
        st.markdown("### 🧠 Model's Query Generation Plan")
        st.markdown(f"🔹 **AI Provider Used:** {ai_provider}")
        st.markdown(f"🔹 **Model Used:** {model_name}")
        st.markdown(f"🔹 **Target Number of Queries Decided:** {target_num}")
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

        # Analysis section
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
