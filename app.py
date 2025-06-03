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

# --- Healthcare/Pharma Enhancements Start ---

# 1. Regulatory Compliance Mode
def is_compliant(query):
    # Placeholder: real implementation would use a compliance API or rules
    non_compliant_terms = ['guaranteed cure', 'miracle', 'confidential', 'private info']
    return not any(term in query.lower() for term in non_compliant_terms)

def compliance_audit(query):
    if is_compliant(query):
        return "Compliant"
    else:
        return "⚠️ Non-compliant: Check for regulatory risks"

# 2. Medical Entity Recognition and Expansion
def medical_entity_recognition(query):
    # Placeholder: use a real medical NER model or API in production
    drugs = ['aspirin', 'metformin', 'ibuprofen']
    diseases = ['diabetes', 'hypertension', 'covid-19']
    found = []
    for term in drugs + diseases:
        if term in query.lower():
            found.append(term)
    return found

def expand_medical_entities(entities):
    # Placeholder: expand with synonyms/ICD-10 codes
    mapping = {
        'aspirin': ['acetylsalicylic acid', 'ASA'],
        'diabetes': ['type 2 diabetes', 'T2D', 'E11']
    }
    expanded = []
    for ent in entities:
        expanded.extend(mapping.get(ent, []))
    return expanded

# 3. Persona-Based Query Simulation
def persona_simulation(query, persona):
    persona_prompts = {
        'Patient': f"How would a patient ask: {query}?",
        'Caregiver': f"How would a caregiver ask: {query}?",
        'HCP': f"How would a healthcare professional ask: {query}?",
        'Payer': f"How would an insurance payer ask: {query}?"
    }
    return persona_prompts.get(persona, query)

# 4. Localized Healthcare SEO Insights
def add_localization(query, location):
    if location:
        return f"{query} in {location}"
    return query

# 5. Structured Data and Schema Integration
def suggest_schema(query):
    # Suggest schema types based on query content
    if any(x in query.lower() for x in ['doctor', 'physician', 'clinic']):
        return "Physician, MedicalOrganization"
    if any(x in query.lower() for x in ['drug', 'treatment']):
        return "Drug, MedicalWebPage"
    return "MedicalWebPage"

# 6. Content Gap and Authority Analysis
def authority_sites():
    return ['nih.gov', 'mayoclinic.org', 'webmd.com']

# 7. Patient Education and Accessibility Filters
def is_accessible(query):
    # Placeholder: check for jargon, reading level, etc.
    long_words = [w for w in query.split() if len(w) > 14]
    return len(long_words) == 0

# 8. Real-Time Reputation Monitoring Integration
def monitor_reputation(query):
    # Placeholder: in production, connect to review/sentiment APIs
    if 'side effects' in query.lower():
        return "Trending concern: Side effects"
    return ""

# 9. Clinical Trial and Drug Information Expansion
def expand_for_clinical_trials(query):
    if 'trial' in query.lower():
        return [query + " inclusion criteria", query + " locations", query + " enrollment"]
    return []

# 10. Compliance-Friendly Content Export and Audit
def export_with_compliance_audit(df):
    df['compliance_status'] = df['query'].apply(compliance_audit)
    return df

# --- Healthcare/Pharma Enhancements End ---

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

# --- Reasoning function for natural language explanations ---
def intent_reasoning(intent):
    explanations = {
        "informational": "This addresses informational needs by providing general knowledge or answers to questions.",
        "commercial": "This compares commercial options, helping users evaluate alternatives.",
        "transactional": "This targets transactional intent, guiding users toward making a purchase or taking action.",
        "navigational": "This directs users to a specific website or brand destination.",
        "local": "This focuses on local intent, helping users find nearby locations or services."
    }
    return explanations.get(intent, f"This covers {intent} intent.")

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
    st.sidebar.subheader("Choose AI Provider")
    ai_provider = st.sidebar.selectbox(
        "Select AI Provider",
        options=["Gemini", "OpenAI (ChatGPT)"],
        help="Choose between Google Gemini or OpenAI's ChatGPT models",
        index=0
    )
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
        model_name = None
        if available_models:
            model_name = st.sidebar.selectbox("Select Gemini Model", available_models)
        else:
            st.sidebar.info("ℹ️ Enter your API key to see available models.")
    elif ai_provider == "OpenAI (ChatGPT)":
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

    # Healthcare/Pharma settings
    st.sidebar.markdown("---")
    st.sidebar.header("⚕️ Healthcare/Pharma Enhancements")
    enable_compliance = st.sidebar.checkbox("Enable Regulatory Compliance Mode", value=True)
    enable_medical_ner = st.sidebar.checkbox("Medical Entity Recognition & Expansion", value=True)
    persona = st.sidebar.selectbox("Persona Simulation", ["None", "Patient", "Caregiver", "HCP", "Payer"])
    location = st.sidebar.text_input("Localize queries to (city/state)", "")
    enable_schema = st.sidebar.checkbox("Suggest Schema Markup", value=True)
    enable_accessibility = st.sidebar.checkbox("Patient Education/Accessibility Filter", value=True)
    enable_reputation = st.sidebar.checkbox("Monitor Reputation/Concerns", value=True)
    enable_clinical_trials = st.sidebar.checkbox("Expand for Clinical Trials", value=True)
    enable_compliance_export = st.sidebar.checkbox("Compliance-Friendly Export", value=True)

    # Input section
    st.header("Seed Query or Upload")
    col1, col2 = st.columns(2)
    with col1:
        seed_query = st.text_input("Enter a seed query (e.g. 'best electric SUV for mountains')", "")
    with col2:
        uploaded_file = st.file_uploader("Or upload a CSV of queries", type=["csv"])

    # Target number of queries
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

        analyzer = QueryAnalyzer()
        rows = []
        for q in queries:
            original_q = q
            # Persona simulation
            if persona and persona != "None":
                q = persona_simulation(q, persona)
            # Localization
            if location:
                q = add_localization(q, location)
            # Clinical trial expansion
            expanded = []
            if enable_clinical_trials:
                expanded = expand_for_clinical_trials(q)
            # Medical entity recognition & expansion
            entities, expanded_entities = [], []
            if enable_medical_ner:
                entities = medical_entity_recognition(q)
                expanded_entities = expand_medical_entities(entities)
            # Accessibility
            accessible = True
            if enable_accessibility:
                accessible = is_accessible(q)
            # Compliance
            compliance_status = ""
            if enable_compliance:
                compliance_status = compliance_audit(q)
            # Schema
            schema_type = ""
            if enable_schema:
                schema_type = suggest_schema(q)
            # Reputation
            reputation = ""
            if enable_reputation:
                reputation = monitor_reputation(q)
            # Analysis
            analysis = analyzer.analyze_query_intent(q)
            reason = intent_reasoning(analysis['primary_intent'])
            row = {
                "query": q,
                "type": analysis['primary_intent'],
                "user_inten": analysis['primary_intent'],
                "reasoning": reason,
                "entities": ", ".join(entities),
                "entity_expansion": ", ".join(expanded_entities),
                "schema": schema_type,
                "accessible": "Yes" if accessible else "No",
                "compliance_status": compliance_status,
                "reputation_signal": reputation,
                "clinical_trial_expansion": ", ".join(expanded) if expanded else ""
            }
            rows.append(row)

        df_queries = pd.DataFrame(rows, columns=[
            "query", "type", "user_inten", "reasoning", "entities", "entity_expansion",
            "schema", "accessible", "compliance_status", "reputation_signal", "clinical_trial_expansion"
        ])

        # Compliance-friendly export
        if enable_compliance_export:
            df_queries = export_with_compliance_audit(df_queries)

        st.markdown("#### Generated Queries (with Healthcare/Pharma Enhancements)")
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
