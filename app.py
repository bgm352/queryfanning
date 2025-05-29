import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import re
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

# --- Configuration ---
st.set_page_config(
    page_title="Relevance Engineering Tool",
    page_icon="🤖",
    layout="wide"
)

# --- Safe import for BeautifulSoup ---
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    st.warning("beautifulsoup4 not installed. Passage extraction from URLs will be disabled.")

# --- Load model ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- Extract passages from URL ---
def extract_passages_from_url(url: str, max_length: int = 200) -> List[str]:
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

# --- Vectorize and score passages ---
def vectorize_and_score(query: str, passages: List[str], model):
    if not passages:
        return []
    query_emb = model.encode([query])
    passage_embs = model.encode(passages)
    scores = cosine_similarity(query_emb, passage_embs)[0]
    return list(zip(passages, scores))

# --- Suggest improved passage ---
def suggest_improved_passage(query: str, competitor_passage: str, your_passage: str, openai_key: str, demographics: str = ""):
    if not openai_key:
        return "OpenAI API key required for optimization."
    try:
        import openai
        prompt = (
            f"Query: {query}\n\n"
            f"Target patient demographics: {demographics}\n\n"
            f"Competitor's top passage:\n{competitor_passage}\n\n"
            f"Your current passage:\n{your_passage}\n\n"
            "Rewrite your passage to be more relevant to the query and competitive with the top passage, "
            "while maintaining your unique voice and accuracy. "
            "Tailor the language and content to the specified patient demographics."
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
    except ImportError:
        return "openai library not installed."
    except Exception as e:
        return f"Error: {str(e)}"

# --- Workflow Visualization ---
def show_workflow():
    st.markdown("""
    ### 🚀 Relevance Engineering Workflow

    **1. Upload & Analyze**  
    Upload a CSV of your queries and URLs. The app analyzes your and competitors' content.

    **2. Extract & Score**  
    The app extracts passages from your and competitors' pages, then scores them for relevance to your query.

    **3. Optimize**  
    Enter patient demographics and use AI to rewrite your content for better relevance and engagement.

    **4. Deploy**  
    Copy the improved content and update your website for higher rankings and better patient experience.

    ---
    """)

# --- Main App ---
def main():
    st.title("Relevance Engineering Tool")
    show_workflow()

    query = st.text_input("Enter your target query:")
    openai_key = st.text_input("OpenAI API Key (for optimization):", type="password", key="openai_key")
    demographics = st.text_input("Patient demographics (e.g., 'adults with asthma, uninsured patients'):", key="demographics")

    uploaded_file = st.file_uploader(
        "Upload CSV with ranking URLs (columns: query, your_url, competitor_urls)",
        type="csv"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())

        model = load_model()

        for idx, row in df.iterrows():
            query = row['query'] if 'query' in df.columns else query
            your_url = row['your_url']
            competitor_urls = row['competitor_urls'].split('|') if 'competitor_urls' in df.columns else []

            st.subheader(f"Query: {query}")
            st.write(f"Your URL: {your_url}")
            st.write("Competitor URLs:")
            st.write(competitor_urls)

            # Extract your passages
            your_passages = extract_passages_from_url(your_url)
            st.write("Your passages:")
            st.write(your_passages)

            # Extract competitor passages
            comp_passages = []
            for url in competitor_urls:
                passages = extract_passages_from_url(url)
                comp_passages.extend(passages)
            st.write("Competitor passages (first 10):")
            st.write(comp_passages[:10])

            # Score your passages
            your_scores = vectorize_and_score(query, your_passages, model) if your_passages else []
            comp_scores = vectorize_and_score(query, comp_passages, model) if comp_passages else []

            your_top = sorted(your_scores, key=lambda x: -x[1])[0] if your_scores else ("", 0)
            comp_top = sorted(comp_scores, key=lambda x: -x[1])[0] if comp_scores else ("", 0)

            st.markdown(f"**Your top passage (score {your_top[1]:.2f}):**\n\n{your_top[0]}")
            st.markdown(f"**Competitor's top passage (score {comp_top[1]:.2f}):**\n\n{comp_top[0]}")

            if st.button(f"Suggest Improved Passage for Query: {query}", key=f"improve_{idx}"):
                improved = suggest_improved_passage(query, comp_top[0], your_top[0], openai_key, demographics)
                st.markdown("**Improved Passage:**")
                st.write(improved)

if __name__ == "__main__":
    main()




