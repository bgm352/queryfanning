# Advanced SEO AI Citations Analyzer

## Overview

This application is a powerful tool for analyzing how different AI language models (such as OpenAI GPT-4, Google Gemini, and Anthropic Claude) respond to various types of search queries. By uploading a CSV file containing queries and their intent types, users can automatically query multiple AI models, collect their responses, and analyze the results for insights such as response quality, common topics, and intent-specific metrics.

## Features

- **CSV Upload:** Supports CSV files with `query` and `type` columns.
- **Multi-LLM Queries:** Automatically queries OpenAI, Google Gemini, and Anthropic Claude (API keys required).
- **Response Analysis:** Analyzes responses for quality, length, and success rate.
- **Type-Specific Insights:** Breaks down analysis by query type (e.g., informational, transactional).
- **Visualizations:** Displays query type distribution and analysis results.
- **Configurable:** Allows users to set request delay, passage length, and similarity thresholds.

## Requirements

- **Python 3.8+**
- **Required Libraries:**  
  - `streamlit`
  - `pandas`
  - `numpy`
  - `sentence-transformers`
  - `nltk`
  - `scikit-learn`
  - `plotly`
  - `openai`
  - `google.generativeai`
  - `requests`

  *Note: Optional imports are handled gracefully if some libraries are missing.*

## Installation

1. **Clone the repository:**
