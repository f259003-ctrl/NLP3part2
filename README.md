# Simple Contract Compliance Checker

A lightweight Streamlit app that checks contract PDFs for compliance using Google Gemini AI.

## Features

- ✅ PDF text extraction
- ✅ 5 essential compliance rules
- ✅ Google Gemini AI analysis
- ✅ Export results (JSON/CSV)
- ✅ No complex dependencies

## Quick Deploy

[![Deploy](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

1. **Fork this repository**
2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**
3. **Connect your GitHub repository**
4. **Set `GOOGLE_API_KEY` in secrets**
5. **Deploy!**

## Local Development

```bash
git clone <your-repo-url>
cd <repo-name>

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
