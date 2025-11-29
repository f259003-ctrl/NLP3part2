import streamlit as st
import json
import tempfile
import os
from pypdf import PdfReader
import google.generativeai as genai

# App configuration
st.set_page_config(
    page_title="Contract Compliance Checker",
    page_icon="📋",
    layout="wide"
)

# Simple rules definition
COMPLIANCE_RULES = [
    {
        "id": "confidentiality",
        "name": "Confidentiality Clause",
        "description": "Document contains confidentiality provisions",
        "category": "Confidentiality",
        "severity": "High"
    },
    {
        "id": "term_duration", 
        "name": "Contract Term",
        "description": "Specifies start/end dates or duration",
        "category": "Term",
        "severity": "High"
    },
    {
        "id": "termination",
        "name": "Termination Clause",
        "description": "Includes termination conditions",
        "category": "Termination", 
        "severity": "High"
    },
    {
        "id": "governing_law",
        "name": "Governing Law",
        "description": "Specifies legal jurisdiction",
        "category": "Legal",
        "severity": "Medium"
    },
    {
        "id": "payment_terms",
        "name": "Payment Terms", 
        "description": "Specifies payment amounts and schedule",
        "category": "Financial",
        "severity": "High"
    }
]

def extract_pdf_text(uploaded_file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def analyze_compliance(api_key, rule, contract_text):
    """Analyze compliance for a single rule"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Analyze this contract text for compliance with this rule:
        
        RULE: {rule['name']}
        DESCRIPTION: {rule['description']}
        
        CONTRACT TEXT:
        {contract_text[:10000]}  # Limit text length
        
        Answer with ONLY
