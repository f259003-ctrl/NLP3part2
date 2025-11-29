# app.py
import streamlit as st
import json
import pandas as pd
import tempfile
import os
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai

# Load rules definition
RULES_DEFINITION = {
    "confidentiality_clause": {
        "name": "Confidentiality Clause Presence",
        "description": "Document must contain a confidentiality clause protecting sensitive information",
        "category": "Confidentiality",
        "severity": "High"
    },
    "term_duration": {
        "name": "Contract Term Duration",
        "description": "Contract must specify a clear start and end date or duration",
        "category": "Term",
        "severity": "High"
    },
    "termination_clause": {
        "name": "Termination Clause",
        "description": "Document must include termination conditions and notice period",
        "category": "Termination",
        "severity": "High"
    },
    "governing_law": {
        "name": "Governing Law Clause",
        "description": "Contract must specify the governing law jurisdiction",
        "category": "Legal",
        "severity": "Medium"
    },
    "indemnification": {
        "name": "Indemnification Clause",
        "description": "Must include indemnification provisions for liability protection",
        "category": "Liability",
        "severity": "High"
    },
    "ip_ownership": {
        "name": "Intellectual Property Ownership",
        "description": "Clearly defines ownership of intellectual property created during contract",
        "category": "Intellectual Property",
        "severity": "High"
    },
    "payment_terms": {
        "name": "Payment Terms",
        "description": "Specifies payment amounts, schedules, and methods",
        "category": "Financial",
        "severity": "High"
    },
    "warranties": {
        "name": "Warranties and Representations",
        "description": "Includes appropriate warranties and representations",
        "category": "Liability",
        "severity": "Medium"
    },
    "limitation_liability": {
        "name": "Limitation of Liability",
        "description": "Includes reasonable limitation of liability clauses",
        "category": "Liability",
        "severity": "Medium"
    },
    "dispute_resolution": {
        "name": "Dispute Resolution Mechanism",
        "description": "Specifies dispute resolution process (arbitration, mediation, litigation)",
        "category": "Legal",
        "severity": "Medium"
    },
    "assignment_clause": {
        "name": "Assignment Clause",
        "description": "Addresses whether contract can be assigned to third parties",
        "category": "Transfer",
        "severity": "Low"
    },
    "force_majeure": {
        "name": "Force Majeure Clause",
        "description": "Includes force majeure provisions for unforeseen circumstances",
        "category": "Risk",
        "severity": "Medium"
    },
    "notices": {
        "name": "Notices Provision",
        "description": "Specifies how formal notices should be delivered",
        "category": "Administrative",
        "severity": "Low"
    },
    "entire_agreement": {
        "name": "Entire Agreement Clause",
        "description": "States that the document represents the entire agreement",
        "category": "Legal",
        "severity": "Medium"
    },
    "severability": {
        "name": "Severability Clause",
        "description": "Includes severability clause for invalid provisions",
        "category": "Legal",
        "severity": "Low"
    },
    "amendment_process": {
        "name": "Amendment Process",
        "description": "Specifies how the contract can be amended",
        "category": "Administrative",
        "severity": "Low"
    }
}

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def extract_text_from_pdf(self, pdf_file) -> List[Document]:
        """Extract text from PDF and convert to LangChain Documents"""
        documents = []
        reader = PdfReader(pdf_file)
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": pdf_file.name,
                        "page": page_num + 1
                    }
                ))
        
        return documents
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents"""
        chunks = self.text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        return vector_store
    
    def process_pdf(self, pdf_file) -> FAISS:
        """Complete PDF processing pipeline"""
        documents = self.extract_text_from_pdf(pdf_file)
        vector_store = self.create_vector_store(documents)
        return vector_store

class ComplianceChecker:
    def __init__(self, vector_store: FAISS):
        self.vector_store = vector_store
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel('gemini-pro')
    
    def retrieve_relevant_sections(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant sections from the document"""
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def check_rule_compliance(self, rule_id: str, rule_definition: Dict) -> Dict[str, Any]:
        """Check compliance for a specific rule"""
        query = f"{rule_definition['description']} What specific clauses or language addresses this requirement?"
        
        # Retrieve relevant sections
        relevant_sections = self.retrieve_relevant_sections(query)
        context = "\n\n".join(relevant_sections)
        
        # Create prompt for Gemini
        prompt = f"""
        Analyze the following contract sections and determine if they satisfy this compliance rule:
        
        RULE: {rule_definition['name']}
        DESCRIPTION: {rule_definition['description']}
        CATEGORY: {rule_definition['category']}
        SEVERITY: {rule_definition['severity']}
        
        CONTRACT SECTIONS:
        {context}
        
        Please provide:
        1. Compliance Status: [Compliant/Non-Compliant/Partially Compliant]
        2. Evidence: Specific text from the contract that supports your assessment
        3. Confidence Level: [High/Medium/Low]
        4. Remediation Steps: If non-compliant, what should be added or modified?
        
        Format your response as JSON:
        {{
            "rule_id": "{rule_id}",
            "rule_name": "{rule_definition['name']}",
            "compliance_status": "status",
            "evidence": "text evidence",
            "confidence": "confidence level",
            "remediation": "remediation steps",
            "category": "{rule_definition['category']}",
            "severity": "{rule_definition['severity']}"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result['relevant_sections'] = relevant_sections
                return result
            else:
                return self._create_error_response(rule_id, rule_definition, "Failed to parse response")
        except Exception as e:
            return self._create_error_response(rule_id, rule_definition, str(e))
    
    def _create_error_response(self, rule_id: str, rule_definition: Dict, error: str) -> Dict[str, Any]:
        """Create error response for failed rule checks"""
        return {
            "rule_id": rule_id,
            "rule_name": rule_definition['name'],
            "compliance_status": "Error",
            "evidence": f"Error during analysis: {error}",
            "confidence": "Low",
            "remediation": "Unable to provide remediation due to analysis error",
            "category": rule_definition['category'],
            "severity": rule_definition['severity'],
            "relevant_sections": []
        }

def main():
    st.set_page_config(
        page_title="Contract Compliance Checker",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 Contract Compliance Checking Agent")
    st.markdown("Upload a contract PDF to check compliance against 16 standard rules")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Contract PDF",
        type="pdf",
        help="Upload the contract PDF you want to analyze"
    )
    
    # API key input
    api_key = st.sidebar.text_input(
        "Google API Key",
        type="password",
        help="Enter your Google Gemini API key"
    )
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Compliance Rules")
        
        # Display rules by category
        categories = set(rule['category'] for rule in RULES_DEFINITION.values())
        
        for category in categories:
            with st.expander(f"{category} Rules"):
                for rule_id, rule in RULES_DEFINITION.items():
                    if rule['category'] == category:
                        st.write(f"**{rule['name']}**")
                        st.caption(f"Severity: {rule['severity']}")
                        st.write(rule['description'])
                        st.divider()
    
    with col2:
        st.subheader("Document Analysis")
        
        if uploaded_file and api_key:
            if st.button("Run Compliance Check", type="primary"):
                with st.spinner("Processing document and analyzing compliance..."):
                    try:
                        # Process PDF
                        processor = PDFProcessor()
                        vector_store = processor.process_pdf(uploaded_file)
                        st.session_state.vector_store = vector_store
                        
                        # Initialize checker
                        checker = ComplianceChecker(vector_store)
                        
                        # Run compliance checks
                        progress_bar = st.progress(0)
                        results = {}
                        
                        for i, (rule_id, rule_def) in enumerate(RULES_DEFINITION.items()):
                            result = checker.check_rule_compliance(rule_id, rule_def)
                            results[rule_id] = result
                            progress_bar.progress((i + 1) / len(RULES_DEFINITION))
                        
                        st.session_state.results = results
                        st.success("Compliance analysis completed!")
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
        
        # Display results if available
        if st.session_state.results:
            display_results(st.session_state.results)

def display_results(results: Dict[str, Any]):
    st.subheader("Compliance Results")
    
    # Summary statistics
    status_counts = {}
    for result in results.values():
        status = result['compliance_status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Create summary columns
    col1, col2, col3, col4 = st.columns(4)
    
    total_rules = len(results)
    compliant_count = status_counts.get('Compliant', 0)
    non_compliant_count = status_counts.get('Non-Compliant', 0)
    partial_count = status_counts.get('Partially Compliant', 0)
    
    with col1:
        st.metric("Total Rules", total_rules)
    with col2:
        st.metric("Compliant", compliant_count)
    with col3:
        st.metric("Non-Compliant", non_compliant_count)
    with col4:
        st.metric("Partial Compliance", partial_count)
    
    # Detailed results table
    st.subheader("Detailed Compliance Analysis")
    
    rows = []
    for rule_id, result in results.items():
        rows.append({
            'Rule Name': result['rule_name'],
            'Category': result['category'],
            'Severity': result['severity'],
            'Status': result['compliance_status'],
            'Confidence': result['confidence'],
            'Evidence': result['evidence'][:150] + '...' if len(result['evidence']) > 150 else result['evidence']
        })
    
    df = pd.DataFrame(rows)
    
    # Color code the status column
    def color_status(val):
        if val == 'Compliant':
            return 'color: green'
        elif val == 'Non-Compliant':
            return 'color: red'
        elif val == 'Partially Compliant':
            return 'color: orange'
        else:
            return ''
    
    styled_df = df.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Download results
    st.subheader("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON download
        json_results = json.dumps(results, indent=2)
        st.download_button(
            label="Download JSON Results",
            data=json_results,
            file_name="compliance_results.json",
            mime="application/json"
        )
    
    with col2:
        # CSV download
        csv_results = df.to_csv(index=False)
        st.download_button(
            label="Download CSV Report",
            data=csv_results,
            file_name="compliance_report.csv",
            mime="text/csv"
        )
    
    # Show remediation steps for non-compliant rules
    non_compliant_rules = {k: v for k, v in results.items() 
                          if v['compliance_status'] in ['Non-Compliant', 'Partially Compliant']}
    
    if non_compliant_rules:
        st.subheader("🚨 Remediation Required")
        
        for rule_id, result in non_compliant_rules.items():
            with st.expander(f"🛠️ {result['rule_name']} - {result['compliance_status']}"):
                st.write(f"**Evidence:** {result['evidence']}")
                st.write(f"**Remediation Steps:** {result['remediation']}")
                st.write(f"**Confidence:** {result['confidence']}")

if __name__ == "__main__":
    main()
