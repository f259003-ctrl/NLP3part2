import streamlit as st
import json
import pandas as pd
import tempfile
import os
import re
from pypdf import PdfReader
import google.generativeai as genai

# Simple rules definition
RULES_DEFINITION = {
    "confidentiality_clause": {
        "name": "Confidentiality Clause",
        "description": "Document must contain confidentiality provisions",
        "category": "Confidentiality",
        "severity": "High",
        "keywords": ["confidential", "non-disclosure", "proprietary"]
    },
    "term_duration": {
        "name": "Contract Term Duration", 
        "description": "Must specify start and end dates or duration",
        "category": "Term",
        "severity": "High",
        "keywords": ["term", "duration", "effective date", "expiration"]
    },
    "termination_clause": {
        "name": "Termination Clause",
        "description": "Must include termination conditions and notice period",
        "category": "Termination", 
        "severity": "High",
        "keywords": ["termination", "terminate", "notice period"]
    },
    "governing_law": {
        "name": "Governing Law",
        "description": "Must specify governing law jurisdiction",
        "category": "Legal",
        "severity": "Medium",
        "keywords": ["governing law", "jurisdiction", "laws of"]
    },
    "payment_terms": {
        "name": "Payment Terms",
        "description": "Must specify payment amounts and schedules",
        "category": "Financial",
        "severity": "High",
        "keywords": ["payment", "fee", "compensation", "invoice"]
    }
}

class SimplePDFProcessor:
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF without complex dependencies"""
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

class SimpleComplianceChecker:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def check_rule_compliance(self, rule_id, rule_definition, contract_text):
        """Simple rule checking using Gemini"""
        if not contract_text:
            return self._create_error_response(rule_id, rule_definition, "No contract text provided")
        
        # Use first 15000 chars to avoid token limits
        preview_text = contract_text[:15000] + "..." if len(contract_text) > 15000 else contract_text
        
        prompt = f"""
        Analyze this contract text for compliance with the rule below:
        
        RULE: {rule_definition['name']}
        DESCRIPTION: {rule_definition['description']}
        CATEGORY: {rule_definition['category']}
        SEVERITY: {rule_definition['severity']}
        
        CONTRACT TEXT:
        {preview_text}
        
        Provide a JSON response with:
        - compliance_status: [Compliant/Non-Compliant/Partially Compliant]
        - evidence: specific text that supports your assessment
        - confidence: [High/Medium/Low] 
        - remediation: steps to fix if non-compliant
        
        Response format (JSON only):
        {{
            "rule_id": "{rule_id}",
            "rule_name": "{rule_definition['name']}",
            "compliance_status": "status",
            "evidence": "text evidence", 
            "confidence": "confidence level",
            "remediation": "remediation steps"
        }}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_response(response.text, rule_id, rule_definition)
        except Exception as e:
            return self._create_error_response(rule_id, rule_definition, str(e))
    
    def _parse_response(self, response_text, rule_id, rule_definition):
        """Parse the Gemini response"""
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\{.*\}[^{}]*\}|\{.*\}', response_text, re.DOTALL)
            if json_match:
                result_text = json_match.group()
                result_text = result_text.replace('```json', '').replace('```', '').strip()
                result = json.loads(result_text)
                
                # Add missing fields
                result['category'] = rule_definition['category']
                result['severity'] = rule_definition['severity']
                return result
            else:
                return self._create_error_response(rule_id, rule_definition, "No valid JSON in response")
        except Exception as e:
            return self._create_error_response(rule_id, rule_definition, f"Parse error: {e}")
    
    def _create_error_response(self, rule_id, rule_definition, error):
        return {
            "rule_id": rule_id,
            "rule_name": rule_definition['name'],
            "compliance_status": "Error",
            "evidence": f"Analysis error: {error}",
            "confidence": "Low",
            "remediation": "Please check the contract manually",
            "category": rule_definition['category'],
            "severity": rule_definition['severity']
        }

def main():
    st.set_page_config(
        page_title="Simple Compliance Checker",
        page_icon="📋",
        layout="wide"
    )
    
    st.title("📋 Simple Contract Compliance Checker")
    st.markdown("Upload a contract PDF to check basic compliance rules")
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'contract_text' not in st.session_state:
        st.session_state.contract_text = None
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    api_key = st.sidebar.text_input(
        "Google API Key",
        type="password",
        help="Enter your Google Gemini API key",
        value=st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, 'secrets') else ""
    )
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Contract PDF",
        type="pdf",
        help="Upload the contract PDF to analyze"
    )
    
    # Rule selection
    st.sidebar.subheader("Select Rules to Check")
    selected_rules = {}
    for rule_id, rule in RULES_DEFINITION.items():
        if st.sidebar.checkbox(f"{rule['name']} ({rule['severity']})", value=True, key=rule_id):
            selected_rules[rule_id] = rule
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Selected Rules")
        if not selected_rules:
            st.warning("Please select at least one rule")
        else:
            for rule_id, rule in selected_rules.items():
                with st.expander(f"📝 {rule['name']}"):
                    st.write(f"**Category:** {rule['category']}")
                    st.write(f"**Severity:** {rule['severity']}")
                    st.write(f"**Description:** {rule['description']}")
                    st.write(f"**Keywords:** {', '.join(rule['keywords'])}")
    
    with col2:
        st.subheader("Document Analysis")
        
        if uploaded_file and api_key and selected_rules:
            if st.button("🔍 Run Compliance Check", type="primary", use_container_width=True):
                with st.spinner("Processing document..."):
                    try:
                        # Extract text from PDF
                        processor = SimplePDFProcessor()
                        contract_text = processor.extract_text_from_pdf(uploaded_file)
                        
                        if not contract_text:
                            st.error("Could not extract text from PDF. The file might be scanned or corrupted.")
                            return
                        
                        st.session_state.contract_text = contract_text
                        
                        # Show text preview
                        with st.expander("📄 View Extracted Text Preview"):
                            st.text_area("Contract Text", contract_text[:2000] + "..." if len(contract_text) > 2000 else contract_text, height=200)
                        
                        # Initialize checker
                        checker = SimpleComplianceChecker(api_key)
                        
                        # Run compliance checks
                        progress_bar = st.progress(0)
                        results = {}
                        
                        for i, (rule_id, rule_def) in enumerate(selected_rules.items()):
                            st.info(f"Checking: {rule_def['name']}...")
                            result = checker.check_rule_compliance(rule_id, rule_def, contract_text)
                            results[rule_id] = result
                            progress_bar.progress((i + 1) / len(selected_rules))
                        
                        st.session_state.results = results
                        st.success("✅ Compliance analysis completed!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
        
        # Display results
        if st.session_state.results:
            display_results(st.session_state.results)

def display_results(results):
    st.subheader("📊 Compliance Results")
    
    # Summary statistics
    status_counts = {}
    for result in results.values():
        status = result['compliance_status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Summary metrics
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
        st.metric("Non-Compliant", non_compliant_count, delta_color="inverse")
    with col4:
        st.metric("Partial", partial_count)
    
    # Results table
    st.subheader("📋 Detailed Results")
    
    rows = []
    for rule_id, result in results.items():
        rows.append({
            'Rule': result['rule_name'],
            'Category': result['category'],
            'Severity': result['severity'],
            'Status': result['compliance_status'],
            'Confidence': result['confidence'],
            'Evidence': result['evidence'][:100] + '...' if len(result['evidence']) > 100 else result['evidence']
        })
    
    df = pd.DataFrame(rows)
    
    # Color coding
    def color_row(row):
        if row['Status'] == 'Compliant':
            return ['background-color: #d4edda'] * len(row)
        elif row['Status'] == 'Non-Compliant':
            return ['background-color: #f8d7da'] * len(row)
        elif row['Status'] == 'Partially Compliant':
            return ['background-color: #fff3cd'] * len(row)
        else:
            return ['background-color: #f0f0f0'] * len(row)
    
    styled_df = df.style.apply(color_row, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Remediation section
    non_compliant = {k: v for k, v in results.items() 
                    if v['compliance_status'] in ['Non-Compliant', 'Partially Compliant']}
    
    if non_compliant:
        st.subheader("🚨 Required Remediation")
        
        for rule_id, result in non_compliant.items():
            with st.expander(f"🔧 {result['rule_name']} - {result['compliance_status']}", expanded=True):
                st.write(f"**Evidence:** {result['evidence']}")
                st.write(f"**Remediation Steps:** {result['remediation']}")
                st.write(f"**Confidence:** {result['confidence']}")
    
    # Export options
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON export
        json_data = json.dumps(results, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="compliance_results.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # CSV export
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download CSV", 
            data=csv_data,
            file_name="compliance_report.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
