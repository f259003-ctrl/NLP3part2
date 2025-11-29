"""
Custom compliance checker tool connected to Gemini
"""

import os
import json
import re
import logging
from typing import List, Dict, Any
import google.generativeai as genai
from langchain.vectorstores import FAISS

logger = logging.getLogger(__name__)

class ComplianceChecker:
    """Checks compliance against defined rules using Gemini"""
    
    def __init__(self, vector_store: FAISS, api_key: str):
        self.vector_store = vector_store
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def retrieve_relevant_sections(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant document sections for a query"""
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.warning(f"Error in similarity search: {e}")
            return []
    
    def check_rule_compliance(self, rule_id: str, rule_definition: Dict) -> Dict[str, Any]:
        """Check compliance for a specific rule"""
        query = self._build_compliance_query(rule_definition)
        relevant_sections = self.retrieve_relevant_sections(query)
        context = self._build_context(relevant_sections)
        
        prompt = self._build_prompt(rule_id, rule_definition, context)
        
        try:
            response = self.model.generate_content(prompt)
            result = self._parse_response(response.text, rule_id, rule_definition)
            result['relevant_sections'] = relevant_sections
            return result
            
        except Exception as e:
            logger.error(f"Error checking rule {rule_id}: {e}")
            return self._create_error_response(rule_id, rule_definition, str(e))
    
    def _build_compliance_query(self, rule_definition: Dict) -> str:
        """Build query for retrieving relevant sections"""
        return f"{rule_definition['description']} What specific clauses or language addresses this requirement?"
    
    def _build_context(self, sections: List[str]) -> str:
        """Build context from relevant sections"""
        return "\n\n".join(sections) if sections else "No relevant sections found."
    
    def _build_prompt(self, rule_id: str, rule_definition: Dict, context: str) -> str:
        """Build prompt for Gemini"""
        return f"""
        Analyze the contract sections and determine compliance with this rule:
        
        RULE: {rule_definition['name']}
        DESCRIPTION: {rule_definition['description']}
        CATEGORY: {rule_definition['category']}
        SEVERITY: {rule_definition['severity']}
        
        CONTRACT SECTIONS:
        {context}
        
        Provide JSON response with:
        - compliance_status: [Compliant/Non-Compliant/Partially Compliant]
        - evidence: specific text supporting assessment
        - confidence: [High/Medium/Low]
        - remediation: steps to fix if non-compliant
        
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
    
    def _parse_response(self, response_text: str, rule_id: str, rule_definition: Dict) -> Dict[str, Any]:
        """Parse Gemini response"""
        try:
            json_match = re.search(r'\{[^{}]*\{.*\}[^{}]*\}|\{.*\}', response_text, re.DOTALL)
            if json_match:
                result_text = json_match.group()
                result_text = result_text.replace('```json', '').replace('```', '').strip()
                return json.loads(result_text)
            else:
                raise ValueError("No JSON found in response")
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}")
    
    def _create_error_response(self, rule_id: str, rule_definition: Dict, error: str) -> Dict[str, Any]:
        """Create error response"""
        return {
            "rule_id": rule_id,
            "rule_name": rule_definition['name'],
            "compliance_status": "Error",
            "evidence": f"Analysis error: {error}",
            "confidence": "Low",
            "remediation": "Check the contract manually",
            "category": rule_definition['category'],
            "severity": rule_definition['severity']
        }
