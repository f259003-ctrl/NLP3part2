"""
Main entry point for the Compliance Checking Agent
"""

import logging
from src.pdf_processor import PDFProcessor
from src.compliance_checker import ComplianceChecker
from src.agent_workflow import ComplianceAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main execution function"""
    print("Contract Compliance Checking Agent")
    print("=" * 40)
    
    # Initialize components
    processor = PDFProcessor()
    # ... rest of main execution logic

if __name__ == "__main__":
    main()
