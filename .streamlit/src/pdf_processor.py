"""
PDF ingestion and vector store pipeline
"""

import os
import logging
from typing import List, Optional
from pypdf import PdfReader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF processing and vector store creation"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Document]:
        """Extract text from PDF file"""
        try:
            reader = PdfReader(pdf_path)
            documents = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path,
                            "page": page_num + 1,
                            "total_pages": len(reader.pages)
                        }
                    ))
            
            logger.info(f"Extracted {len(documents)} pages from {pdf_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise
    
    def create_vector_store(self, documents: List[Document], save_path: Optional[str] = None) -> FAISS:
        """Create and optionally save vector store"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            vector_store = FAISS.from_documents(chunks, self.embeddings)
            
            if save_path:
                vector_store.save_local(save_path)
                logger.info(f"Vector store saved to {save_path}")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def load_vector_store(self, load_path: str) -> FAISS:
        """Load existing vector store"""
        try:
            vector_store = FAISS.load_local(load_path, self.embeddings)
            logger.info(f"Vector store loaded from {load_path}")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
