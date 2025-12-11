"""Setup script for Multi-Modal RAG System."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multimodal-rag",
    version="1.0.0",
    author="Your Name",
    description="Multi-Modal RAG System for Financial Documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "openai>=1.12.0",
        "langchain>=0.1.10",
        "pdfplumber>=0.10.3",
        "chromadb>=0.4.22",
        "rank-bm25>=0.2.2",
        "streamlit>=1.31.0",
        "pandas>=2.2.0",
        "numpy>=1.26.3",
        "pyyaml>=6.0.1",
        "pytesseract>=0.3.10",
        "pdf2image>=1.16.3",
        "pillow>=10.2.0",
    ],
)

