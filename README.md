# 📄 RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that allows users to upload PDF documents and ask natural language questions about their contents.

## 🚀 Features

- Upload any PDF
- Extract text automatically
- Split text into semantic chunks
- Generate embeddings using Hugging Face (`all-MiniLM-L6-v2`)
- Store vectors in ChromaDB
- Perform semantic similarity search
- Generate context-aware answers using Gemini 2.5 Flash
- Maintain conversational chat history

## 🛠 Tech Stack

- Python
- Streamlit
- LangChain
- Hugging Face Sentence Transformers
- ChromaDB
- Google Gemini API
- PyPDF

## 📌 Architecture

PDF Upload
↓
Text Extraction
↓
Chunking
↓
Embeddings
↓
ChromaDB
↓
Similarity Search
↓
Retrieved Context
↓
Gemini 2.5 Flash
↓
Final Answer

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📷 Demo

Upload a PDF and ask questions such as:

- Explain SIMD systems
- What are the advantages of MIMD?
- Compare SIMD and MIMD
- Summarize Chapter 1

The chatbot retrieves relevant chunks from the document and generates answers grounded in the uploaded PDF.
