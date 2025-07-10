# 📚 QueryGen — PDF-Based Question Answering System

QueryGen is a Streamlit-based application that allows users to:
- Upload PDF documents
- Generate a semantic datastore using sentence embeddings
- Query the content using natural language questions
- Retrieve summarized answers with source references

This system combines NLP capabilities (via HuggingFace Transformers) with vector-based semantic search (custom in-memory implementation using cosine similarity).

---

## ✨ Features

- 📄 Upload multiple PDFs and extract their text
- 🔍 Create document embeddings using Sentence Transformers
- 🧠 Perform semantic similarity search
- 📝 Summarize context using BART model
- 🖥️ Intuitive and dark-themed Streamlit UI

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **NLP Models**: HuggingFace Transformers (`all-MiniLM-L6-v2`, `facebook/bart-large-cnn`)
- **PDF Parsing**: PyPDF2
- **Embedding & Search**: Custom cosine similarity + serialized persistence
- **Vector Store**: Custom in-memory store with `pickle`

---

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/querygen.git
cd querygen
