import os
import shutil
import numpy as np
import PyPDF2
from transformers import AutoTokenizer, AutoModel, BartTokenizer, BartForConditionalGeneration
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
import pickle
import streamlit as st

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #2e2e2e; /* Dark background color */
            color: #f5f5f5; /* Light text color */
        }
        .stButton > button {
            width: 100%;
            background-color: #4CAF50; /* Button background color */
            color: white; /* Button text color */
            font-size: 16px;
        }
        .stTextInput > div > div > input {
            font-size: 16px;
            background-color: #3c3c3c; /* Input field background color */
            color: #f5f5f5; /* Input field text color */
        }
        .stFileUploader > div {
            font-size: 16px;
            background-color: #3c3c3c; /* File uploader background color */
            color: #f5f5f5; /* File uploader text color */
        }
        .stMarkdown {
            background-color: #3c3c3c; /* Markdown background color */
            color: #f5f5f5; /* Markdown text color */
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
        }
        .main {
            padding: 20px;
            background-color: #2e2e2e; /* Main content background color */
            color: #f5f5f5; /* Main content text color */
        }
    </style>
""", unsafe_allow_html=True)

class Encoder:
    def __init__(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy().tolist()
        return embedding

CHROMA_PATH = "chroma"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

class Chroma:
    def __init__(self, persist_directory, embedding_function):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.documents = []

    def similarity_search_with_relevance_scores(self, query_text, k=3):
        query_embedding = self.embedding_function.embed_query(query_text)
        similarity_scores = []

        for document in self.documents:
            document_embedding = self.embedding_function.embed_query(document.page_content)
            similarity_score = self.calculate_cosine_similarity(query_embedding, document_embedding)
            similarity_scores.append((document, similarity_score))

        if not similarity_scores:
            print("No similarity scores found. Ensure documents are loaded and embeddings are computed correctly.")
            return []

        # Normalize similarity scores
        max_score = max(similarity_scores, key=lambda x: x[1])[1]
        min_score = min(similarity_scores, key=lambda x: x[1])[1]
        normalized_scores = [(doc, (score - min_score) / (max_score - min_score)) for doc, score in similarity_scores]

        # Filter results based on threshold
        filtered_scores = [(doc, score) for doc, score in normalized_scores if score >= 0.7]

        # Sort by similarity score in descending order
        filtered_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results
        return filtered_scores[:k]

    def calculate_cosine_similarity(self, vector1, vector2):
        dot_product = np.dot(vector1, vector2)
        norm_vector1 = np.linalg.norm(vector1)
        norm_vector2 = np.linalg.norm(vector2)
        cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
        return cosine_similarity

    def persist(self):
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory)

        with open(os.path.join(self.persist_directory, 'documents.pkl'), 'wb') as f:
            pickle.dump(self.documents, f)

        print(f"Persisted {len(self.documents)} documents to {self.persist_directory}.")

def generate_data_store(documents):
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            documents.append(Document(page_content=text, metadata={"source": uploaded_file.name}))
    print(f"Loaded {len(documents)} PDF documents.")
    return documents

def split_text(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: List[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding_function = HuggingFaceEmbeddings()
    embeddings = embedding_function.embed_documents([chunk.page_content for chunk in chunks])
    print(f"Calculated embeddings for {len(chunks)} chunks.")
    
    chroma = Chroma(CHROMA_PATH, embedding_function)
    chroma.documents = chunks
    chroma.persist_directory = CHROMA_PATH

    chroma.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

class HuggingFaceEmbeddings:
    def __init__(self):
        self.encoder = Encoder(embedding_model_name)

    def embed_query(self, text):
        return self.encoder.encode(text)

    def embed_documents(self, texts):
        embeddings = [self.encoder.encode(text) for text in texts]
        print(f"Calculated embeddings for {len(embeddings)} documents.")
        return embeddings

def generate_response(context_text, query_text):
    summarization_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    summarization_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    
    prompt = f"Summarize the following context and answer the question.\n\nContext:\n{context_text}\n\nQuestion:\n{query_text}"
    inputs = summarization_tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = summarization_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def query_database(query_text):
    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    if not os.path.exists(os.path.join(CHROMA_PATH, 'documents.pkl')):
        print("Error: No documents found in the Chroma database. Please run the script in 'generate' mode first.")
        return

    with open(os.path.join(CHROMA_PATH, 'documents.pkl'), 'rb') as f:
        db.documents = pickle.load(f)

    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0:
        print("Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(f"Context: {context_text}\nQuestion: {query_text}")

    response_text = generate_response(context_text, query_text)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"**Response:** {response_text}\n\n**Sources:** {sources}"
    print(formatted_response)

    return formatted_response

def main():
    st.title("📚 QueryGen")
    st.markdown("""
        <style>
            .main {
                background-color: #2e2e2e; /* Main content background color */
                font-family: Arial, sans-serif;
                color: #f5f5f5; /* Main content text color */
                padding: 20px;
            }
            .stButton > button {
                width: 100%;
                background-color: #4CAF50; /* Button background color */
                color: white; /* Button text color */
                font-size: 16px;
            }
        </style>
    """, unsafe_allow_html=True)

    mode = st.sidebar.selectbox("Choose Mode", ["Generate Datastore", "Query Database"])

    if mode == "Generate Datastore":
        st.header("Generate Datastore")
        st.write("Upload PDF files to generate a searchable datastore.")
        uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type="pdf")
        if st.button("Generate Datastore"):
            if uploaded_files:
                with st.spinner("Processing files..."):
                    documents = load_documents(uploaded_files)
                    generate_data_store(documents)
                st.success("Datastore generated successfully!")
            else:
                st.error("Please upload at least one PDF file.")

    elif mode == "Query Database":
        st.header("Query Database")
        st.write("Enter your query to search the datastore.")
        query_text = st.text_input("Enter your query")
        if st.button("Search"):
            if query_text:
                with st.spinner("Searching datastore..."):
                    response = query_database(query_text)
                st.markdown(response, unsafe_allow_html=True)
            else:
                st.error("Please enter a query.")

if __name__ == "__main__":
    main()
