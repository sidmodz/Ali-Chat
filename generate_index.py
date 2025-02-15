import os
import faiss
import numpy as np
import google.generativeai as genai
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Missing API key! Set GOOGLE_API_KEY as an environment variable.")

def get_embedding(text):
    """Generate an embedding for a given text using Gemini."""
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(response["embedding"])

def load_txt_documents(directory="documents"):
    """Load and split text documents from the folder."""
    documents = []

    # Add your .txt resume path
    path = '/path'
    loader = TextLoader(path)
    documents.extend(loader.load())
    return documents

# Load, split, and process documents
documents = load_txt_documents()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Convert documents to embeddings
doc_texts = [doc.page_content for doc in texts]
embeddings = np.array([get_embedding(text) for text in doc_texts])

# Save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "faiss_index.bin")
np.save("doc_texts.npy", doc_texts)  # Save texts for retrieval
print("Documents indexed successfully!")
