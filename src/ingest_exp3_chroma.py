# Port is 8000

## DS 4300 Example - from docs

import ollama
import chromadb
from chromadb.config import Settings
import numpy as np
import os
import fitz
import re
from nltk.corpus import stopwords

# Initialize chroma connection
chroma_client = chromadb.HttpClient(
    host="localhost",
    port=8000
    )

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# used to clear the chroma vector store
def clear_chroma():
    print("Clearing existing Chroma store...")
    for col_name in chroma_client.list_collections():
        chroma_client.delete_collection(name=col_name)
    print("Chroma store cleared.")


# Create an HNSW index in chroma
def create_hnsw_index():
    try:
        for col in chroma_client.list_collections():
            chroma_client.delete_collection(name=col.name)
    except:
        pass
    chroma_client.create_collection(name="Notes")
    print("Collection created successfully.")


# Generate an embedding using nomic-embed-text - makes 768 dim vector
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# store the embedding in Chroma
def store_embedding(file: str, page: str, chunk: str, embedding: list, db: str):
    collection = chroma_client.get_collection(name="Notes")

    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"

    collection.add(
        ids=key,
        embeddings=np.array(embedding, dtype=np.float32),
        metadatas=[{
            "file": file,
            "chunk": chunk,
            "page": page
        }],
    )
    print(f"Stored embedding for: {chunk}")


# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

def preprocess_text(text):
    """Optionally remove whitespace and stop words from text."""
    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # get set of English stop words
    stop_words = set(stopwords.words('english'))
    
    # filter out stop words
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    
    return " ".join(filtered_tokens)


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


def process_pdfs_alt(data_dir, chunk_size, overlap, embed, preprocess, db):

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)

            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                if preprocess!=0:
                    text = preprocess_text(text)  

                chunks = split_text_into_chunks(text, chunk_size, overlap)
    
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk, embed)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                        db=db,
                    )
            print(f" -----> Processed {file_name}")


def query_chroma(query_text: str):
    query_text = "Efficient search in vector databases"
    embedding = get_embedding(query_text)
    chroma_collection = chroma_client.get_collection(name="Notes")
    res = chroma_collection.query(
        query_embeddings=embedding,
        n_results=3,
        include=["metadatas", "distances"]
    )


def test_preproc_vars(chunk_size, overlap, embed, preprocess=0, db='chroma'):
    clear_chroma()
    print("Cleared chroma")
    create_hnsw_index()
    print("Created Index")
    process_pdfs_alt("../data/", chunk_size, overlap, embed, preprocess, db)
    print("\n---Done processing PDFs---\n")
    query_chroma("What is the capital of France?")


def main():
    test_preproc_vars(500, 100, "nomic-embed-text")


if __name__ == "__main__":
    main()



