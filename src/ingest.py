## ingest.py

import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import re
from nltk.corpus import stopwords

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# ORIGINAL
# VECTOR_DIM = 768

# after experiment
VECTOR_DIM = 1024
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")


# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


# Generate an embedding using nomic-embed-text - makes 768 dim vector ORIGINAL
#def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

#after experiment
def get_embedding(text: str, model: str = "mxbai-embed-large") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embedding: list, db: str):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"

    if db=='redis':
        redis_client.hset(
            key,
            mapping={
                "file": file,
                "page": page,
                "chunk": chunk,
                "embedding": np.array(
                    embedding, dtype=np.float32
                ).tobytes(),  # Store as byte array
            },
        )
        print(f"Stored embedding for: {chunk}")
    else:
        print('This db is not configured.')


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


# two pdf readers (fitz, pdf plumber), whitespace on or off (preproc?), three overlap sizes, three chunk sizes, three embedding models (nomic, Instructor XL, 1 more)
# prompt tweaks, three DBs (Redis, Chroma, 1 more), LLMs (current: llama3.2:latest, could also use Mistral)
 
# Process all PDF files in a given directory
def process_pdfs(data_dir):

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                # print(f"  Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")

def process_pdfs_alt(data_dir, chunk_size, overlap, embed, preprocess, db):

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)

            # pass pdf reader here, as well as flag to preprocess or not
            text_by_page = extract_text_from_pdf(pdf_path)

            for page_num, text in text_by_page:
                if preprocess!=0:
                    text = preprocess_text(text)  

                chunks = split_text_into_chunks(text, chunk_size, overlap)
                # print(f"  Chunks: {chunks}")
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


def query_redis(query_text: str):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    query_text = "Efficient search in vector databases"
    embedding = get_embedding(query_text)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    # print(res.docs)

    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")

def test_preproc_vars(chunk_size, overlap, embed, preprocess=0, db='redis'):
    clear_redis_store()
    create_hnsw_index()
    # OpenWebUI
    process_pdfs_alt("../data/", chunk_size, overlap, embed, preprocess, db)
    print("\n---Done processing PDFs---\n")
    query_redis("What is the capital of France?")


def main(): 
    # original
    #test_preproc_vars(500, 100, "nomic-embed-text")

    # reccomendations from experiment 1 & 2
    test_preproc_vars(1000, 100, "nomic-embed-text", 1)


if __name__ == "__main__":
    main()


