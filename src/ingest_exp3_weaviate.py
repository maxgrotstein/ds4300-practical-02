# Need to run this in terminal to get container to spin up properly
# docker run -d \
#   --name weaviate \
#   -p 8080:8080 \
#   -p 50051:50051
#   -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
#   -e QUERY_DEFAULTS_LIMIT=25 \
#   semitechnologies/weaviate:latest

# Port is 8080

## DS 4300 Example - from docs

import ollama
import weaviate
from weaviate import WeaviateClient
from weaviate.classes.query import Filter
import weaviate.classes.config as wvcc
from weaviate.classes.query import MetadataQuery
from weaviate.classes.config import Configure, VectorDistances
import numpy as np
import os
import fitz
import re
from nltk.corpus import stopwords

CLASS_NAME = "Document"

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# used to clear the chroma vector store
def clear_weaviate():
    print("Clearing existing Weaviate store...")
    weaviate_client = weaviate.connect_to_local()

    try:
        if weaviate_client.collections.exists(CLASS_NAME):
            weaviate_client.collections.delete(CLASS_NAME)
            weaviate_client.close()
            print("Weaviate store cleared.")
    except:
        weaviate_client.close()
        return
    
    
# Create an HNSW index in Weaviate
def create_hnsw_index():
    clear_weaviate()
    try:
        weaviate_client = weaviate.connect_to_local()
        collection = weaviate_client.collections.create(
            name=CLASS_NAME,
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            properties=[
                wvcc.Property(
                    name="file",
                    data_type=wvcc.DataType.TEXT
                ),
                wvcc.Property(
                    name="page",
                    data_type=wvcc.DataType.TEXT
                ),
                wvcc.Property(
                    name="chunk",
                    data_type=wvcc.DataType.TEXT
                )
            ]
        )
        weaviate_client.close()
    except:
        weaviate_client.close()
        pass
    finally:
        weaviate_client.close()
        print("Collection created successfully.")
    

# Generate an embedding using nomic-embed-text - makes 768 dim vector
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# store the embedding in Weaviate
def store_embedding(file: str, page: str, chunk: str, embedding: list, db: str):
    weaviate_client = weaviate.connect_to_local()
    docs = weaviate_client.collections.get(CLASS_NAME)
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"

    new_uuid = docs.data.insert(
        properties={
            "file": file,
            "page": page,
            "chunk": chunk
        },
        vector=embedding
    )
    weaviate_client.close()
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


def query_weaviate(query_text: str):
    weaviate_client = weaviate.connect_to_local()
    query_text = "Efficient search in vector databases"
    collection = weaviate_client.collections.get(CLASS_NAME)
    response = collection.query.bm25(
        query=query_text,
        limit=3,
        offset=1,
        return_metadata=MetadataQuery(score=True),
    )
    weaviate_client.close()

    for o in response.objects:
        print(f"{o.properties} \n ----> {o.metadata.score}\n")


def test_preproc_vars(chunk_size, overlap, embed, preprocess=0, db='chroma'):
    clear_weaviate()
    print("Cleared Weaviate")
    create_hnsw_index()
    print("Created Index")
    process_pdfs_alt("../data/", chunk_size, overlap, embed, preprocess, db)
    print("\n---Done processing PDFs---\n")
    query_weaviate("What is the capital of France?")


def main():
    test_preproc_vars(500, 100, "nomic-embed-text")


if __name__ == "__main__":
    main()



