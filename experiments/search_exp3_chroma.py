

import chromadb
import json
import numpy as np
import ollama
from redis.commands.search.field import VectorField, TextField

# ollama pull mistral

# Initialize models
chroma_client = chromadb.HttpClient(
    host="localhost",
    port=8000
    )

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"
LLAMA_MODEL="llama3.2:latest"




def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3):

    query_embedding = get_embedding(query)

    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32)
    chroma_collection = chroma_client.get_collection(name="Notes")

    try:
        # Construct the vector similarity search query
        # Use a more standard RediSearch vector search syntax
        # q = Query("*").sort_by("embedding", query_vector)

        # q = (
        #     Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        #     .sort_by("vector_distance")
        #     .return_fields("id", "file", "page", "chunk", "vector_distance")
        #     .dialect(2)
        # )

        # Perform the search
        res = chroma_collection.query(
            query_embeddings=query_vector,
            n_results=top_k,
            include=["metadatas", "distances"]
        )

        for i in range(len(res.get("ids")[0])):
            print(
                # f"---> File: {res["metadatas"][0][i]['file']}, Page: {res["metadatas"][0][i]['page']}, Chunk: {res["metadatas"][0][i]['chunk']}"
            )
        return res

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results, conversation_history):

   

    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('distance', 0)):.2f}"
            for result in context_results
        ]
    )

    # print(f"context_str: {context_str}")

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.


Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model=LLAMA_MODEL, messages=[{"role": "user", "content": prompt}]
        # was using mistral:lastest
    )

    return response["message"]["content"]

## Converts the Chroma DB format of results to the Redis format
def convert_chroma_response_to_redis(results):
    final_list = []

    ids_list = results['ids'][0]
    distance_list = results['distances'][0]
    files_list = [results['metadatas'][0][i]['file'] for i in range(len(results['ids'][0]))]
    pages_list = [results['metadatas'][0][i]['page'] for i in range(len(results['ids'][0]))]
    chunk_list = [results['metadatas'][0][i]['chunk'] for i in range(len(results['ids'][0]))]
    
    for i in range(len(ids_list)):
        final_list.append({
            "id": ids_list[i],
            "file": files_list[i],
            "page": pages_list[i],
            "chunk": chunk_list[i],
            "distance": distance_list[i]
        })
    
    return final_list


def search_and_generate(query, embedding_model, llm_model):
    raw_results = search_embeddings(query)
    formatted = convert_chroma_response_to_redis(raw_results)
    return generate_rag_response(query, formatted, conversation_history=[])


def interactive_search():
    conversation_history=[]

    """Interactive search interface."""
    print("Model:", LLAMA_MODEL)
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")
    

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        if query.lower() == "clear":
            # reset the conversation history
            conversation_history = []  
            print("Conversation history cleared.")
            continue
    

        # Search for relevant embeddings
        context_results = search_embeddings(query)
        context_results = convert_chroma_response_to_redis(context_results)


        # Generate RAG response
        response = generate_rag_response(query, context_results, conversation_history)

        print("\n--- Response ---")
        print(response)

        # add to conversation history
        conversation_history.append({"user": query, "assistant": response})


if __name__ == "__main__":
    interactive_search()
