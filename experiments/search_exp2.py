# search.py

import redis
import json
import numpy as np
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField

# ollama pull mistral

# Initialize models
redis_client = redis.StrictRedis(host="localhost", port=6379, decode_responses=True)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"



def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, embedding_model="nomic-embed-text", top_k=3):

    query_embedding = get_embedding(query, model=embedding_model)

    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Construct the vector similarity search query
        # Use a more standard RediSearch vector search syntax
        # q = Query("*").sort_by("embedding", query_vector)

        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "overlap", "chunk_size", "preproc", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into the expected format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "overlap": result.overlap,
                "chunk_size": result.chunk_size,
                "preproc": result.preproc,
                "similarity": float(result.vector_distance),
            }
            for result in results.docs
        ][:top_k]

        # Print results for debugging
        for result in top_results:
            print(
                #f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def search_and_generate(query, embedding_model, llm_model):
    context_results = search_embeddings(query, top_k=5)
    return generate_rag_response(query, context_results, "llama3.2:latest")


def generate_rag_response(query, context_results, llm_model):



    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    # print(f"context_str: {context_str}")

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'. The character '~' must NOT be used anywhere in your response.


Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model=llm_model, messages=[{"role": "user", "content": prompt}]
    
    )

    return response["message"]["content"]


def interactive_search():
    conversation_history=[]

    """Interactive search interface."""
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

        # Generate RAG response
        response = generate_rag_response(query, context_results, conversation_history)

        print("\n--- Response ---")
        print(response)

        # add to conversation history
        conversation_history.append({"user": query, "assistant": response})



# def store_embedding(file, page, chunk, embedding):
#     """
#     Store an embedding in Redis using a hash with vector field.

#     Args:
#         file (str): Source file name
#         page (str): Page number
#         chunk (str): Chunk index
#         embedding (list): Embedding vector
#     """
#     key = f"{file}_page_{page}_chunk_{chunk}"
#     redis_client.hset(
#         key,
#         mapping={
#             "embedding": np.array(embedding, dtype=np.float32).tobytes(),
#             "file": file,
#             "page": page,
#             "chunk": chunk,
#         },
#     )


if __name__ == "__main__":
    interactive_search()
