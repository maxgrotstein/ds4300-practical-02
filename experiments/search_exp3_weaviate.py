import weaviate
from weaviate import WeaviateClient
from weaviate.classes.query import MetadataQuery
import json
import numpy as np

import ollama
from redis.commands.search.field import VectorField, TextField

# Initialize models
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"
LLAMA_MODEL="llama3.2:latest"


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

def search_embeddings(query, top_k=3):
    weaviate_client = weaviate.connect_to_local()
    query_embedding = get_embedding(query)

    query_vector = np.array(query_embedding, dtype=np.float32)
    collection = weaviate_client.collections.get("Document")

    try:
        # Perform the search
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            offset=1,
            return_metadata=MetadataQuery(score=True, distance=True),
        )
        weaviate_client.close()

        for o in response.objects:
            print(
                # f"---> File: {o.properties["file"]}, Page: {o.properties["page"]}, Chunk: {o.properties["chunk"]}"
                )
        return response

    except Exception as e:
        weaviate_client.close()
        print(f"Search error: {e}")
        return []

def search_and_generate(query, embedding_model, llm_model):
    context_results = search_embeddings(query)
    return generate_rag_response(query, context_results, conversation_history=[])


def generate_rag_response(query, context_results, conversation_history):

   
    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.properties.get('file', 'Unknown file')} (page {result.properties.get('page', 'Unknown page')}, chunk {result.properties.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.metadata.score):.2f}"
            for result in context_results.objects
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

        # Generate RAG response
        response = generate_rag_response(query, context_results, conversation_history)

        print("\n--- Response ---")
        print(response)

        # add to conversation history
        conversation_history.append({"user": query, "assistant": response})


if __name__ == "__main__":
    interactive_search()
