import csv
import time
import os
import psutil
from ingest_exp2 import clear_redis_store, create_hnsw_index, process_pdfs_alt
from search_exp2 import search_embeddings, generate_rag_response

# results folder
os.makedirs("results", exist_ok=True)

# constants
CHUNK_SIZE = 1000
OVERLAP = 100
PREPROC = 1
DB = "redis"

# parameters
embedding_models = ["nomic-embed-text", "mxbai-embed-large", "bge-m3"]
ollama_models = ["llama3.2:latest", "mistral:latest"]

# questions
questions = [
    "1) What is Redis?",
    "1) Describe ACID compliance.",
    "1) Describe a B+ Tree.",
    "2) What are the tradeoffs between B+ Tree and AVL trees?",
    "2) Write a MongoDB aggregation pipeline to find the top 5 customers with the highest total spend. Assume the orders collection contains documents with fields: customerId, items (an array of objects with price and quantity), and status. Only include orders where status is 'completed'. Return the customerId and their total spend, sorted from highest to lowest.",
    "2) What are the inherent CAP theorem tradeoffs associated with different types of database systems, such as relational databases (RDBMS), document stores (e.g., MongoDB), vector databases (e.g., Redis with vector support), and graph databases (e.g., Neo4j)?"
]

output_csv = os.path.join("results", "experiment2_results.csv")

with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "question", "response", "ollama", "generation_time", "generation_memory",
        "embedding", "search_time", "search_memory",
        "vector_similarity_min", "vector_similarity_max", "vector_similarity_avg"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='~', quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()

    # loop over each embedding model
    for embedding_model in embedding_models:
        print(f"\n--- Processing ingestion for embedding model: {embedding_model} ---")
        clear_redis_store()
        create_hnsw_index(embedding_model)
        process_pdfs_alt("../data/", CHUNK_SIZE, OVERLAP, embedding_model, PREPROC, DB)
    

        # for each LLM model, run all questions
        for ollama_model in ollama_models:
            print(f"\n=== Evaluating with Ollama model: {ollama_model} and embedding: {embedding_model} ===")
            for question in questions:
                print(f"\nQuery: {question}")

                # record search start time and memory
                search_start_time = time.perf_counter()

                context_results = search_embeddings(question, embedding_model, top_k=5)

                search_end_time = time.perf_counter()
                search_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


                search_time = search_end_time - search_start_time
                

                # compute vector similarity metrics 
                similarities = [float(result.get("similarity", 0)) for result in context_results]
                if similarities:
                    sim_min = min(similarities)
                    sim_max = max(similarities)
                    sim_avg = sum(similarities) / len(similarities)
                else:
                    sim_min = sim_max = sim_avg = None

                # record generation start time and memory
                gen_start_time = time.perf_counter()

                response = generate_rag_response(question, context_results, llm_model=ollama_model)

                gen_end_time = time.perf_counter()
                generation_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

                generation_time = gen_end_time - gen_start_time
                

                # record results in CSV
                writer.writerow({
                    "question": question,
                    "response": response,
                    "ollama": ollama_model,
                    "generation_time": generation_time,
                    "generation_memory": generation_memory,
                    "embedding": embedding_model,
                    "search_time": search_time,
                    "search_memory": search_memory,
                    "vector_similarity_min": sim_min,
                    "vector_similarity_max": sim_max,
                    "vector_similarity_avg": sim_avg
                })
                print("Response recorded.")

        clear_redis_store()

print(f"\nExperiment completed. Results saved to {output_csv}")
