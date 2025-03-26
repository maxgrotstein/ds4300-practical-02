# For set 1, evaluating ingestion
# Constants: llama 3:2, nomic-embed-text, Redis
# variables to record: question, response, vector simlarity, overlap, chunk, preproc

# run ingest with each combination of params and search for each question


import csv
from ingest_exp1 import clear_redis_store, create_hnsw_index, process_pdfs_alt
from search_exp1 import search_embeddings, generate_rag_response
import os

# results folder
os.makedirs("results", exist_ok=True)

# constants 
EMBEDDING_MODEL = "nomic-embed-text"
DB = "redis"

# param grid
chunk_sizes = [200, 500, 1000]       
overlap_values = [0, 50, 100]       
preproc_flags = [0, 1]             

# evaluation questions (3 general retrieval + 3 critical thinking)
questions = [
    "What is Redis?",
    #"Describe ACID compliance.",
    #"Describe a B+ Tree."
    #"What are the tradeoffs between B+ Tree and AVL trees?",
    #"Write a MongoDB aggregation pipeline to find the top 5 customers with the highest total spend. "
    #"Assume the orders collection contains documents with fields: customerId, items (an array of objects with price and quantity), "
    #"and status. Only include orders where status is 'completed'. Return the customerId and their total spend, sorted from highest to lowest.",
    "What are the inherent CAP theorem tradeoffs associated with different types of database systems, such as relational databases (RDBMS), "
    "document stores (e.g., MongoDB), vector databases (e.g., Redis with vector support), and graph databases (e.g., Neo4j)?"
]

output_csv = os.path.join("results", "experiment0101_results.csv")

with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "question", "response", 
        "vector_similarity_min", "vector_similarity_max", "vector_similarity_avg", 
        "overlap", "chunk_size", "preproc"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='~', quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()

    # loop over every combination of parameters
    for chunk_size in chunk_sizes:
        for overlap in overlap_values:
            for preproc in preproc_flags:
                print(f"\n--- Experiment run: chunk_size={chunk_size}, overlap={overlap}, preproc={preproc} ---")
                clear_redis_store()
                create_hnsw_index()

                # process PDFs from data directory
                process_pdfs_alt("../data/", chunk_size, overlap, EMBEDDING_MODEL, preproc, DB)

                # evaluate each question
                for question in questions:
                    print(f"\nQuery: {question}")
                    # get k nearest neighbors 
                    context_results = search_embeddings(question, top_k=3)

                    # compute similarity metrics
                    similarities = [float(result.get("similarity", 0)) for result in context_results]
                    if similarities:
                        sim_min = min(similarities)
                        sim_max = max(similarities)
                        sim_avg = sum(similarities) / len(similarities)
                    else:
                        sim_min = sim_max = sim_avg = None

                    response = generate_rag_response(question, context_results, [])

                    # record results in CSV
                    writer.writerow({
                        "question": question,
                        "response": response,
                        "vector_similarity_min": sim_min,
                        "vector_similarity_max": sim_max,
                        "vector_similarity_avg": sim_avg,
                        "overlap": overlap,
                        "chunk_size": chunk_size,
                        "preproc": preproc
                    })
                    print("Response recorded.")

                clear_redis_store()

print(f"\nExperiment completed. Results saved to {output_csv}")

