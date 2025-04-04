# For set 1, evaluating ingestion
# Constants: llama 3:2, nomic-embed-text, Redis
# variables to record: question, response, overlap, chunk, preproc

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

# evaluation questions (1 general retrieval + 1 critical thinking, need to keep it reasonable for a pure qualitative analysis)
questions = [
    "What is Redis?",
    "What are the inherent CAP theorem tradeoffs associated with different types of database systems, such as relational databases (RDBMS), "
    "document stores (e.g., MongoDB), vector databases (e.g., Redis with vector support), and graph databases (e.g., Neo4j)?"
]

output_csv = os.path.join("results", "experiment1_results.csv")

with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "question", "response", 
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

                    response = generate_rag_response(question, context_results, [])

                    # record results in CSV
                    writer.writerow({
                        "question": question,
                        "response": response,
                        "overlap": overlap,
                        "chunk_size": chunk_size,
                        "preproc": preproc
                    })
                    print("Response recorded.")

                clear_redis_store()

print(f"\nExperiment completed. Results saved to {output_csv}")

