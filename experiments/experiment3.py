# experiment3.py

import csv
import time
import os
import psutil

# Sets path to results folder
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(results_dir, exist_ok=True)

# constants
CHUNK_SIZE = 1000
OVERLAP = 100
PREPROC = 1
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_MODEL = "llama3.2:latest"
DB = os.getenv("EXPERIMENT_DB", "redis").lower()  # set manually before each run
DATA_DIR = "../data/"

# questions
questions = [
    "1) What is Redis?",
    "1) Describe ACID compliance.",
    "1) Describe a B+ Tree.",
    "2) What are the tradeoffs between B+ Tree and AVL trees?",
    "2) Write a MongoDB aggregation pipeline to find the top 5 customers with the highest total spend. Assume the orders collection contains documents with fields: customerId, items (an array of objects with price and quantity), and status. Only include orders where status is 'completed'. Return the customerId and their total spend, sorted from highest to lowest.",
    "2) What are the inherent CAP theorem tradeoffs associated with different types of database systems, such as relational databases (RDBMS), document stores (e.g., MongoDB), vector databases (e.g., Redis with vector support), and graph databases (e.g., Neo4j)?"
]

# import db-specific functions
if DB == "redis":
    from ingest_exp1 import clear_redis_store as clear_db, create_hnsw_index as create_index, process_pdfs_alt as ingest
    from search_exp2 import search_and_generate
elif DB == "chroma":
    from ingest_exp3_chroma import clear_chroma as clear_db, create_hnsw_index as create_index, process_pdfs_alt as ingest
    from search_exp3_chroma import search_and_generate
elif DB == "weaviate":
    from ingest_exp3_weaviate import clear_weaviate as clear_db, create_hnsw_index as create_index, process_pdfs_alt as ingest
    from search_exp3_weaviate import search_and_generate
else:
    raise ValueError("Invalid DB selection. Must be 'redis', 'chroma', or 'weaviate'.")

output_csv = os.path.join(results_dir, 'experiment3_results.csv')


# run the experiment
with open(output_csv, mode="a", newline="", encoding="utf-8") as csvfile:
    fieldnames = [
        "question", "response", "db",
        "index_time", "index_memory", 
        "search_time", "search_memory"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='~', quoting=csv.QUOTE_MINIMAL)

    # write header only if file is empty
    if os.stat(output_csv).st_size == 0:
        writer.writeheader()

    print(f"\n--- Running Experiment 3 for DB: {DB} ---")

    # clear and build index
    clear_db()
    create_index()
    process = psutil.Process()

    # index timing and memory
    index_start = time.perf_counter()
    ingest(DATA_DIR, CHUNK_SIZE, OVERLAP, EMBEDDING_MODEL, PREPROC, DB)
    index_end = time.perf_counter()

    index_time = index_end - index_start
    index_memory = process.memory_info().rss / (1024 * 1024)
   

    # loop over all questions
    for question in questions:
        print(f"\nQuery: {question}")

        # record search time and memory
        search_start_time = time.perf_counter()
        context_results = search_and_generate(question, EMBEDDING_MODEL, OLLAMA_MODEL)
        search_end_time = time.perf_counter()
        search_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        search_time = search_end_time - search_start_time

        # extract response
        response = context_results if isinstance(context_results, str) else "RESPONSE FORMAT ERROR"

        # record results in CSV
        writer.writerow({
            "question": question,
            "response": response,
            "db": DB,
            "index_time": index_time,
            "index_memory": index_memory,
            "search_time": search_time,
            "search_memory": search_memory
        })
        print("Response recorded.")

print(f"\nExperiment completed. Results saved to {output_csv}")
