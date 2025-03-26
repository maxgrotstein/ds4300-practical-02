# General Retrieval

# What is Redis?
# Describe ACID compliance.
# Describe a B+ Tree.


# Critical thinking

# What are the tradeoffs between B+ Tree and AVL trees?
# Write a MongoDB aggregation pipeline to find the top 5 customers with the highest total spend. Assume the orders collection contains documents with fields: customerId, items (an array of objects with price and quantity), and status. Only include orders where status is "completed". Return the customerId and their total spend, sorted from highest to lowest.
# What are the inherent CAP theorem tradeoffs associated with different types of database systems, such as relational databases (RDBMS), document stores (e.g., MongoDB), vector databases (e.g., Redis with vector support), and graph databases (e.g., Neo4j)?

# For set 1, evaluating ingestion
# Constants: llama 3:2, nomic-embed-text, Redis
# variables to record: question, response, vector simlarity, overlap, chunk, preproc
# 3 overlap * 3 chunks * 2 preproc * 6 questions = 108 responses

# For set 2:
# Constants: overlap, chunk, preproc
# variables to record: question, response, vector simlarity, ollama, embedding, speed(time) to search, memory_usage to search,
# 2 ollama * 3 embedding * 6 questions = 36 responses

# For set 3
# Constants: overlap, chunk, preproc, ollama, embedding
# variables to record: question, response, vector similarity, db, speed(time) to index, memory_usage to index, size(storage) of index, speed(time) to query, memory_usage to query
# 3 dbs * 6 questions = 18 responses

# Anything from here


