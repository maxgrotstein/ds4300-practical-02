# ds4300-practical-02

## Project Overview
This project implements a Retrieval Augmented Generation (RAG) pipeline with the primary focus being on creating a model based off class notes to serve as a sort of study guide / cheat sheet. It has built in custom vector stores and LLMs as well as experimentation tools in order to evaluate different configurations (e.g. chunk size, embedding model, chunk overlap, etc.) for analysis of optimal performance and use cases.

## Project Structure
<pre>
ds4300-practical-02/  
├── data/                    # Input .pdf here for use in the RAG  
├── src/                     # Core RAG logic (ingestion and searching)  
│   ├── ingest.py  
│   └── search.py  
├── experiments/             # Scripts to run tests with different configs  
│   └── experiment.py  
├── results/                 # Output of test runs for analysis
├── requirements.txt  
├── docker-compose.yaml  
└── README.md  
</pre>

## Quickstart Guide
### 1. Install Dependencies
```bash
pip install -r requirement.txt
```
Note: you may need to install Ollama separately  
### 2. Spin up appropriate Docker containers
Enter the following command into the terminal, with the specific database type replacing ***db_type***. Options are the following:
- redis
- chroma
- weaviate
```bash
docker compose --profile db_type up 
```
Note: you can spin up multiple containers at once by adding more ```--profile db_type up``` after the first

### 3. Input Data Sources
Place any .pdf files that you wish to be ingested and used for the RAG into the /data folder

### 4. Ingest the data
Depending on which database type you are using, you will need to run different commands:  
For Redis:   
```bash
python ingest.py
```  
For Chroma:  
```bash
python chroma_ingest.py
```  
For Weaviate:  
```bash
python rag_weaviate_ingest.py
```  

### 5. Start up Model
Again depending on the database type used, you will need to run one of the following different commands:
For Redis:   
```bash
python search.py
```  
For Chroma:   
```bash
python chroma_search.py
```  
For Weaviate:   
```bash
python rag_weaviate_search.py
```  

### 6. Ask Questions
From here you can simply type in the terminal in order to ask the model a question which it will answer based off of the provided data files.  
Use:
- exit: to close out of the model  
- clear: to remove the context of previous questions

## Requirements
See requirements.txt for full list of dependencies. Key items include:  
- ollama  
- weaviate-client
- redis
- chromadb  

You will also need Docker installed and running in order to create the containers based on the provided code. They can be set up without Docker, but it is not advised.

## Experimental Methodology

### Model Refinement Process
1. Test the PDF module first to check efficacy
2. Check the chunk size, overlay, and embedding model
3. Try tweaking K nearest neighbors in search

### Data Collection (METHODOLOGY, why we broke these variables into these sets)
1. Record for response quality chunk-size, chunk overlap, pre-processing (whitespace), embedding model, llm, prompt memory
2. Experiment Set #1: check chunk-size, chunk overlap, pre-proc, pdf for response quality
3. Experiment Set #2: embedding model, llm, prompt memory for response quality
4. Experiment Set #3: track db size, construction time, search time (need to track db type and llm)

## Credits
Built by Max Grotstein & Ben Pierce for experimenting with real-world RAG configurations and vector DB performance.