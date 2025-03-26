# ds4300-practical-02

# TODO List

1. Get document list together (including Redis/Mongo documentation)
2. Make cheat sheet and feed into document list
3. Setup code for pre-processing
4. Set up code for data testing and collection
5. Track performance across chunk-size, chunk overlap, pre-processing (whitespace), embedding model
6. Test with other factors (can expand on this process)

# Model Refinement Process
1. Test the PDF module first to check efficacy
- Determined that Fitz is fine
2. Check the chunk size, overlay, and embedding model
3. Try tweaking K nearest neighbors in search

# Data Collection
1. Record for response quality chunk-size, chunk overlap, pre-processing (whitespace), embedding model, llm, prompt memory
2. Experiment Set #1: check chunk-size, chunk overlap, pre-proc, pdf for response quality
3. Experiment Set #2: embedding model, llm, prompt memory for response quality
4. Experiment Set #3: track db size, construction time, search time (need to track db type and llm)