from Text_process import VectorDB_Retriever

query = "A brief summary of Hashimoto's thyroiditis"
vectordb = VectorDB_Retriever()
retrieved_chunks = vectordb.retrieve(query, 5)

for i, chunk in enumerate(retrieved_chunks):
    print(f"[{i}] {chunk}\n")
