from Bi_Encoder.VectorDB.Text_process import VectorDB_Retriever


def Bi_Encoder_Retriever(query, top_k=5):
    vectordb = VectorDB_Retriever()
    retrieved_chunks = vectordb.retrieve(query, top_k)
    return retrieved_chunks


# query = "A brief summary of Hashimoto's thyroiditis"
# print(Bi_Encoder_Retriever(query))

