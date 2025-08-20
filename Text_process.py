from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # 显示进度条


class VectorDB_Retriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer(
            "/Users/zhangpengfei/bge-large-zh",
        )

    def split_into_chunks(self, doc_file: str) -> List[str]:
        with open(doc_file, "r", encoding="utf-8") as file:
            content = file.read()
        return [chunk for chunk in content.split("\n") if chunk.strip()]

    def embed_chunk(self, chunk: str) -> List[float]:
        embedding = self.embedding_model.encode(chunk, normalize_embeddings=True)
        return embedding.tolist()

    def save_embeddings(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        chromadb_client = chromadb.PersistentClient("/Users/zhangpengfei/PycharmProjects/RAG/Bi_Encoder/VectorDB/chroma.db")
        chromadb_collection = chromadb_client.get_or_create_collection(name="default")

        for i, (chunk, embedding) in enumerate(tqdm(zip(chunks, embeddings), total=len(chunks))):
            chromadb_collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[str(i)]
            )

    def retrieve(self, query: str, top_k: int) -> List[str]:
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True).tolist()
        chromadb_client = chromadb.PersistentClient("/Users/zhangpengfei/PycharmProjects/RAG/Bi_Encoder/VectorDB/chroma.db")
        chromadb_collection = chromadb_client.get_or_create_collection(name="default")
        results = chromadb_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results['documents'][0]


# vectordb = VectorDB_Retriever()
#
# chunks = vectordb.split_into_chunks("/Users/zhangpengfei/PycharmProjects/RAG/DataBase/output.txt")
# embeddings = vectordb.embedding_model.encode(
#         chunks,
#         normalize_embeddings=True,
#         show_progress_bar=True
#     ).tolist()
# vectordb.save_embeddings(chunks, embeddings)
