from Bi_Encoder.Bi_Encoder_retriever import Bi_Encoder_Retriever
from BM25.BM25 import BM25_retriever
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
import torch


class Retriever:
    def all_retriever(self, query: str, score_threshold=0.5):

        bi_Encoder_Retriever = Bi_Encoder_Retriever(query, top_k=10)

        BM25_Retriever = BM25_retriever(query, k=10)

        Retriever = list(set(BM25_Retriever + bi_Encoder_Retriever))

        trained_model_path = "/Users/zhangpengfei/PycharmProjects/RAG/Cross-Encoder/cross-encoder-distilroberta-base-2025-08-17_03-01-41"  # 或你保存的路径
        model = CrossEncoder(trained_model_path)

        test_pairs = []
        for p in Retriever:
            test_pairs.append([query, p])

        batch_size = 16
        scores = model.predict(test_pairs, batch_size=batch_size)

        scored_results = sorted(zip(scores, test_pairs), key=lambda x: x[0], reverse=True)
        retriever = ""
        results = []
        for score, (q, p) in scored_results:
            if score > score_threshold:
                results.append(p)
        #         retriever += p
        # results.append(retriever)
        print(results)
        return results


# query = "What information can you provide about Maple Syrup Urine Disease?"
# R = Retriever()
# result = R.all_retriever(query, score_threshold=0.5)
# print(result)

# Bi_Encoder_Retriever = Bi_Encoder_Retriever(query, top_k=10)
#
# BM25_Retriever = BM25_retriever(query, k=10)
#
# Retriever = list(set(BM25_Retriever + Bi_Encoder_Retriever))
#
# trained_model_path = "/Users/zhangpengfei/PycharmProjects/RAG/Cross-Encoder/cross-encoder-distilroberta-base-2025-08-17_03-01-41"  # 或你保存的路径
# model = CrossEncoder(trained_model_path)
#
# test_pairs = []
# for p in Retriever:
#     test_pairs.append([query, p])
#
# batch_size = 16
# scores = model.predict(test_pairs, batch_size=batch_size)
#
# scored_results = sorted(zip(scores, test_pairs), key=lambda x: x[0], reverse=True)
#
# for score, (q, p) in scored_results:
#     if score > 0.5:
#         print(f"Query: {q}")
#         print(f"Passage: {p}")
#         print(f"Score: {score:.4f}")
#         print("------")
