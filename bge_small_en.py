import json
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

model_id = "BAAI/bge-small-en"
model = SentenceTransformer(model_id)

TRAIN_DATASET_FPATH = '/Users/zhangpengfei/PycharmProjects/RAG/Bi_Encoder/rag_test.json'
VAL_DATASET_FPATH = '/Users/zhangpengfei/PycharmProjects/RAG/Bi_Encoder/rag_train.json'

BATCH_SIZE = 10

with open(TRAIN_DATASET_FPATH, 'r+') as f:
    train_dataset = json.load(f)

with open(VAL_DATASET_FPATH, 'r+') as f:
    val_dataset = json.load(f)

dataset = train_dataset

corpus = dataset['corpus']
queries = dataset['queries']
relevant_docs = dataset['relevant_docs']

examples = []
for query_id, query in queries.items():
    node_id = relevant_docs[query_id][0]
    text = corpus[node_id]
    example = InputExample(texts=[query, text])
    examples.append(example)

loader = DataLoader(
    examples, batch_size=BATCH_SIZE
)

loss = losses.MultipleNegativesRankingLoss(model)

dataset = val_dataset

corpus = dataset['corpus']
queries = dataset['queries']
relevant_docs = dataset['relevant_docs']

evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs)

EPOCHS = 2

warmup_steps = int(len(loader) * EPOCHS * 0.1)

model.fit(
    train_objectives=[(loader, loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    output_path='exp_finetune_20',
    show_progress_bar=True,
    evaluator=evaluator,
    evaluation_steps=50,
)
