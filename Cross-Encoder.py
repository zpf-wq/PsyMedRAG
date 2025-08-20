from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util, InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from datetime import datetime
from torch.utils.data import IterableDataset
import gzip
import os
import tarfile
import tqdm
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

train_batch_size = 32
num_epochs = 1
pos_neg_ration = 4
max_train_samples = 200_000
model_name = 'distilroberta-base'

base_path = './'
model_save_path = base_path + 'finetuned_models/cross-encoder-' + model_name.replace("/", "-") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

model = CrossEncoder(model_name, num_labels=1, max_length=512, device="cuda")

data_folder = 'msmarco-data'
os.makedirs(data_folder, exist_ok=True)

corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        util.http_get(
            'https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz',
            tar_filepath
        )
    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage

queries = {}
queries_filepath = os.path.join('queries.train.tsv')
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

dev_samples = {}
num_dev_queries = 50
num_max_dev_negatives = 50

train_eval_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
if not os.path.exists(train_eval_filepath):
    util.http_get(
        'https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz',
        train_eval_filepath
    )

with gzip.open(train_eval_filepath, 'rt') as fIn:
    for line in fIn:
        qid, pos_id, neg_id = line.strip().split()

        if qid not in dev_samples and len(dev_samples) < num_dev_queries:
            dev_samples[qid] = {
                'query': queries[qid],
                'positive': [],
                'negative': []
            }

        if qid in dev_samples:
            if len(dev_samples[qid]['positive']) < 1:   # 一个 query 只留 1 个正样本
                dev_samples[qid]['positive'].append(corpus[pos_id])

            if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
                dev_samples[qid]['negative'].append(corpus[neg_id])


class MSMARCOStreamDataset(IterableDataset):
    def __init__(self, queries, corpus, filepath, dev_samples,
                 pos_neg_ration=4, max_samples=None):
        self.queries = queries
        self.corpus = corpus
        self.filepath = filepath
        self.dev_samples = dev_samples
        self.pos_neg_ration = pos_neg_ration
        self.max_samples = max_samples

    def __iter__(self):
        cnt = 0
        with gzip.open(self.filepath, 'rt') as fIn:
            for line in tqdm.tqdm(fIn, unit_scale=True, desc="Streaming dataset"):
                qid, pos_id, neg_id = line.strip().split()

                # 跳过 dev 集
                if qid in self.dev_samples:
                    continue

                query = self.queries[qid]
                if (cnt % (self.pos_neg_ration + 1)) == 0:
                    passage = self.corpus[pos_id]
                    label = 1
                else:
                    passage = self.corpus[neg_id]
                    label = 0

                yield InputExample(texts=[query, passage], label=label)

                cnt += 1
                if self.max_samples and cnt >= self.max_samples:
                    break


train_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train.tsv.gz')
if not os.path.exists(train_filepath):
    util.http_get(
        'https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz',
        train_filepath
    )

train_dataset = MSMARCOStreamDataset(
    queries=queries,
    corpus=corpus,
    filepath=train_filepath,
    dev_samples=dev_samples,
    pos_neg_ration=pos_neg_ration,
    max_samples=max_train_samples
)

train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size)
evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=500,
    warmup_steps=1000,
    output_path=model_save_path,
    use_amp=True,
)
