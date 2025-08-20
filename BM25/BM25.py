import math
import json
import jieba
from typing import List
import numpy as np
from multiprocessing import Pool, cpu_count

"""
All of these algorithms have been taken from the paper:
Trotmam et al, Improvements to BM25 and Language Models Examined

Here we implement all the BM25 variations mentioned. 
"""


class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0  # 文档总数
        self.avgdl = 0  # 平均文档长度
        self.doc_freqs = []  # 每篇文档的词频字典
        self.idf = {}  # 每个词的逆文档频率
        self.doc_len = []   #每篇文档长度
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)  # 统计文档长度、词频和词出现文档数
        self._calc_idf(nd)  # 计算每个词的 IDF

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))  # 保存每篇文档长度（词数）
            num_doc += len(document)  # 计算平均文档长度

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)  # 统计每篇文档的词频

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size  # 计算平均文档长度
        return nd

    def _tokenize_corpus(self, corpus):  # 多进程批量分词语料库
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):  # 返回 query 与语料库所有文档的 BM25 分数
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):  # 只计算 query 与指定文档子集的分数
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
        # 检查传入的文档列表长度是否和语料库一致
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)  # 调用 get_scores(query) 获取分数。
        top_n = np.argsort(scores)[::-1][:n]  # 按分数从高到低取前 n 个索引
        return [documents[i] for i in top_n]  # 返回对应的文档内容列表


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1  # BM25 的调节参数，控制词频饱和速度。
        self.b = b  # 控制文档长度归一化
        self.epsilon = epsilon  # 防止 IDF 为负时设置一个下限
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()


class BM25L(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=0.5):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq + 0.5)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            ctd = q_freq / (1 - self.b + self.b * doc_len / self.avgdl)
            score += (self.idf.get(q) or 0) * (self.k1 + 1) * (ctd + self.delta) / \
                     (self.k1 + ctd + self.delta)
        return score.tolist()


class BM25Plus(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, delta=1):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(self.corpus_size + 1) - math.log(freq)
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq * (self.k1 + 1)) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score.tolist()


def BM25_sort(file_path: str, query: str, k=10) -> list[tuple]:
    with open(file_path, "r", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]

    tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = list(jieba.cut(query))

    doc_scores = bm25.get_scores(tokenized_query)

    result = [(corpu, doc_score) for corpu, doc_score in zip(corpus, doc_scores)]

    return sorted(result, key=lambda x: x[1], reverse=True)[0: k]


def load_json(json_path: str) -> dict:
    """读取 json，返回 input→output 的映射"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["input"]: item["output"] for item in data}


def BM25_retriever(query, k=10):
    file_path = "/Users/zhangpengfei/PycharmProjects/RAG/BM25/input.txt"
    json_path = "/Users/zhangpengfei/PycharmProjects/RAG/BM25/medical_meadow_wikidoc.json"
    res = BM25_sort(file_path, query, k)

    # 载入 input→output 映射
    qa_map = load_json(json_path)

    # 输出检索到的问题和对应答案
    retriever_result = []
    for q, score in res:
        answer = qa_map.get(q, "⚠️ 未找到答案")
        retriever_result.append(answer)
    return retriever_result


# if __name__ == "__main__":
#     file_path = "/Users/zhangpengfei/PycharmProjects/RAG/BM25/input.txt"
#     json_path = "/Users/zhangpengfei/PycharmProjects/RAG/BM25/medical_meadow_wikidoc.json"
#
#     query = "A brief summary of Hashimoto's thyroiditis"
#     print(BM25_retriever(file_path, json_path, query, k=10))

    # corpus = [
    #     "清晨的海边，天空刚刚泛起鱼肚白，薄雾像一层轻纱笼罩在静谧的水面上。远处的渔船缓缓驶出港口，桅杆在雾中若隐若现。海浪轻拍礁石，溅起细碎的浪花，发出低沉而有节奏的声响。偶尔传来几声海鸟的鸣叫，打破了清晨的寂静，带来一丝生机。空气中弥漫着湿润而带咸味的海风，让人忍不住深吸一口，仿佛能尝到海的味道。",
    #
    #     "在老旧的图书馆角落里，一位满头白发的老人正专注地翻阅一本泛黄的诗集。木质书架散发出淡淡的陈年气息，仿佛每一道划痕都记录着岁月的痕迹。阳光透过高高的拱形窗户洒下，斑驳的光影落在书页上，把字句镀上一层温暖的光泽。老人时而轻轻点头，时而微微皱眉，仿佛在和诗人隔空对话。翻页的沙沙声与窗外树叶的摇曳声交织在一起，静谧而动人。",
    #
    #     "一辆鲜红的越野车在蜿蜒的山路上疾驰，车轮碾过碎石，扬起大片尘土，被晨风卷向山谷。驾驶座上的年轻人双手紧握方向盘，目光专注而炽热，脸上写满了兴奋与一丝紧张。山路两侧的野花在风中微微点头，仿佛在为这场速度的较量加油助威。拐弯处，阳光透过稀疏的树叶洒在车身上，映出跳动的光影。远方，山峰在薄雾中若隐若现，像在召唤他继续前行。",
    #
    #     "在幽静的图书馆深处，一位须发皆白的老人坐在磨得光亮的木椅上，手中捧着一本封皮磨损的旧书。空气中弥漫着纸张与木头交织的陈香，仿佛连呼吸都能触到岁月的纹理。阳光透过高窗的彩色玻璃洒下，折射出细碎的光斑，在书页间缓缓流淌。老人偶尔停下阅读，抬眼凝望天花板的雕花梁木，仿佛在寻找某段久远的记忆。书页翻动的声音，与远处钟摆的滴答声交织成一首安静的时光之歌。",
    #
    #     "我精心挑选了一条丝绸围巾作为给妈妈的生日礼物，包裹在柔软的礼盒里。每一层纸都小心折好，仿佛把我的心意都包裹其中。妈妈打开礼物的那一刻，她的眼睛微微闪亮，嘴角泛起温暖的笑容。我能感觉到，她的喜悦穿透礼物，直抵我的心里，让我忍不住也跟着笑了。",
    #
    #     "夕阳渐渐沉入海平面，天边被染成橘红与紫色的渐变。海风轻拂脸颊，带来咸味与海藻的香气。礁石间的浪花反射着落日的余晖，闪烁着金色光点。偶尔有渔船缓慢划过远方的水面，发出轻微的摇橹声。海鸟盘旋在天空，鸣叫声与浪声交织成一首悠长的黄昏旋律，让人心神宁静。",
    #
    #     "厚重的乌云笼罩在城市上空，雨点敲打窗户，发出轻柔而均匀的节奏。在图书馆的木质桌旁，一位中年学者低头专注阅读着厚厚的历史书籍。空气中混合着纸张的香气和雨水带来的清新湿气。偶尔翻页的沙沙声与窗外雨声交错，让整个空间显得静谧而充满思绪的流动。",
    #
    #     "清晨的山林中，薄雾在树梢间缭绕，阳光穿过枝叶洒下斑驳的光影。小径蜿蜒而上，两侧野花点缀其间，露珠在叶片上闪烁。远处传来松涛与鸟鸣声，空气中混合着泥土与青草的清香。一个年轻人背着轻便的登山包，缓步沿路前行，偶尔停下拍照，仿佛想把这一刻的宁静与美好留存下来。",
    #
    #     "深夜的实验室里，只有柔和的白色灯光映照着长长的操作台。玻璃器皿里闪烁着微弱光芒，蒸汽缓缓上升。年轻的研究员戴着护目镜，专注地操作仪器，记录实验数据。空气中弥漫着轻微的化学气息，伴随着试剂滴入烧瓶的细微声响，显得神秘而专注。墙上的时钟滴答作响，与实验的节奏默契呼应。",
    #
    #     "过节时，妈妈给了我一个礼物，那是一个小巧的木质首饰盒，里面整齐地摆放着我喜欢的小饰品。她轻轻拍了拍我的手，说：“这是给你的。”我打开首饰盒，闻到木头散发的淡淡香气，感受到妈妈满满的爱意。那一刻，我的心被暖流包围，仿佛这个小小的礼物里装满了她无声的关怀和陪伴。"
    # ]
    #
    # tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
    #
    # bm25 = BM25Okapi(tokenized_corpus)
    #
    # query = "妈妈 礼物"
    # tokenized_query = list(jieba.cut(query))
    #
    # doc_scores = bm25.get_scores(tokenized_query)
    #
    # result = [(corpu, doc_score) for corpu, doc_score in zip(corpus, doc_scores)]
    # print(sorted(result, key=lambda x: x[1], reverse=True)[0: 3])

    # result = bm25.get_top_n(tokenized_query, corpus, n=2)
    # print(result)

