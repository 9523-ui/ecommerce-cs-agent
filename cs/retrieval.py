"""
统一检索层 —— TF-IDF / Embedding / Hybrid 三种后端可切换。

设计要点（面试可讲）：
1. 同一个 Retriever 类同时服务线上检索与规模实验。实验直接传入
   「31 条真 FAQ + N 条干扰项」构造实例，保证对照实验与线上跑的是同一套代码，
   避免「实验里一套实现、线上另一套」导致数字不可信。
2. Embedding 用 ONNX 量化的 BAAI/bge-small-zh-v1.5（512 维、90MB、CPU 推理）。
   不引入 torch，适配无独显的机器；模型懒加载，默认后端不加载模型。
3. 不同后端的分数尺度不同，因此各后端有独立阈值（由标注数据校准）。
   Hybrid 用 RRF（倒数排名融合）排序，只看排名、不看分数，天然免疫尺度失衡；
   返回的置信度仍是 Embedding 语义分数，因此与 embedding 共用同一阈值。

   曾踩过的坑：早期版本用「各后端分数 / 各自阈值后加权求和」，
   TF-IDF 除以 0.15 被放大 6.7 倍、Embedding 除以 0.5 只放大 2 倍，
   结果 TF-IDF 主导融合，Hybrid 反而比纯 Embedding 更差（改写 Recall@1 45% vs 77.5%）。

后端切换：环境变量 CS_RETRIEVER = tfidf | embedding | hybrid
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

jieba.setLogLevel(20)

BASE = Path(__file__).resolve().parent
FAQ_PATH = BASE / "data" / "faq.json"
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_CACHE = BASE.parent / ".model_cache"

# 阈值由 cs/eval/calibrate.py 用 50 正例 + 7 负例实测校准后填入
THRESHOLDS = {
    "tfidf": 0.15,       # 沿用线上已验证值（原评估集上命中/未命中完全可分）
    "embedding": 0.50,   # 实测最优 0.483，差异 <0.02，取整数便于解释
    # Hybrid 的 RRF 排序改变了 top1 的选择，分数分布与纯 embedding 不同，
    # 故独立校准：实测最优 0.415（区间 0.407~0.422，较窄，存在过拟合风险，
    # 后续应扩充标注集验证）。取 0.42。
    "hybrid": 0.42,
}

RRF_K0 = 60  # RRF 标准平滑常数

BACKEND = os.getenv("CS_RETRIEVER", "tfidf").strip().lower()
if BACKEND not in THRESHOLDS:
    BACKEND = "tfidf"

# ── 中文分词（jieba + 停用词，沿用已验证方案）──
_STOP = set(
    "的了吗呢吧啊呀哦嗯是你我他她它这那有在会和与及或就都也很还请问一下"
    "帮我给把被让对从到为以个些点种样些什么怎么怎样多少几能否要不要"
)


def _tokenize(text: str) -> list[str]:
    return [w for w in jieba.lcut(text) if len(w) > 1 and w not in _STOP]


# ── Embedding 模型：全局单例 + 双重检查锁，避免并发重复加载 ──
_model = None
_model_lock = threading.Lock()


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                # 国内网络走镜像；已缓存时不产生网络请求
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                from fastembed import TextEmbedding
                _model = TextEmbedding(
                    model_name=MODEL_NAME,
                    cache_dir=str(MODEL_CACHE),
                    threads=2,
                )
    return _model


_emb_cache: dict[str, np.ndarray] = {}
_emb_cache_lock = threading.Lock()


def embed_texts(texts: list[str], cache_key: str | None = None) -> np.ndarray:
    """返回已 L2 归一化的向量矩阵，点积即余弦相似度。

    cache_key 相同则复用矩阵：规模实验里 tfidf/embedding/hybrid 三个后端
    会在同一批文档上反复建索引，避免对 1 万条语料重复编码。
    """
    if cache_key is not None:
        with _emb_cache_lock:
            if cache_key in _emb_cache:
                return _emb_cache[cache_key]
    mat = np.asarray(list(get_model().embed(texts)), dtype=np.float32)
    if cache_key is not None:
        with _emb_cache_lock:
            _emb_cache[cache_key] = mat
    return mat


class Retriever:
    """可接收任意语料的检索器。payloads[i] 与 documents[i] 一一对应。"""

    def __init__(self, documents: list[str], payloads: list[dict], backend: str = BACKEND):
        if len(documents) != len(payloads):
            raise ValueError("documents 与 payloads 长度必须一致")
        self.documents = documents
        self.payloads = payloads
        self.backend = backend if backend in THRESHOLDS else "tfidf"
        self.threshold = THRESHOLDS[self.backend]

        # TF-IDF 索引（所有后端都建，Hybrid 需要）
        self.vec = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None, min_df=1)
        self.tmat = self.vec.fit_transform(documents)

        # Embedding 索引（按需；同一批文档跨后端复用）
        self.emat = None
        if self.backend in ("embedding", "hybrid"):
            key = hashlib.md5("\x1f".join(documents).encode("utf-8")).hexdigest()
            self.emat = embed_texts(documents, cache_key=key)

    def search(self, query: str, k: int = 3) -> list[tuple[dict, float]]:
        if self.backend == "hybrid":
            # RRF：只看两个后端给出的全库排名，避免分数尺度差异导致某一端主导
            t = cosine_similarity(self.vec.transform([query]), self.tmat)[0]
            qv = embed_texts([query])[0]
            e = self.emat @ qv
            rank_t = np.argsort(np.argsort(-t))
            rank_e = np.argsort(np.argsort(-e))
            rrf = 1.0 / (RRF_K0 + rank_t + 1) + 1.0 / (RRF_K0 + rank_e + 1)
            order = np.asarray(rrf).argsort()[::-1][:k]
            # 置信度用语义分数，与 embedding 后端共用阈值
            return [(self.payloads[i], float(e[i])) for i in order]

        if self.backend == "tfidf":
            scores = cosine_similarity(self.vec.transform([query]), self.tmat)[0]
        else:  # embedding
            qv = embed_texts([query])[0]
            scores = self.emat @ qv

        order = np.asarray(scores).argsort()[::-1][:k]
        return [(self.payloads[i], float(scores[i])) for i in order]


# ── 线上默认实例（懒加载：只有真正调用 search_faq 才建索引）──
_default = None
_default_lock = threading.Lock()


def _get_default() -> Retriever:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                faqs = json.loads(FAQ_PATH.read_text(encoding="utf-8"))["faqs"]
                docs = [f'{x["question"]} {x["answer"]}' for x in faqs]
                _default = Retriever(docs, faqs, BACKEND)
    return _default


def search_faq(query: str, k: int = 3) -> list[tuple[dict, float]]:
    """与旧接口完全一致：返回 [(faq_dict, score), ...]"""
    return _get_default().search(query, k)


def get_threshold() -> float:
    return THRESHOLDS[BACKEND]


def get_backend() -> str:
    return BACKEND


# 兼容旧代码：模块级常量（导入时确定）
MATCH_THRESHOLD = get_threshold()
