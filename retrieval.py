import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite, log1p

import numpy as np
from video_processing import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.casefold())


class BM25:
    """BM25 with positive Robertson/Lucene IDF and document-length normalization."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.documents = [Counter(tokenize(text)) for text in documents]
        self.lengths = np.array([sum(doc.values()) for doc in self.documents], dtype=float)
        self.average_length = float(self.lengths.mean()) if documents else 0.0
        self.k1, self.b = k1, b
        self.postings = defaultdict(list)
        for i, doc in enumerate(self.documents):
            for term, count in doc.items():
                self.postings[term].append((i, count))

    def scores(self, query: str) -> np.ndarray:
        result = np.zeros(len(self.documents), dtype=float)
        if not self.average_length:
            return result
        for term in set(tokenize(query)):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            idf = log1p((len(self.documents) - len(postings) + 0.5) / (len(postings) + 0.5))
            for i, frequency in postings:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.average_length)
                result[i] += idf * frequency * (self.k1 + 1) / (frequency + norm)
        return result


def reciprocal_rank_fusion(
    rankings: dict[str, list[int]], weights: dict[str, float], k: int = 60
) -> list[tuple[int, float, dict[str, int]]]:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    scores = defaultdict(float)
    ranks = defaultdict(dict)
    for name, ranking in rankings.items():
        weight = weights.get(name, 1.0)
        if not isfinite(weight) or weight < 0:
            raise ValueError("RRF weights must be finite and nonnegative")
        if weight == 0:
            continue
        seen = set()
        for item in ranking:
            if item in seen:
                continue
            seen.add(item)
            rank = len(seen)
            scores[item] += weight / (k + rank)
            ranks[item][name] = rank
    return [(i, scores[i], ranks[i]) for i in sorted(scores, key=lambda i: (-scores[i], i))]


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a finite matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def validate_retrieval(
    mode: str, top_k: int, candidate_k: int, dense_weight: float, sparse_weight: float
) -> None:
    if mode not in {"hybrid", "dense", "bm25"}:
        raise ValueError("mode must be hybrid, dense, or bm25")
    if top_k <= 0 or candidate_k <= 0:
        raise ValueError("Retrieval limits must be positive")
    if any(not isfinite(w) or w < 0 for w in (dense_weight, sparse_weight)):
        raise ValueError("Retrieval weights must be finite and nonnegative")
    if (
        (mode == "dense" and dense_weight == 0)
        or (mode == "bm25" and sparse_weight == 0)
        or (dense_weight + sparse_weight == 0)
    ):
        raise ValueError("At least one active retrieval branch must have a positive weight")


class HybridRetriever:
    def __init__(
        self, chunks: list[Chunk], text_vectors: np.ndarray, frame_vectors: np.ndarray, models
    ):
        self.chunks = chunks
        self.text_vectors = unit_rows(text_vectors)
        self.frame_vectors = unit_rows(frame_vectors)
        if len(self.text_vectors) != len(chunks):
            raise ValueError("Text embedding count does not match chunks")
        self.frame_owners = np.array([i for i, c in enumerate(chunks) for _ in c.frames], dtype=int)
        if len(self.frame_vectors) != len(self.frame_owners):
            raise ValueError("Image embedding count does not match frames")
        self.models = models
        self.sparse = BM25([chunk.text for chunk in chunks])

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "hybrid",
        candidate_k: int = 50,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[Hit]:
        validate_retrieval(mode, top_k, candidate_k, dense_weight, sparse_weight)
        if not query.strip():
            raise ValueError("A nonempty query is required")
        limit = min(len(self.chunks), max(candidate_k, top_k))
        rankings, weights = {}, {}
        if mode != "bm25" and dense_weight > 0:
            query_text = unit_rows(self.models.encode_text([query]))[0]
            text_scores = self.text_vectors @ query_text
            valid = [i for i, chunk in enumerate(self.chunks) if chunk.text.strip()]
            rankings["dense_text"] = sorted(valid, key=lambda i: (-float(text_scores[i]), i))[
                :limit
            ]
            query_image = unit_rows(np.asarray([self.models.encode_visual_query(query)]))[0]
            frame_scores = self.frame_vectors @ query_image
            image_scores = np.full(len(self.chunks), -np.inf)
            np.maximum.at(image_scores, self.frame_owners, frame_scores)
            valid = [i for i, chunk in enumerate(self.chunks) if chunk.frames]
            rankings["dense_image"] = sorted(valid, key=lambda i: (-float(image_scores[i]), i))[
                :limit
            ]
            weights.update(dense_text=dense_weight / 2, dense_image=dense_weight / 2)
        if mode != "dense" and sparse_weight > 0:
            sparse_scores = self.sparse.scores(query)
            valid = [i for i, score in enumerate(sparse_scores) if score > 0]
            rankings["bm25"] = sorted(valid, key=lambda i: (-float(sparse_scores[i]), i))[:limit]
            weights["bm25"] = sparse_weight
        return [
            Hit(self.chunks[i], score, ranks)
            for i, score, ranks in reciprocal_rank_fusion(rankings, weights)[:top_k]
        ]
