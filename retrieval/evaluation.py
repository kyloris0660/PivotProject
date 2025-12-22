import logging
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .config import RetrievalConfig


def _recall_at(pred: List[str], gt: str, k: int) -> int:
    return int(gt in pred[:k])


def evaluate_retrieval(
    caption_pairs: Sequence[Tuple[str, str]],
    caption_embs: np.ndarray,
    image_embs: np.ndarray,
    pivot_vectors: np.ndarray,
    image_ids: List[str],
    hnsw_index,
    config: RetrievalConfig,
) -> Dict[str, float]:
    recalls = {1: 0, 5: 0, 10: 0}
    hnsw_times: List[float] = []
    rerank_times: List[float] = []

    for idx, (_, gt_image_id) in tqdm(
        enumerate(caption_pairs), total=len(caption_pairs), desc="eval"
    ):
        query_vec = caption_embs[idx]
        pivot_query = (1.0 - np.dot(pivot_vectors, query_vec)).reshape(1, -1)

        t0 = time.perf_counter()
        cand_labels, _ = hnsw_index.knn_query(pivot_query, k=config.topC)
        hnsw_times.append((time.perf_counter() - t0) * 1000)

        cand_ids = cand_labels[0]
        rerank_start = time.perf_counter()
        sims = image_embs[cand_ids] @ query_vec
        top_indices = np.argsort(-sims)[: config.k]
        result_ids = [image_ids[cand_ids[i]] for i in top_indices]
        rerank_times.append((time.perf_counter() - rerank_start) * 1000)

        for k in [1, 5, 10]:
            top_k_ids = result_ids[: min(k, len(result_ids))]
            recalls[k] += _recall_at(top_k_ids, gt_image_id, k)

    n = len(caption_pairs)
    metrics = {
        "Recall@1": recalls[1] / n,
        "Recall@5": recalls[5] / n,
        "Recall@10": recalls[10] / n,
        "avg_hnsw_ms": float(np.mean(hnsw_times)),
        "avg_rerank_ms": float(np.mean(rerank_times)),
        "avg_total_ms": float(np.mean(hnsw_times) + np.mean(rerank_times)),
    }
    logging.info(
        "Recall@1=%.4f Recall@5=%.4f Recall@10=%.4f | HNSW %.2f ms | rerank %.2f ms",
        metrics["Recall@1"],
        metrics["Recall@5"],
        metrics["Recall@10"],
        metrics["avg_hnsw_ms"],
        metrics["avg_rerank_ms"],
    )
    return metrics
