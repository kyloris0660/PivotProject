"""Benchmark baseline methods on Flickr30k.

Compares three retrieval methods on the same image/query set:
 A) Brute-force exact (original space dot product)
 B) HNSW on original embeddings (cosine)
 C) Pivot + HNSW (pivot-space ANN, then original-space rerank)

Outputs per-method recall/latency/index stats, saves CSV/JSON and a latency-vs-recall plot.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import hnswlib

from retrieval.config import RetrievalConfig
from retrieval.data import build_caption_pairs, load_flickr30k
from retrieval.embeddings import (
    load_clip,
    load_or_compute_caption_embeddings,
    load_or_compute_image_embeddings,
)
from retrieval.pivots import compute_pivot_coordinates, select_pivots
from retrieval.utils import ensure_dir, save_json, set_seed, setup_logging


def parse_list(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",") if x]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark brute-force vs HNSW vs Pivot+HNSW"
    )
    p.add_argument("--source", default="hf", choices=["hf"], help="data source")
    p.add_argument("--split", default="test")
    p.add_argument("--model_name", default="openai/clip-vit-base-patch32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--max_captions", type=int, default=None)
    p.add_argument("--n_queries", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--pivot_m", type=int, default=16)
    p.add_argument("--topC", type=int, default=1200)
    p.add_argument("--pivot_efSearch", type=int, default=128)
    p.add_argument("--pivot_M", type=int, default=24)
    p.add_argument("--orig_hnsw_efSearch", type=int, default=128)
    p.add_argument("--orig_hnsw_M", type=int, default=24)
    p.add_argument("--pivot_norm", choices=["none", "zscore"], default="none")
    p.add_argument("--pivot_weight", choices=["none", "variance"], default="none")
    p.add_argument("--batch_size_text", type=int, default=64)
    p.add_argument("--batch_size_image", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pivot_sample", type=int, default=5000)
    p.add_argument("--efc", dest="ef_construction", type=int, default=200)
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument(
        "--warmup", type=int, default=10, help="warmup queries before timing"
    )
    return p.parse_args()


def sample_queries(
    pairs: Sequence[Tuple[str, str]], n_queries: int | None, seed: int
) -> Tuple[List[Tuple[str, str]], np.ndarray]:
    total = len(pairs)
    if n_queries is None or n_queries >= total:
        return list(pairs), np.arange(total, dtype=int)
    rng = np.random.RandomState(seed)
    idx = rng.choice(total, size=n_queries, replace=False)
    idx.sort()
    sampled = [pairs[i] for i in idx]
    return sampled, idx


def compute_recalls(
    pred_ids: List[List[str]], gt_ids: List[str], ks: Sequence[int] = (1, 5, 10)
) -> Dict[str, float]:
    counts = {k: 0 for k in ks}
    for preds, gt in zip(pred_ids, gt_ids):
        for k in ks:
            if gt in preds[: min(k, len(preds))]:
                counts[k] += 1
    n = len(gt_ids)
    return {f"Recall@{k}": counts[k] / n for k in ks}


def compute_candidate_hits(
    labels: np.ndarray, gt_indices: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Compute candidate recall and rank positions within candidate lists.

    labels: (Q, C) candidate indices; gt_indices: (Q,) ground-truth image indices.
    Returns (cand_recall, ranks) where ranks are 0-based positions or -1 if miss.
    """
    if labels.shape[0] != gt_indices.shape[0]:
        raise ValueError("labels and gt_indices must have the same number of queries")

    matches = labels == gt_indices[:, None]
    hit_mask = matches.any(axis=1)
    ranks = np.where(hit_mask, np.argmax(matches, axis=1), -1)
    cand_recall = float(hit_mask.mean()) if hit_mask.size else 0.0
    return cand_recall, ranks


def brute_force_search(
    caption_embs: np.ndarray,
    image_embs: np.ndarray,
    image_ids: List[str],
    k: int,
    batch_q: int,
    warmup: int,
) -> Tuple[List[List[str]], float]:
    preds: List[List[str]] = []
    timings: List[Tuple[float, int]] = []  # (seconds, batch_size)
    n_queries = caption_embs.shape[0]
    for start in range(0, n_queries, batch_q):
        end = min(start + batch_q, n_queries)
        batch = caption_embs[start:end]
        t0 = time.perf_counter()
        sims = batch @ image_embs.T  # (B, N)
        topk_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
        # sort within topk
        topk_scores = np.take_along_axis(sims, topk_idx, axis=1)
        order = np.argsort(-topk_scores, axis=1)
        sorted_idx = np.take_along_axis(topk_idx, order, axis=1)
        timings.append((time.perf_counter() - t0, batch.shape[0]))
        for row in sorted_idx:
            preds.append([image_ids[j] for j in row])
    # drop warmup timings from avg
    start_idx = (warmup + batch_q - 1) // batch_q if warmup > 0 else 0
    eff = timings[start_idx:]
    if eff:
        total_time = sum(t for t, _ in eff)
        total_q = sum(b for _, b in eff)
        avg_ms = (total_time / total_q) * 1000
    else:
        avg_ms = 0.0
    return preds, avg_ms


def build_hnsw_index(
    data: np.ndarray,
    space: str,
    M: int,
    ef_construction: int,
    ef_search: int,
    index_path: Path,
) -> Tuple[hnswlib.Index, float, int]:
    ensure_dir(index_path.parent)
    index = hnswlib.Index(space=space, dim=data.shape[1])
    t0 = time.perf_counter()
    index.init_index(max_elements=data.shape[0], ef_construction=ef_construction, M=M)
    index.add_items(data, np.arange(data.shape[0]))
    index.set_ef(ef_search)
    build_time = time.perf_counter() - t0
    index.save_index(str(index_path))
    size_bytes = os.path.getsize(index_path)
    return index, build_time, size_bytes


def hnsw_search(
    index: hnswlib.Index,
    queries: np.ndarray,
    topk: int,
    warmup: int,
) -> float:
    timings: List[float] = []
    for q in queries:
        t0 = time.perf_counter()
        index.knn_query(q, k=topk)
        timings.append(time.perf_counter() - t0)
    effective = timings[warmup:] if warmup > 0 else timings
    return (np.mean(effective) * 1000) if effective else 0.0


def hnsw_search_batch(
    index: hnswlib.Index,
    queries: np.ndarray,
    topk: int,
    batch_q: int,
    warmup: int,
) -> Tuple[np.ndarray, float]:
    # Ensure ef >= topk to avoid hnswlib contiguous buffer errors when k is large.
    try:
        current_ef = index.get_ef()
    except AttributeError:
        current_ef = None
    target_ef = max(topk * 2, current_ef or 0, topk)
    if current_ef is None or current_ef < target_ef:
        index.set_ef(target_ef)

    labels_all: List[np.ndarray] = []
    timings: List[Tuple[float, int]] = []
    n_queries = queries.shape[0]
    warmup_queries = warmup
    for start in range(0, n_queries, batch_q):
        end = min(start + batch_q, n_queries)
        batch = queries[start:end]
        t0 = time.perf_counter()
        labels, _ = index.knn_query(batch, k=topk)
        timings.append((time.perf_counter() - t0, batch.shape[0]))
        labels_all.append(labels)
    start_idx = (warmup_queries + batch_q - 1) // batch_q if warmup_queries > 0 else 0
    eff = timings[start_idx:]
    if eff:
        total_time = sum(t for t, _ in eff)
        total_q = sum(b for _, b in eff)
        avg_ms = (total_time / total_q) * 1000
    else:
        avg_ms = 0.0
    return np.vstack(labels_all), avg_ms


def apply_pivot_transform(
    coords: np.ndarray,
    norm: str,
    weight: str,
    stats: Dict[str, np.ndarray] | None = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray] | None]:
    if norm == "none" and weight == "none":
        return coords, stats

    if stats is None:
        mean = coords.mean(axis=0)
        std = coords.std(axis=0) + 1e-6
    else:
        mean = stats["mean"]
        std = stats["std"]

    if norm == "zscore":
        coords = (coords - mean) / std

    if weight == "variance":
        weight_vec = 1.0 / std  # variance-based scaling
        coords = coords * weight_vec

    return coords, {"mean": mean, "std": std}


def orig_hnsw_method(
    caption_embs: np.ndarray,
    image_embs: np.ndarray,
    image_ids: List[str],
    gt_indices: np.ndarray,
    base_config: RetrievalConfig,
    args: argparse.Namespace,
) -> Dict[str, float | int | str]:
    cfg = replace(
        base_config,
        ef_search=args.orig_hnsw_efSearch,
        M=args.orig_hnsw_M,
        force_recompute=args.force_recompute,
    )

    subset_tag = f"_n{image_embs.shape[0]}" if cfg.max_images is not None else ""
    index_path = cfg.cache_path("index") / (
        f"benchmark_orig_hnsw_{cfg.split}_M{cfg.M}_efc{cfg.ef_construction}_efs{cfg.ef_search}{subset_tag}.bin"
    )
    index, build_time, index_size = build_hnsw_index(
        image_embs,
        space="cosine",
        M=cfg.M,
        ef_construction=cfg.ef_construction,
        ef_search=cfg.ef_search,
        index_path=index_path,
    )

    labels, hnsw_ms = hnsw_search_batch(
        index,
        caption_embs,
        topk=cfg.k,
        batch_q=args.batch_size_text,
        warmup=args.warmup,
    )
    preds = [[image_ids[j] for j in row] for row in labels]
    cand_recall, cand_ranks = compute_candidate_hits(labels, gt_indices)
    hit_ranks = cand_ranks[cand_ranks >= 0]
    hit_rank_mean = float(hit_ranks.mean()) if hit_ranks.size else -1.0
    hit_rank_median = float(np.median(hit_ranks)) if hit_ranks.size else -1.0

    result: Dict[str, float | int | str] = {
        "method": "orig_hnsw",
        "avg_total_ms": hnsw_ms,
        "hnsw_ms": hnsw_ms,
        "pivot_map_ms": 0.0,
        "rerank_ms": 0.0,
        "brute_force_ms": 0.0,
        "build_index_time_sec": build_time,
        "index_size_bytes": index_size,
        "m": 0,
        "topC": cfg.k,
        "efSearch": cfg.ef_search,
        "M": cfg.M,
        "pivot_norm": "none",
        "pivot_weight": "none",
        "CandRecall@topC": cand_recall,
        "cand_hit_rank_mean": hit_rank_mean,
        "cand_hit_rank_median": hit_rank_median,
    }
    result["preds"] = preds
    return result


def pivot_hnsw_method(
    caption_embs: np.ndarray,
    image_embs: np.ndarray,
    image_ids: List[str],
    gt_indices: np.ndarray,
    base_config: RetrievalConfig,
    args: argparse.Namespace,
) -> Dict[str, float | int | str]:
    # Build a config for pivot path
    cfg = replace(
        base_config,
        m=args.pivot_m,
        topC=args.topC,
        ef_search=args.pivot_efSearch,
        M=args.pivot_M,
        force_recompute=args.force_recompute,
    )

    pivots, _ = select_pivots(image_embs, cfg)
    pivot_coords = compute_pivot_coordinates(image_embs, pivots, cfg, cfg.split)

    pivot_coords, stats = apply_pivot_transform(
        pivot_coords, args.pivot_norm, args.pivot_weight, None
    )

    index_path = cfg.cache_path("index") / (
        f"benchmark_pivot_hnsw_{cfg.split}_m{cfg.m}_M{cfg.M}_efc{cfg.ef_construction}_efs{cfg.ef_search}_seed{cfg.seed}_n{image_embs.shape[0]}"
        f"_{args.pivot_norm}_{args.pivot_weight}.bin"
    )
    index, build_time, index_size = build_hnsw_index(
        pivot_coords,
        space="l2",
        M=cfg.M,
        ef_construction=cfg.ef_construction,
        ef_search=cfg.ef_search,
        index_path=index_path,
    )

    # Map queries to pivot space (with optional normalization/weight)
    t0 = time.perf_counter()
    pivot_queries = 1.0 - caption_embs @ pivots.T
    pivot_queries, _ = apply_pivot_transform(
        pivot_queries, args.pivot_norm, args.pivot_weight, stats
    )
    pivot_map_ms = (time.perf_counter() - t0) * 1000 / caption_embs.shape[0]

    labels, hnsw_ms = hnsw_search_batch(
        index,
        pivot_queries,
        topk=cfg.topC,
        batch_q=args.batch_size_text,
        warmup=args.warmup,
    )

    cand_recall, cand_ranks = compute_candidate_hits(labels, gt_indices)
    hit_ranks = cand_ranks[cand_ranks >= 0]
    hit_rank_mean = float(hit_ranks.mean()) if hit_ranks.size else -1.0
    hit_rank_median = float(np.median(hit_ranks)) if hit_ranks.size else -1.0

    # Rerank in original space (batched + argpartition for equivalence)
    rerank_times: List[Tuple[float, int]] = []  # (seconds, batch_size)
    preds: List[List[str]] = []
    n_queries = caption_embs.shape[0]
    for start in range(0, n_queries, args.batch_size_text):
        end = min(start + args.batch_size_text, n_queries)
        lab_batch = labels[start:end]
        q_batch = caption_embs[start:end]
        t1 = time.perf_counter()
        cand_emb_batch = image_embs[lab_batch]  # (B, topC, D)
        sims = np.einsum("bkd,bd->bk", cand_emb_batch, q_batch)
        topk_idx_batch = np.argpartition(-sims, base_config.k - 1, axis=1)[
            :, : base_config.k
        ]
        topk_scores = np.take_along_axis(sims, topk_idx_batch, axis=1)
        order = np.argsort(-topk_scores, axis=1)
        sorted_idx = np.take_along_axis(topk_idx_batch, order, axis=1)
        rerank_times.append((time.perf_counter() - t1, q_batch.shape[0]))
        for row_sorted, lab_row in zip(sorted_idx, lab_batch):
            preds.append([image_ids[lab_row[i]] for i in row_sorted])

    if rerank_times:
        total_time = sum(t for t, _ in rerank_times)
        total_q = sum(b for _, b in rerank_times)
        rerank_ms = (total_time / total_q) * 1000
    else:
        rerank_ms = 0.0

    result: Dict[str, float | int | str] = {
        "method": "pivot_hnsw",
        "avg_total_ms": pivot_map_ms + hnsw_ms + rerank_ms,
        "pivot_map_ms": pivot_map_ms,
        "hnsw_ms": hnsw_ms,
        "rerank_ms": rerank_ms,
        "brute_force_ms": 0.0,
        "build_index_time_sec": build_time,
        "index_size_bytes": index_size,
        "m": cfg.m,
        "topC": cfg.topC,
        "efSearch": cfg.ef_search,
        "M": cfg.M,
        "pivot_norm": args.pivot_norm,
        "pivot_weight": args.pivot_weight,
        "CandRecall@topC": cand_recall,
        "cand_hit_rank_mean": hit_rank_mean,
        "cand_hit_rank_median": hit_rank_median,
    }
    # recalls filled by caller
    result["preds"] = preds  # temporary; stripped later
    return result


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    base_config = RetrievalConfig(
        source=args.source,
        split=args.split,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size_image,
        num_workers=args.num_workers,
        pivot_sample=args.pivot_sample,
        seed=args.seed,
        ef_construction=args.ef_construction,
        force_recompute=args.force_recompute,
        max_images=args.max_images,
        max_captions=args.max_captions,
        k=args.k,
    )

    logging.info("Loading dataset and embeddings")
    records = load_flickr30k(base_config)
    caption_pairs = build_caption_pairs(records, base_config.max_captions)

    model, processor = load_clip(base_config.model_name, base_config.device)
    image_embs, image_ids = load_or_compute_image_embeddings(
        records, base_config, model, processor
    )

    # For captions we want batch size text
    config_text = replace(base_config, batch_size=args.batch_size_text)
    caption_embs = load_or_compute_caption_embeddings(
        caption_pairs, config_text, model, processor
    )

    # Sample queries
    sampled_pairs, sampled_idx = sample_queries(
        caption_pairs, args.n_queries, args.seed
    )
    caption_embs_sample = caption_embs[sampled_idx]
    gt_ids = [pair[1] for pair in sampled_pairs]
    id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}
    gt_indices = np.array([id_to_index[g] for g in gt_ids], dtype=int)

    results: List[Dict[str, float | int | str]] = []

    # Method A: brute-force exact
    preds_brute, brute_ms = brute_force_search(
        caption_embs_sample,
        image_embs,
        image_ids,
        k=base_config.k,
        batch_q=args.batch_size_text,
        warmup=args.warmup,
    )
    recalls_brute = compute_recalls(preds_brute, gt_ids)
    res_a: Dict[str, float | int | str] = {
        "method": "brute_force",
        "avg_total_ms": brute_ms,
        "brute_force_ms": brute_ms,
        "hnsw_ms": 0.0,
        "pivot_map_ms": 0.0,
        "rerank_ms": 0.0,
        "build_index_time_sec": 0.0,
        "index_size_bytes": 0,
        "m": 0,
        "topC": base_config.k,
        "efSearch": 0,
        "M": 0,
        "pivot_norm": "none",
        "pivot_weight": "none",
        "CandRecall@topC": 1.0,
        "cand_hit_rank_mean": -1.0,
        "cand_hit_rank_median": -1.0,
        **recalls_brute,
    }
    results.append(res_a)

    # Method B: HNSW on original embeddings
    res_b = orig_hnsw_method(
        caption_embs_sample, image_embs, image_ids, gt_indices, base_config, args
    )
    recalls_b = compute_recalls(res_b.pop("preds"), gt_ids)
    res_b.update(recalls_b)
    results.append(res_b)

    # Method C: Pivot + HNSW
    res_c = pivot_hnsw_method(
        caption_embs_sample, image_embs, image_ids, gt_indices, base_config, args
    )
    recalls_c = compute_recalls(res_c.pop("preds"), gt_ids)
    res_c.update(recalls_c)
    results.append(res_c)

    # Add common fields
    for r in results:
        r.update(
            {
                "N_images": image_embs.shape[0],
                "n_queries": len(gt_ids),
                "D": image_embs.shape[1],
                "model_name": base_config.model_name,
                "device": base_config.device,
                "seed": base_config.seed,
                "max_images": base_config.max_images,
                "max_captions": base_config.max_captions,
                "k": base_config.k,
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    summary_path = Path("results") / f"benchmark_{timestamp}.csv"
    json_path = Path("results") / f"benchmark_{timestamp}.json"
    ensure_dir(summary_path.parent)

    headers = list(results[0].keys())
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            f.write(",".join(str(r[h]) for h in headers) + "\n")

    save_json(results, json_path)

    # Plot R@10 vs latency
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [float(r.get("avg_total_ms", 0)) for r in results]
    y = [float(r.get("Recall@10", 0)) for r in results]
    labels = [str(r.get("method", "")) for r in results]
    ax.scatter(x, y, color="#4c72b0")
    for xi, yi, lbl in zip(x, y, labels):
        ax.text(xi, yi, lbl, fontsize=8)
    ax.set_xlabel("Avg total ms per query")
    ax.set_ylabel("Recall@10")
    ax.set_title("Benchmark: R@10 vs latency")
    fig.tight_layout()
    plot_path = Path("plots") / f"benchmark_latency_vs_recall_{timestamp}.png"
    ensure_dir(plot_path.parent)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    logging.info(
        "Benchmark saved: %s, %s, plot: %s", summary_path, json_path, plot_path
    )


if __name__ == "__main__":
    main()
