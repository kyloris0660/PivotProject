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
import torch

from retrieval.config import RetrievalConfig
from retrieval.data import build_caption_pairs, load_dataset_records
from retrieval.embeddings import (
    load_clip,
    load_or_compute_caption_embeddings,
    load_or_compute_image_embeddings,
)
from retrieval.pivots import compute_pivot_coordinates, select_pivots
from retrieval.pivots import pivot_weight_paths
from retrieval.utils import ensure_dir, save_json, set_seed, setup_logging


def parse_list(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",") if x]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark brute-force vs HNSW vs Pivot+HNSW"
    )
    p.add_argument(
        "--dataset",
        default="flickr30k",
        choices=["flickr30k", "coco_captions", "conceptual_captions"],
        help="dataset",
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
    p.add_argument("--orig_topC", type=int, default=None)
    p.add_argument("--orig_hnsw_efSearch", type=int, default=128)
    p.add_argument("--orig_hnsw_M", type=int, default=24)
    p.add_argument("--pivot_norm", choices=["none", "zscore"], default="none")
    p.add_argument(
        "--pivot_weight",
        choices=["none", "variance", "learned"],
        default="none",
    )
    p.add_argument(
        "--pivot_source",
        choices=["images", "captions", "union", "mixture", "caption_guided_images"],
        default="images",
    )
    p.add_argument("--pivot_mix_ratio", type=float, default=0.5)
    p.add_argument("--pivot_pool_size", type=int, default=50000)
    p.add_argument("--pivot_coord", choices=["sim", "dist"], default="sim")
    p.add_argument("--pivot_metric", choices=["l2", "cosine", "ip"], default="l2")
    p.add_argument("--pivot_weight_eps", type=float, default=1e-6)
    p.add_argument("--pivot_learn_pairs", type=int, default=20000)
    p.add_argument("--pivot_learn_queries", type=int, default=2000)
    p.add_argument("--pivot_learn_negs", type=int, default=8)
    p.add_argument("--batch_size_text", type=int, default=64)
    p.add_argument("--batch_size_image", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_threads", type=int, default=4)
    p.add_argument(
        "--allow_coco_train_download",
        action="store_true",
        help="Allow downloading large COCO train2017 images in fallback mode",
    )
    p.add_argument("--pivot_sample", type=int, default=5000)
    p.add_argument("--efc", dest="ef_construction", type=int, default=200)
    p.add_argument("--pivot_prune_to", type=int, default=0)
    p.add_argument(
        "--pivot_preset",
        choices=["none", "shortlist_g8"],
        default="none",
        help="Preset pivot config; shortlist_g8 enforces the shortlist G8 winner",
    )
    p.add_argument(
        "--rerank_device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="device for rerank; auto picks cuda if available",
    )
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument(
        "--warmup", type=int, default=10, help="warmup queries before timing"
    )
    return p.parse_args()


def apply_pivot_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.pivot_preset == "shortlist_g8":
        # Align with shortlist G8 winner (sim + cosine + zscore + learned, m=16)
        args.pivot_source = "images"
        args.pivot_coord = "sim"
        args.pivot_metric = "cosine"
        args.pivot_norm = "zscore"
        args.pivot_weight = "learned"
        args.pivot_m = 16
        args.topC = 600
        args.pivot_efSearch = 1200
        args.pivot_M = 24
        # Allow user override of prune depth via CLI (default 0)
    return args


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


def ensure_float32_contig(arr: np.ndarray) -> np.ndarray:
    # Ensure downstream math uses contiguous float32 buffers
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr)


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
    ef_search: int,
    num_threads: int | None = None,
) -> Tuple[np.ndarray, float]:
    # Clip topk to available elements to avoid contiguous buffer errors
    current_count = index.get_current_count()
    effective_topk = min(topk, current_count)
    if effective_topk <= 0:
        return np.empty((queries.shape[0], 0), dtype=int), 0.0

    # Ensure ef >= requested search depth
    target_ef = max(ef_search, effective_topk)
    try:
        index.set_ef(target_ef)
    except AttributeError:
        pass

    if num_threads is not None and num_threads > 0:
        index.set_num_threads(num_threads)

    labels_all: List[np.ndarray] = []
    timings: List[Tuple[float, int]] = []
    n_queries = queries.shape[0]
    warmup_queries = warmup
    for start in range(0, n_queries, batch_q):
        end = min(start + batch_q, n_queries)
        batch = queries[start:end]
        t0 = time.perf_counter()
        labels, _ = index.knn_query(batch, k=effective_topk)
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
    weight_vec: np.ndarray | None = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray] | None]:
    """Apply zscore + diagonal weights (variance/learned) in fixed order."""

    if stats is None:
        mean = coords.mean(axis=0)
        std = coords.std(axis=0) + eps
        stats = {"mean": mean, "std": std, "std_raw": std.copy()}
    else:
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is None or std is None:
            raise ValueError("stats must contain mean and std when provided")

    if norm == "zscore":
        coords = (coords - mean) / (std + eps)

    applied_weight: np.ndarray | None = None
    if weight == "variance":
        base_std = stats.get("std_raw", std)
        applied_weight = (
            weight_vec if weight_vec is not None else 1.0 / (base_std + eps)
        )
    elif weight == "learned":
        if weight_vec is None:
            weight_vec = stats.get("weight")
        if weight_vec is None:
            raise ValueError("learned weights requested but weight_vec is None")
        applied_weight = weight_vec

    if applied_weight is not None:
        coords = coords * np.sqrt(applied_weight)
        stats["weight"] = applied_weight

    stats["mean"] = mean
    stats["std"] = std
    return coords, stats


def _learn_diagonal_weights(
    pivot_coords_img: np.ndarray,
    pivot_queries: np.ndarray,
    image_embs: np.ndarray,
    caption_embs: np.ndarray,
    gt_indices: np.ndarray,
    config: RetrievalConfig,
) -> np.ndarray:
    """Compute diagonal weights by contrasting positives vs hard negatives."""

    rng = np.random.RandomState(config.seed)
    num_queries = min(
        config.pivot_learn_queries, config.pivot_learn_pairs, pivot_queries.shape[0]
    )
    if num_queries <= 0:
        return np.ones(pivot_coords_img.shape[1], dtype=np.float32)

    sample_idx = rng.choice(pivot_queries.shape[0], size=num_queries, replace=False)
    q_coords = pivot_queries[sample_idx]
    q_embs = caption_embs[sample_idx]
    pos_idx = gt_indices[sample_idx]

    # Hard negatives from original space similarity
    sims = q_embs @ image_embs.T
    topk = min(config.pivot_learn_negs * 4 + 1, sims.shape[1])
    top_idx = np.argpartition(-sims, topk - 1, axis=1)[:, :topk]

    neg_coords_all: List[np.ndarray] = []
    for qi, (row, pos) in enumerate(zip(top_idx, pos_idx)):
        scores_row = sims[qi, row]
        order = np.argsort(-scores_row)
        ordered = row[order]
        ordered = [idx for idx in ordered if idx != pos][: config.pivot_learn_negs]
        if len(ordered) < config.pivot_learn_negs:
            candidates = np.setdiff1d(np.arange(image_embs.shape[0]), np.array([pos]))
            need = config.pivot_learn_negs - len(ordered)
            if candidates.size > 0 and need > 0:
                size_take = min(need, candidates.size)
                extra = rng.choice(candidates, size=size_take, replace=False)
                ordered.extend(list(extra))
        while len(ordered) < config.pivot_learn_negs:
            ordered.append(ordered[-1] if ordered else 0)
        neg_coords_all.append(pivot_coords_img[ordered])

    neg_coords = np.stack(neg_coords_all, axis=0)  # (Q, negs, D)
    pos_coords = pivot_coords_img[pos_idx]

    diff2_pos = (q_coords - pos_coords) ** 2
    diff2_neg = (q_coords[:, None, :] - neg_coords) ** 2

    mu_pos = diff2_pos.mean(axis=0)
    mu_neg = diff2_neg.mean(axis=(0, 1))

    raw = np.clip(mu_neg - mu_pos, 0, None)
    mean_raw = float(raw.mean())
    if mean_raw <= 0:
        return np.ones(pivot_coords_img.shape[1], dtype=np.float32)

    w = raw / (mean_raw + config.pivot_weight_eps)
    w = np.maximum(w, config.pivot_weight_eps)
    return w.astype(np.float32, copy=False)


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

    image_embs = ensure_float32_contig(image_embs)
    caption_embs = ensure_float32_contig(caption_embs)

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

    search_topk = max(cfg.k, args.orig_topC) if args.orig_topC is not None else cfg.k

    labels, hnsw_ms = hnsw_search_batch(
        index,
        caption_embs,
        topk=search_topk,
        batch_q=args.batch_size_text,
        warmup=args.warmup,
        ef_search=cfg.ef_search,
        num_threads=args.num_threads,
    )
    labels = labels.astype(np.int64, copy=False)
    rerank_device = args.rerank_device
    if rerank_device == "auto":
        rerank_device = "cuda" if torch.cuda.is_available() else "cpu"
    if rerank_device == "cuda" and not torch.cuda.is_available():
        logging.warning(
            "CUDA requested for rerank but not available; falling back to CPU"
        )
        rerank_device = "cpu"

    preds: List[List[str]] = []
    rerank_times: List[Tuple[float, int]] = []
    image_embs_torch = None
    if rerank_device == "cuda":
        image_embs_torch = torch.as_tensor(image_embs, device="cuda").contiguous()

    n_queries = caption_embs.shape[0]
    for start in range(0, n_queries, args.batch_size_text):
        end = min(start + args.batch_size_text, n_queries)
        lab_batch = labels[start:end]
        q_batch = caption_embs[start:end]
        t1 = time.perf_counter()
        if rerank_device == "cuda":
            lab_tensor = torch.as_tensor(lab_batch, device="cuda", dtype=torch.long)
            q_tensor = torch.as_tensor(q_batch, device="cuda")
            cand_emb = image_embs_torch[lab_tensor]
            scores = torch.einsum("bkd,bd->bk", cand_emb, q_tensor)
            k_eff = min(base_config.k, scores.shape[1])
            _, topk_idx = torch.topk(scores, k=k_eff, dim=1)
            rerank_times.append((time.perf_counter() - t1, q_batch.shape[0]))
            topk_idx_cpu = topk_idx.cpu().numpy()
            for row_sorted, lab_row in zip(topk_idx_cpu, lab_batch):
                preds.append([image_ids[lab_row[i]] for i in row_sorted])
        else:
            cand_emb_batch = image_embs[lab_batch]
            sims = np.einsum("bkd,bd->bk", cand_emb_batch, q_batch)
            k_eff = min(base_config.k, sims.shape[1])
            topk_idx_batch = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
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

    cand_recall, cand_ranks = compute_candidate_hits(labels, gt_indices)
    hit_ranks = cand_ranks[cand_ranks >= 0]
    hit_rank_mean = float(hit_ranks.mean()) if hit_ranks.size else -1.0
    hit_rank_median = float(np.median(hit_ranks)) if hit_ranks.size else -1.0
    cand_recall_pruned = cand_recall
    hit_rank_mean_pruned = hit_rank_mean
    hit_rank_median_pruned = hit_rank_median
    prune_ms = 0.0

    sum_components_ms = hnsw_ms + rerank_ms + prune_ms

    result: Dict[str, float | int | str] = {
        "method": "orig_hnsw",
        "avg_total_ms": sum_components_ms,
        "sum_components_ms": sum_components_ms,
        "hnsw_ms": hnsw_ms,
        "pivot_map_ms": 0.0,
        "rerank_ms": rerank_ms,
        "brute_force_ms": 0.0,
        "build_index_time_sec": build_time,
        "index_size_bytes": index_size,
        "m": 0,
        "topC": search_topk,
        "efSearch": cfg.ef_search,
        "M": cfg.M,
        "pivot_norm": "none",
        "pivot_weight": "none",
        "pivot_coord": "none",
        "pivot_metric": "none",
        "pivot_source": "none",
        "pivot_prune_to": 0,
        "prune_ms": prune_ms,
        "rerank_device": rerank_device,
        "pivot_preset": args.pivot_preset,
        "CandRecall@topC": cand_recall,
        "CandRecall@pruned": cand_recall_pruned,
        "cand_hit_rank_mean": hit_rank_mean,
        "cand_hit_rank_median": hit_rank_median,
        "cand_hit_rank_mean_pruned": hit_rank_mean_pruned,
        "cand_hit_rank_median_pruned": hit_rank_median_pruned,
    }
    result["preds"] = preds
    return result


def pivot_hnsw_method(
    caption_embs: np.ndarray,
    caption_image_ids: List[str],
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
        pivot_norm=args.pivot_norm,
        pivot_weight=args.pivot_weight,
        pivot_source=args.pivot_source,
        pivot_mix_ratio=args.pivot_mix_ratio,
        pivot_pool_size=args.pivot_pool_size,
        pivot_coord=args.pivot_coord,
        pivot_metric=args.pivot_metric,
        pivot_weight_eps=args.pivot_weight_eps,
        pivot_learn_pairs=args.pivot_learn_pairs,
        pivot_learn_queries=args.pivot_learn_queries,
        pivot_learn_negs=args.pivot_learn_negs,
    )
    image_embs = ensure_float32_contig(image_embs)
    caption_embs = ensure_float32_contig(caption_embs)

    pivots, _ = select_pivots(
        image_embs, image_ids, caption_embs, caption_image_ids, cfg
    )
    pivots = ensure_float32_contig(pivots)

    pivot_coords = compute_pivot_coordinates(
        image_embs, pivots, cfg, cfg.split, kind="images"
    )

    # Query pivot coordinates
    t_q = time.perf_counter()
    pivot_queries_base = compute_pivot_coordinates(
        caption_embs, pivots, cfg, cfg.split, kind="captions"
    )

    # Normalize (zscore) first, then weights
    pivot_coords, stats = apply_pivot_transform(
        pivot_coords, args.pivot_norm, "none", None, eps=cfg.pivot_weight_eps
    )
    pivot_queries, _ = apply_pivot_transform(
        pivot_queries_base,
        args.pivot_norm,
        "none",
        stats,
        eps=cfg.pivot_weight_eps,
    )

    weight_vec = None
    if args.pivot_weight == "variance":
        weight_vec = 1.0 / (stats.get("std_raw", stats["std"]) + cfg.pivot_weight_eps)
    elif args.pivot_weight == "learned":
        weight_path, weight_meta = pivot_weight_paths(cfg)
        if weight_path.exists() and weight_meta.exists() and not cfg.force_recompute:
            logging.info("Loading learned pivot weights from %s", weight_path)
            weight_vec = np.load(weight_path)
        else:
            weight_vec = _learn_diagonal_weights(
                pivot_coords,
                pivot_queries,
                image_embs,
                caption_embs,
                gt_indices,
                cfg,
            )
            np.save(weight_path, weight_vec)
            save_json(
                {
                    "mode": "learned",
                    "pivot_coord": cfg.pivot_coord,
                    "pivot_metric": cfg.pivot_metric,
                },
                weight_meta,
            )

    pivot_coords, stats = apply_pivot_transform(
        pivot_coords,
        "none",
        args.pivot_weight,
        stats,
        weight_vec,
        eps=cfg.pivot_weight_eps,
    )
    pivot_queries, _ = apply_pivot_transform(
        pivot_queries,
        "none",
        args.pivot_weight,
        stats,
        weight_vec,
        eps=cfg.pivot_weight_eps,
    )

    pivot_coords = ensure_float32_contig(pivot_coords)
    pivot_queries = ensure_float32_contig(pivot_queries)

    pivot_map_ms = (time.perf_counter() - t_q) * 1000 / max(caption_embs.shape[0], 1)

    space = "l2" if cfg.pivot_metric == "l2" else cfg.pivot_metric
    index_path = cfg.cache_path("index") / (
        f"benchmark_pivot_hnsw_{cfg.dataset}_{cfg.split}_{cfg.pivot_source}_{cfg.pivot_coord}_{cfg.pivot_metric}_m{cfg.m}_M{cfg.M}_efc{cfg.ef_construction}_efs{cfg.ef_search}_seed{cfg.seed}_n{image_embs.shape[0]}"
        f"_{args.pivot_norm}_{args.pivot_weight}.bin"
    )
    index, build_time, index_size = build_hnsw_index(
        pivot_coords,
        space=space,
        M=cfg.M,
        ef_construction=cfg.ef_construction,
        ef_search=cfg.ef_search,
        index_path=index_path,
    )

    # Optional multithreading for search
    if args.num_threads and args.num_threads > 0:
        index.set_num_threads(args.num_threads)

    labels, hnsw_ms = hnsw_search_batch(
        index,
        pivot_queries,
        topk=cfg.topC,
        batch_q=args.batch_size_text,
        warmup=args.warmup,
        ef_search=cfg.ef_search,
        num_threads=args.num_threads,
    )
    labels = labels.astype(np.int64, copy=False)

    prune_ms = 0.0
    labels_for_rerank = labels
    actual_prune_to = (
        min(args.pivot_prune_to, labels.shape[1]) if args.pivot_prune_to > 0 else 0
    )
    if 0 < actual_prune_to < labels.shape[1]:
        t_prune = time.perf_counter()
        dists = np.sum((pivot_coords[labels] - pivot_queries[:, None, :]) ** 2, axis=2)
        idx_keep = np.argpartition(dists, actual_prune_to - 1, axis=1)[
            :, :actual_prune_to
        ]
        dist_keep = np.take_along_axis(dists, idx_keep, axis=1)
        order_keep = np.argsort(dist_keep, axis=1)
        sorted_keep = np.take_along_axis(idx_keep, order_keep, axis=1)
        labels_for_rerank = np.take_along_axis(labels, sorted_keep, axis=1)
        prune_ms = (time.perf_counter() - t_prune) * 1000 / labels.shape[0]
    else:
        actual_prune_to = labels.shape[1]

    rerank_device = args.rerank_device
    if rerank_device == "auto":
        rerank_device = "cuda" if torch.cuda.is_available() else "cpu"
    if rerank_device == "cuda" and not torch.cuda.is_available():
        logging.warning(
            "CUDA requested for rerank but not available; falling back to CPU"
        )
        rerank_device = "cpu"

    preds: List[List[str]] = []
    rerank_times: List[Tuple[float, int]] = []
    n_queries = caption_embs.shape[0]

    image_embs_torch = None
    if rerank_device == "cuda":
        image_embs_torch = torch.as_tensor(image_embs, device="cuda").contiguous()

    for start in range(0, n_queries, args.batch_size_text):
        end = min(start + args.batch_size_text, n_queries)
        lab_batch = labels_for_rerank[start:end]
        q_batch = caption_embs[start:end]
        t1 = time.perf_counter()
        if rerank_device == "cuda":
            lab_tensor = torch.as_tensor(lab_batch, device="cuda", dtype=torch.long)
            q_tensor = torch.as_tensor(q_batch, device="cuda")
            cand_emb = image_embs_torch[lab_tensor]
            scores = torch.einsum("bkd,bd->bk", cand_emb, q_tensor)
            k_eff = min(base_config.k, scores.shape[1])
            _, topk_idx = torch.topk(scores, k=k_eff, dim=1)
            rerank_times.append((time.perf_counter() - t1, q_batch.shape[0]))
            topk_idx_cpu = topk_idx.cpu().numpy()
            for row_sorted, lab_row in zip(topk_idx_cpu, lab_batch):
                preds.append([image_ids[lab_row[i]] for i in row_sorted])
        else:
            cand_emb_batch = image_embs[lab_batch]
            sims = np.einsum("bkd,bd->bk", cand_emb_batch, q_batch)
            k_eff = min(base_config.k, sims.shape[1])
            topk_idx_batch = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
            topk_scores = np.take_along_axis(sims, topk_idx_batch, axis=1)
            order = np.argsort(-topk_scores, axis=1)
            sorted_idx = np.take_along_axis(topk_idx_batch, order, axis=1)
            rerank_times.append((time.perf_counter() - t1, q_batch.shape[0]))
            for row_sorted, lab_row in zip(sorted_idx, lab_batch):
                preds.append([image_ids[lab_row[i]] for i in row_sorted])

    cand_recall_top, cand_ranks_top = compute_candidate_hits(labels, gt_indices)
    hit_top = cand_ranks_top[cand_ranks_top >= 0]
    hit_rank_mean = float(hit_top.mean()) if hit_top.size else -1.0
    hit_rank_median = float(np.median(hit_top)) if hit_top.size else -1.0

    cand_recall_pruned, cand_ranks_pruned = compute_candidate_hits(
        labels_for_rerank, gt_indices
    )
    hit_pruned = cand_ranks_pruned[cand_ranks_pruned >= 0]
    hit_rank_mean_pruned = float(hit_pruned.mean()) if hit_pruned.size else -1.0
    hit_rank_median_pruned = float(np.median(hit_pruned)) if hit_pruned.size else -1.0

    if rerank_times:
        total_time = sum(t for t, _ in rerank_times)
        total_q = sum(b for _, b in rerank_times)
        rerank_ms = (total_time / total_q) * 1000
    else:
        rerank_ms = 0.0

    sum_components_ms = pivot_map_ms + hnsw_ms + prune_ms + rerank_ms

    result: Dict[str, float | int | str] = {
        "method": "pivot_hnsw",
        "avg_total_ms": sum_components_ms,
        "sum_components_ms": sum_components_ms,
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
        "pivot_coord": cfg.pivot_coord,
        "pivot_metric": cfg.pivot_metric,
        "pivot_source": cfg.pivot_source,
        "pivot_pool_size": cfg.pivot_pool_size,
        "pivot_mix_ratio": cfg.pivot_mix_ratio,
        "pivot_prune_to": actual_prune_to,
        "prune_ms": prune_ms,
        "rerank_device": rerank_device,
        "pivot_preset": args.pivot_preset,
        "CandRecall@topC": cand_recall_top,
        "CandRecall@pruned": cand_recall_pruned,
        "cand_hit_rank_mean": hit_rank_mean,
        "cand_hit_rank_median": hit_rank_median,
        "cand_hit_rank_mean_pruned": hit_rank_mean_pruned,
        "cand_hit_rank_median_pruned": hit_rank_median_pruned,
    }
    # recalls filled by caller
    result["preds"] = preds  # temporary; stripped later
    return result


def main() -> None:
    args = parse_args()
    args = apply_pivot_preset(args)
    setup_logging()
    set_seed(args.seed)

    base_config = RetrievalConfig(
        dataset=args.dataset,
        source=args.source,
        split=args.split,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size_image,
        num_workers=args.num_workers,
        pivot_sample=args.pivot_sample,
        pivot_source=args.pivot_source,
        pivot_mix_ratio=args.pivot_mix_ratio,
        pivot_pool_size=args.pivot_pool_size,
        pivot_coord=args.pivot_coord,
        pivot_metric=args.pivot_metric,
        pivot_weight=args.pivot_weight,
        pivot_weight_eps=args.pivot_weight_eps,
        pivot_learn_pairs=args.pivot_learn_pairs,
        pivot_learn_queries=args.pivot_learn_queries,
        pivot_learn_negs=args.pivot_learn_negs,
        seed=args.seed,
        ef_construction=args.ef_construction,
        force_recompute=args.force_recompute,
        max_images=args.max_images,
        max_captions=args.max_captions,
        k=args.k,
        allow_coco_train_download=args.allow_coco_train_download,
    )

    logging.info("Loading dataset and embeddings")
    records = load_dataset_records(base_config)
    caption_pairs = build_caption_pairs(records, base_config.max_captions)

    model, processor = load_clip(base_config.model_name, base_config.device)
    t_embed_img = time.perf_counter()
    image_embs, image_ids = load_or_compute_image_embeddings(
        records, base_config, model, processor
    )
    embed_time_img = time.perf_counter() - t_embed_img
    image_embs = ensure_float32_contig(image_embs)

    # For captions we want batch size text
    config_text = replace(base_config, batch_size=args.batch_size_text)
    t_embed_cap = time.perf_counter()
    caption_embs = load_or_compute_caption_embeddings(
        caption_pairs, config_text, model, processor
    )
    embed_time_cap = time.perf_counter() - t_embed_cap
    caption_embs = ensure_float32_contig(caption_embs)

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
        "sum_components_ms": brute_ms,
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
        "pivot_coord": "none",
        "pivot_metric": "none",
        "pivot_source": "none",
        "pivot_prune_to": 0,
        "prune_ms": 0.0,
        "rerank_device": "cpu",
        "pivot_preset": args.pivot_preset,
        "CandRecall@topC": 1.0,
        "CandRecall@pruned": 1.0,
        "cand_hit_rank_mean": -1.0,
        "cand_hit_rank_median": -1.0,
        "cand_hit_rank_mean_pruned": -1.0,
        "cand_hit_rank_median_pruned": -1.0,
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
        caption_embs_sample,
        [pair[1] for pair in caption_pairs],
        image_embs,
        image_ids,
        gt_indices,
        base_config,
        args,
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
                "dataset": base_config.dataset,
                "seed": base_config.seed,
                "max_images": base_config.max_images,
                "max_captions": base_config.max_captions,
                "k": base_config.k,
                "embedding_build_time_img_sec": embed_time_img,
                "embedding_build_time_cap_sec": embed_time_cap,
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
