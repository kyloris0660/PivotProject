"""Scaling benchmark for text-to-image retrieval.

Runs brute-force, original-space HNSW, and pivot+HNSW across increasing N,
reusing the same sampled queries (filtered to images within each N) and
supporting multiple datasets (flickr30k, coco_captions).
Outputs per-method metrics to CSV/JSON and plots latency and Recall@10 vs N.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from retrieval.config import RetrievalConfig
from retrieval.data import build_caption_pairs, load_dataset_records
from retrieval.embeddings import (
    load_clip,
    load_or_compute_caption_embeddings,
    load_or_compute_image_embeddings,
)
from retrieval.utils import ensure_dir, save_json, set_seed, setup_logging

from scripts import benchmark_baselines as base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scaling benchmark for retrieval")
    p.add_argument(
        "--dataset",
        default="flickr30k",
        choices=["flickr30k", "coco_captions", "conceptual_captions"],
        help="dataset",
    )
    p.add_argument("--source", default="hf", choices=["hf"], help="data source")
    p.add_argument("--split", default="train")
    p.add_argument("--model_name", default="openai/clip-vit-base-patch32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--Ns", default="5000,10000,20000,50000,80000")
    p.add_argument("--n_queries", type=int, default=2000)
    p.add_argument("--max_captions", type=int, default=10000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--pivot_m", type=int, default=16)
    p.add_argument("--topC", type=int, default=1200)
    p.add_argument("--pivot_efSearch", type=int, default=128)
    p.add_argument("--pivot_M", type=int, default=24)
    p.add_argument("--pivot_prune_to", type=int, default=0)
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
        choices=[
            "images",
            "captions",
            "union",
            "mixture",
            "caption_guided_images",
            "caption_cluster_guided_images",
        ],
        default="images",
    )
    p.add_argument("--pivot_caption_sample", type=int, default=50000)
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
        "--rerank_device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="device for rerank; auto picks cuda if available",
    )
    p.add_argument(
        "--allow_coco_train_download",
        action="store_true",
        help="Allow downloading large COCO train2017 images in fallback mode",
    )
    p.add_argument("--pivot_sample", type=int, default=5000)
    p.add_argument("--efc", dest="ef_construction", type=int, default=200)
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    Ns = base.parse_list(args.Ns)
    if not Ns:
        raise ValueError("Ns must not be empty")
    max_N = max(Ns)

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
        pivot_caption_sample=args.pivot_caption_sample,
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
        max_images=max_N,
        max_captions=args.max_captions,
        k=args.k,
        allow_coco_train_download=args.allow_coco_train_download,
    )

    logging.info("Loading dataset and embeddings (max_images=%d)", max_N)
    records = load_dataset_records(base_config)
    caption_pairs = build_caption_pairs(records, base_config.max_captions)
    caption_image_ids = [pair[1] for pair in caption_pairs]

    model, processor = load_clip(base_config.model_name, base_config.device)
    image_embs, image_ids = load_or_compute_image_embeddings(
        records, base_config, model, processor
    )
    image_embs = base.ensure_float32_contig(image_embs)

    config_text = replace(base_config, batch_size=args.batch_size_text)
    caption_embs = load_or_compute_caption_embeddings(
        caption_pairs, config_text, model, processor
    )
    caption_embs = base.ensure_float32_contig(caption_embs)

    sampled_pairs, sampled_idx = base.sample_queries(
        caption_pairs, args.n_queries, args.seed
    )
    gt_ids = [pair[1] for pair in sampled_pairs]
    id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}

    results: List[Dict[str, float | int | str]] = []

    for N in Ns:
        logging.info("Running scaling experiment for N=%d", N)
        actual_N = min(N, len(image_ids))
        if actual_N < N:
            logging.warning(
                "Requested N=%d exceeds dataset size %d; using N=%d",
                N,
                len(image_ids),
                actual_N,
            )
        allowed_ids = set(image_ids[:actual_N])
        keep_mask = [gt in allowed_ids for gt in gt_ids]
        kept_indices = sampled_idx[keep_mask]
        if kept_indices.size == 0:
            logging.warning("No queries retained for N=%d; skipping", N)
            continue

        caption_embs_subset = caption_embs[kept_indices]
        caption_image_ids_subset = [caption_image_ids[i] for i in kept_indices]
        gt_ids_subset = [gt for keep, gt in zip(keep_mask, gt_ids) if keep]
        gt_indices_subset = np.array([id_to_index[g] for g in gt_ids_subset], dtype=int)

        image_embs_subset = image_embs[:actual_N]
        image_ids_subset = image_ids[:actual_N]

        cfg_N = replace(base_config, max_images=actual_N)

        preds_brute, brute_ms = base.brute_force_search(
            caption_embs_subset,
            image_embs_subset,
            image_ids_subset,
            k=base_config.k,
            batch_q=args.batch_size_text,
            warmup=args.warmup,
        )
        recalls_brute = base.compute_recalls(preds_brute, gt_ids_subset)
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
            "CandRecall@topC": 1.0,
            "CandRecall@pruned": 1.0,
            "cand_hit_rank_mean": -1.0,
            "cand_hit_rank_median": -1.0,
            "cand_hit_rank_mean_pruned": -1.0,
            "cand_hit_rank_median_pruned": -1.0,
            **recalls_brute,
        }
        results.append(res_a)

        res_b = base.orig_hnsw_method(
            caption_embs_subset,
            image_embs_subset,
            image_ids_subset,
            gt_indices_subset,
            cfg_N,
            args,
        )
        recalls_b = base.compute_recalls(res_b.pop("preds"), gt_ids_subset)
        res_b.update(recalls_b)
        results.append(res_b)

        res_c = base.pivot_hnsw_method(
            caption_embs_subset,
            caption_image_ids_subset,
            image_embs_subset,
            image_ids_subset,
            gt_indices_subset,
            cfg_N,
            args,
        )
        recalls_c = base.compute_recalls(res_c.pop("preds"), gt_ids_subset)
        res_c.update(recalls_c)
        results.append(res_c)

        for r in (res_a, res_b, res_c):
            r.update(
                {
                    "N_images": N,
                    "n_queries": len(gt_ids_subset),
                    "D": image_embs.shape[1],
                    "model_name": base_config.model_name,
                    "device": base_config.device,
                    "dataset": base_config.dataset,
                    "seed": base_config.seed,
                    "N_images": actual_N,
                    "max_images": actual_N,
                    "max_captions": base_config.max_captions,
                    "k": base_config.k,
                    "pivot_norm": r.get("pivot_norm", "none"),
                    "pivot_weight": r.get("pivot_weight", "none"),
                }
            )

    if not results:
        logging.error("No results produced; check dataset and Ns")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    summary_path = Path("results") / f"scaling_{timestamp}.csv"
    json_path = Path("results") / f"scaling_{timestamp}.json"
    ensure_dir(summary_path.parent)

    headers = list(results[0].keys())
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in results:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")

    save_json(results, json_path)

    # Build lookup for plotting
    by_method: Dict[str, List[Dict[str, float | int | str]]] = {
        m: [] for m in ("brute_force", "orig_hnsw", "pivot_hnsw")
    }
    for r in results:
        by_method[r["method"]].append(r)
    for method in by_method:
        by_method[method].sort(key=lambda x: x["N_images"])

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, label in (
        ("brute_force", "Brute force"),
        ("orig_hnsw", "Orig HNSW"),
        ("pivot_hnsw", "Pivot+HNSW"),
    ):
        Ns_sorted = [r["N_images"] for r in by_method[method]]
        lat = [float(r["avg_total_ms"]) for r in by_method[method]]
        ax.plot(Ns_sorted, lat, marker="o", label=label)
    ax.set_xlabel("#Images (N)")
    ax.set_ylabel("Avg total ms per query")
    ax.set_title("Latency vs N")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_latency_path = Path("plots") / f"scaling_latency_vs_N_{timestamp}.png"
    ensure_dir(plot_latency_path.parent)
    fig.savefig(plot_latency_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, label in (
        ("brute_force", "Brute force"),
        ("orig_hnsw", "Orig HNSW"),
        ("pivot_hnsw", "Pivot+HNSW"),
    ):
        Ns_sorted = [r["N_images"] for r in by_method[method]]
        rec = [float(r.get("Recall@10", 0)) for r in by_method[method]]
        ax.plot(Ns_sorted, rec, marker="o", label=label)
    ax.set_xlabel("#Images (N)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall@10 vs N")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_rec_path = Path("plots") / f"scaling_R10_vs_N_{timestamp}.png"
    ensure_dir(plot_rec_path.parent)
    fig.savefig(plot_rec_path, dpi=200)
    plt.close(fig)

    logging.info(
        "Scaling benchmark saved: %s, %s; plots: %s, %s",
        summary_path,
        json_path,
        plot_latency_path,
        plot_rec_path,
    )


if __name__ == "__main__":
    main()
