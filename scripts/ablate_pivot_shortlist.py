"""Run a fixed shortlist of pivot_hnsw configs to quickly check candidate recall and latency.

Outputs CSV/JSON plus two scatter plots (CandRecall@topC vs latency, Recall@10 vs latency).
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
from scripts.benchmark_baselines import (
    compute_recalls,
    ensure_float32_contig,
    pivot_hnsw_method,
    sample_queries,
)


SHORTLIST: List[Dict[str, object]] = [
    {
        "name": "G1",
        "pivot_source": "union",
        "pivot_coord": "sim",
        "pivot_metric": "cosine",
        "pivot_norm": "zscore",
        "pivot_weight": "variance",
        "pivot_m": 16,
    },
    {
        "name": "G2",
        "pivot_source": "union",
        "pivot_coord": "sim",
        "pivot_metric": "cosine",
        "pivot_norm": "zscore",
        "pivot_weight": "learned",
        "pivot_m": 16,
    },
    {
        "name": "G3",
        "pivot_source": "union",
        "pivot_coord": "sim",
        "pivot_metric": "l2",
        "pivot_norm": "zscore",
        "pivot_weight": "variance",
        "pivot_m": 16,
    },
    {
        "name": "G4",
        "pivot_source": "union",
        "pivot_coord": "sim",
        "pivot_metric": "l2",
        "pivot_norm": "zscore",
        "pivot_weight": "learned",
        "pivot_m": 16,
    },
    {
        "name": "G5",
        "pivot_source": "mixture",
        "pivot_mix_ratio": 0.5,
        "pivot_coord": "sim",
        "pivot_metric": "cosine",
        "pivot_norm": "zscore",
        "pivot_weight": "learned",
        "pivot_m": 16,
    },
    {
        "name": "G6",
        "pivot_source": "union",
        "pivot_coord": "sim",
        "pivot_metric": "cosine",
        "pivot_norm": "zscore",
        "pivot_weight": "variance",
        "pivot_m": 32,
    },
    {
        "name": "G7",
        "pivot_source": "union",
        "pivot_coord": "sim",
        "pivot_metric": "cosine",
        "pivot_norm": "zscore",
        "pivot_weight": "learned",
        "pivot_m": 32,
    },
    {
        "name": "G8",
        "pivot_source": "images",
        "pivot_coord": "sim",
        "pivot_metric": "cosine",
        "pivot_norm": "zscore",
        "pivot_weight": "learned",
        "pivot_m": 16,
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Shortlist ablation for pivot_hnsw")
    p.add_argument(
        "--dataset",
        default="coco_captions",
        choices=["flickr30k", "coco_captions", "conceptual_captions"],
        help="dataset",
    )
    p.add_argument("--source", default="hf", choices=["hf"], help="data source")
    p.add_argument("--split", default="val")
    p.add_argument("--model_name", default="openai/clip-vit-base-patch32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--max_captions", type=int, default=None)
    p.add_argument("--n_queries", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--batch_size_text", type=int, default=128)
    p.add_argument("--batch_size_image", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--allow_coco_train_download", action="store_true")
    p.add_argument("--pivot_sample", type=int, default=5000)
    p.add_argument("--pivot_pool_size", type=int, default=50000)
    p.add_argument("--pivot_mix_ratio", type=float, default=0.5)
    p.add_argument("--pivot_weight_eps", type=float, default=1e-6)
    p.add_argument("--pivot_learn_pairs", type=int, default=20000)
    p.add_argument("--pivot_learn_queries", type=int, default=2000)
    p.add_argument("--pivot_learn_negs", type=int, default=8)
    p.add_argument("--topC", type=int, default=600)
    p.add_argument("--pivot_prune_to", type=int, default=0)
    p.add_argument("--pivot_efSearch", type=int, default=1200)
    p.add_argument("--pivot_M", type=int, default=24)
    p.add_argument("--rerank_device", choices=["cpu", "cuda", "auto"], default="cuda")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--force_recompute", action="store_true")
    return p.parse_args()


def pick_best(rows: List[Dict[str, object]]) -> Dict[str, object]:
    def better(a: Dict[str, object], b: Dict[str, object]) -> bool:
        cr_a = float(a.get("CandRecall@topC", 0))
        cr_b = float(b.get("CandRecall@topC", 0))
        if cr_a - cr_b > 0.02:
            return True
        if cr_b - cr_a > 0.02:
            return False
        r10_a = float(a.get("Recall@10", 0))
        r10_b = float(b.get("Recall@10", 0))
        if r10_a - r10_b > 0.02:
            return True
        if r10_b - r10_a > 0.02:
            return False
        return float(a.get("avg_total_ms", 1e9)) < float(b.get("avg_total_ms", 1e9))

    best = rows[0]
    for row in rows[1:]:
        if better(row, best):
            best = row
    return best


def main() -> None:
    args = parse_args()
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
        pivot_source="images",
        pivot_mix_ratio=args.pivot_mix_ratio,
        pivot_pool_size=args.pivot_pool_size,
        pivot_coord="sim",
        pivot_metric="cosine",
        pivot_weight="none",
        pivot_weight_eps=args.pivot_weight_eps,
        pivot_learn_pairs=args.pivot_learn_pairs,
        pivot_learn_queries=args.pivot_learn_queries,
        pivot_learn_negs=args.pivot_learn_negs,
        seed=args.seed,
        ef_construction=200,
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
    image_embs, image_ids = load_or_compute_image_embeddings(
        records, base_config, model, processor
    )
    image_embs = ensure_float32_contig(image_embs)

    config_text = replace(base_config, batch_size=args.batch_size_text)
    caption_embs = load_or_compute_caption_embeddings(
        caption_pairs, config_text, model, processor
    )
    caption_embs = ensure_float32_contig(caption_embs)

    sampled_pairs, sampled_idx = sample_queries(
        caption_pairs, args.n_queries, args.seed
    )
    caption_embs_sample = caption_embs[sampled_idx]
    gt_ids = [pair[1] for pair in sampled_pairs]
    id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}
    gt_indices = np.array([id_to_index[g] for g in gt_ids], dtype=int)

    rows: List[Dict[str, object]] = []

    for conf in SHORTLIST:
        run_args = argparse.Namespace(**vars(args))
        run_args.pivot_source = conf.get("pivot_source", run_args.pivot_source)
        run_args.pivot_mix_ratio = conf.get("pivot_mix_ratio", run_args.pivot_mix_ratio)
        run_args.pivot_coord = conf.get("pivot_coord", run_args.pivot_coord)
        run_args.pivot_metric = conf.get("pivot_metric", run_args.pivot_metric)
        run_args.pivot_norm = conf.get(
            "pivot_norm",
            run_args.pivot_norm if hasattr(run_args, "pivot_norm") else "none",
        )
        run_args.pivot_weight = conf.get(
            "pivot_weight",
            run_args.pivot_weight if hasattr(run_args, "pivot_weight") else "none",
        )
        run_args.pivot_m = conf.get(
            "pivot_m", run_args.pivot_m if hasattr(run_args, "pivot_m") else 16
        )
        run_args.topC = args.topC
        run_args.pivot_prune_to = args.pivot_prune_to
        run_args.pivot_efSearch = args.pivot_efSearch
        run_args.pivot_M = args.pivot_M
        run_args.rerank_device = args.rerank_device
        run_args.num_threads = args.num_threads

        res = pivot_hnsw_method(
            caption_embs_sample,
            image_embs,
            image_ids,
            gt_indices,
            base_config,
            run_args,
        )
        recalls = compute_recalls(res.pop("preds"), gt_ids)
        res.update(recalls)

        row: Dict[str, object] = {
            "group": conf["name"],
            "dataset": args.dataset,
            "split": args.split,
            "N_images": image_embs.shape[0],
            "n_queries": len(gt_ids),
            "pivot_source": res.get("pivot_source", run_args.pivot_source),
            "pivot_mix_ratio": run_args.pivot_mix_ratio,
            "pivot_coord": res.get("pivot_coord", run_args.pivot_coord),
            "pivot_metric": res.get("pivot_metric", run_args.pivot_metric),
            "pivot_norm": res.get("pivot_norm", run_args.pivot_norm),
            "pivot_weight": res.get("pivot_weight", run_args.pivot_weight),
            "pivot_m": res.get("m", run_args.pivot_m),
            "topC": res.get("topC", run_args.topC),
            "pivot_prune_to": res.get("pivot_prune_to", run_args.pivot_prune_to),
            "CandRecall@topC": res.get("CandRecall@topC", 0.0),
            "CandRecall@pruned": res.get("CandRecall@pruned", 0.0),
            "Recall@1": res.get("Recall@1", recalls.get("Recall@1", 0.0)),
            "Recall@5": res.get("Recall@5", recalls.get("Recall@5", 0.0)),
            "Recall@10": res.get("Recall@10", recalls.get("Recall@10", 0.0)),
            "hit_rank_mean_topC": res.get("cand_hit_rank_mean", -1.0),
            "hit_rank_median_topC": res.get("cand_hit_rank_median", -1.0),
            "avg_total_ms": res.get("avg_total_ms", 0.0),
            "sum_components_ms": res.get("sum_components_ms", 0.0),
            "pivot_map_ms": res.get("pivot_map_ms", 0.0),
            "hnsw_ms": res.get("hnsw_ms", 0.0),
            "prune_ms": res.get("prune_ms", 0.0),
            "rerank_ms": res.get("rerank_ms", 0.0),
            "method": res.get("method", "pivot_hnsw"),
        }
        rows.append(row)
        logging.info(
            "Finished %s: CandRecall@topC=%.3f, Recall@10=%.3f, avg_total_ms=%.1f",
            row["group"],
            row["CandRecall@topC"],
            row["Recall@10"],
            row["avg_total_ms"],
        )

    best = pick_best(rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    results_path = Path("results") / f"shortlist_{timestamp}.csv"
    json_path = Path("results") / f"shortlist_{timestamp}.json"
    plot_cand_path = Path("plots") / f"shortlist_candrecall_vs_latency_{timestamp}.png"
    plot_r10_path = Path("plots") / f"shortlist_r10_vs_latency_{timestamp}.png"
    ensure_dir(results_path.parent)
    ensure_dir(plot_cand_path.parent)

    headers = [
        "group",
        "dataset",
        "split",
        "N_images",
        "n_queries",
        "pivot_source",
        "pivot_mix_ratio",
        "pivot_coord",
        "pivot_metric",
        "pivot_norm",
        "pivot_weight",
        "pivot_m",
        "topC",
        "pivot_prune_to",
        "CandRecall@topC",
        "CandRecall@pruned",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "hit_rank_mean_topC",
        "hit_rank_median_topC",
        "avg_total_ms",
        "sum_components_ms",
        "pivot_map_ms",
        "hnsw_ms",
        "prune_ms",
        "rerank_ms",
        "method",
    ]

    with results_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")

    save_json({"runs": rows, "best": best}, json_path)

    # Plot CandRecall@topC vs latency
    x = [float(r.get("avg_total_ms", 0)) for r in rows]
    y = [float(r.get("CandRecall@topC", 0)) for r in rows]
    labels = [str(r.get("group", "")) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, color="#4c72b0")
    for xi, yi, lbl in zip(x, y, labels):
        ax.text(xi, yi, lbl, fontsize=8)
    ax.set_xlabel("Avg total ms per query")
    ax.set_ylabel("CandRecall@topC")
    ax.set_title("Shortlist: CandRecall@topC vs latency")
    fig.tight_layout()
    fig.savefig(plot_cand_path, dpi=200)
    plt.close(fig)

    # Plot Recall@10 vs latency
    y2 = [float(r.get("Recall@10", 0)) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y2, color="#dd8452")
    for xi, yi, lbl in zip(x, y2, labels):
        ax.text(xi, yi, lbl, fontsize=8)
    ax.set_xlabel("Avg total ms per query")
    ax.set_ylabel("Recall@10")
    ax.set_title("Shortlist: R@10 vs latency")
    fig.tight_layout()
    fig.savefig(plot_r10_path, dpi=200)
    plt.close(fig)

    logging.info("Saved shortlist results: %s, %s", results_path, json_path)
    logging.info(
        "Best config: %s (CandRecall@topC=%.3f, Recall@10=%.3f, avg_total_ms=%.1f)",
        best.get("group"),
        best.get("CandRecall@topC"),
        best.get("Recall@10"),
        best.get("avg_total_ms"),
    )


if __name__ == "__main__":
    main()
