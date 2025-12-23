"""Recall-first pivot_hnsw shortlist to push candidate recall on COCO train (N=20k).

Runs a fixed set of high-recall configs on the same query batch and writes CSV/JSON summaries.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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


SHORTLIST: List[Dict[str, int]] = [
    {"name": "G1", "pivot_m": 32, "topC": 4000, "pivot_efSearch": 8000, "pivot_prune_to": 0},
    {"name": "G2", "pivot_m": 32, "topC": 8000, "pivot_efSearch": 16000, "pivot_prune_to": 0},
    {"name": "G3", "pivot_m": 64, "topC": 4000, "pivot_efSearch": 8000, "pivot_prune_to": 0},
    {"name": "G4", "pivot_m": 64, "topC": 8000, "pivot_efSearch": 16000, "pivot_prune_to": 0},
    {"name": "L1", "pivot_m": 32, "topC": 2000, "pivot_efSearch": 4000, "pivot_prune_to": 0},
    {"name": "L2", "pivot_m": 64, "topC": 2000, "pivot_efSearch": 4000, "pivot_prune_to": 0},
]

TOPC_SWEEP = [2000, 4000, 8000, 12000]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recall-focused shortlist for pivot_hnsw")
    p.add_argument("--dataset", default="coco_captions", choices=["flickr30k", "coco_captions", "conceptual_captions"])
    p.add_argument("--source", default="hf", choices=["hf"], help="data source")
    p.add_argument("--split", default="train")
    p.add_argument("--model_name", default="openai/clip-vit-base-patch32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--rerank_device", choices=["cpu", "cuda", "auto"], default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=20000)
    p.add_argument("--max_captions", type=int, default=None)
    p.add_argument("--n_queries", type=int, default=2000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--batch_size_text", type=int, default=128)
    p.add_argument("--batch_size_image", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--allow_coco_train_download", action="store_true")
    p.add_argument("--coco_root", default="", help="Optional local COCO root; uses existing data if set")
    p.add_argument("--pivot_norm", choices=["none", "zscore"], default="zscore")
    p.add_argument("--pivot_weight", choices=["none", "variance", "learned"], default="variance")
    p.add_argument("--pivot_coord", choices=["sim", "dist"], default="sim")
    p.add_argument("--pivot_metric", choices=["l2", "cosine", "ip"], default="cosine")
    p.add_argument("--pivot_weight_eps", type=float, default=1e-6)
    p.add_argument("--pivot_learn_pairs", type=int, default=20000)
    p.add_argument("--pivot_learn_queries", type=int, default=2000)
    p.add_argument("--pivot_learn_negs", type=int, default=8)
    p.add_argument("--pivot_pool_size", type=int, default=50000)
    p.add_argument("--pivot_mix_ratio", type=float, default=0.5)
    p.add_argument("--pivot_source", choices=["images", "captions", "union", "mixture"], default="images")
    p.add_argument("--pivot_M", type=int, default=24)
    p.add_argument("--pivot_sample", type=int, default=5000)
    p.add_argument("--pivot_m", type=int, default=32)
    p.add_argument("--topC", type=int, default=4000)
    p.add_argument("--pivot_efSearch", type=int, default=8000)
    p.add_argument("--pivot_prune_to", type=int, default=0)
    p.add_argument("--recall_first", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--recall_efsearch_multiplier", type=int, default=1, choices=[1, 2])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--enable_topc_sweep", action="store_true", help="Also sweep topC for best m using efSearch=2*topC, prune=0")
    return p.parse_args()


def run_group(
    name: str,
    overrides: Dict[str, int],
    args: argparse.Namespace,
    base_config: RetrievalConfig,
    caption_embs: np.ndarray,
    caption_image_ids: List[str],
    image_embs: np.ndarray,
    image_ids: List[str],
    gt_indices: np.ndarray,
    gt_ids: List[str],
) -> Dict[str, object]:
    run_args = argparse.Namespace(**vars(args))
    for k, v in overrides.items():
        setattr(run_args, k, v)

    res = pivot_hnsw_method(
        caption_embs,
        caption_image_ids,
        image_embs,
        image_ids,
        gt_indices,
        base_config,
        run_args,
    )
    recalls = compute_recalls(res.pop("preds"), gt_ids)
    res.update(recalls)

    row: Dict[str, object] = {
        "group": name,
        "m": res.get("m", overrides.get("pivot_m")),
        "topC": res.get("topC", overrides.get("topC")),
        "efSearch": res.get("efSearch", overrides.get("pivot_efSearch")),
        "prune_to": res.get("pivot_prune_to", overrides.get("pivot_prune_to", 0)),
        "CandRecall@topC": res.get("CandRecall@topC", 0.0),
        "Recall@1": res.get("Recall@1", 0.0),
        "Recall@5": res.get("Recall@5", 0.0),
        "Recall@10": res.get("Recall@10", 0.0),
        "avg_total_ms": res.get("avg_total_ms", 0.0),
        "hnsw_ms": res.get("hnsw_ms", 0.0),
        "pivot_map_ms": res.get("pivot_map_ms", 0.0),
        "rerank_ms": res.get("rerank_ms", 0.0),
        "pivot_prune_to_arg": res.get("pivot_prune_to_arg", overrides.get("pivot_prune_to", 0)),
        "pivot_prune_to_effective": res.get("pivot_prune_to_effective", res.get("pivot_prune_to", 0)),
        "pivot_source": res.get("pivot_source", run_args.pivot_source),
        "pivot_coord": res.get("pivot_coord", run_args.pivot_coord),
        "pivot_metric": res.get("pivot_metric", run_args.pivot_metric),
        "pivot_norm": res.get("pivot_norm", run_args.pivot_norm),
        "pivot_weight": res.get("pivot_weight", run_args.pivot_weight),
        "seed": args.seed,
        "n_queries": args.n_queries,
        "max_images": args.max_images,
    }
    logging.info(
        "Finished %s: CandRecall@topC=%.3f Recall@10=%.3f (topC=%s, ef=%s, m=%s)",
        name,
        float(row["CandRecall@topC"]),
        float(row["Recall@10"]),
        row["topC"],
        row["efSearch"],
        row["m"],
    )
    return row


def pick_best(rows: List[Dict[str, object]], key: str) -> Dict[str, object]:
    return max(rows, key=lambda r: (float(r.get(key, 0.0)), float(r.get("Recall@10", 0.0)), -float(r.get("avg_total_ms", 1e9))))


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
        ef_construction=200,
        force_recompute=args.force_recompute,
        max_images=args.max_images,
        max_captions=args.max_captions,
        k=args.k,
        allow_coco_train_download=args.allow_coco_train_download,
        coco_root=args.coco_root or None,
        drive_sync=None,
    )

    logging.info("Loading dataset and embeddings (max_images=%s)", args.max_images)
    records = load_dataset_records(base_config)
    caption_pairs = build_caption_pairs(records, base_config.max_captions)
    caption_image_ids = [pair[1] for pair in caption_pairs]

    model, processor = load_clip(base_config.model_name, base_config.device)
    image_embs, image_ids = load_or_compute_image_embeddings(records, base_config, model, processor)
    image_embs = ensure_float32_contig(image_embs)

    config_text = replace(base_config, batch_size=args.batch_size_text)
    caption_embs = load_or_compute_caption_embeddings(caption_pairs, config_text, model, processor)
    caption_embs = ensure_float32_contig(caption_embs)

    sampled_pairs, sampled_idx = sample_queries(caption_pairs, args.n_queries, args.seed)
    caption_embs_sample = caption_embs[sampled_idx]
    caption_image_ids_sample = [caption_image_ids[i] for i in sampled_idx]
    gt_ids = [pair[1] for pair in sampled_pairs]
    id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}
    gt_indices = np.array([id_to_index[g] for g in gt_ids], dtype=int)

    rows: List[Dict[str, object]] = []
    for group in SHORTLIST:
        rows.append(
            run_group(
                group["name"],
                group,
                args,
                base_config,
                caption_embs_sample,
                caption_image_ids_sample,
                image_embs,
                image_ids,
                gt_indices,
                gt_ids,
            )
        )

    if args.enable_topc_sweep and rows:
        best_by_cand = pick_best(rows, "CandRecall@topC")
        best_m = int(best_by_cand.get("m", args.pivot_m))
        logging.info("TopC sweep enabled; using best m=%d from shortlist", best_m)
        for topC in TOPC_SWEEP:
            name = f"sweep_topC{topC}"
            if any(r.get("group") == name for r in rows):
                continue
            rows.append(
                run_group(
                    name,
                    {
                        "pivot_m": best_m,
                        "topC": topC,
                        "pivot_efSearch": topC * 2,
                        "pivot_prune_to": 0,
                    },
                    args,
                    base_config,
                    caption_embs_sample,
                    caption_image_ids_sample,
                    image_embs,
                    image_ids,
                    gt_indices,
                    gt_ids,
                )
            )

    if not rows:
        logging.error("No results produced; check dataset or parameters")
        return

    best_by_recall10 = pick_best(rows, "Recall@10")
    best_by_cand = pick_best(rows, "CandRecall@topC")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = Path("results") / f"recall_shortlist_{timestamp}.csv"
    json_path = Path("results") / f"recall_shortlist_{timestamp}.json"
    ensure_dir(csv_path.parent)

    headers = [
        "group",
        "m",
        "topC",
        "efSearch",
        "prune_to",
        "CandRecall@topC",
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "avg_total_ms",
        "hnsw_ms",
        "pivot_map_ms",
        "rerank_ms",
        "pivot_prune_to_arg",
        "pivot_prune_to_effective",
        "pivot_source",
        "pivot_coord",
        "pivot_metric",
        "pivot_norm",
        "pivot_weight",
        "seed",
        "n_queries",
        "max_images",
    ]

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(h, "")) for h in headers) + "\n")

    save_json(
        {
            "rows": rows,
            "best_by_recall10": best_by_recall10,
            "best_by_candrecall": best_by_cand,
            "args": vars(args),
        },
        json_path,
    )

    print(
        "best setting: name=%s m=%s topC=%s efSearch=%s prune=%s CandRecall@topC=%.3f Recall@10=%.3f"
        % (
            best_by_recall10.get("group"),
            best_by_recall10.get("m"),
            best_by_recall10.get("topC"),
            best_by_recall10.get("efSearch"),
            best_by_recall10.get("prune_to"),
            float(best_by_recall10.get("CandRecall@topC", 0.0)),
            float(best_by_recall10.get("Recall@10", 0.0)),
        )
    )


if __name__ == "__main__":
    main()
