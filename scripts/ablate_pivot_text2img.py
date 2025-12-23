"""Systematic ablation for pivot text->image retrieval (COCO val 5k).

Runs pivot+HNSW with configurable pivot sources/coords/metrics/weights and
outputs a CSV plus two scatter plots:
1) CandRecall@topC vs avg_total_ms
2) Recall@10 vs avg_total_ms
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
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


def parse_list(arg: str) -> List[str]:
    return [x for x in arg.split(",") if x]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablate pivot settings for text->image")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dataset", default="coco_captions")
    p.add_argument("--source", default="hf")
    p.add_argument("--split", default="val")
    p.add_argument("--model_name", default="openai/clip-vit-base-patch32")
    p.add_argument("--batch_size_text", type=int, default=64)
    p.add_argument("--batch_size_image", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_threads", type=int, default=4)
    p.add_argument("--n_queries", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--pivot_prune_to", type=int, default=0)
    p.add_argument("--pivot_efSearch", type=int, default=128)
    p.add_argument("--pivot_M", type=int, default=24)
    p.add_argument("--topC", type=int, default=600)
    p.add_argument("--efc", dest="ef_construction", type=int, default=200)
    p.add_argument("--pivot_sample", type=int, default=5000)
    p.add_argument("--pivot_pool_size", type=int, default=50000)
    p.add_argument("--pivot_mix_ratio", type=float, default=0.5)
    p.add_argument("--pivot_weight_eps", type=float, default=1e-6)
    p.add_argument("--pivot_learn_pairs", type=int, default=20000)
    p.add_argument("--pivot_learn_queries", type=int, default=2000)
    p.add_argument("--pivot_learn_negs", type=int, default=8)
    p.add_argument("--pivot_sources", type=parse_list, default="images,union,mixture")
    p.add_argument("--pivot_coords", type=parse_list, default="dist,sim")
    p.add_argument("--pivot_metrics", type=parse_list, default="l2,cosine")
    p.add_argument("--pivot_weights", type=parse_list, default="none,variance,learned")
    p.add_argument("--pivot_norms", type=parse_list, default="none,zscore")
    p.add_argument("--ms", type=parse_list, default="16,32")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_images", type=int, default=5000)
    p.add_argument("--max_captions", type=int, default=None)
    p.add_argument("--rerank_device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--force_recompute", action="store_true")
    p.add_argument("--allow_coco_train_download", action="store_true")
    return p.parse_args()


def build_args_template(cli: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=cli.dataset,
        source=cli.source,
        split=cli.split,
        model_name=cli.model_name,
        device=cli.device,
        seed=cli.seed,
        pivot_prune_to=cli.pivot_prune_to,
        pivot_efSearch=cli.pivot_efSearch,
        pivot_M=cli.pivot_M,
        topC=cli.topC,
        batch_size_text=cli.batch_size_text,
        warmup=cli.warmup,
        num_threads=cli.num_threads,
        rerank_device=cli.rerank_device,
        pivot_norm="none",
        pivot_weight="none",
        pivot_source="images",
        pivot_mix_ratio=cli.pivot_mix_ratio,
        pivot_pool_size=cli.pivot_pool_size,
        pivot_coord="sim",
        pivot_metric="l2",
        pivot_weight_eps=cli.pivot_weight_eps,
        pivot_learn_pairs=cli.pivot_learn_pairs,
        pivot_learn_queries=cli.pivot_learn_queries,
        pivot_learn_negs=cli.pivot_learn_negs,
        pivot_m=16,
        topC_override=None,
        orig_hnsw_efSearch=128,
        orig_hnsw_M=24,
        pivot_sample=cli.pivot_sample,
        pivot_pool_size_override=None,
        pivot_weight_override=None,
        pivot_coord_override=None,
        pivot_metric_override=None,
        pivot_source_override=None,
        pivot_mix_ratio_override=None,
        pivot_norm_override=None,
        pivot_m_override=None,
        pivot_weight_eps_override=None,
        pivot_prune_to_override=None,
        force_recompute=cli.force_recompute,
    )


def main() -> None:
    cli = parse_args()
    setup_logging()
    set_seed(cli.seed)

    base_config = RetrievalConfig(
        dataset=cli.dataset,
        source=cli.source,
        split=cli.split,
        model_name=cli.model_name,
        device=cli.device,
        batch_size=cli.batch_size_image,
        num_workers=cli.num_workers,
        pivot_sample=cli.pivot_sample,
        pivot_source="images",
        pivot_mix_ratio=cli.pivot_mix_ratio,
        pivot_pool_size=cli.pivot_pool_size,
        pivot_coord="sim",
        pivot_metric="l2",
        pivot_weight="none",
        pivot_weight_eps=cli.pivot_weight_eps,
        pivot_learn_pairs=cli.pivot_learn_pairs,
        pivot_learn_queries=cli.pivot_learn_queries,
        pivot_learn_negs=cli.pivot_learn_negs,
        seed=cli.seed,
        ef_construction=cli.ef_construction,
        force_recompute=cli.force_recompute,
        max_images=cli.max_images,
        max_captions=cli.max_captions,
        k=cli.k,
        allow_coco_train_download=cli.allow_coco_train_download,
    )

    logging.info("Loading dataset and embeddings")
    records = load_dataset_records(base_config)
    caption_pairs = build_caption_pairs(records, base_config.max_captions)
    caption_image_ids = [pair[1] for pair in caption_pairs]

    model, processor = load_clip(base_config.model_name, base_config.device)
    image_embs, image_ids = load_or_compute_image_embeddings(
        records, base_config, model, processor
    )
    image_embs = base.ensure_float32_contig(image_embs)

    config_text = replace(base_config, batch_size=cli.batch_size_text)
    caption_embs = load_or_compute_caption_embeddings(
        caption_pairs, config_text, model, processor
    )
    caption_embs = base.ensure_float32_contig(caption_embs)

    sampled_pairs, sampled_idx = base.sample_queries(
        caption_pairs, cli.n_queries, cli.seed
    )
    caption_embs_sample = caption_embs[sampled_idx]
    caption_image_ids_sample = [caption_image_ids[i] for i in sampled_idx]
    gt_ids = [pair[1] for pair in sampled_pairs]
    id_to_index = {img_id: idx for idx, img_id in enumerate(image_ids)}
    gt_indices = np.array([id_to_index[g] for g in gt_ids], dtype=int)

    results: List[Dict[str, float | int | str]] = []
    arg_template = build_args_template(cli)

    for source in cli.pivot_sources:
        for coord in cli.pivot_coords:
            for metric in cli.pivot_metrics:
                for weight in cli.pivot_weights:
                    for norm in cli.pivot_norms:
                        for m in [int(x) for x in cli.ms]:
                            run_args = SimpleNamespace(**vars(arg_template))
                            run_args.pivot_source = source
                            run_args.pivot_coord = coord
                            run_args.pivot_metric = metric
                            run_args.pivot_weight = weight
                            run_args.pivot_norm = norm
                            run_args.pivot_m = m
                            run_args.pivot_mix_ratio = cli.pivot_mix_ratio
                            run_args.pivot_pool_size = cli.pivot_pool_size

                            res = base.pivot_hnsw_method(
                                caption_embs_sample,
                                caption_image_ids_sample,
                                image_embs,
                                image_ids,
                                gt_indices,
                                base_config,
                                run_args,
                            )
                            recalls = base.compute_recalls(res.pop("preds"), gt_ids)
                            res.update(recalls)
                            res.update(
                                {
                                    "pivot_source": source,
                                    "pivot_coord": coord,
                                    "pivot_metric": metric,
                                    "pivot_weight": weight,
                                    "pivot_norm": norm,
                                    "m": m,
                                    "topC": cli.topC,
                                }
                            )
                            results.append(res)
                            logging.info(
                                "source=%s coord=%s metric=%s weight=%s norm=%s m=%d | CandRecall@topC=%.3f Recall@10=%.3f avg_ms=%.2f",
                                source,
                                coord,
                                metric,
                                weight,
                                norm,
                                m,
                                res.get("CandRecall@topC", 0.0),
                                res.get("Recall@10", 0.0),
                                res.get("avg_total_ms", 0.0),
                            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    summary_path = Path("results") / f"ablate_pivot_{timestamp}.csv"
    json_path = Path("results") / f"ablate_pivot_{timestamp}.json"
    ensure_dir(summary_path.parent)

    if results:
        headers = list(results[0].keys())
        with summary_path.open("w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for r in results:
                f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
        save_json(results, json_path)

        # Plots
        def scatter(x_key: str, y_key: str, title: str, fname: str) -> None:
            fig, ax = plt.subplots(figsize=(6, 4))
            x = [float(r.get(x_key, 0)) for r in results]
            y = [float(r.get(y_key, 0)) for r in results]
            labels = [
                f"{r.get('pivot_source')}-{r.get('pivot_coord')}-{r.get('pivot_metric')}-{r.get('pivot_weight')}-m{r.get('m')}"
                for r in results
            ]
            ax.scatter(x, y, c="#4c72b0")
            for xi, yi, lbl in zip(x, y, labels):
                ax.text(xi, yi, lbl, fontsize=7)
            ax.set_xlabel(x_key)
            ax.set_ylabel(y_key)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plot_path = Path("plots") / fname
            ensure_dir(plot_path.parent)
            fig.savefig(plot_path, dpi=200)
            plt.close(fig)

        scatter(
            "avg_total_ms",
            "CandRecall@topC",
            "CandRecall@topC vs latency",
            f"ablate_candrecall_vs_latency_{timestamp}.png",
        )
        scatter(
            "avg_total_ms",
            "Recall@10",
            "Recall@10 vs latency",
            f"ablate_R10_vs_latency_{timestamp}.png",
        )

    logging.info(
        "Ablation saved: %s, %s and plots in plots/ablate_*_%s.png",
        summary_path,
        json_path,
        timestamp,
    )


if __name__ == "__main__":
    main()
