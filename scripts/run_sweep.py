"""Grid sweep for pivot/HNSW params on a small subset.

Runs multiple configurations on the same subset (max_images/max_captions) and
reports the best by Recall@10. Results are aggregated to CSV and a scatter
plot of Recall@10 vs latency is saved.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from retrieval.config import RetrievalConfig
from retrieval.data import build_caption_pairs, load_flickr30k
from retrieval.embeddings import (
    load_clip,
    load_or_compute_caption_embeddings,
    load_or_compute_image_embeddings,
)
from retrieval.evaluation import evaluate_retrieval
from retrieval.index import build_or_load_index
from retrieval.pivots import compute_pivot_coordinates, select_pivots
from retrieval.utils import ensure_dir, save_json, set_seed, setup_logging


def parse_list(arg: str) -> List[int]:
    return [int(x) for x in arg.split(",") if x]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep pivot/HNSW params on subset")
    p.add_argument("--device", default="cuda")
    p.add_argument("--split", default="test")
    p.add_argument("--model", dest="model_name", default="openai/clip-vit-base-patch32")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_images", type=int, default=2000)
    p.add_argument("--max_captions", type=int, default=10000)
    p.add_argument("--ms", type=parse_list, default="16,20,24,32")
    p.add_argument("--topCs", type=parse_list, default="500,800,1000")
    p.add_argument("--efSearches", type=parse_list, default="128,256")
    p.add_argument("--Ms", type=parse_list, default="32,48")
    p.add_argument("--efc", dest="ef_construction", type=int, default=200)
    p.add_argument("--pivot_sample", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force_recompute", action="store_true")
    return p.parse_args()


def run_one(
    config: RetrievalConfig, records, caption_pairs, image_embs, image_ids, caption_embs
) -> Dict[str, float]:
    caption_image_ids = [pair[1] for pair in caption_pairs]
    pivots, _, _ = select_pivots(
        image_embs, image_ids, caption_embs, caption_image_ids, config
    )
    pivot_coords = compute_pivot_coordinates(
        image_embs, pivots, config, config.split, kind="images"
    )
    hnsw_index, build_time = build_or_load_index(pivot_coords, config)
    metrics = evaluate_retrieval(
        caption_pairs=caption_pairs,
        caption_embs=caption_embs,
        image_embs=image_embs,
        pivot_vectors=pivots,
        image_ids=image_ids,
        hnsw_index=hnsw_index,
        config=config,
    )
    metrics["build_index_time_sec"] = build_time
    return metrics


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)

    base_config = RetrievalConfig(
        source="hf",
        split=args.split,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pivot_sample=args.pivot_sample,
        seed=args.seed,
        ef_construction=args.ef_construction,
        force_recompute=args.force_recompute,
        max_images=args.max_images,
        max_captions=args.max_captions,
    )

    # Load data and embeddings once.
    records = load_flickr30k(base_config)
    caption_pairs = build_caption_pairs(records, base_config.max_captions)
    model, processor = load_clip(base_config.model_name, base_config.device)
    image_embs, image_ids = load_or_compute_image_embeddings(
        records, base_config, model, processor
    )
    caption_embs = load_or_compute_caption_embeddings(
        caption_pairs, base_config, model, processor
    )

    grid: List[Tuple[int, int, int, int]] = []
    for m in args.ms:
        for topC in args.topCs:
            for ef_s in args.efSearches:
                for M in args.Ms:
                    grid.append((m, topC, ef_s, M))

    results: List[Dict[str, float | int | str]] = []
    logging.info("Running %d configs", len(grid))
    for m, topC, ef_s, M in grid:
        cfg = base_config
        cfg.m = m
        cfg.topC = topC
        cfg.ef_search = ef_s
        cfg.M = M
        metrics = run_one(
            cfg, records, caption_pairs, image_embs, image_ids, caption_embs
        )
        row: Dict[str, float | int | str] = {
            **metrics,
            "m": m,
            "topC": topC,
            "efSearch": ef_s,
            "M": M,
            "model_name": base_config.model_name,
            "seed": base_config.seed,
            "device": base_config.device,
            "max_images": base_config.max_images,
            "max_captions": base_config.max_captions,
        }
        results.append(row)
        logging.info(
            "m=%d topC=%d efS=%d M=%d | R@1=%.4f R@5=%.4f R@10=%.4f total_ms=%.3f",
            m,
            topC,
            ef_s,
            M,
            metrics.get("Recall@1", 0.0),
            metrics.get("Recall@5", 0.0),
            metrics.get("Recall@10", 0.0),
            metrics.get("avg_total_ms", 0.0),
        )

    # Pick best by Recall@10, then Recall@5, then avg_total_ms (lower is better).
    results_sorted = sorted(
        results,
        key=lambda r: (
            -float(r.get("Recall@10", 0)),
            -float(r.get("Recall@5", 0)),
            float(r.get("avg_total_ms", 1e9)),
        ),
    )
    best = results_sorted[0]
    logging.info("Best config: %s", best)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    summary_path = Path("results") / f"sweep_{timestamp}.csv"
    ensure_dir(summary_path.parent)
    headers = list(results[0].keys())
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in results_sorted:
            f.write(",".join(str(r[h]) for h in headers) + "\n")

    # Scatter plot Recall@10 vs latency.
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [float(r.get("avg_total_ms", 0)) for r in results_sorted]
    y = [float(r.get("Recall@10", 0)) for r in results_sorted]
    labels = [
        f"m{r['m']}_C{r['topC']}_ef{r['efSearch']}_M{r['M']}" for r in results_sorted
    ]
    ax.scatter(x, y, c="#4c72b0")
    for xi, yi, lbl in zip(x, y, labels):
        ax.text(xi, yi, lbl, fontsize=7)
    ax.set_xlabel("Avg total ms")
    ax.set_ylabel("Recall@10")
    ax.set_title("Sweep on subset")
    fig.tight_layout()
    plot_path = Path("plots") / f"sweep_{timestamp}.png"
    ensure_dir(plot_path.parent)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    best_path = Path("results") / f"sweep_{timestamp}_best.json"
    save_json(best, best_path)
    logging.info("Saved sweep summary to %s and plot to %s", summary_path, plot_path)
    logging.info("Best config saved to %s", best_path)


if __name__ == "__main__":
    main()
