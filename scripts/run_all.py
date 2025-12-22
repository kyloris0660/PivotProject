import argparse
import logging
from datetime import datetime
from typing import Dict


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
from retrieval.plotting import plot_latency_vs_recall
from retrieval.utils import ensure_dir, save_json, set_seed, setup_logging


def parse_args() -> RetrievalConfig:
    parser = argparse.ArgumentParser(
        description="Pivot-based coordinate reduction + HNSW on Flickr30k"
    )
    parser.add_argument("--source", default="hf", choices=["hf"], help="data source")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--model", dest="model_name", default="openai/clip-vit-base-patch32"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--m", type=int, default=16, help="number of pivots")
    parser.add_argument(
        "--topC", type=int, default=500, help="candidate pool from HNSW"
    )
    parser.add_argument("--k", type=int, default=10, help="final top-k")
    parser.add_argument("--pivot_sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--efc", dest="ef_construction", type=int, default=200)
    parser.add_argument("--efs", dest="ef_search", type=int, default=128)
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--max_captions", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)

    args = parser.parse_args()
    return RetrievalConfig(
        source=args.source,
        split=args.split,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        m=args.m,
        topC=args.topC,
        k=args.k,
        pivot_sample=args.pivot_sample,
        seed=args.seed,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        force_recompute=args.force_recompute,
        max_captions=args.max_captions,
        max_images=args.max_images,
    )


def main() -> None:
    config = parse_args()
    setup_logging()
    set_seed(config.seed)

    ensure_dir(config.cache_dir)
    ensure_dir(config.results_dir)
    ensure_dir(config.plots_dir)

    records = load_flickr30k(config)
    caption_pairs = build_caption_pairs(records, config.max_captions)
    logging.info("Caption pairs: %d", len(caption_pairs))

    model, processor = load_clip(config.model_name, config.device)
    image_embs, image_ids = load_or_compute_image_embeddings(
        records, config, model, processor
    )
    caption_embs = load_or_compute_caption_embeddings(
        caption_pairs, config, model, processor
    )

    pivots, pivot_indices = select_pivots(image_embs, config)
    pivot_coords = compute_pivot_coordinates(image_embs, pivots, config, config.split)

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    meta: Dict[str, str | int | float] = {
        "m": config.m,
        "topC": config.topC,
        "k": config.k,
        "model_name": config.model_name,
        "M": config.M,
        "efConstruction": config.ef_construction,
        "efSearch": config.ef_search,
        "seed": config.seed,
        "device": config.device,
    }

    all_results = {**metrics, **meta}
    stem = config.result_stem(timestamp)
    json_path = config.results_dir / f"{stem}.json"
    csv_path = config.results_dir / f"{stem}.csv"
    save_json(all_results, json_path)
    with csv_path.open("w", encoding="utf-8") as f:
        headers = list(all_results.keys())
        f.write(",".join(headers) + "\n")
        f.write(",".join(str(all_results[h]) for h in headers) + "\n")

    plot_path = plot_latency_vs_recall(metrics, config, timestamp)

    logging.info("Summary: %s", all_results)
    logging.info("Results saved: %s | %s", json_path, csv_path)
    logging.info("Plot saved: %s", plot_path)


if __name__ == "__main__":
    main()
