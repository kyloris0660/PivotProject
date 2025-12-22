"""Pivot coordinate reduction utilities.

Maps high-dimensional, L2-normalized embeddings into a low-dimensional pivot-distance
space T(x) = [1 - x·p1, ..., 1 - x·pm] so HNSW can operate cheaply before reranking
in the original space.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from tqdm import tqdm

from .config import RetrievalConfig
from .utils import ensure_dir, load_json, save_json


def _pivot_dir(config: RetrievalConfig) -> Path:
    path = config.cache_path("pivots")
    ensure_dir(path)
    return path


def select_pivots(
    embeddings: np.ndarray,
    config: RetrievalConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    pivot_dir = _pivot_dir(config)
    pivot_path = pivot_dir / f"pivots_m{config.m}_seed{config.seed}.npy"
    pivot_meta = pivot_dir / f"pivots_m{config.m}_seed{config.seed}.json"

    if pivot_path.exists() and pivot_meta.exists() and not config.force_recompute:
        logging.info("Loading pivots from %s", pivot_path)
        return np.load(pivot_path), np.array(load_json(pivot_meta)["indices"])

    logging.info("Selecting %d pivots using farthest-point selection", config.m)
    rng = np.random.RandomState(config.seed)
    num_points = embeddings.shape[0]

    if config.pivot_sample and config.pivot_sample < num_points:
        sample_indices = rng.choice(num_points, size=config.pivot_sample, replace=False)
    else:
        sample_indices = np.arange(num_points)

    sample_emb = embeddings[sample_indices]
    pivots: list[int] = []

    first_idx = int(rng.choice(len(sample_emb)))
    pivots.append(first_idx)

    min_dists = np.full(len(sample_emb), np.inf)
    for _ in tqdm(range(1, config.m), desc="pivots"):
        last_pivot = sample_emb[pivots[-1]]
        dists = 1.0 - sample_emb @ last_pivot
        min_dists = np.minimum(min_dists, dists)
        next_idx = int(np.argmax(min_dists))
        pivots.append(next_idx)

    pivot_vectors = sample_emb[pivots]
    pivot_indices_global = sample_indices[pivots]

    np.save(pivot_path, pivot_vectors)
    save_json({"indices": pivot_indices_global.tolist()}, pivot_meta)
    logging.info("Saved pivots to %s", pivot_path)
    return pivot_vectors, pivot_indices_global


def compute_pivot_coordinates(
    embeddings: np.ndarray, pivots: np.ndarray, config: RetrievalConfig, split: str
) -> np.ndarray:
    pivot_dir = _pivot_dir(config)
    coord_path = pivot_dir / f"pivot_coords_images_{split}_m{config.m}.npy"
    if coord_path.exists() and not config.force_recompute:
        logging.info("Loading pivot coordinates from %s", coord_path)
        return np.load(coord_path)

    logging.info("Computing pivot coordinates for %d vectors", embeddings.shape[0])
    coords = 1.0 - embeddings @ pivots.T
    np.save(coord_path, coords)
    logging.info("Saved pivot coordinates to %s", coord_path)
    return coords
