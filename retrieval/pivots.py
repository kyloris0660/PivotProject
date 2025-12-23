"""Pivot utilities: pool building, pivot selection, coordinate mapping."""

from __future__ import annotations

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


def _l2_normalize(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr / norms


def _subset(arr: np.ndarray, size: int, rng: np.random.RandomState) -> np.ndarray:
    if arr.shape[0] <= size:
        return arr
    idx = rng.choice(arr.shape[0], size=size, replace=False)
    idx.sort()
    return arr[idx]


def build_pivot_pool(
    image_embs: np.ndarray,
    caption_embs: np.ndarray | None,
    config: RetrievalConfig,
) -> Tuple[np.ndarray, str]:
    """Build a normalized pivot pool according to the configured source."""

    rng = np.random.RandomState(config.seed)
    pool_size = config.pivot_pool_size

    img_pool = _l2_normalize(image_embs.astype(np.float32, copy=False))
    img_pool = np.ascontiguousarray(img_pool)

    cap_pool = None
    if caption_embs is not None:
        cap_pool = _l2_normalize(caption_embs.astype(np.float32, copy=False))
        cap_pool = np.ascontiguousarray(cap_pool)

    source = config.pivot_source
    if source == "images":
        pool = _subset(img_pool, min(pool_size, img_pool.shape[0]), rng)
    elif source == "captions":
        if cap_pool is None:
            raise ValueError("pivot_source=captions requires caption embeddings")
        pool = _subset(cap_pool, min(pool_size, cap_pool.shape[0]), rng)
    elif source == "union":
        if cap_pool is None:
            raise ValueError("pivot_source=union requires caption embeddings")
        combined = np.concatenate([img_pool, cap_pool], axis=0)
        pool = _subset(combined, min(pool_size, combined.shape[0]), rng)
    elif source == "mixture":
        if cap_pool is None:
            raise ValueError("pivot_source=mixture requires caption embeddings")
        cap_target = int(pool_size * config.pivot_mix_ratio)
        img_target = max(pool_size - cap_target, 0)
        cap_part = _subset(cap_pool, min(cap_target, cap_pool.shape[0]), rng)
        img_part = _subset(img_pool, min(img_target, img_pool.shape[0]), rng)
        deficit = pool_size - (cap_part.shape[0] + img_part.shape[0])
        if deficit > 0:
            extra_source = cap_pool if cap_part.shape[0] < cap_target else img_pool
            extra = _subset(extra_source, min(deficit, extra_source.shape[0]), rng)
            pool = np.concatenate([img_part, cap_part, extra], axis=0)
        else:
            pool = np.concatenate([img_part, cap_part], axis=0)
        if pool.shape[0] > pool_size:
            pool = _subset(pool, pool_size, rng)
    else:
        raise ValueError(f"Unknown pivot_source '{source}'")

    tag = f"src{source}_mix{config.pivot_mix_ratio}_pool{config.pivot_pool_size}"
    return pool, tag


def _pivot_cache_suffix(config: RetrievalConfig, tag: str) -> str:
    subset_tag = f"_n{config.max_images}" if config.max_images is not None else ""
    return f"{config.dataset}_{config.split}_{tag}_m{config.m}_seed{config.seed}{subset_tag}"


def select_pivots(
    image_embs: np.ndarray,
    caption_embs: np.ndarray | None,
    config: RetrievalConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select pivots via farthest-point sampling from the configured pool."""

    pivot_dir = _pivot_dir(config)
    pool, pool_tag = build_pivot_pool(image_embs, caption_embs, config)
    suffix = _pivot_cache_suffix(config, pool_tag)
    pivot_path = pivot_dir / f"pivots_{suffix}.npy"
    pivot_meta = pivot_dir / f"pivots_{suffix}.json"

    if pivot_path.exists() and pivot_meta.exists() and not config.force_recompute:
        logging.info("Loading pivots from %s", pivot_path)
        return np.load(pivot_path), np.array(load_json(pivot_meta).get("indices", []))

    logging.info(
        "Selecting %d pivots using farthest-point sampling from %d pool items",
        config.m,
        pool.shape[0],
    )
    rng = np.random.RandomState(config.seed)
    pivots: list[int] = []

    first_idx = int(rng.choice(pool.shape[0]))
    pivots.append(first_idx)

    min_dists = np.full(pool.shape[0], np.inf, dtype=np.float32)
    for _ in tqdm(range(1, config.m), desc="pivots"):
        last_pivot = pool[pivots[-1]]
        dists = 1.0 - pool @ last_pivot
        min_dists = np.minimum(min_dists, dists)
        next_idx = int(np.argmax(min_dists))
        pivots.append(next_idx)

    pivot_vectors = pool[pivots].astype(np.float32, copy=False)
    pivot_indices = np.array(pivots, dtype=int)

    np.save(pivot_path, pivot_vectors)
    save_json({"indices": pivot_indices.tolist(), "pool_tag": pool_tag}, pivot_meta)
    logging.info("Saved pivots to %s", pivot_path)
    return pivot_vectors, pivot_indices


def compute_pivot_coordinates(
    embeddings: np.ndarray,
    pivots: np.ndarray,
    config: RetrievalConfig,
    split: str,
    kind: str,
) -> np.ndarray:
    """Map embeddings to pivot coordinates (sim or dist), cached per config."""

    pivot_dir = _pivot_dir(config)
    pool_tag = _pivot_cache_suffix(
        config,
        f"src{config.pivot_source}_mix{config.pivot_mix_ratio}_pool{config.pivot_pool_size}",
    )
    coord_tag = f"{kind}_{config.dataset}_{split}_{config.pivot_coord}_{config.pivot_metric}_{pool_tag}"
    coord_path = pivot_dir / f"pivot_coords_{coord_tag}.npy"
    if coord_path.exists() and not config.force_recompute:
        logging.info("Loading pivot coordinates from %s", coord_path)
        return np.load(coord_path)

    logging.info(
        "Computing pivot coordinates (%s/%s) for %d vectors",
        config.pivot_coord,
        config.pivot_metric,
        embeddings.shape[0],
    )
    base = embeddings @ pivots.T
    if config.pivot_coord == "dist":
        coords = 1.0 - base
    else:
        coords = base

    if config.pivot_metric == "cosine":
        coords = _l2_normalize(coords.astype(np.float32, copy=False))
    else:
        coords = coords.astype(np.float32, copy=False)
    coords = np.ascontiguousarray(coords)

    np.save(coord_path, coords)
    logging.info("Saved pivot coordinates to %s", coord_path)
    return coords


def pivot_weight_paths(config: RetrievalConfig) -> Tuple[Path, Path]:
    pivot_dir = _pivot_dir(config)
    suffix = _pivot_cache_suffix(
        config,
        f"src{config.pivot_source}_mix{config.pivot_mix_ratio}_pool{config.pivot_pool_size}",
    )
    fname = f"pivot_weights_{config.pivot_weight}_{config.pivot_coord}_{config.pivot_metric}_{suffix}.npy"
    meta = f"pivot_weights_{config.pivot_weight}_{config.pivot_coord}_{config.pivot_metric}_{suffix}.json"
    return pivot_dir / fname, pivot_dir / meta
