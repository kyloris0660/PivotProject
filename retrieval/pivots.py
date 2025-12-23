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


def _pool_tag(config: RetrievalConfig) -> str:
    extra = ""
    if config.pivot_source == "caption_cluster_guided_images":
        extra = f"_capsamp{config.pivot_caption_sample}"
    return f"src{config.pivot_source}_mix{config.pivot_mix_ratio}_pool{config.pivot_pool_size}{extra}"


def _fps_with_seed(
    pool: np.ndarray, seed_indices: list[int], target_m: int, rng: np.random.RandomState
) -> list[int]:
    """Farthest-point sampling continuing from provided seeds (on normalized pool)."""

    if target_m <= 0:
        return []

    selected: list[int] = list(seed_indices)
    if not selected:
        selected.append(int(rng.choice(pool.shape[0])))

    min_dists = np.full(pool.shape[0], np.inf, dtype=np.float32)
    for idx in selected:
        # cosine distance on normalized vectors
        min_dists = np.minimum(min_dists, 1.0 - pool @ pool[idx])
        min_dists[idx] = -np.inf  # do not reselect the same point

    while len(selected) < target_m:
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        min_dists = np.minimum(min_dists, 1.0 - pool @ pool[next_idx])
        min_dists[next_idx] = -np.inf

    return selected


def build_pivot_pool(
    image_embs: np.ndarray,
    image_ids: list[str],
    caption_embs: np.ndarray | None,
    caption_image_ids: list[str] | None,
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
    elif source == "caption_guided_images":
        if cap_pool is None or caption_image_ids is None:
            raise ValueError(
                "pivot_source=caption_guided_images requires caption embeddings and caption_image_ids"
            )
        id_to_idx = {img_id: idx for idx, img_id in enumerate(image_ids)}
        sums = np.zeros_like(img_pool)
        counts = np.zeros((img_pool.shape[0],), dtype=np.int32)
        for emb, img_id in zip(cap_pool, caption_image_ids):
            idx = id_to_idx.get(img_id)
            if idx is None:
                continue
            sums[idx] += emb
            counts[idx] += 1
        mask = counts > 0
        guided = img_pool.copy()
        guided[mask] = sums[mask] / counts[mask][:, None]
        guided = _l2_normalize(guided)
        guided = np.ascontiguousarray(guided)
        pool = _subset(guided, min(pool_size, guided.shape[0]), rng)
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

    tag = _pool_tag(config)
    return pool, tag


def _select_caption_cluster_guided_images(
    image_embs: np.ndarray,
    image_ids: list[str],
    caption_embs: np.ndarray,
    config: RetrievalConfig,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Cluster captions, pick nearest images per centroid, then FPS fill to m."""

    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "scikit-learn is required for caption_cluster_guided_images"
        ) from exc

    rng = np.random.RandomState(config.seed)
    img_pool = _l2_normalize(image_embs.astype(np.float32, copy=False))
    cap_pool = _l2_normalize(caption_embs.astype(np.float32, copy=False))

    sample_size = min(cap_pool.shape[0], max(1, config.pivot_caption_sample))
    cap_sample = cap_pool
    if cap_pool.shape[0] > sample_size:
        idx = rng.choice(cap_pool.shape[0], size=sample_size, replace=False)
        idx.sort()
        cap_sample = cap_pool[idx]

    logging.info(
        "Clustering %d sampled captions into %d centroids for pivot guidance",
        cap_sample.shape[0],
        config.m,
    )
    kmeans = MiniBatchKMeans(
        n_clusters=config.m,
        random_state=config.seed,
        batch_size=2048,
        n_init="auto",
    )
    kmeans.fit(cap_sample)
    centroids = _l2_normalize(kmeans.cluster_centers_.astype(np.float32, copy=False))

    sims = centroids @ img_pool.T  # (m, N)
    best_idx = np.argmax(sims, axis=1)

    pivot_indices: list[int] = []
    seen = set()
    for idx in best_idx:
        if idx not in seen:
            pivot_indices.append(int(idx))
            seen.add(int(idx))

    if len(pivot_indices) < config.m:
        logging.info(
            "Filling %d remaining pivots using farthest-point sampling on images",
            config.m - len(pivot_indices),
        )
        pivot_indices = _fps_with_seed(img_pool, pivot_indices, config.m, rng)

    pivot_vectors = img_pool[pivot_indices].astype(np.float32, copy=False)
    pool_tag = _pool_tag(config)

    stats: dict = {}
    unique_count = len(set(pivot_indices))
    stats["pivot_unique_count"] = unique_count
    stats["pivot_duplicate_count"] = len(pivot_indices) - unique_count

    # Coverage self-check on a small caption sample
    cover_sample = min(cap_pool.shape[0], 200)
    if cover_sample > 0:
        cover_idx = rng.choice(cap_pool.shape[0], size=cover_sample, replace=False)
        cover_caps = cap_pool[cover_idx]
        max_sims = np.max(cover_caps @ pivot_vectors.T, axis=1)
        stats["pivot_cover_maxsim_mean"] = float(np.mean(max_sims))
        stats["pivot_cover_maxsim_p50"] = float(np.percentile(max_sims, 50))
        stats["pivot_cover_maxsim_p90"] = float(np.percentile(max_sims, 90))

    meta = {
        "indices": pivot_indices,
        "pool_tag": pool_tag,
        "pivot_image_ids": [image_ids[i] for i in pivot_indices],
        "stats": stats,
    }
    return pivot_vectors, np.array(pivot_indices, dtype=int), meta


def _pivot_cache_suffix(config: RetrievalConfig, tag: str) -> str:
    subset_tag = f"_n{config.max_images}" if config.max_images is not None else ""
    return f"{config.dataset}_{config.split}_{tag}_m{config.m}_seed{config.seed}{subset_tag}"


def select_pivots(
    image_embs: np.ndarray,
    image_ids: list[str],
    caption_embs: np.ndarray | None,
    caption_image_ids: list[str] | None,
    config: RetrievalConfig,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Select pivots via farthest-point sampling from the configured pool."""

    pivot_dir = _pivot_dir(config)

    if config.pivot_source == "caption_cluster_guided_images":
        if caption_embs is None:
            raise ValueError(
                "pivot_source=caption_cluster_guided_images requires caption embeddings"
            )

        pool_tag = (
            f"srccaption_cluster_guided_images_mix{config.pivot_mix_ratio}_pool{config.pivot_pool_size}"
            f"_capsamp{config.pivot_caption_sample}"
        )
        suffix = _pivot_cache_suffix(config, pool_tag)
        pivot_path = pivot_dir / f"pivots_{suffix}.npy"
        pivot_meta = pivot_dir / f"pivots_{suffix}.json"

        if pivot_path.exists() and pivot_meta.exists() and not config.force_recompute:
            logging.info("Loading pivots from %s", pivot_path)
            pivots = np.load(pivot_path)
            meta = load_json(pivot_meta)
            pivot_indices = np.array(meta.get("indices", []), dtype=int)
            return pivots, pivot_indices, meta

        pivots, pivot_indices, meta = _select_caption_cluster_guided_images(
            image_embs, image_ids, caption_embs, config
        )
        meta["pool_tag"] = pool_tag
        np.save(pivot_path, pivots)
        save_json(meta, pivot_meta)
        logging.info("Saved pivots to %s", pivot_path)
        return pivots, pivot_indices, meta

    pool, pool_tag = build_pivot_pool(
        image_embs, image_ids, caption_embs, caption_image_ids, config
    )
    suffix = _pivot_cache_suffix(config, pool_tag)
    pivot_path = pivot_dir / f"pivots_{suffix}.npy"
    pivot_meta = pivot_dir / f"pivots_{suffix}.json"

    if pivot_path.exists() and pivot_meta.exists() and not config.force_recompute:
        logging.info("Loading pivots from %s", pivot_path)
        pivots = np.load(pivot_path)
        meta = load_json(pivot_meta)
        pivot_indices = np.array(meta.get("indices", []))
        return pivots, pivot_indices, meta

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

    meta = {"indices": pivot_indices.tolist(), "pool_tag": pool_tag}
    np.save(pivot_path, pivot_vectors)
    save_json(meta, pivot_meta)
    logging.info("Saved pivots to %s", pivot_path)
    return pivot_vectors, pivot_indices, meta


def compute_pivot_coordinates(
    embeddings: np.ndarray,
    pivots: np.ndarray,
    config: RetrievalConfig,
    split: str,
    kind: str,
) -> np.ndarray:
    """Map embeddings to pivot coordinates (sim or dist), cached per config."""

    pivot_dir = _pivot_dir(config)
    pool_tag = _pivot_cache_suffix(config, _pool_tag(config))
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
    suffix = _pivot_cache_suffix(config, _pool_tag(config))
    fname = f"pivot_weights_{config.pivot_weight}_{config.pivot_coord}_{config.pivot_metric}_{suffix}.npy"
    meta = f"pivot_weights_{config.pivot_weight}_{config.pivot_coord}_{config.pivot_metric}_{suffix}.json"
    return pivot_dir / fname, pivot_dir / meta
