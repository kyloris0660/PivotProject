import logging
import time
from pathlib import Path
from typing import Tuple

import hnswlib
import numpy as np

from .config import RetrievalConfig
from .utils import ensure_dir, load_json, save_json


def _index_dir(config: RetrievalConfig) -> Path:
    path = config.cache_path("index")
    ensure_dir(path)
    return path


def build_or_load_index(
    data: np.ndarray, config: RetrievalConfig, space: str = "l2"
) -> Tuple[hnswlib.Index, float]:
    idx_dir = _index_dir(config)
    index_path = (
        idx_dir
        / f"hnsw_{config.split}_m{config.m}_M{config.M}_efc{config.ef_construction}.bin"
    )
    meta_path = (
        idx_dir
        / f"hnsw_{config.split}_m{config.m}_M{config.M}_efc{config.ef_construction}.meta.json"
    )

    if index_path.exists() and meta_path.exists() and not config.force_recompute:
        logging.info("Loading HNSW index from %s", index_path)
        meta = load_json(meta_path)
        index = hnswlib.Index(space=space, dim=meta["dim"])
        index.load_index(str(index_path), max_elements=meta["max_elements"])
        index.set_ef(config.ef_search)
        return index, 0.0

    logging.info(
        "Building new HNSW index: dim=%d, M=%d, efC=%d",
        data.shape[1],
        config.M,
        config.ef_construction,
    )
    index = hnswlib.Index(space=space, dim=data.shape[1])
    start = time.perf_counter()
    index.init_index(
        max_elements=data.shape[0], ef_construction=config.ef_construction, M=config.M
    )
    index.add_items(data, np.arange(data.shape[0]))
    index.set_ef(config.ef_search)
    build_time = time.perf_counter() - start

    index.save_index(str(index_path))
    save_json(
        {"dim": int(data.shape[1]), "max_elements": int(data.shape[0])}, meta_path
    )
    logging.info("Saved HNSW index to %s", index_path)
    return index, build_time


def query_index(
    index: hnswlib.Index, queries: np.ndarray, topC: int, ef_search: int
) -> Tuple[np.ndarray, np.ndarray]:
    index.set_ef(ef_search)
    labels, distances = index.knn_query(queries, k=topC)
    return labels, distances
