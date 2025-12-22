import logging
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

from .config import RetrievalConfig
from .utils import ensure_dir, load_json, save_json


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "_")


def load_clip(model_name: str, device: str) -> tuple[CLIPModel, AutoProcessor]:
    logging.info("Loading model %s on %s", model_name, device)
    model = CLIPModel.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return model, processor


def _collate_images(batch: Sequence[Image.Image]) -> List[Image.Image]:
    # Ensure RGB to avoid mode issues.
    return [img.convert("RGB") for img in batch]


def _collate_identity(batch: Sequence[str]) -> List[str]:
    # Top-level picklable collate for Windows multiprocessing; keeps text batch as list.
    return list(batch)


def _embed_image_batch(
    images: List[Image.Image], model: CLIPModel, processor: AutoProcessor, device: str
) -> torch.Tensor:
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt").to(device)
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    return feats


def _embed_text_batch(
    texts: List[str], model: CLIPModel, processor: AutoProcessor, device: str
) -> torch.Tensor:
    with torch.no_grad():
        inputs = processor(
            text=texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    return feats


def _embedding_dir(config: RetrievalConfig) -> Path:
    path = config.cache_path("embeddings")
    ensure_dir(path)
    return path


def load_or_compute_image_embeddings(
    records: List[dict],
    config: RetrievalConfig,
    model: CLIPModel,
    processor: AutoProcessor,
) -> tuple[np.ndarray, List[str]]:
    embed_dir = _embedding_dir(config)
    model_tag = _sanitize_model_name(config.model_name)
    subset_tag = f"_n{len(records)}" if config.max_images is not None else ""
    emb_path = embed_dir / f"images_{config.split}_{model_tag}{subset_tag}.npy"
    ids_path = embed_dir / f"image_ids_{config.split}{subset_tag}.json"

    if emb_path.exists() and ids_path.exists() and not config.force_recompute:
        logging.info("Loading cached image embeddings from %s", emb_path)
        embs = np.load(emb_path)
        ids = load_json(ids_path)
        if embs.shape[0] == len(records) and len(ids) == len(records):
            return embs, ids
        logging.info(
            "Cached image embeddings length %d does not match requested records %d; recomputing",
            embs.shape[0],
            len(records),
        )

    logging.info("Computing image embeddings for %d images", len(records))
    imgs = [rec["image"] for rec in records]
    ids = [rec["image_id"] for rec in records]

    dataloader = DataLoader(
        imgs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=_collate_images,
    )
    all_feats: List[torch.Tensor] = []
    for batch in tqdm(dataloader, desc="embed_images"):
        feats = _embed_image_batch(batch, model, processor, config.device)
        all_feats.append(feats.cpu())

    feats_cat = torch.cat(all_feats, dim=0).numpy()
    np.save(emb_path, feats_cat)
    save_json(ids, ids_path)
    logging.info("Saved image embeddings to %s", emb_path)
    return feats_cat, ids


def load_or_compute_caption_embeddings(
    pairs: List[Tuple[str, str]],
    config: RetrievalConfig,
    model: CLIPModel,
    processor: AutoProcessor,
) -> np.ndarray:
    embed_dir = _embedding_dir(config)
    model_tag = _sanitize_model_name(config.model_name)
    subset_tag = f"_n{len(pairs)}" if config.max_captions is not None else ""
    emb_path = embed_dir / f"captions_{config.split}_{model_tag}{subset_tag}.npy"
    mapping_path = embed_dir / f"caption_to_image_{config.split}{subset_tag}.json"

    if emb_path.exists() and mapping_path.exists() and not config.force_recompute:
        logging.info("Loading cached caption embeddings from %s", emb_path)
        embs = np.load(emb_path)
        if embs.shape[0] == len(pairs):
            return embs
        logging.info(
            "Cached caption embeddings length %d does not match requested pairs %d; recomputing",
            embs.shape[0],
            len(pairs),
        )

    logging.info("Computing caption embeddings for %d captions", len(pairs))
    captions = [text for text, _ in pairs]
    dataloader = DataLoader(
        captions,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=_collate_identity,
    )

    all_feats: List[torch.Tensor] = []
    for batch in tqdm(dataloader, desc="embed_captions"):
        feats = _embed_text_batch(batch, model, processor, config.device)
        all_feats.append(feats.cpu())

    feats_cat = torch.cat(all_feats, dim=0).numpy()
    np.save(emb_path, feats_cat)
    save_json({"caption_to_image": [img_id for _, img_id in pairs]}, mapping_path)
    logging.info("Saved caption embeddings to %s", emb_path)
    return feats_cat
