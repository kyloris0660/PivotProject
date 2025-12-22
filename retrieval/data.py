import logging
from typing import Dict, List, Tuple

import datasets
from datasets import load_dataset
from packaging import version
from PIL import Image
from tqdm import tqdm

from .config import RetrievalConfig


DATASET_REGISTRY = {
    "flickr30k": "nlphuji/flickr30k",
    # coco captions variants on HF; we'll try these in order
    "coco_captions": ["coco_captions", "HuggingFaceM4/coco_captions"],
}


def _extract_captions(record: Dict) -> List[str]:
    # Common HF variants: captions (list[str]), caption (str), sentences (list[dict|str]) with raw/sentence/tokens.
    if "captions" in record and isinstance(record["captions"], list):
        caps = [c for c in record["captions"] if isinstance(c, str) and c.strip()]
        if caps:
            return caps

    if "caption" in record:
        cap_val = record["caption"]
        if isinstance(cap_val, str) and cap_val.strip():
            return [cap_val]
        if isinstance(cap_val, list):
            caps: List[str] = []
            for s in cap_val:
                if isinstance(s, str) and s.strip():
                    caps.append(s)
                elif isinstance(s, dict):
                    if "raw" in s and isinstance(s["raw"], str) and s["raw"].strip():
                        caps.append(s["raw"])
                    elif (
                        "sentence" in s
                        and isinstance(s["sentence"], str)
                        and s["sentence"].strip()
                    ):
                        caps.append(s["sentence"])
                    elif "tokens" in s and isinstance(s["tokens"], list):
                        joined = " ".join(
                            str(tok) for tok in s["tokens"] if isinstance(tok, str)
                        )
                        if joined.strip():
                            caps.append(joined)
            if caps:
                return caps

    if "sentences" in record and isinstance(record["sentences"], list):
        caps: List[str] = []
        for s in record["sentences"]:
            if isinstance(s, str) and s.strip():
                caps.append(s)
            elif isinstance(s, dict):
                if "raw" in s and isinstance(s["raw"], str) and s["raw"].strip():
                    caps.append(s["raw"])
                elif (
                    "sentence" in s
                    and isinstance(s["sentence"], str)
                    and s["sentence"].strip()
                ):
                    caps.append(s["sentence"])
                elif "tokens" in s and isinstance(s["tokens"], list):
                    joined = " ".join(
                        str(tok) for tok in s["tokens"] if isinstance(tok, str)
                    )
                    if joined.strip():
                        caps.append(joined)
        if caps:
            return caps

    available = ", ".join(record.keys())
    raise KeyError(
        "Dataset record missing caption(s). Expected fields like 'captions', 'caption', or 'sentences' with raw/sentence/tokens. "
        f"Available keys: {available}"
    )


def _extract_image(record: Dict) -> Image.Image:
    if "image" not in record:
        raise KeyError("Dataset record missing 'image' field (PIL.Image expected).")
    return record["image"]


def _extract_image_id(record: Dict, idx: int) -> str:
    candidates = ["image_id", "id", "filename"]
    for key in candidates:
        if key in record:
            return str(record[key])
    return str(idx)


def load_flickr30k(config: RetrievalConfig) -> List[Dict]:
    logging.info("Loading Flickr30k from Hugging Face: split=%s", config.split)

    if version.parse(datasets.__version__).major >= 3:
        raise RuntimeError(
            "datasets>=3.0 removes support for dataset scripts. Install datasets<3.0 (e.g., pip install 'datasets<3.0.0') "
            "and clear any cached flickr30k artifacts under ~/.cache/huggingface/datasets."
        )

    try:
        ds = load_dataset(
            "nlphuji/flickr30k",
            split=config.split,
        )
    except RuntimeError as e:
        msg = (
            "Failed to load dataset nlphuji/flickr30k because dataset scripts are blocked. "
            "Install datasets<3.0.0 (e.g., 2.19.x) and clear local cache for flickr30k under ~/.cache/huggingface/datasets."
        )
        raise RuntimeError(msg) from e

    records: List[Dict] = []
    max_items = config.max_images if config.max_images is not None else len(ds)
    for idx, item in tqdm(enumerate(ds), total=min(len(ds), max_items), desc="dataset"):
        if idx >= max_items:
            break
        captions = _extract_captions(item)
        image = _extract_image(item)
        image_id = _extract_image_id(item, idx)
        records.append(
            {
                "image_id": image_id,
                "image": image,
                "captions": captions,
                "split": config.split,
            }
        )
    logging.info("Loaded %d images", len(records))
    return records


def load_coco_captions(config: RetrievalConfig) -> List[Dict]:
    logging.info("Loading COCO Captions from Hugging Face: split=%s", config.split)

    tried: List[str] = []
    last_err: Exception | None = None
    ds_all = None
    for name in DATASET_REGISTRY["coco_captions"]:
        tried.append(name)
        try:
            ds_all = load_dataset(name)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            logging.warning("Failed to load dataset '%s': %s", name, e)
            continue

    if ds_all is None:
        raise RuntimeError(
            f"Unable to load COCO captions dataset. Tried: {tried}. Last error: {last_err}"
        )

    available_splits = sorted(ds_all.keys())
    if config.split not in ds_all:
        raise ValueError(
            f"Split '{config.split}' not found for coco_captions. Available splits: {available_splits}"
        )

    ds = ds_all[config.split]
    records: List[Dict] = []
    max_items = config.max_images if config.max_images is not None else len(ds)
    for idx, item in tqdm(enumerate(ds), total=min(len(ds), max_items), desc="dataset"):
        if idx >= max_items:
            break
        captions = _extract_captions(item)
        image = _extract_image(item)
        image_id = _extract_image_id(item, idx)
        records.append(
            {
                "image_id": image_id,
                "image": image,
                "captions": captions,
                "split": config.split,
            }
        )
    logging.info("Loaded %d images", len(records))
    return records


def build_caption_pairs(
    records: List[Dict], max_captions: int | None = None
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for rec in records:
        for caption in rec["captions"]:
            pairs.append((caption, rec["image_id"]))
            if max_captions is not None and len(pairs) >= max_captions:
                return pairs
    return pairs


def load_dataset_records(config: RetrievalConfig) -> List[Dict]:
    if config.dataset == "flickr30k":
        return load_flickr30k(config)
    if config.dataset == "coco_captions":
        return load_coco_captions(config)
    raise ValueError(
        f"Unknown dataset '{config.dataset}'. Supported: {list(DATASET_REGISTRY.keys())}"
    )
