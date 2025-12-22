import logging
from typing import Dict, List, Tuple

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from .config import RetrievalConfig


def _extract_captions(record: Dict) -> List[str]:
    if "captions" in record and isinstance(record["captions"], list):
        return [c for c in record["captions"] if isinstance(c, str)]
    if "caption" in record and isinstance(record["caption"], str):
        return [record["caption"]]
    raise KeyError(
        "Dataset record missing caption(s). Expected 'captions' (List[str]) or 'caption' (str)."
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
    ds = load_dataset("nlphuji/flickr30k", split=config.split)

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
