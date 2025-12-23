import json
import logging
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import datasets
from datasets import load_dataset
from packaging import version
from PIL import Image
import requests
from tqdm import tqdm

from .config import RetrievalConfig


DATASET_REGISTRY = {
    "flickr30k": "nlphuji/flickr30k",
    # coco captions variants on HF; we'll try these in order (with optional configs)
    "coco_captions": [
        ("coco_captions", "2017"),
        ("coco_captions", "2014"),
        ("coco_captions", None),
        ("HuggingFaceM4/coco_captions", None),
    ],
    # widely available web-caption dataset with train/validation
    "conceptual_captions": [
        ("conceptual_captions", None),
    ],
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
    if "image" in record and isinstance(record["image"], Image.Image):
        return record["image"]

    url = record.get("image_url") or record.get("url")
    if url:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise KeyError(f"Failed to fetch image from url {url}: {exc}") from exc

    raise KeyError("Dataset record missing 'image' field (PIL.Image expected).")


def _download_file(url: str, dest: Path, desc: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return

    last_exc: Exception | None = None
    current_url = url
    tried_http = False
    for attempt in range(3):
        try:
            with requests.get(current_url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with dest.open("wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=desc
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            return
        except requests.exceptions.SSLError as exc:  # type: ignore[attr-defined]
            last_exc = exc
            if not tried_http and current_url.startswith(
                "https://images.cocodataset.org/"
            ):
                logging.warning(
                    "SSL failed for %s (%s); retrying over http", current_url, exc
                )
                current_url = current_url.replace("https://", "http://", 1)
                tried_http = True
                time.sleep(1)
                continue
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

        wait = 2**attempt
        logging.warning(
            "Download failed (%s) [%s], retry %d/3 in %ds",
            desc,
            last_exc,
            attempt + 1,
            wait,
        )
        time.sleep(wait)

    raise RuntimeError(f"Failed to download {url}: {last_exc}") from last_exc


def _download_and_extract_zip(
    url: str, zip_path: Path, extract_dir: Path, desc: str
) -> None:
    """Download a zip once and extract; reuse existing artifacts."""

    extract_dir.parent.mkdir(parents=True, exist_ok=True)
    if extract_dir.exists():
        return

    if not zip_path.exists():
        _download_file(url, zip_path, desc=desc)

    if not extract_dir.exists():
        logging.info("Extracting %s to %s", zip_path, extract_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir.parent)


def _load_coco_annotations(cache_root: Path, split_key: str) -> Tuple[Dict, Dict]:
    ann_dir = cache_root / "annotations"
    ann_zip = ann_dir / "annotations_trainval2017.zip"
    extract_dir = ann_dir
    _download_and_extract_zip(
        url="http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        zip_path=ann_zip,
        extract_dir=extract_dir,
        desc="download_annotations",
    )

    ann_map = {
        "train2017": extract_dir / "annotations" / "captions_train2017.json",
        "val2017": extract_dir / "annotations" / "captions_val2017.json",
    }
    if split_key not in ann_map:
        raise ValueError(f"Unsupported COCO split '{split_key}' for annotations")

    ann_path = ann_map[split_key]
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file missing: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data.get("images", [])}
    captions_by_image: Dict[int, List[str]] = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        captions_by_image.setdefault(img_id, []).append(ann.get("caption", ""))
    return images, captions_by_image


def _ensure_coco_images(cache_root: Path, split_key: str) -> Path:
    zips_dir = cache_root / "zips"
    zips_dir.mkdir(parents=True, exist_ok=True)
    zip_path = zips_dir / f"{split_key}.zip"
    split_dir = cache_root / split_key

    url = f"http://images.cocodataset.org/zips/{split_key}.zip"
    _download_and_extract_zip(url, zip_path, split_dir, desc=f"download_{split_key}")
    return split_dir


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
    for name, cfg in DATASET_REGISTRY["coco_captions"]:
        tag = name if cfg is None else f"{name}:{cfg}"
        tried.append(tag)
        try:
            if cfg is None:
                ds_all = load_dataset(name)
            else:
                ds_all = load_dataset(name, cfg)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            logging.warning("Failed to load dataset '%s': %s", tag, e)
            continue

    if ds_all is None:
        logging.warning(
            "HF COCO captions not available; falling back to official COCO download. Tried: %s; last error: %s",
            tried,
            last_err,
        )
        return load_coco_captions_fallback(config)

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


def load_coco_captions_fallback(config: RetrievalConfig) -> List[Dict]:
    split_map = {
        "train": "train2017",
        "training": "train2017",
        "val": "val2017",
        "validation": "val2017",
    }
    if config.split not in split_map:
        raise ValueError(
            f"Split '{config.split}' not supported for COCO fallback. Available splits: {list(split_map.keys())}"
        )

    split_key = split_map[config.split]
    if split_key == "train2017" and not config.allow_coco_train_download:
        raise RuntimeError(
            "train2017 is large (~118k images, 18GB). Pass --allow_coco_train_download to enable downloading train images."
        )

    cache_root = config.cache_path("datasets", "coco2017")
    images_meta, captions_by_image = _load_coco_annotations(cache_root, split_key)

    all_image_ids = list(images_meta.keys())
    max_images = (
        config.max_images if config.max_images is not None else len(all_image_ids)
    )
    if split_key == "val2017":
        max_images = min(max_images, 5000)
    selected_ids = all_image_ids[:max_images]
    logging.info(
        "COCO fallback using split=%s, images=%d", split_key, len(selected_ids)
    )

    split_dir = _ensure_coco_images(cache_root, split_key)

    records: List[Dict] = []
    for img_id in tqdm(selected_ids, desc="dataset"):
        meta = images_meta[img_id]
        file_name = meta.get("file_name")
        if not file_name:
            continue
        img_path = split_dir / file_name
        if not img_path.exists():
            logging.warning("Image missing after extract: %s", img_path)
            continue
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to open image %s: %s", img_path, exc)
            continue

        caps = captions_by_image.get(img_id, [])
        if not caps:
            continue

        records.append(
            {
                "image_id": str(img_id),
                "image": image,
                "captions": caps,
                "split": config.split,
            }
        )

    logging.info("Loaded %d images via COCO fallback", len(records))
    return records


def load_conceptual_captions(config: RetrievalConfig) -> List[Dict]:
    logging.info(
        "Loading Conceptual Captions from Hugging Face: split=%s", config.split
    )

    tried: List[str] = []
    last_err: Exception | None = None
    ds_all = None
    for name, cfg in DATASET_REGISTRY["conceptual_captions"]:
        tag = name if cfg is None else f"{name}:{cfg}"
        tried.append(tag)
        try:
            if cfg is None:
                ds_all = load_dataset(name)
            else:
                ds_all = load_dataset(name, cfg)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            logging.warning("Failed to load dataset '%s': %s", tag, e)
            continue

    if ds_all is None:
        raise RuntimeError(
            f"Unable to load conceptual_captions dataset. Tried: {tried}. Last error: {last_err}"
        )

    available_splits = sorted(ds_all.keys())
    if config.split not in ds_all:
        raise ValueError(
            f"Split '{config.split}' not found for conceptual_captions. Available splits: {available_splits}"
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
    if config.dataset == "conceptual_captions":
        return load_conceptual_captions(config)
    raise ValueError(
        f"Unknown dataset '{config.dataset}'. Supported: {list(DATASET_REGISTRY.keys())}"
    )
