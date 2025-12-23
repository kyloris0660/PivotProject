import json
import logging
import os
import shutil
import time
import zipfile
from datetime import datetime
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


def _coco_split_paths(root: Path, split_key: str) -> Tuple[Path, Path]:
    split_dir = root / split_key
    captions_path = root / "annotations" / f"captions_{split_key}.json"
    return split_dir, captions_path


def _drive_paths(drive_root: Path, split_key: str, tag: str) -> Dict[str, Path]:
    dataset_root = drive_root / "datasets" / "coco2017"
    return {
        "dataset_root": dataset_root,
        "split_dir": dataset_root / split_key,
        "ann_dir": dataset_root / "annotations",
        "manifest_dir": dataset_root / "manifests",
        "manifest_path": dataset_root / "manifests" / f"{tag}.json",
    }


def _materialize_from_drive(
    drive_root: Path,
    local_root: Path,
    split_key: str,
    tag: str,
    tolerance: float = 0.01,
) -> Tuple[Dict | None, bool]:
    paths = _drive_paths(drive_root, split_key, tag)
    manifest_path = paths["manifest_path"]
    if not manifest_path.exists():
        return None, False

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Failed to read Drive manifest %s: %s", manifest_path, exc)
        return None, False

    drive_split_dir = paths["split_dir"]
    if not drive_split_dir.exists():
        logging.warning("Drive split dir missing: %s", drive_split_dir)
        return None, False

    local_split_dir, local_captions_path = _coco_split_paths(local_root, split_key)
    local_split_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not local_split_dir.exists():
            os.symlink(drive_split_dir, local_split_dir, target_is_directory=True)
        elif not local_split_dir.is_symlink():
            shutil.copytree(drive_split_dir, local_split_dir, dirs_exist_ok=True)
    except Exception:
        shutil.copytree(drive_split_dir, local_split_dir, dirs_exist_ok=True)

    drive_captions = paths["ann_dir"] / local_captions_path.name
    if drive_captions.exists():
        local_captions_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_captions_path.exists():
            shutil.copy2(drive_captions, local_captions_path)

    file_names = manifest.get("file_names", [])
    missing = [fn for fn in file_names if not (local_split_dir / fn).exists()]
    if file_names:
        miss_ratio = len(missing) / len(file_names)
        if miss_ratio > tolerance:
            logging.warning(
                "Drive subset %s missing %d/%d files (>%.1f%%); will redownload",
                tag,
                len(missing),
                len(file_names),
                tolerance * 100,
            )
            return None, False
        if missing:
            logging.warning(
                "Drive subset %s missing %d/%d files; will fetch missing via HTTP",
                tag,
                len(missing),
                len(file_names),
            )

    logging.info(
        "Found persisted COCO subset in Drive: %s; reusing without HTTP download",
        manifest_path,
    )
    return manifest, True


def _persist_subset_to_drive(
    source_root: Path,
    drive_root: Path,
    split_key: str,
    tag: str,
    manifest: Dict,
    captions_path: Path,
) -> None:
    paths = _drive_paths(drive_root, split_key, tag)
    paths["ann_dir"].mkdir(parents=True, exist_ok=True)
    paths["split_dir"].mkdir(parents=True, exist_ok=True)
    paths["manifest_dir"].mkdir(parents=True, exist_ok=True)

    drive_captions = paths["ann_dir"] / captions_path.name
    if captions_path.exists() and not drive_captions.exists():
        shutil.copy2(captions_path, drive_captions)

    src_split_dir = source_root / split_key
    for fn in manifest.get("file_names", []):
        src = src_split_dir / fn
        dst = paths["split_dir"] / fn
        if dst.exists() or not src.exists():
            continue
        shutil.copy2(src, dst)

    manifest_path = paths["manifest_path"]
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logging.info("Persisted COCO subset to Drive: %s", manifest_path)


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

    # If directory exists but is empty or missing target files, still extract
    dir_has_content = extract_dir.exists() and any(extract_dir.rglob("*"))

    if not zip_path.exists():
        _download_file(url, zip_path, desc=desc)

    if not dir_has_content:
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
        "train2017": extract_dir / "captions_train2017.json",
        "val2017": extract_dir / "captions_val2017.json",
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


def _ensure_coco_images_zip(cache_root: Path, split_key: str) -> Path:
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

    local_cache_root = config.cache_path("datasets", "coco2017")
    cache_root = Path(config.coco_root) if config.coco_root else local_cache_root

    drive_sync = config.drive_sync or {}
    persist_to_drive = bool(drive_sync.get("persist_to_drive"))
    drive_root_str = drive_sync.get("drive_root")
    drive_root = Path(drive_root_str) if drive_root_str else None
    max_images_requested = config.max_images
    default_tag = f"coco2017_{split_key}_n{max_images_requested or 'all'}_seed{config.seed}"
    drive_tag = drive_sync.get("tag") or default_tag
    drive_available = bool(persist_to_drive and drive_root and drive_root.exists())
    if persist_to_drive and not drive_available:
        logging.info(
            "persist_to_drive enabled but Drive root %s not found; in Colab run drive.mount('/content/drive').",
            drive_root_str or "/content/drive",
        )

    manifest_from_drive: Dict | None = None
    if cache_root != local_cache_root:
        split_dir_override, captions_override = _coco_split_paths(cache_root, split_key)
        if split_dir_override.exists() and captions_override.exists():
            logging.info("Using coco_root=%s for %s", cache_root, split_key)
        elif drive_available:
            manifest_from_drive, reused = _materialize_from_drive(
                drive_root, local_cache_root, split_key, drive_tag
            )
            if reused:
                cache_root = local_cache_root
    elif drive_available:
        manifest_from_drive, reused = _materialize_from_drive(
            drive_root, local_cache_root, split_key, drive_tag
        )
        if reused:
            cache_root = local_cache_root

    images_meta, captions_by_image = _load_coco_annotations(cache_root, split_key)

    all_image_ids = list(images_meta.keys())
    max_images = (
        config.max_images if config.max_images is not None else len(all_image_ids)
    )
    if split_key == "val2017":
        max_images = min(max_images, 5000)

    manifest_data = manifest_from_drive
    if manifest_data and manifest_data.get("split") != split_key:
        manifest_data = None

    if manifest_data:
        manifest_ids = manifest_data.get("image_ids", [])
        manifest_files = manifest_data.get("file_names", [])
        target_count = min(max_images, len(manifest_ids))
        selected_items = list(zip(manifest_ids[:target_count], manifest_files))
        logging.info(
            "COCO fallback using Drive manifest %s with %d images",
            drive_tag,
            len(selected_items),
        )
    else:
        selected_ids = all_image_ids[:max_images]
        selected_items = [
            (img_id, images_meta[img_id].get("file_name")) for img_id in selected_ids
        ]
        logging.info(
            "COCO fallback using split=%s, images=%d", split_key, len(selected_items)
        )

    records: List[Dict] = []
    used_ids: List[str] = []
    used_files: List[str] = []

    split_dir, captions_path = _coco_split_paths(cache_root, split_key)
    split_dir.mkdir(parents=True, exist_ok=True)

    if split_key == "val2017":
        split_dir = _ensure_coco_images_zip(cache_root, split_key)
        iterator_items = selected_items
        desc = "dataset"
    else:
        iterator_items = selected_items
        desc = "download_train_subset"

    for img_id, file_name in tqdm(iterator_items, desc=desc):
        if not file_name:
            continue

        img_path = split_dir / file_name

        if not img_path.exists():
            url = f"http://images.cocodataset.org/{split_key}/{file_name}"
            try:
                with requests.get(url, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                    with img_path.open("wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to download image %s: %s", url, exc)
                continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            logging.warning("Failed to open image %s: %s", img_path, exc)
            continue

        caps = captions_by_image.get(int(img_id) if isinstance(img_id, str) and img_id.isdigit() else img_id, [])
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
        used_ids.append(str(img_id))
        used_files.append(file_name)

        if len(records) >= max_images:
            break

    if persist_to_drive and drive_available and records:
        manifest_payload = {
            "split": split_key,
            "max_images": len(records),
            "seed": config.seed,
            "timestamp": datetime.utcnow().isoformat(),
            "tag": drive_tag,
            "image_ids": used_ids,
            "file_names": used_files,
        }
        _persist_subset_to_drive(cache_root, drive_root, split_key, drive_tag, manifest_payload, captions_path)

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
