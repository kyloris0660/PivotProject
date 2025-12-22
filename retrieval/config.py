from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RetrievalConfig:
    dataset: str = "flickr30k"
    source: str = "hf"
    split: str = "test"
    model_name: str = "openai/clip-vit-base-patch32"
    device: str = "cpu"
    batch_size: int = 32
    num_workers: int = 4
    m: int = 16
    topC: int = 500
    k: int = 10
    pivot_sample: int = 5000
    seed: int = 42
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 128
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    plots_dir: Path = field(default_factory=lambda: Path("plots"))
    force_recompute: bool = False
    max_captions: Optional[int] = None
    max_images: Optional[int] = None

    def cache_path(self, *parts: str) -> Path:
        return self.cache_dir.joinpath(*parts)

    def result_stem(self, timestamp: str) -> str:
        return f"exp_{timestamp}"
