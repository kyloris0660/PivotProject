from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from .config import RetrievalConfig
from .utils import ensure_dir


def plot_latency_vs_recall(
    metrics: Dict[str, float], config: RetrievalConfig, timestamp: str
) -> Path:
    ensure_dir(config.plots_dir)
    fig, ax1 = plt.subplots(figsize=(6, 4))

    recalls = [
        metrics.get("Recall@1", 0),
        metrics.get("Recall@5", 0),
        metrics.get("Recall@10", 0),
    ]
    labels = ["R@1", "R@5", "R@10"]
    ax1.bar(labels, recalls, color="#4c72b0")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Recall")

    ax2 = ax1.twinx()
    ax2.plot(
        labels,
        [metrics.get("avg_total_ms", 0)] * 3,
        color="#dd8452",
        marker="o",
        label="Avg latency (ms)",
    )
    ax2.set_ylabel("Latency (ms)")

    fig.tight_layout()
    plot_path = config.plots_dir / f"plot_{timestamp}.png"
    ensure_dir(plot_path.parent)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path
