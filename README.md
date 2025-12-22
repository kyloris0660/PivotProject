# Pivot-Based Coordinate Reduction + HNSW on Flickr30k

This project implements the core method of **Pivot-Based Coordinate Reduction and Hierarchical Indexing for High-Dimensional Image Retrieval**. The goal is to make text-to-image retrieval efficient by first mapping high-dimensional embeddings (e.g., CLIP 512-d) into a low-dimensional pivot-distance coordinate space, building an ANN index there, and then re-ranking in the original space to recover accuracy.

## Problem Definition
- Directly indexing high-dimensional embeddings is costly.
- We select `m` pivots (farthest-point sampling) from image embeddings.
- Each vector `x` is mapped to pivot space via $T(x) = [d(x,p_1), ..., d(x,p_m)]$ with cosine distance `d(x,p)=1 - x·p` (embeddings are L2-normalized so cosine is dot product).
- ANN (HNSW, `space='l2'`) runs in pivot space to retrieve a candidate pool, then exact re-ranking is done in the original embedding space to produce final top-k results.

## Dataset
- Default source: `load_dataset("nlphuji/flickr30k")`.
- Robust to caption field names (`caption` or `captions` list) and uses `image` (PIL.Image). Each sample is normalized to `{image_id, image, captions, split}`.
- Evaluation flattens captions into `(caption_text, gt_image_id)` pairs.

## Pipeline (matches the research logic)
1. **Load data** (Flickr30k test split by default).
2. **Embed images/text** with CLIP; cache to `cache/embeddings/`.
3. **Select pivots** via farthest-point sampling (cosine distance) with fixed seed; cache to `cache/pivots/`.
4. **Project to pivot space**: compute `1 - dot` to pivots; cache pivot coordinates.
5. **Build/load HNSW** in pivot space (L2) from `cache/index/`.
6. **Query**: map caption embedding to pivot space, get topC candidates from HNSW, rerank in original space by cosine similarity, output top-k.
7. **Report metrics**: Recall@{1,5,10}, avg latency components, build time; save JSON+CSV to `results/` and a simple plot to `plots/`.

## Quick Start
# From project root (datasets must be <3.0 for flickr30k script)
```bash
python -m pip install -r requirements.txt
python -m scripts.run_all --source hf --split test --m 8 --topC 500 --k 10 --device cuda
```
Key flags: `--pivot_sample` (sampling for pivot selection), `--seed`, `--M`, `--efc`/`--efs`, `--force_recompute`, `--max_images`, `--max_captions`.

## Benchmark baselines
使用 `scripts/benchmark_baselines.py` 对同一批查询/图库公平对比三种方法：
- A) brute-force（原空间精确点积）
- B) 原空间 HNSW（cosine）
- C) Pivot+HNSW（pivot 低维 ANN → 原空间精排）

示例（子集 2k 图 / 1 万 caption，默认 CLIP-B/32）：
```bash
python -m scripts.benchmark_baselines --device cuda \
	--max_images 2000 --max_captions 10000 --n_queries 1000 \
	--pivot_m 16 --topC 1200 --pivot_efSearch 128 --pivot_M 24 \
	--orig_hnsw_efSearch 128 --orig_hnsw_M 24 --force_recompute
```
输出：`results/benchmark_*.json`、`results/benchmark_*.csv`，以及 `plots/benchmark_latency_vs_recall_*.png`。每种方法记录 Recall@1/5/10、平均查询延迟（分拆 brute/hnsw/pivot_map/rerank）、索引构建时间、索引大小等。

可选 pivot 表达组合（`--pivot_norm`, `--pivot_weight`），建议跑四组：
- `(none, none)`（基线）
- `(zscore, none)`
- `(none, variance)`
- `(zscore, variance)`

候选集指标：在 pivot+HNSW 中额外输出 `CandRecall@topC`（gt 是否出现在 topC 候选）和命中名次统计（mean/median，未命中为 -1），便于区分候选阶段 vs rerank 阶段的损失。

扩展：新增 `scripts/benchmark_scaling.py` 跑规模曲线（默认 split=train，N=2000/5000/10000/20000/29000，n_queries=1000）：
```bash
python -m scripts.benchmark_scaling --device cuda --pivot_m 16 --topC 1200 \
	--pivot_efSearch 128 --pivot_M 24 --orig_hnsw_efSearch 128 --orig_hnsw_M 24
```
输出：`results/scaling_*.csv|json`，以及 `plots/scaling_latency_vs_N_*.png`、`plots/scaling_R10_vs_N_*.png`（三条曲线：brute、orig HNSW、pivot+HNSW）。同一 N 内三种方法共享同一批 sampled queries（过滤掉不在前 N 图中的查询）。

## Caching Layout
- Image embeddings: `cache/embeddings/images_{split}_{model}.npy`
- Image IDs: `cache/embeddings/image_ids_{split}.json`
- Caption embeddings: `cache/embeddings/captions_{split}_{model}.npy`
- Caption→image mapping: `cache/embeddings/caption_to_image_{split}.json`
- Pivots: `cache/pivots/pivots_m{m}_seed{seed}.npy` (+ meta JSON)
- Pivot coords (images): `cache/pivots/pivot_coords_images_{split}_m{m}.npy`
- HNSW index: `cache/index/hnsw_{split}_m{m}_M{M}_efc{efc}.bin` (+ meta)
- Results: `results/exp_YYYYmmdd_HHMM.(json|csv)`; plots in `plots/`.

## Output
Console summary includes recalls and latency. Files record:
- `Recall@1/5/10`, `CandRecall@topC`, candidate hit rank (mean/median), `avg_hnsw_ms`, `avg_rerank_ms`, `avg_total_ms`, `build_index_time_sec`, `index_size_bytes`。
- Parameters: `m, topC, k, model_name, M, efConstruction, efSearch, seed, device, pivot_norm, pivot_weight`。

## Notes
- All randomness (pivot start, sampling) is controlled by `--seed`.
- Pivot distance computation uses cosine distance on normalized embeddings.
- Clear error messages will be raised if dataset fields are missing; adjust field names or preprocess accordingly.
