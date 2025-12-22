# Pivot-Based Coordinate Reduction + HNSW on Flickr30k

This project implements the core method of **Pivot-Based Coordinate Reduction and Hierarchical Indexing for High-Dimensional Image Retrieval**. The goal is to make text-to-image retrieval efficient by first mapping high-dimensional embeddings (e.g., CLIP 512-d) into a low-dimensional pivot-distance coordinate space, building an ANN index there, and then re-ranking in the original space to recover accuracy.

## Problem Definition
- Directly indexing high-dimensional embeddings is costly.
- We select `m` pivots (farthest-point sampling) from image embeddings.
- Each vector `x` is mapped to pivot space via $T(x) = [d(x,p_1), ..., d(x,p_m)]$ with cosine distance `d(x,p)=1 - x·p` (embeddings are L2-normalized so cosine is dot product).
- ANN (HNSW, `space='l2'`) runs in pivot space to retrieve a candidate pool, then exact re-ranking is done in the original embedding space to produce final top-k results.

## Datasets
- `--dataset flickr30k` (default): uses `nlphuji/flickr30k` (requires `datasets<3.0` because dataset scripts were removed in 3.x). Splits: `train/test`.
- `--dataset coco_captions`: uses `HuggingFaceM4/coco_captions` with `train/validation`. If an invalid split is provided, the runner prints available splits and exits.
- Robust caption parsing: supports `caption`, `captions` (list), or `sentences` with `raw/sentence/tokens`. Images come from the `image` column (PIL.Image). Each record is normalized to `{image_id, image, captions, split}` and captions are flattened into `(caption_text, gt_image_id)` pairs.

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

额外加速/诊断参数：
- `--rerank_device {cpu,cuda,auto}`（默认 auto，CUDA 可选 GPU rerank）
- `--pivot_prune_to N`：pivot 空间二阶段裁剪（仅保留 N 个候选进入原空间 rerank）
- `--num_threads`：传给 hnswlib `set_num_threads`
- `--orig_topC`：原空间 HNSW 的候选深度，用于记录 CandRecall@topC（默认等于 k）

可选 pivot 表达组合（`--pivot_norm`, `--pivot_weight`），建议跑四组：
- `(none, none)`（基线）
- `(zscore, none)`
- `(none, variance)`
- `(zscore, variance)`

候选集指标：在 pivot+HNSW 中额外输出 `CandRecall@topC`（gt 是否出现在 topC 候选）和命中名次统计（mean/median，未命中为 -1），便于区分候选阶段 vs rerank 阶段的损失。

扩展：新增 `scripts/benchmark_scaling.py` 跑规模曲线（默认 split=train，N=5000/10000/20000/50000/80000，n_queries=1000，可按实际图像数自动截断）：
```bash
python -m scripts.benchmark_scaling --device cuda --pivot_m 16 --topC 1200 \
	--pivot_efSearch 128 --pivot_M 24 --orig_hnsw_efSearch 128 --orig_hnsw_M 24
```
输出：`results/scaling_*.csv|json`，以及 `plots/scaling_latency_vs_N_*.png`、`plots/scaling_R10_vs_N_*.png`（三条曲线：brute、orig HNSW、pivot+HNSW）。同一 N 内三种方法共享同一批 sampled queries（过滤掉不在前 N 图中的查询）。

## Speed-first preset（大规模优先）
- pivot_m=16
- topC=400–600，pivot_prune_to=200（pivot 空间精排后再进原空间 rerank）
- pivot_efSearch=max(topC, 512)
- rerank_device=cuda
- num_threads=8

在更大规模（>50k 图）下，brute-force 延迟线性增长，而 ANN（orig HNSW / pivot+HNSW）保持更平缓；pivot 方法在 topC 较小、prune 后 rerank 负载更低，应出现延迟优势拐点。

## Colab 一键命令
基准（validation split，max_images=50k，n_queries=2000）：
```bash
python -m scripts.benchmark_baselines --dataset coco_captions --split validation --device cuda \
	--max_images 50000 --n_queries 2000 --batch_size_text 128 --batch_size_image 128 \
	--pivot_m 16 --topC 600 --pivot_prune_to 200 --pivot_efSearch 600 --pivot_M 24 \
	--orig_topC 600 --orig_hnsw_efSearch 256 --orig_hnsw_M 32 --num_threads 8 --rerank_device cuda
```

规模曲线（Ns 自动截断到可用图像总数）：
```bash
python -m scripts.benchmark_scaling --dataset coco_captions --split validation --device cuda \
	--Ns 5000,10000,20000,50000,80000 --n_queries 2000 --batch_size_text 128 --batch_size_image 128 \
	--pivot_m 16 --topC 600 --pivot_prune_to 200 --pivot_efSearch 600 --pivot_M 24 \
	--orig_topC 600 --orig_hnsw_efSearch 256 --orig_hnsw_M 32 --num_threads 8 --rerank_device cuda
```

输出保存在 `results/`（CSV/JSON）和 `plots/`（PNG）下。

## Caching Layout
- Image embeddings: `cache/embeddings/{dataset}_images_{split}_{model}.npy`
- Image IDs: `cache/embeddings/{dataset}_image_ids_{split}.json`
- Caption embeddings: `cache/embeddings/{dataset}_captions_{split}_{model}.npy`
- Caption→image mapping: `cache/embeddings/{dataset}_caption_to_image_{split}.json`
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
