# Syc Grid

This repo has two main workflows.

## 1. Feature Extraction

Build positive/negative text sets and compute mean-difference SAE features.

First, generate positive and negative examples:

```bash
python MD_feature_extraction/step01_build_posneg.py \
  --level pragmatic \
  --input-data MD_feature_extraction/data/input/trivia_qa.jsonl \
  --output-dir MD_feature_extraction/outputs/llama31_8b_instruct/level2 \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --model-type instruct \
  --device cuda:0
```

Then compute mean-difference features from those sets:

```bash
python MD_feature_extraction/step02_mean_diff.py \
  --pos-dir MD_feature_extraction/outputs/llama31_8b_instruct/level2/pos \
  --neg-dir MD_feature_extraction/outputs/llama31_8b_instruct/level2/neg \
  --output-dir mean_diff_latents/llama31_8b_instruct/level2 \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --device cuda:0
```

## 2. Main Experiments

Run baseline, steering, or ablation experiments on sycophancy benchmarks.

Edit a config in `configs/`, especially:

```yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"

steering:
  mean_diff_path: "{work_dir}/../mean_diff_latents/llama31_8b_instruct/level2"
  mean_diff_direction: "all"

benchmarks:
  - mmlu
  - elephant_oeq
  - elephant_aita
  - elephant_ss
  - syconbench_debate
```

Run the experiment:

```bash
python run.py --config configs/llama31_8b_instruct.yaml
```

Outputs are written under the config's `output.base_dir`, with `generations.jsonl`, `judge_results.jsonl`, `scores.json`, and `master_summary.csv`.
