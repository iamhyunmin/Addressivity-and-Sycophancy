# Model Configs for MD Feature Extraction

## Supported Models & SAE Checkpoints

### Llama 3.1 8B Instruct
- `--model-name meta-llama/Llama-3.1-8B-Instruct --model-type instruct`
- SAE: `--release llama-3.1-8b-instruct-andyrdt --sae-id resid_post_layer_3_trainer_1 --layers 3 7 11 15 19 23 27`

### Llama 3.1 8B Base
- `--model-name meta-llama/Llama-3.1-8B --model-type base`
- SAE: `--release llama_scope_lxr_32x --sae-id l0r_32x --layers 0 1 2 ... 31`

### Gemma 2 2B IT
- `--model-name google/gemma-2-2b-it --model-type instruct`
- SAE: `--release gemma-scope-2b-pt-res-canonical --sae-id layer_0/width_16k/canonical --layers 0 1 2 ... 25`

### Gemma 2 9B Base (PT)
- `--model-name google/gemma-2-9b --model-type base`
- SAE: `--release gemma-scope-9b-pt-res-canonical --sae-id layer_0/width_16k/canonical --layers 0 1 2 ... 41`
- All 42 layers available (width_16k/canonical)

### Gemma 2 9B IT
- `--model-name google/gemma-2-9b-it --model-type instruct`
- SAE: `--release gemma-scope-9b-it-res-canonical --sae-id layer_9/width_16k/canonical --layers 9 20 31`
- Note: only layers 9, 20, 31 have canonical SAEs (16k width)

---

## Input Data

All input data is included in `data/` for self-contained execution.

### data/inputs/ (Level 2 & 3 pos inputs)
Base conversation datasets, 500 samples each:
- `soda.jsonl` — SODA dialogues
- `daily_dialog.jsonl` — DailyDialog conversations
- `taskmaster.jsonl` — Taskmaster task-oriented dialogues
- `trivia_qa.jsonl` — TriviaQA questions

Each row has `head` or `text` or `input_text` field containing the conversation input.

### data/principal_variants/ (Level 3 neg inputs)
Principal-rewritten variants where the speaker identity is modified:
- `soda_principal_variants.jsonl`
- `daily_dialog_principal_rewritten_gpt-4o-mini.jsonl`
- `taskmaster_principal_rewritten_gpt-4o-mini.jsonl`
- `trivia_qa_principal_variants.jsonl`

Each row has `rewritten_text` (modified input) and `principal_prefix` (e.g. "My cousin", "The person next to me").

---

## Level Definitions & Data Usage

### Level 1: Template Breakage (instruct models only)
- **Pos**: original text + native chat template → generate → no judge filtering
- **Neg**: original text + 5 alternate templates → generate → broken judge → keep "Not Broken"
- **Data**: `data/inputs/*.jsonl`

```bash
python step01_build_posneg.py --level template --model-type instruct \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --input-data data/inputs/soda.jsonl data/inputs/daily_dialog.jsonl data/inputs/taskmaster.jsonl data/inputs/trivia_qa.jsonl \
  --output-dir output/llama_it/template --device cuda:0
```

### Level 2: Pragmatic vs Generation
- **Pos**: 5 pragmatic prompts → generate → pragmatic judge → keep "Yes"
- **Neg**: 5 generation prompts → generate → pragmatic judge → keep "No"
- **Data**: `data/inputs/*.jsonl`
- **Base model format**: `"{instruct_prompt}\n\n{input}"` for both pos and neg
- **Instruct model format**: chat template with system message

```bash
# Instruct
python step01_build_posneg.py --level pragmatic --model-type instruct \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --input-data data/inputs/soda.jsonl data/inputs/daily_dialog.jsonl data/inputs/taskmaster.jsonl data/inputs/trivia_qa.jsonl \
  --output-dir output/llama_it/pragmatic --device cuda:0

# Base
python step01_build_posneg.py --level pragmatic --model-type base \
  --model-name meta-llama/Llama-3.1-8B \
  --input-data data/inputs/soda.jsonl data/inputs/daily_dialog.jsonl data/inputs/taskmaster.jsonl data/inputs/trivia_qa.jsonl \
  --output-dir output/llama_base/pragmatic --device cuda:0
```

### Level 3: Principal Awareness
- **Pos**: original text + 5 pragmatic prompts → generate → pragmatic judge → keep "Yes"
- **Neg**: principal variant text + 5 pragmatic prompts (round-robin) → generate → pragmatic judge (keep "Yes") → principal judge (keep "Y")
- **Pos data**: `data/inputs/*.jsonl`
- **Neg data**: `data/principal_variants/*.jsonl`
- **Base model format**: `"{instruct_prompt}\n\nQuestion: {input}\nAnswer:"`
- **Instruct model format**: chat template with system message

```bash
# Instruct (all 4 datasets)
python step01_build_posneg.py --level principal --model-type instruct \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --input-data data/inputs/soda.jsonl data/inputs/daily_dialog.jsonl data/inputs/taskmaster.jsonl data/inputs/trivia_qa.jsonl \
  --neg-input-data data/principal_variants/soda_principal_variants.jsonl data/principal_variants/daily_dialog_principal_rewritten_gpt-4o-mini.jsonl data/principal_variants/taskmaster_principal_rewritten_gpt-4o-mini.jsonl data/principal_variants/trivia_qa_principal_variants.jsonl \
  --output-dir output/llama_it/principal --device cuda:0

# Base
python step01_build_posneg.py --level principal --model-type base \
  --model-name meta-llama/Llama-3.1-8B \
  --input-data data/inputs/soda.jsonl data/inputs/daily_dialog.jsonl data/inputs/taskmaster.jsonl data/inputs/trivia_qa.jsonl \
  --neg-input-data data/principal_variants/soda_principal_variants.jsonl data/principal_variants/daily_dialog_principal_rewritten_gpt-4o-mini.jsonl data/principal_variants/taskmaster_principal_rewritten_gpt-4o-mini.jsonl data/principal_variants/trivia_qa_principal_variants.jsonl \
  --output-dir output/llama_base/principal --device cuda:0
```

---

## Step 2: Mean-Diff Computation

Takes pos/neg directories from Step 1 output (or any existing selected data).

```bash
# Llama 3.1 8B Instruct
python step02_mean_diff.py --pos-dir output/llama_it/pragmatic/pos --neg-dir output/llama_it/pragmatic/neg \
  --output-dir output/llama_it/pragmatic/mean_diff \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --release llama-3.1-8b-instruct-andyrdt --sae-id resid_post_layer_3_trainer_1 \
  --layers 3 7 11 15 19 23 27 --device cuda:0

# Llama 3.1 8B Base
python step02_mean_diff.py --pos-dir output/llama_base/pragmatic/pos --neg-dir output/llama_base/pragmatic/neg \
  --output-dir output/llama_base/pragmatic/mean_diff \
  --model-name meta-llama/Llama-3.1-8B \
  --release llama_scope_lxr_32x --sae-id l0r_32x \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --device cuda:0

# Gemma 2 9B IT (only 3 layers)
python step02_mean_diff.py --pos-dir output/gemma9b_it/pragmatic/pos --neg-dir output/gemma9b_it/pragmatic/neg \
  --output-dir output/gemma9b_it/pragmatic/mean_diff \
  --model-name google/gemma-2-9b-it \
  --release gemma-scope-9b-it-res-canonical --sae-id layer_9/width_16k/canonical \
  --layers 9 20 31 --device cuda:0

# Gemma 2 9B Base (all 42 layers)
python step02_mean_diff.py --pos-dir output/gemma9b_pt/pragmatic/pos --neg-dir output/gemma9b_pt/pragmatic/neg \
  --output-dir output/gemma9b_pt/pragmatic/mean_diff \
  --model-name google/gemma-2-9b \
  --release gemma-scope-9b-pt-res-canonical --sae-id layer_0/width_16k/canonical \
  --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 --device cuda:0
```

### Using Existing Selected Data

If pos/neg selected data already exists:

```bash
python step02_mean_diff.py \
  --pos-dir data/selected/pos \
  --neg-dir data/selected/neg \
  --output-dir output/llama_it/principal/mean_diff \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --release llama-3.1-8b-instruct-andyrdt --sae-id resid_post_layer_3_trainer_1 \
  --layers 3 7 11 15 19 23 27 --device cuda:0
```
