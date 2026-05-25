#!/usr/bin/env python3
"""
Compute mean-difference latent scores directly during SAE extraction.
No intermediate JSONL — accumulates set-level means on GPU in real time.

Reads annotated JSONL files (instruction_set_experiment), runs model forward
+ SAE encode, and accumulates per-feature mean activations into pos (pragmatic)
and neg (generation) accumulators. Outputs mean-diff vectors as .npy files.

Usage:
  python compute_mean_diff_direct.py \
    --input-dir /dataset/Addressivity/pipeline_4_13_llama31_8b_instruct/outputs/instruction_set_experiment \
    --output-dir mean_diff_latents/llama31_8b_instruct/direct \
    --device cuda:1 --batch-size 16

  # Specific files
  python compute_mean_diff_direct.py \
    --input-dir /path/to/instruction_set_experiment \
    --input-files daily_dialog_pragmatic_annotated.jsonl daily_dialog_generation_annotated.jsonl \
    --output-dir mean_diff_latents/daily_dialog \
    --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from sae_lens import SAE
except ImportError as exc:
    raise SystemExit("sae_lens required: pip install sae-lens") from exc


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_RELEASE = "llama-3.1-8b-instruct-andyrdt"
DEFAULT_SAE_ID = "resid_post_layer_3_trainer_1"
ANDYRDT_LAYERS = [3, 7, 11, 15, 19, 23, 27]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default=None,
                    help="Directory with *_annotated.jsonl files (auto-classified by filename keyword).")
    p.add_argument("--input-files", nargs="*", default=None,
                    help="Specific JSONL filenames within --input-dir.")
    p.add_argument("--pos-dir", default=None,
                    help="Directory with positive (pragmatic) JSONL files. All *.jsonl inside are treated as pos.")
    p.add_argument("--neg-dir", default=None,
                    help="Directory with negative (generation) JSONL files. All *.jsonl inside are treated as neg.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--release", default=DEFAULT_RELEASE)
    p.add_argument("--sae-id", default=DEFAULT_SAE_ID)
    p.add_argument("--layers", nargs="*", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--token-batch-size", type=int, default=512)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eot-token", default="<|eot_id|>")
    p.add_argument("--top-k", type=int, default=500, help="Top features to save in JSON summary.")
    p.add_argument("--pos-keyword", default="pragmatic",
                    help="Filename keyword for positive set (default: pragmatic)")
    p.add_argument("--neg-keyword", default="generation",
                    help="Filename keyword for negative set (default: generation)")
    p.add_argument("--sae-group-size", type=int, default=0,
                    help="If >0, use 2-phase mode: forward once, cache hidden states to CPU, "
                         "then SAE encode in groups of this size. Saves VRAM for many layers.")
    return p.parse_args()


def build_sae_ids(base_sae_id: str, layers: list[int]) -> dict[int, str]:
    import re
    result = {}
    # LlamaScope format: 'l0r_8x' → 'l{layer}r_8x'
    test = re.sub(r"(?<=l)\d+(?=r)", "0", base_sae_id)
    if test != base_sae_id or re.match(r"l\d+r", base_sae_id):
        for layer in layers:
            result[layer] = re.sub(r"(?<=l)\d+(?=r)", str(layer), base_sae_id)
        return result
    # andyrdt format: 'resid_post_layer_3_trainer_1' or 'resid_post_layer_0/trainer_0'
    prefix = "resid_post_layer_"
    if base_sae_id.startswith(prefix):
        rest = base_sae_id[len(prefix):]
        if "/" in rest:
            _, _, suffix = rest.partition("/")
            for layer in layers:
                result[layer] = f"{prefix}{layer}/{suffix}"
        else:
            _, _, suffix = rest.partition("_")
            for layer in layers:
                result[layer] = f"{prefix}{layer}_{suffix}"
        return result
    # Gemma Scope format: 'layer_0/width_16k/canonical'
    if base_sae_id.startswith("layer_"):
        for layer in layers:
            result[layer] = re.sub(r"(?<=layer_)\d+", str(layer), base_sae_id)
        return result
    raise ValueError(f"Unsupported sae_id format: {base_sae_id}")


def chunk_tensor(tensor: torch.Tensor, size: int):
    for start in range(0, tensor.shape[0], size):
        yield tensor[start:start + size]


def classify_file(path: Path, pos_keyword: str, neg_keyword: str) -> str:
    stem = path.stem.lower()
    if pos_keyword in stem:
        return "pos"
    if neg_keyword in stem:
        return "neg"
    raise ValueError(f"Cannot classify {path.name} — must contain '{pos_keyword}' or '{neg_keyword}'")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = args.layers or ANDYRDT_LAYERS
    eot_token = args.eot_token

    # Collect and classify input files
    file_labels = {}
    if args.pos_dir and args.neg_dir:
        # Explicit pos/neg directories
        for f in sorted(Path(args.pos_dir).glob("*.jsonl")):
            file_labels[f] = "pos"
        for f in sorted(Path(args.neg_dir).glob("*.jsonl")):
            file_labels[f] = "neg"
    elif args.input_dir:
        # Auto-classify by filename keyword
        input_dir = Path(args.input_dir)
        if args.input_files:
            input_files = [input_dir / f for f in args.input_files]
        else:
            input_files = sorted(input_dir.glob("*_annotated.jsonl"))
        for f in input_files:
            file_labels[f] = classify_file(f, args.pos_keyword, args.neg_keyword)
    else:
        raise SystemExit("Either --input-dir or --pos-dir/--neg-dir is required.")

    if not file_labels:
        raise SystemExit("No input files found.")
    for f, label in file_labels.items():
        print(f"  {label}: {f.name}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load model
    print(f"Loading model: {args.model_name}  device={args.device}")
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map=None,
    ).to(args.device)
    model.eval()

    sae_id_map = build_sae_ids(args.sae_id, layers)

    if args.sae_group_size > 0:
        # ── 2-phase mode: forward once → cache hidden states → SAE in groups ──
        print(f"\n[2-phase mode] sae_group_size={args.sae_group_size}")

        # Phase 1: Forward pass, cache per-row hidden states & metadata to CPU
        print("[Phase 1] Forward pass — caching hidden states to CPU")
        # cached_data: list of (label, prompt_length, valid_length, {layer: flat_hidden_cpu})
        cached_data = []

        for input_path, label in file_labels.items():
            print(f"\n[{label}] {input_path.name}")
            rows = []
            with input_path.open(encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
            if args.limit:
                rows = rows[:args.limit]
            print(f"  rows: {len(rows)}")

            for batch_start in tqdm(range(0, len(rows), args.batch_size), desc=f"  {input_path.stem}"):
                batch_rows = rows[batch_start:batch_start + args.batch_size]

                full_texts = []
                prompt_lengths = []
                for row in batch_rows:
                    prompt = row.get("prompt_text", "")
                    generated = row.get("generated_next_sentence", "")
                    full_texts.append(prompt + generated + eot_token)
                    prompt_lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))

                encoded = tokenizer(
                    full_texts, return_tensors="pt", padding=True, add_special_tokens=False,
                )
                encoded = {k: v.to(args.device) for k, v in encoded.items()}
                valid_lengths = [int(v) for v in encoded["attention_mask"].sum(dim=1).tolist()]
                mask_bool = encoded["attention_mask"].bool()

                with torch.inference_mode():
                    outputs = model(**encoded, output_hidden_states=True, use_cache=False)

                for ri in range(len(batch_rows)):
                    row_hiddens = {}
                    for layer in layers:
                        layer_hidden = outputs.hidden_states[layer + 1]
                        # Extract only valid tokens for this row
                        row_mask = mask_bool[ri]
                        row_hiddens[layer] = layer_hidden[ri][row_mask].cpu()
                    cached_data.append({
                        "label": label,
                        "prompt_length": prompt_lengths[ri],
                        "valid_length": valid_lengths[ri],
                        "hiddens": row_hiddens,
                    })

                del outputs, encoded
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()

        print(f"\nCached {len(cached_data)} rows")

        # Phase 2: SAE encode in groups
        print(f"\n[Phase 2] SAE encode (group_size={args.sae_group_size})")
        d_sae = None
        accum = {}
        counts = {}

        for group_start in range(0, len(layers), args.sae_group_size):
            group_layers = layers[group_start:group_start + args.sae_group_size]
            print(f"\n  Loading SAE group: layers {group_layers}")

            saes = {}
            for layer in group_layers:
                sae_id = sae_id_map[layer]
                sae = SAE.from_pretrained(release=args.release, sae_id=sae_id, device=args.device)
                if isinstance(sae, tuple):
                    sae = sae[0]
                sae.eval()
                saes[layer] = sae
                if d_sae is None:
                    d_sae = sae.cfg.d_sae
                    print(f"  d_sae: {d_sae}")
                    for lbl in ("pos", "neg"):
                        accum[lbl] = {}
                        counts[lbl] = {}

            # Init accumulators for this group
            for lbl in ("pos", "neg"):
                for layer in group_layers:
                    accum[lbl][layer] = {
                        "input": torch.zeros(d_sae, device=args.device, dtype=torch.float64),
                        "output": torch.zeros(d_sae, device=args.device, dtype=torch.float64),
                        "both": torch.zeros(d_sae, device=args.device, dtype=torch.float64),
                    }
                    counts[lbl][layer] = 0

            for cd in tqdm(cached_data, desc=f"  encode layers {group_layers}"):
                label = cd["label"]
                pl = cd["prompt_length"]
                vl = cd["valid_length"]
                n_out = vl - pl

                for layer in group_layers:
                    flat_hidden = cd["hiddens"][layer].to(args.device)
                    total_tokens = flat_hidden.shape[0]

                    with torch.inference_mode():
                        row_input_sum = torch.zeros(d_sae, device=args.device, dtype=torch.float32)
                        row_output_sum = torch.zeros(d_sae, device=args.device, dtype=torch.float32)

                        for chunk_start in range(0, total_tokens, args.token_batch_size):
                            chunk_end = min(chunk_start + args.token_batch_size, total_tokens)
                            acts = saes[layer].encode(flat_hidden[chunk_start:chunk_end]).float()

                            inp_end = max(0, min(pl - chunk_start, acts.shape[0]))
                            if inp_end > 0:
                                row_input_sum += acts[:inp_end].sum(dim=0)
                            if inp_end < acts.shape[0]:
                                row_output_sum += acts[inp_end:].sum(dim=0)
                            del acts

                    if pl > 0:
                        accum[label][layer]["input"] += (row_input_sum / pl).double()
                    if n_out > 0:
                        accum[label][layer]["output"] += (row_output_sum / n_out).double()
                    if vl > 0:
                        accum[label][layer]["both"] += ((row_input_sum + row_output_sum) / vl).double()
                    counts[label][layer] += 1

                    del flat_hidden

            del saes
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()

        del cached_data
        gc.collect()

    else:
        # ── Original mode: load all SAEs at once ──
        saes: dict[int, SAE] = {}
        for layer in layers:
            sae_id = sae_id_map[layer]
            print(f"  Loading SAE: layer={layer}  sae_id={sae_id}")
            sae, _, _ = SAE.from_pretrained(release=args.release, sae_id=sae_id, device=args.device)
            sae.eval()
            saes[layer] = sae

        d_sae = saes[layers[0]].cfg.d_sae
        print(f"d_sae: {d_sae}")

        accum = {}
        counts = {}
        for label in ("pos", "neg"):
            accum[label] = {}
            counts[label] = {}
            for layer in layers:
                accum[label][layer] = {
                    "input": torch.zeros(d_sae, device=args.device, dtype=torch.float64),
                    "output": torch.zeros(d_sae, device=args.device, dtype=torch.float64),
                    "both": torch.zeros(d_sae, device=args.device, dtype=torch.float64),
                }
                counts[label][layer] = 0

        for input_path, label in file_labels.items():
            print(f"\n[{label}] {input_path.name}")
            rows = []
            with input_path.open(encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
            if args.limit:
                rows = rows[:args.limit]
            print(f"  rows: {len(rows)}")

            for batch_start in tqdm(range(0, len(rows), args.batch_size), desc=f"  {input_path.stem}"):
                batch_rows = rows[batch_start:batch_start + args.batch_size]

                full_texts = []
                prompt_lengths = []
                for row in batch_rows:
                    prompt = row.get("prompt_text", "")
                    generated = row.get("generated_next_sentence", "")
                    full_texts.append(prompt + generated + eot_token)
                    prompt_lengths.append(len(tokenizer.encode(prompt, add_special_tokens=False)))

                encoded = tokenizer(
                    full_texts, return_tensors="pt", padding=True, add_special_tokens=False,
                )
                encoded = {k: v.to(args.device) for k, v in encoded.items()}
                valid_lengths = [int(v) for v in encoded["attention_mask"].sum(dim=1).tolist()]

                with torch.inference_mode():
                    outputs = model(**encoded, output_hidden_states=True, use_cache=False)

                for layer in layers:
                    layer_hidden = outputs.hidden_states[layer + 1]
                    flat_hidden = layer_hidden[encoded["attention_mask"].bool()].to(saes[layer].device)
                    total_tokens = flat_hidden.shape[0]

                    cum = [0]
                    for vl in valid_lengths:
                        cum.append(cum[-1] + vl)

                    row_input_sums = [torch.zeros(d_sae, device=args.device, dtype=torch.float32) for _ in batch_rows]
                    row_output_sums = [torch.zeros(d_sae, device=args.device, dtype=torch.float32) for _ in batch_rows]

                    with torch.inference_mode():
                        for chunk_start in range(0, total_tokens, args.token_batch_size):
                            chunk_end = min(chunk_start + args.token_batch_size, total_tokens)
                            acts = saes[layer].encode(flat_hidden[chunk_start:chunk_end]).float()

                            for ri in range(len(batch_rows)):
                                ov_start = max(chunk_start, cum[ri])
                                ov_end = min(chunk_end, cum[ri + 1])
                                if ov_start >= ov_end:
                                    continue
                                a_s = ov_start - chunk_start
                                a_e = ov_end - chunk_start
                                row_acts = acts[a_s:a_e]

                                local_start = ov_start - cum[ri]
                                pl = prompt_lengths[ri]
                                inp_end = max(0, min(pl - local_start, row_acts.shape[0]))
                                if inp_end > 0:
                                    row_input_sums[ri] += row_acts[:inp_end].sum(dim=0)
                                if inp_end < row_acts.shape[0]:
                                    row_output_sums[ri] += row_acts[inp_end:].sum(dim=0)

                            del acts

                    for ri in range(len(batch_rows)):
                        pl = prompt_lengths[ri]
                        vl = valid_lengths[ri]
                        n_out = vl - pl

                        inp = row_input_sums[ri]
                        out = row_output_sums[ri]

                        if pl > 0:
                            accum[label][layer]["input"] += (inp / pl).double()
                        if n_out > 0:
                            accum[label][layer]["output"] += (out / n_out).double()
                        if vl > 0:
                            accum[label][layer]["both"] += ((inp + out) / vl).double()

                    counts[label][layer] += len(batch_rows)
                    del row_input_sums, row_output_sums, flat_hidden, layer_hidden

                del outputs, encoded
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()

    # Compute mean-diff and save
    print("\nComputing mean-diff...")
    modes = ["input", "output", "both"]
    for layer in layers:
        n_pos = counts["pos"][layer]
        n_neg = counts["neg"][layer]
        print(f"  layer {layer}: n_pos={n_pos}, n_neg={n_neg}")

        layer_result = {
            "layer": layer,
            "n_pos_samples": n_pos,
            "n_neg_samples": n_neg,
            "d_sae": d_sae,
            "modes": {},
        }

        for mode in modes:
            pos_mean = (accum["pos"][layer][mode] / max(n_pos, 1)).cpu().numpy().astype(np.float32)
            neg_mean = (accum["neg"][layer][mode] / max(n_neg, 1)).cpu().numpy().astype(np.float32)
            mean_diff = pos_mean - neg_mean

            # Save npy
            npy_dir = output_dir / "npy" / mode
            npy_dir.mkdir(parents=True, exist_ok=True)
            np.save(npy_dir / f"layer_{layer:02d}_mean_diff.npy", mean_diff)
            np.save(npy_dir / f"layer_{layer:02d}_pos_mean.npy", pos_mean)
            np.save(npy_dir / f"layer_{layer:02d}_neg_mean.npy", neg_mean)

            # Top-k for JSON summary
            pos_order = np.argsort(-mean_diff)[:args.top_k]
            neg_order = np.argsort(mean_diff)[:args.top_k]
            abs_order = np.argsort(-np.abs(mean_diff))[:args.top_k]

            layer_result["modes"][mode] = {
                "n_nonzero_diff": int(np.count_nonzero(mean_diff)),
                "max_diff": float(mean_diff.max()),
                "min_diff": float(mean_diff.min()),
                "top_positive": [{"feature_id": int(i), "mean_diff": float(mean_diff[i])} for i in pos_order],
                "top_negative": [{"feature_id": int(i), "mean_diff": float(mean_diff[i])} for i in neg_order],
                "top_abs": [{"feature_id": int(i), "mean_diff": float(mean_diff[i])} for i in abs_order],
            }

        write_json(output_dir / f"layer_{layer:02d}_mean_diff.json", layer_result)

    # Summary
    write_json(output_dir / "summary.json", {
        "input_files": {str(f): label for f, label in file_labels.items()},
        "d_sae": d_sae,
        "layers": layers,
        "top_k": args.top_k,
        "pos_keyword": args.pos_keyword,
        "neg_keyword": args.neg_keyword,
        "counts": {label: {str(l): counts[label][l] for l in layers} for label in ("pos", "neg")},
    })

    print(f"\nDone. Output: {output_dir}")


if __name__ == "__main__":
    main()
