
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from dotenv import load_dotenv


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from syc.config import load_yaml, resolve_config, parse_layers, parse_list
from syc.models.loader import load_model_and_tokenizer, get_model_input_device
from syc.steering.feature_selector import (
    load_mean_diff_feature, available_mean_diff_layers,
)
from syc.steering.sae_loader import load_sae, get_decoder_directions
from syc.steering.steerer import all_token_steering_hook, zero_ablation_hook, no_steering
from syc.benchmarks.mmlu import MMLUBenchmark
from syc.benchmarks.elephant import ElephantBenchmark
from syc.benchmarks.syconbench import SYCONBenchBenchmark
from syc.evaluation import judge_mc, judge_oeq, judge_sycon
from syc.evaluation.scoring import compute_scores
from syc.utils.io import read_jsonl, write_jsonl, write_json, read_json, scores_exist


BENCHMARK_REGISTRY = {
    "mmlu": MMLUBenchmark(),
    "elephant_oeq": ElephantBenchmark("elephant_oeq"),
    "elephant_aita": ElephantBenchmark("elephant_aita"),
    "elephant_ss": ElephantBenchmark("elephant_ss"),
    "syconbench_debate": SYCONBenchBenchmark(),
}

ALL_BENCHMARKS = list(BENCHMARK_REGISTRY.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAE steering / ablation runs over behavior benchmarks.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--layers", nargs="*", type=int, default=None,
                        help="Override layers from config.")
    parser.add_argument("--top-k", nargs="*", type=int, default=None,
                        help="Override top-k values.")
    parser.add_argument("--alpha", nargs="*", type=float, default=None,
                        help="Override alpha values.")
    parser.add_argument("--direction", default=None, choices=["pos", "neg", "all"],
                        help="Mean-diff direction override: pos, neg, or all (default: from config).")
    parser.add_argument("--mean-diff-path", default=None,
                        help="Override steering.mean_diff_path from config.")
    parser.add_argument("--ablate", action="store_true",
                        help="Zero-ablation mode: encode through SAE, zero target feature, decode. "
                             "Uses mean-diff top-1 feature. No alpha needed.")
    parser.add_argument("--benchmarks", nargs="*", default=None, choices=ALL_BENCHMARKS,
                        help="Benchmarks to run (default: all in config).")
    parser.add_argument("--output-dir", default=None,
                        help="Override output base directory.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size.")
    parser.add_argument("--judge-model", default=None,
                        help="Override judge model (default: gpt-4o-mini).")
    parser.add_argument("--judge-concurrency", type=int, default=32,
                        help="OpenAI API concurrency.")
    parser.add_argument("--device", default=None,
                        help="Device override: 'auto' (all GPUs via device_map), 'cuda:0', 'cuda:1', etc. "
                             "Defaults to model.device in config, or 'cuda' if available.")
    parser.add_argument("--model-name", default=None,
                        help="Override model.name, e.g. a local export-epoch directory.")
    parser.add_argument("--system-prompt", default=None,
                        help="Override model.system_prompt.")
    parser.add_argument("--local-files-only", action="store_true",
                        help="Load models only from local cache.")
    parser.add_argument("--skip-judge", action="store_true",
                        help="Skip judging step (generate only).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing scores.json instead of skipping.")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run baseline (no steering) only.")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline and run intervention combos only.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a single combo (first layer, k=1, alpha=1) on first benchmark only.")
    parser.add_argument("--rewrite", action="store_true",
                        help="Rewrite mode: load existing T1 predictions from generations.jsonl, "
                             "apply cutoff, re-generate T2, re-judge, and re-score. "
                             "Only applies to MC benchmarks (mmlu).")
    return parser.parse_args()


def normalize_device(device: object) -> str:

    if device == "auto":
        return "cuda"
    if isinstance(device, int):
        return f"cuda:{device}"

    device_str = str(device)
    if device_str.isdigit():
        return f"cuda:{device_str}"
    return device_str


def load_env():
    load_dotenv(SCRIPT_DIR / ".env")
    load_dotenv(SCRIPT_DIR.parent / ".env")
    load_dotenv()


def resolve_output_dir(cfg: dict, cli_override: str | None, config_path: Path) -> Path:
    if cli_override:
        return Path(cli_override).resolve()
    base = cfg.get("output", {}).get("base_dir")
    if base:
        return Path(base).resolve()
    exp_name = cfg.get("experiment_name", "run")
    return (config_path.parent / "results" / exp_name).resolve()


_DEFAULT_MAIN_PATHS = {
    "mmlu": str(SCRIPT_DIR / "main_sets" / "raw" / "mmlu_full_test_2000.pkl"),
    "elephant_oeq": str(SCRIPT_DIR / "main_sets" / "raw" / "OEQ.csv"),
    "elephant_aita": str(SCRIPT_DIR / "main_sets" / "raw" / "AITA-YTA.csv"),
    "elephant_ss": str(SCRIPT_DIR / "main_sets" / "raw" / "SS.csv"),
    "syconbench_debate": str(SCRIPT_DIR / "main_sets" / "syconbench_debate"),
}


def get_main_set_path(cfg: dict, benchmark_name: str) -> Path:
    main_sets = cfg.get("main_sets", {})
    path_str = main_sets.get(benchmark_name) or _DEFAULT_MAIN_PATHS.get(benchmark_name)
    if not path_str:
        raise SystemExit(
            f"No main_set path configured for benchmark '{benchmark_name}'. "
            f"Add it under 'main_sets:' in your config."
        )
    return Path(path_str)


def run_judge(records: list[dict], benchmark_name: str, judge_model: str, concurrency: int) -> list[dict]:
    if benchmark_name == "mmlu":
        return judge_mc.score_records(records, judge_model=judge_model, concurrency=concurrency)
    elif benchmark_name in ("elephant_oeq", "elephant_aita", "elephant_ss"):
        return judge_oeq.score_records(records, dataset_name=benchmark_name, judge_model=judge_model, concurrency=concurrency)
    elif benchmark_name == "syconbench_debate":
        return judge_sycon.score_records(records, judge_model=judge_model, concurrency=concurrency)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")


def run_combo(
    *,
    model,
    tokenizer,
    cfg: dict,
    benchmark_name: str,
    records: list[dict],
    combo_dir: Path,
    layer: int | None,
    top_k: int | None,
    alpha: float | None,
    direction: torch.Tensor | None,
    mode: str | None,
    judge_model: str,
    judge_concurrency: int,
    skip_judge: bool,
    overwrite: bool,
    device: str,
    meta: dict,
    sae_for_ablation: object | None = None,
    ablation_feature_ids: list[int] | None = None,
) -> dict:

    scores_path = combo_dir / "scores.json"
    if scores_path.exists() and not overwrite:
        print(f"  [skip] {combo_dir.name} (scores.json exists)")
        return read_json(scores_path)

    combo_dir.mkdir(parents=True, exist_ok=True)
    bench = BENCHMARK_REGISTRY[benchmark_name]


    if sae_for_ablation is not None and ablation_feature_ids:
        ctx = zero_ablation_hook(model, layer, sae_for_ablation, ablation_feature_ids)
    elif direction is not None:
        ctx = all_token_steering_hook(model, layer, direction, alpha)
    else:
        ctx = no_steering()
    with ctx:
        generations = bench.generate(model, tokenizer, records, cfg, device)
    write_jsonl(combo_dir / "generations.jsonl", generations)

    if skip_judge:
        return {}


    print(f"  [judge] {benchmark_name} n={len(generations)}")
    judged = run_judge(generations, benchmark_name, judge_model, judge_concurrency)
    write_jsonl(combo_dir / "judge_results.jsonl", judged)


    scores = compute_scores(judged, benchmark_name)
    scores["layer"] = layer
    scores["top_k"] = top_k
    scores["alpha"] = alpha
    scores["mode"] = mode
    scores["meta"] = {**meta, "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
    write_json(scores_path, scores)
    print(f"  [scores] {scores}")
    return scores


MC_BENCHMARKS = {"mmlu"}


def rewrite_combo(
    *,
    model,
    tokenizer,
    cfg: dict,
    benchmark_name: str,
    combo_dir: Path,
    layer: int | None,
    top_k: int | None,
    alpha: float | None,
    mode: str | None,
    judge_model: str,
    judge_concurrency: int,
    skip_judge: bool,
    device: str,
    meta: dict,
) -> dict | None:

    gen_path = combo_dir / "generations.jsonl"
    if not gen_path.exists():
        print(f"  [skip-rewrite] {combo_dir.name} (no generations.jsonl)")
        return None

    bench = BENCHMARK_REGISTRY[benchmark_name]
    existing = read_jsonl(gen_path)
    print(f"  [rewrite] {combo_dir.name}  n={len(existing)}")


    if layer is not None:
        sae = load_sae(cfg, layer, device=device)
        mean_diff_path = cfg.get("steering", {}).get("mean_diff_path", "")
        direction_str = "neg" if mode and mode.endswith("_neg") else "pos"
        feature_id, _ = load_mean_diff_feature(mean_diff_path, layer, direction_str)
        model_device = get_model_input_device(model)
        if mode and mode.startswith("ablate_"):
            ctx = zero_ablation_hook(model, layer, sae, [feature_id])
        else:
            direction = get_decoder_directions(sae, [feature_id], device=model_device)
            ctx = all_token_steering_hook(model, layer, direction, alpha)
    else:
        ctx = no_steering()

    with ctx:
        updated = bench.rewrite_t2(model, tokenizer, existing, cfg, device)
    write_jsonl(gen_path, updated)

    if skip_judge:
        return {}

    print(f"  [judge] {benchmark_name} n={len(updated)}")
    judged = run_judge(updated, benchmark_name, judge_model, judge_concurrency)
    write_jsonl(combo_dir / "judge_results.jsonl", judged)

    scores = compute_scores(judged, benchmark_name)
    scores["layer"] = layer
    scores["top_k"] = top_k
    scores["alpha"] = alpha
    scores["mode"] = mode
    scores["meta"] = {**meta, "timestamp": datetime.datetime.utcnow().isoformat() + "Z"}
    write_json(combo_dir / "scores.json", scores)
    print(f"  [scores] {scores}")


    if layer is not None:
        del sae
        if device == "cuda" or device.startswith("cuda:"):
            torch.cuda.empty_cache()

    return scores


def build_summary_csv(output_dir: Path) -> None:

    rows = []
    for scores_path in sorted(output_dir.rglob("scores.json")):
        rel = scores_path.relative_to(output_dir)
        try:
            data = read_json(scores_path)
        except Exception:
            continue
        benchmark = data.get("benchmark", rel.parts[0] if rel.parts else "")
        layer = data.get("layer")
        top_k = data.get("top_k")
        alpha = data.get("alpha")
        mode = data.get("mode")
        meta = data.get("meta", {})
        for key, val in data.items():
            if key in ("benchmark", "layer", "top_k", "alpha", "mode", "meta", "n_samples"):
                continue
            if isinstance(val, (int, float)) or val is None:
                rows.append({
                    "benchmark": benchmark,
                    "layer": layer,
                    "top_k": top_k,
                    "alpha": alpha,
                    "mode": mode,
                    "n_samples": data.get("n_samples"),
                    "metric_name": key,
                    "metric_value": val,
                    "model": meta.get("model", ""),
                })

    if not rows:
        return
    out_path = output_dir / "master_summary.csv"
    fieldnames = ["benchmark", "layer", "top_k", "alpha", "mode", "n_samples", "metric_name", "metric_value", "model"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] {out_path}  ({len(rows)} rows)")


def main() -> None:
    args = parse_args()
    load_env()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)
    cfg = resolve_config(cfg, config_path)


    if args.batch_size:
        cfg.setdefault("run", {})["batch_size"] = args.batch_size
    if args.model_name is not None:
        cfg.setdefault("model", {})["name"] = args.model_name
    if args.system_prompt is not None:
        cfg.setdefault("model", {})["system_prompt"] = args.system_prompt


    _cfg_device = cfg.get("model", {}).get("device", None)
    _cli_device = args.device
    if _cli_device is not None:
        device = normalize_device(_cli_device)
    elif _cfg_device is not None:
        device = normalize_device(_cfg_device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    judge_model = args.judge_model or cfg.get("judge", {}).get("model", "gpt-4o-mini")
    output_dir = resolve_output_dir(cfg, args.output_dir, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)


    shutil.copy2(config_path, output_dir / "config.yaml")


    configured_benchmarks = cfg.get("benchmarks", ALL_BENCHMARKS)
    benchmarks_to_run = args.benchmarks or configured_benchmarks


    model_type = cfg.get("model", {}).get("type", "instruct")
    if model_type == "base" and "syconbench_debate" in benchmarks_to_run:
        print("[info] Skipping syconbench_debate: not supported for base models.")
        benchmarks_to_run = [b for b in benchmarks_to_run if b != "syconbench_debate"]


    datasets: dict[str, list[dict]] = {}
    for bname in benchmarks_to_run:
        path = get_main_set_path(cfg, bname)
        if not path.exists():
            raise SystemExit(f"Main set not found: {path}")
        datasets[bname] = BENCHMARK_REGISTRY[bname].load_main_set(path)
        print(f"[main] {bname}: {len(datasets[bname])} records from {path}")


    if not args.skip_judge and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set. Export it or add to .env")


    model, tokenizer = load_model_and_tokenizer(cfg, device=device, local_files_only=args.local_files_only)
    model_device = get_model_input_device(model)

    meta = {
        "model": cfg["model"]["name"],
        "experiment": cfg.get("experiment_name", ""),
    }


    if args.rewrite:
        rewrite_benchmarks = [b for b in benchmarks_to_run if b in MC_BENCHMARKS]
        if not rewrite_benchmarks:
            raise SystemExit("--rewrite only applies to MC benchmarks (mmlu)")
        print(f"\n[rewrite] Re-running T2 for: {rewrite_benchmarks}")

        for bname in rewrite_benchmarks:

            baseline_dir = output_dir / bname / "baseline"
            rewrite_combo(
                model=model, tokenizer=tokenizer, cfg=cfg,
                benchmark_name=bname, combo_dir=baseline_dir,
                layer=None, top_k=None, alpha=None, mode=None,
                judge_model=judge_model, judge_concurrency=args.judge_concurrency,
                skip_judge=args.skip_judge, device=model_device, meta=meta,
            )



            mean_diff_path = cfg.get("steering", {}).get("mean_diff_path", "")
            avail = available_mean_diff_layers(mean_diff_path) if mean_diff_path else []
            rewrite_layers = args.layers or parse_layers(cfg, avail) if avail else None
            rewrite_top_k = set(args.top_k) if args.top_k else None
            rewrite_alpha = set(args.alpha) if args.alpha else None

            bm_dir = output_dir / bname
            for mode_dir in sorted(bm_dir.iterdir()):
                if not mode_dir.is_dir() or mode_dir.name == "baseline":
                    continue


                combos = []
                for combo_dir in mode_dir.iterdir():
                    if not combo_dir.is_dir():
                        continue
                    parts = combo_dir.name.split("_")
                    try:
                        layer = int(parts[1])
                        top_k = int(parts[3])
                        alpha = float(parts[5])
                    except (IndexError, ValueError):
                        continue
                    combos.append((layer, top_k, alpha, combo_dir))


                if rewrite_layers is not None:
                    layer_set = set(rewrite_layers)
                    combos = [(l, k, a, d) for l, k, a, d in combos if l in layer_set]
                if rewrite_top_k is not None:
                    combos = [(l, k, a, d) for l, k, a, d in combos if k in rewrite_top_k]
                if rewrite_alpha is not None:
                    combos = [(l, k, a, d) for l, k, a, d in combos if a in rewrite_alpha]


                layer_order = {l: i for i, l in enumerate(rewrite_layers)} if rewrite_layers else {}
                combos.sort(key=lambda x: (layer_order.get(x[0], x[0]), x[1], x[2]))

                for layer, top_k, alpha, combo_dir in combos:
                    rewrite_combo(
                        model=model, tokenizer=tokenizer, cfg=cfg,
                        benchmark_name=bname, combo_dir=combo_dir,
                        layer=layer, top_k=top_k, alpha=alpha, mode=mode_dir.name,
                        judge_model=judge_model, judge_concurrency=args.judge_concurrency,
                        skip_judge=args.skip_judge, device=model_device, meta=meta,
                    )

        build_summary_csv(output_dir)
        print(f"\n[done] Rewrite results in: {output_dir}")
        return


    if args.skip_baseline:
        print("\n[baseline] skipped (--skip-baseline)")
    else:
        print("\n[baseline] running without steering")
        for bname in benchmarks_to_run:
            baseline_dir = output_dir / bname / "baseline"
            run_combo(
                model=model, tokenizer=tokenizer, cfg=cfg,
                benchmark_name=bname, records=datasets[bname],
                combo_dir=baseline_dir,
                layer=None, top_k=None, alpha=None, direction=None, mode=None,
                judge_model=judge_model, judge_concurrency=args.judge_concurrency,
                skip_judge=args.skip_judge, overwrite=args.overwrite,
                device=model_device, meta=meta,
            )

    if args.baseline_only:
        build_summary_csv(output_dir)
        return


    mean_diff_path = args.mean_diff_path or cfg.get("steering", {}).get("mean_diff_path", "")
    if not mean_diff_path:
        raise SystemExit("steering.mean_diff_path is required.")

    alpha_values = args.alpha or parse_list(cfg.get("steering", {}).get("alpha", [1, 5, 10, 25, 100, 1000]), cast=float)

    if args.ablate:

        mean_diff_direction_cfg = args.direction or cfg.get("steering", {}).get("mean_diff_direction", "pos")
        if mean_diff_direction_cfg == "all":
            directions_to_run = ["pos", "neg"]
        else:
            directions_to_run = [mean_diff_direction_cfg]

        avail_layers = available_mean_diff_layers(mean_diff_path)
        layers = args.layers or parse_layers(cfg, avail_layers)

        if args.smoke_test:
            layers = layers[:1]
            benchmarks_to_run = benchmarks_to_run[:1]

        for mean_diff_direction in directions_to_run:
            mode_label = f"ablate_{mean_diff_direction}"

            print(f"\n[ablation] direction={mean_diff_direction}  layers={layers}")
            print(f"[ablation] benchmarks={benchmarks_to_run}")

            for layer in layers:
                print(f"\n[layer {layer}] loading SAE + mean-diff feature for ablation ({mean_diff_direction})")
                try:
                    sae = load_sae(cfg, layer, device=device)
                except Exception as exc:
                    print(f"  [warn] SAE load failed for layer {layer}: {exc}  (skipping)")
                    continue

                try:
                    feature_id, feature_record = load_mean_diff_feature(mean_diff_path, layer, mean_diff_direction)
                except FileNotFoundError as exc:
                    print(f"  [warn] {exc}  (skipping)")
                    del sae
                    continue

                print(f"  feature_id={feature_id}  mean_diff={feature_record.get('mean_diff', 'N/A')}")

                selected_features_data = {
                    "layer": layer,
                    "top_k": 1,
                    "mode": mode_label,
                    "intervention": "zero_ablation",
                    "mean_diff_direction": mean_diff_direction,
                    "selected_features": [feature_record],
                }


                print(f"\n[ablation] layer={layer} feature={feature_id} dir={mean_diff_direction}")
                for bname in benchmarks_to_run:
                    combo_dir = output_dir / bname / mode_label / f"layer_{layer}_ablate"
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    feat_path = combo_dir / "selected_features.json"
                    if not feat_path.exists():
                        write_json(feat_path, selected_features_data)
                    run_combo(
                        model=model, tokenizer=tokenizer, cfg=cfg,
                        benchmark_name=bname, records=datasets[bname],
                        combo_dir=combo_dir,
                        layer=layer, top_k=1, alpha=None, direction=None, mode=mode_label,
                        judge_model=judge_model, judge_concurrency=args.judge_concurrency,
                        skip_judge=args.skip_judge, overwrite=args.overwrite,
                        device=model_device, meta=meta,
                        sae_for_ablation=sae, ablation_feature_ids=[feature_id],
                    )

                del sae
                if device == "cuda" or device.startswith("cuda:"):
                    torch.cuda.empty_cache()

    else:

        mean_diff_direction_cfg = args.direction or cfg.get("steering", {}).get("mean_diff_direction", "pos")
        if mean_diff_direction_cfg == "all":
            directions_to_run = ["pos", "neg"]
        else:
            directions_to_run = [mean_diff_direction_cfg]

        avail_layers = available_mean_diff_layers(mean_diff_path)
        layers = args.layers or parse_layers(cfg, avail_layers)

        if args.smoke_test:
            layers = layers[:1]
            alpha_values = [alpha_values[0]]
            benchmarks_to_run = benchmarks_to_run[:1]
            print("[smoke-test] Running single combo on first benchmark only.")

        for mean_diff_direction in directions_to_run:
            mode_label = f"mean_diff_{mean_diff_direction}"

            print(f"\n[mean-diff] direction={mean_diff_direction}  layers={layers}  alpha={alpha_values}")
            print(f"[mean-diff] benchmarks={benchmarks_to_run}")

            for layer in layers:
                print(f"\n[layer {layer}] loading SAE + mean-diff feature ({mean_diff_direction})")
                try:
                    sae = load_sae(cfg, layer, device=device)
                except Exception as exc:
                    print(f"  [warn] SAE load failed for layer {layer}: {exc}  (skipping)")
                    continue

                try:
                    feature_id, feature_record = load_mean_diff_feature(mean_diff_path, layer, mean_diff_direction)
                except FileNotFoundError as exc:
                    print(f"  [warn] {exc}  (skipping)")
                    del sae
                    continue

                direction = get_decoder_directions(sae, [feature_id], device=model_device)
                print(f"  feature_id={feature_id}  mean_diff={feature_record.get('mean_diff', 'N/A')}")

                selected_features_data = {
                    "layer": layer,
                    "top_k": 1,
                    "mode": mode_label,
                    "mean_diff_direction": mean_diff_direction,
                    "selected_features": [feature_record],
                }

                for alpha in alpha_values:
                    print(f"\n[combo] layer={layer} feature={feature_id} alpha={alpha} dir={mean_diff_direction}")
                    for bname in benchmarks_to_run:
                        combo_dir = output_dir / bname / mode_label / f"layer_{layer}_topk_1_alpha_{alpha}"
                        combo_dir.mkdir(parents=True, exist_ok=True)
                        feat_path = combo_dir / "selected_features.json"
                        if not feat_path.exists():
                            write_json(feat_path, selected_features_data)
                        run_combo(
                            model=model, tokenizer=tokenizer, cfg=cfg,
                            benchmark_name=bname, records=datasets[bname],
                            combo_dir=combo_dir,
                            layer=layer, top_k=1, alpha=alpha, direction=direction, mode=mode_label,
                            judge_model=judge_model, judge_concurrency=args.judge_concurrency,
                            skip_judge=args.skip_judge, overwrite=args.overwrite,
                            device=model_device, meta=meta,
                        )

                del sae
                if device == "cuda" or device.startswith("cuda:"):
                    torch.cuda.empty_cache()

    build_summary_csv(output_dir)
    print(f"\n[done] Results in: {output_dir}")


if __name__ == "__main__":
    main()
