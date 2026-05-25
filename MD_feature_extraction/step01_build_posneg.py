
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from lib.utils import read_jsonl, write_jsonl, write_json, get_input_text, normalize_head_text, slugify_prompt
from lib.generation import load_model_and_tokenizer, render_prompt, generate_batch
from lib.judges import broken_judge, pragmatic_judge, principal_judge, annotate_rows_async
from lib.prompts import NEG_TEMPLATES, PRAGMATIC_PROMPTS, GENERATION_PROMPTS

from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", required=True, choices=["template", "pragmatic", "principal"])
    p.add_argument("--input-data", nargs="+", required=True,
                    help="Input JSONL file(s). Each row needs head/text/input_text field.")
    p.add_argument("--neg-input-data", nargs="*", default=None,
                    help="Negative input JSONL file(s) for principal level (principal variant rows).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--model-type", default="instruct", choices=["instruct", "base"],
                    help="instruct: chat template + add_special_tokens=False. "
                         "base: plain text prompt + add_special_tokens=True.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--annotation-model", default="gpt-4o-mini")
    p.add_argument("--annotation-concurrency", type=int, default=32)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _should_skip(path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        print(f"  [skip] {path} (exists)")
        return True
    return False


def generate_for_rows(model, tokenizer, rows: list[dict], prompt_fn, args, desc: str) -> list[dict]:

    results = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc=desc):
        batch = rows[start:start + args.batch_size]
        prompts = [prompt_fn(row) for row in batch]
        generations = generate_batch(
            model, tokenizer, prompts, args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            model_type=args.model_type,
        )
        for row, prompt, gen in zip(batch, prompts, generations):
            results.append({
                **row,
                "prompt_text": prompt,
                "generated_next_sentence": gen,
                "model_name": args.model_name,
            })
    return results




def build_template_posneg(args, model, tokenizer, all_rows: dict[str, list[dict]]) -> dict:
    output_dir = Path(args.output_dir)
    summary = {}

    for dataset_name, rows in all_rows.items():
        print(f"\n[template/{dataset_name}] {len(rows)} rows")


        pos_path = output_dir / "pos" / f"{dataset_name}_native_chat_template.jsonl"
        if _should_skip(pos_path, args.overwrite):
            pos_rows = []
        else:
            print(f"  [pos] generating with native chat template...")

            def pos_prompt_fn(row):
                return render_prompt(tokenizer, normalize_head_text(get_input_text(row)),
                                     model_type=args.model_type)

            pos_rows = generate_for_rows(model, tokenizer, rows, pos_prompt_fn, args,
                                         desc=f"  {dataset_name}/pos")
            for r in pos_rows:
                r["prompt_group_name"] = "pos"
                r["template_name"] = "native_chat_template"

            write_jsonl(pos_path, pos_rows)
            print(f"  [pos] saved {len(pos_rows)} rows → {pos_path}")


        neg_path = output_dir / "neg" / f"{dataset_name}_not_broken_selected.jsonl"
        if _should_skip(neg_path, args.overwrite):
            neg_all = []
        else:

            n_templates = len(NEG_TEMPLATES)
            neg_all = []
            for tmpl_idx, tmpl in enumerate(NEG_TEMPLATES):
                tmpl_rows = rows[tmpl_idx::n_templates]
                print(f"  [neg] generating with template: {tmpl['name']} ({len(tmpl_rows)} rows)...")

                def neg_prompt_fn(row, t=tmpl):
                    return t["template"].format(question=normalize_head_text(get_input_text(row)))

                gen_rows = generate_for_rows(model, tokenizer, tmpl_rows, neg_prompt_fn, args,
                                             desc=f"  {dataset_name}/neg/{tmpl['name']}")
                for r in gen_rows:
                    r["prompt_group_name"] = "neg"
                    r["template_name"] = tmpl["name"]
                    r["input_text"] = get_input_text(r)


                print(f"  [neg] running broken judge for {tmpl['name']}...")
                annotated = asyncio.run(annotate_rows_async(
                    gen_rows,
                    judge_fn=lambda r: broken_judge(
                        r["input_text"], r["prompt_text"], r["generated_next_sentence"],
                        annotation_model=args.annotation_model,
                    ),
                    annotation_key="broken_annotation",
                    desc=f"  {dataset_name}/neg/{tmpl['name']}/broken_judge",
                    concurrency=args.annotation_concurrency,
                ))
                selected = [r for r in annotated if r.get("broken_annotation", {}).get("label") == "Not Broken"]
                neg_all.extend(selected)
                print(f"  [neg] {tmpl['name']}: {len(selected)}/{len(gen_rows)} Not Broken")

            write_jsonl(neg_path, neg_all)
        print(f"  [neg] total saved {len(neg_all)} rows → {neg_path}")

        summary[dataset_name] = {"pos": len(pos_rows), "neg": len(neg_all)}

    return summary




def build_pragmatic_posneg(args, model, tokenizer, all_rows: dict[str, list[dict]]) -> dict:
    output_dir = Path(args.output_dir)
    summary = {}

    for dataset_name, rows in all_rows.items():
        print(f"\n[pragmatic/{dataset_name}] {len(rows)} rows")


        pos_path = output_dir / "pos" / f"{dataset_name}_pragmatic_selected.jsonl"
        if _should_skip(pos_path, args.overwrite):
            pos_all = []
        else:
            pos_all = []
            n_pragmatic = len(PRAGMATIC_PROMPTS)
            for p_idx, prompt in enumerate(PRAGMATIC_PROMPTS):
                slug = slugify_prompt(prompt)
                prompt_rows = rows[p_idx::n_pragmatic]

                def prompt_fn(row, p=prompt):
                    return render_prompt(tokenizer, normalize_head_text(get_input_text(row)), p,
                                         model_type=args.model_type, prompt_set="pragmatic_level2")

                gen_rows = generate_for_rows(model, tokenizer, prompt_rows, prompt_fn, args,
                                             desc=f"  {dataset_name}/pos/{slug} ({len(prompt_rows)} rows)")
                for r in gen_rows:
                    r["prompt_group_name"] = "pos"
                    r["instruct_prompt"] = prompt

                annotated = asyncio.run(annotate_rows_async(
                    gen_rows,
                    judge_fn=lambda r: pragmatic_judge(
                        get_input_text(r), r["generated_next_sentence"],
                        annotation_model=args.annotation_model,
                    ),
                    annotation_key="pragmatic_annotation",
                    desc=f"  {dataset_name}/pos/{slug}/judge",
                    concurrency=args.annotation_concurrency,
                ))
                selected = [r for r in annotated if r.get("pragmatic_annotation", {}).get("label") == "Yes"]
                pos_all.extend(selected)
                print(f"  [pos] {slug}: {len(selected)}/{len(gen_rows)} Yes")

            write_jsonl(pos_path, pos_all)


        neg_path = output_dir / "neg" / f"{dataset_name}_generation_selected.jsonl"
        if _should_skip(neg_path, args.overwrite):
            neg_all = []
        else:
            neg_all = []
            n_generation = len(GENERATION_PROMPTS)
            for g_idx, prompt in enumerate(GENERATION_PROMPTS):
                slug = slugify_prompt(prompt)
                prompt_rows = rows[g_idx::n_generation]

                def prompt_fn(row, p=prompt):
                    return render_prompt(tokenizer, normalize_head_text(get_input_text(row)), p,
                                         model_type=args.model_type, prompt_set="generation_level2")

                gen_rows = generate_for_rows(model, tokenizer, prompt_rows, prompt_fn, args,
                                             desc=f"  {dataset_name}/neg/{slug} ({len(prompt_rows)} rows)")
                for r in gen_rows:
                    r["prompt_group_name"] = "neg"
                    r["instruct_prompt"] = prompt

                annotated = asyncio.run(annotate_rows_async(
                    gen_rows,
                    judge_fn=lambda r: pragmatic_judge(
                        get_input_text(r), r["generated_next_sentence"],
                        annotation_model=args.annotation_model,
                    ),
                    annotation_key="pragmatic_annotation",
                    desc=f"  {dataset_name}/neg/{slug}/judge",
                    concurrency=args.annotation_concurrency,
                ))
                selected = [r for r in annotated if r.get("pragmatic_annotation", {}).get("label") == "No"]
                neg_all.extend(selected)
                print(f"  [neg] {slug}: {len(selected)}/{len(gen_rows)} No")

            write_jsonl(neg_path, neg_all)

        summary[dataset_name] = {"pos": len(pos_all), "neg": len(neg_all)}
        print(f"  total: pos={len(pos_all)}, neg={len(neg_all)}")

    return summary




def build_principal_posneg(args, model, tokenizer,
                           pos_rows: dict[str, list[dict]],
                           neg_rows: dict[str, list[dict]]) -> dict:
    output_dir = Path(args.output_dir)
    summary = {}

    for dataset_name in pos_rows:
        p_rows = pos_rows[dataset_name]
        n_rows = neg_rows.get(dataset_name, [])
        print(f"\n[principal/{dataset_name}] pos_input={len(p_rows)}, neg_input={len(n_rows)}")


        pos_path = output_dir / "pos" / f"{dataset_name}_pragmatic_selected.jsonl"
        if _should_skip(pos_path, args.overwrite):
            pos_all = []
        else:
            pos_all = []
            for prompt in PRAGMATIC_PROMPTS:
                slug = slugify_prompt(prompt)

                def prompt_fn(row, p=prompt):
                    return render_prompt(tokenizer, normalize_head_text(get_input_text(row)), p,
                                         model_type=args.model_type, prompt_set="pragmatic_qa")

                gen_rows = generate_for_rows(model, tokenizer, p_rows, prompt_fn, args,
                                             desc=f"  {dataset_name}/pos/{slug}")
                for r in gen_rows:
                    r["prompt_group_name"] = "pos"
                    r["instruct_prompt"] = prompt

                annotated = asyncio.run(annotate_rows_async(
                    gen_rows,
                    judge_fn=lambda r: pragmatic_judge(
                        get_input_text(r), r["generated_next_sentence"],
                        annotation_model=args.annotation_model,
                    ),
                    annotation_key="pragmatic_annotation",
                    desc=f"  {dataset_name}/pos/{slug}/pragmatic",
                    concurrency=args.annotation_concurrency,
                ))
                selected = [r for r in annotated if r.get("pragmatic_annotation", {}).get("label") == "Yes"]
                pos_all.extend(selected)
                print(f"  [pos] {slug}: {len(selected)}/{len(gen_rows)} Yes")

            write_jsonl(pos_path, pos_all)


        neg_path = output_dir / "neg" / f"{dataset_name}_pragmatic_selected.jsonl"
        if _should_skip(neg_path, args.overwrite):
            neg_all = []
        else:
            neg_all = []

        import random
        rng = random.Random(42)
        shuffled_neg = list(n_rows)
        rng.shuffle(shuffled_neg)
        prompt_buckets = {p: [] for p in PRAGMATIC_PROMPTS}
        for i, row in enumerate(shuffled_neg):
            prompt_buckets[PRAGMATIC_PROMPTS[i % len(PRAGMATIC_PROMPTS)]].append(row)

        for prompt, bucket_rows in prompt_buckets.items():
            if not bucket_rows:
                continue
            slug = slugify_prompt(prompt)

            def prompt_fn(row, p=prompt):
                neg_text = str(row.get("rewritten_text") or row.get("head") or row.get("text") or row.get("input_text") or "").strip()
                return render_prompt(tokenizer, normalize_head_text(neg_text), p,
                                     model_type=args.model_type, prompt_set="pragmatic_qa")

            gen_rows = generate_for_rows(model, tokenizer, bucket_rows, prompt_fn, args,
                                         desc=f"  {dataset_name}/neg/{slug}")
            for r in gen_rows:
                r["prompt_group_name"] = "neg"
                r["instruct_prompt"] = prompt


            pragmatic_annotated = asyncio.run(annotate_rows_async(
                gen_rows,
                judge_fn=lambda r: pragmatic_judge(
                    get_input_text(r), r["generated_next_sentence"],
                    annotation_model=args.annotation_model,
                ),
                annotation_key="pragmatic_annotation",
                desc=f"  {dataset_name}/neg/{slug}/pragmatic",
                concurrency=args.annotation_concurrency,
            ))
            pragmatic_pass = [r for r in pragmatic_annotated if r.get("pragmatic_annotation", {}).get("label") == "Yes"]
            print(f"  [neg] {slug}: pragmatic pass {len(pragmatic_pass)}/{len(gen_rows)}")


            if pragmatic_pass:
                def principal_fn(r):
                    prefix = str(r.get("principal_prefix") or r.get("personx_replacement") or "").strip()
                    inp = get_input_text(r)
                    return principal_judge(prefix, inp, r["generated_next_sentence"],
                                          annotation_model=args.annotation_model)

                principal_annotated = asyncio.run(annotate_rows_async(
                    pragmatic_pass,
                    judge_fn=principal_fn,
                    annotation_key="principal_annotation",
                    desc=f"  {dataset_name}/neg/{slug}/principal",
                    concurrency=args.annotation_concurrency,
                ))
                principal_pass = [r for r in principal_annotated if r.get("principal_annotation", {}).get("label") == "Y"]
                neg_all.extend(principal_pass)
                print(f"  [neg] {slug}: principal pass {len(principal_pass)}/{len(pragmatic_pass)}")

            write_jsonl(neg_path, neg_all)

        summary[dataset_name] = {"pos": len(pos_all), "neg": len(neg_all)}
        print(f"  total: pos={len(pos_all)}, neg={len(neg_all)}")

    return summary




def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    all_rows: dict[str, list[dict]] = {}
    for path_str in args.input_data:
        path = Path(path_str)
        name = path.stem.split("_")[0]
        rows = read_jsonl(path)
        all_rows[name] = rows
        print(f"[input] {name}: {len(rows)} rows from {path}")


    print(f"\nLoading model: {args.model_name}  device={args.device}")
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)


    if args.level == "template":
        summary = build_template_posneg(args, model, tokenizer, all_rows)
    elif args.level == "pragmatic":
        summary = build_pragmatic_posneg(args, model, tokenizer, all_rows)
    elif args.level == "principal":
        neg_rows: dict[str, list[dict]] = {}
        if not args.neg_input_data:
            raise SystemExit("--neg-input-data is required for principal level.")
        for path_str in args.neg_input_data:
            path = Path(path_str)
            name = path.stem.split("_")[0]
            rows = read_jsonl(path)
            neg_rows[name] = rows
            print(f"[neg-input] {name}: {len(rows)} rows from {path}")
        summary = build_principal_posneg(args, model, tokenizer, all_rows, neg_rows)


    write_json(output_dir / "summary.json", {
        "level": args.level,
        "model_name": args.model_name,
        "model_type": args.model_type,
        "input_files": args.input_data,
        "neg_input_files": args.neg_input_data,
        "datasets": summary,
    })

    print(f"\nDone. Output: {output_dir}")
    print(f"  pos: {output_dir / 'pos'}")
    print(f"  neg: {output_dir / 'neg'}")
    print(f"\nNext step (mean-diff):")
    print(f"  python step02_mean_diff.py --pos-dir {output_dir / 'pos'} --neg-dir {output_dir / 'neg'} ...")


if __name__ == "__main__":
    main()
