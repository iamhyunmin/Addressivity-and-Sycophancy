
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from .base import (
    Benchmark,
    generation_eos_token_ids,
    resolve_batch_size,
    should_add_special_tokens,
)
from ..utils.io import read_jsonl


def _build_oeq_prompt(tokenizer, record: dict, model_type: str, cfg: dict) -> str:
    question = str(record.get("question", "")).strip()
    if model_type == "instruct":
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return f"Question: {question}\n\nAnswer:"


def _generate_batch(model, tokenizer, prompts: list[str], gen_cfg: dict, device: str, cfg: dict) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=should_add_special_tokens(cfg),
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 512),
            temperature=gen_cfg.get("temperature", 0.6),
            do_sample=gen_cfg.get("do_sample", True),
            top_p=gen_cfg.get("top_p", 0.9),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=generation_eos_token_ids(tokenizer),
        )
    n_input = inputs["input_ids"].shape[1]
    preds = []
    for seq in out:
        preds.append(tokenizer.decode(seq[n_input:], skip_special_tokens=True).strip())
    return preds


def _length_sorted_batches(records: list[dict], prompts: list[str], batch_size: int):

    indexed = sorted(enumerate(zip(records, prompts)), key=lambda x: len(x[1][1]))
    orig_indices, pairs = zip(*indexed) if indexed else ([], [])
    recs_sorted = [p[0] for p in pairs]
    prompts_sorted = [p[1] for p in pairs]
    orig_indices = list(orig_indices)

    rec_batches, prompt_batches, idx_batches = [], [], []
    for i in range(0, len(recs_sorted), batch_size):
        rec_batches.append(recs_sorted[i:i + batch_size])
        prompt_batches.append(prompts_sorted[i:i + batch_size])
        idx_batches.append(orig_indices[i:i + batch_size])
    return rec_batches, prompt_batches, idx_batches


class ElephantBenchmark(Benchmark):


    def __init__(self, dataset_name: str):
        assert dataset_name in ("elephant_oeq", "elephant_aita", "elephant_ss")
        self.dataset_name = dataset_name

    @property
    def name(self):
        return self.dataset_name


    def load_main_set(self, main_set_path: Path) -> list[dict]:
        import pandas as pd
        df = pd.read_csv(main_set_path)
        unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)

        if self.dataset_name == "elephant_oeq":
            field_map = {
                "question": "prompt",
                "human_answer": "human",
                "source": "source",
                "validation_human": "validation_human",
                "indirectness_human": "indirectness_human",
                "framing_human": "framing_human",
            }
            id_prefix = "oeq"
        elif self.dataset_name == "elephant_aita":
            field_map = {
                "question": "prompt",
                "human_answer": "top_comment",
                "is_asshole": "is_asshole",
                "ytanta": "ytanta",
                "validation_human": "validation_human",
                "indirectness_human": "indirectness_human",
                "framing_human": "framing_human",
            }
            id_prefix = "aita"
        else:
            field_map = {
                "question": "sentence",
                "self_attitude": "self_attitude",
                "other_attitude": "other_attitude",
            }
            id_prefix = "ss"

        records = []
        for idx, row in df.iterrows():
            rec = {"sample_id": f"{id_prefix}:{idx}"}
            for out_key, src_key in field_map.items():
                val = row.get(src_key)

                if val is not None and hasattr(val, "item"):
                    val = val.item()
                try:
                    import math
                    if val is not None and isinstance(val, float) and math.isnan(val):
                        val = None
                except (TypeError, ValueError):
                    pass
                rec[out_key] = val
            records.append(rec)
        return records

    def generate(self, model, tokenizer, records: list[dict], cfg: dict, device: str) -> list[dict]:
        model_type = cfg["model"].get("type", "instruct")
        gen_cfg = cfg.get("task", {}).get("open_ended", {}).get("generation", {})
        batch_size = resolve_batch_size(cfg, self.dataset_name, default=8)


        all_prompts = [_build_oeq_prompt(tokenizer, r, model_type, cfg) for r in records]
        rec_batches, prompt_batches, idx_batches = _length_sorted_batches(records, all_prompts, batch_size)


        results = [None] * len(records)
        total_batches = len(rec_batches)

        for batch_recs, batch_prompts, batch_indices in tqdm(
            zip(rec_batches, prompt_batches, idx_batches),
            total=total_batches,
            desc=f"{self.dataset_name}",
        ):
            preds = _generate_batch(model, tokenizer, batch_prompts, gen_cfg, device, cfg)
            for orig_idx, rec, prompt, pred in zip(batch_indices, batch_recs, batch_prompts, preds):
                res = {
                    "sample_id": rec["sample_id"],
                    "question": rec.get("question", ""),
                    "full_prompt": prompt,
                    "prediction": pred,
                }
                for key in ("validation_human", "indirectness_human", "framing_human", "self_attitude", "other_attitude"):
                    if key in rec:
                        res[key] = rec[key]
                results[orig_idx] = res

        return results
