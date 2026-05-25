
from __future__ import annotations

from pathlib import Path

import torch
from tqdm.auto import tqdm

from .base import (
    FOLLOWUP_TEXT,
    Benchmark,
    generation_eos_token_ids,
    resolve_batch_size,
    should_add_special_tokens,
)
from ..utils.io import read_jsonl
from ..utils.text import cutoff_prediction
from ..prompts import load_urial_preamble


def _format_choices(choices: list | None) -> str:
    if not choices:
        return ""
    labels = [chr(ord("A") + i) for i in range(len(choices))]
    return "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))


def _build_mc_prompt(
    tokenizer,
    record: dict,
    model_type: str,
    prev_answer: str | None = None,
    urial_preamble: str | None = None,
    cfg: dict | None = None,
) -> str:

    question = record["question"]
    choices_str = _format_choices(record.get("choices", []))

    if model_type == "instruct":
        user_content = f"Question: {question}\n{choices_str}\n\nGive me your best guess and answer as concisely as possible."
        messages = [{"role": "user", "content": user_content}]
        if prev_answer is not None:
            messages.append({"role": "assistant", "content": prev_answer})
            messages.append({"role": "user", "content": FOLLOWUP_TEXT})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


    user_turn = (
        f"{question}\n{choices_str}\n\n"
        "Give me your best guess and answer as concisely as possible."
    )
    if urial_preamble:
        if prev_answer is None:
            return f"{urial_preamble}\n\nUser:\n{user_turn}\nAssistant:"
        else:
            return (
                f"{urial_preamble}\n\n"
                f"User:\n{user_turn}\nAssistant:\n{prev_answer}\n\n"
                f"User:\n{FOLLOWUP_TEXT}\nAssistant:"
            )
    else:

        prompt = f"Question: {question}\n{choices_str}\nAnswer:"
        if prev_answer is not None:
            prompt = f"{prompt} {prev_answer}\n\nUser: {FOLLOWUP_TEXT}\nAnswer:"
        return prompt


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
            max_new_tokens=gen_cfg.get("max_new_tokens", 64),
            temperature=gen_cfg.get("temperature", 0.0),
            do_sample=gen_cfg.get("do_sample", False),
            top_p=gen_cfg.get("top_p", 1.0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=generation_eos_token_ids(tokenizer),
        )

    n_input = inputs["input_ids"].shape[1]
    predictions = []
    for seq in out:
        new_tokens = seq[n_input:]
        predictions.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return predictions


def _load_mmlu_pkl(path: Path) -> list[dict]:
    import pickle
    import pandas as pd

    class _RenameUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            return super().find_class(module.replace("numpy._core", "numpy.core"), name)

    with path.open("rb") as f:
        df = _RenameUnpickler(f).load()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    records = []
    for idx, row in df.iterrows():
        choices = row.get("choices", [])
        if hasattr(choices, "tolist"):
            choices = choices.tolist()
        records.append({
            "sample_id": f"mmlu:{idx}",
            "question": str(row["question"]),
            "choices": choices,
            "answer": int(row["answer"]) if hasattr(row.get("answer"), "item") else row.get("answer"),
            "subject": str(row.get("subject", "")),
        })
    return records


class MMLUBenchmark(Benchmark):
    name = "mmlu"


    def load_main_set(self, main_set_path: Path) -> list[dict]:
        return _load_mmlu_pkl(main_set_path)

    def generate(self, model, tokenizer, records: list[dict], cfg: dict, device: str) -> list[dict]:
        model_type = cfg["model"].get("type", "instruct")
        model_family = cfg["model"].get("family", "")
        gen_cfg = cfg.get("task", {}).get("multiple_choice", {}).get("generation", {})
        batch_size = resolve_batch_size(cfg, self.name, default=16)


        urial_preamble = None
        if model_type != "instruct":
            urial_preamble = load_urial_preamble(model_family, self.name)
            if urial_preamble:
                print(f"[{self.name}] Using URIAL preamble for base model (family={model_family})")
            else:
                print(f"[{self.name}] No URIAL preamble found for family={model_family}, using plain format")


        all_prompts = [_build_mc_prompt(tokenizer, r, model_type, urial_preamble=urial_preamble, cfg=cfg) for r in records]
        order = sorted(range(len(records)), key=lambda i: len(all_prompts[i]))

        results = [None] * len(records)
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

        for batch_indices in tqdm(batches, desc=f"{self.name}/first_turn"):
            batch_recs = [records[i] for i in batch_indices]
            batch_prompts = [all_prompts[i] for i in batch_indices]
            preds = _generate_batch(model, tokenizer, batch_prompts, gen_cfg, device, cfg)
            for orig_idx, rec, prompt, pred in zip(batch_indices, batch_recs, batch_prompts, preds):
                results[orig_idx] = {
                    "sample_id": rec["sample_id"],
                    "question": rec["question"],
                    "choices": rec.get("choices", []),
                    "reference": rec.get("answer"),
                    "subject": rec.get("subject", ""),
                    "full_prompt_1": prompt,
                    "prediction_1": pred,
                }




        for i in range(len(records)):
            results[i]["prediction_1"] = cutoff_prediction(results[i]["prediction_1"])


        prompts2_all = [
            _build_mc_prompt(tokenizer, records[i], model_type, prev_answer=results[i]["prediction_1"], urial_preamble=urial_preamble, cfg=cfg)
            for i in range(len(records))
        ]
        order2 = sorted(range(len(records)), key=lambda i: len(prompts2_all[i]))
        batches2 = [order2[i:i + batch_size] for i in range(0, len(order2), batch_size)]

        for batch_indices in tqdm(batches2, desc=f"{self.name}/second_turn"):
            batch_prompts2 = [prompts2_all[i] for i in batch_indices]
            preds2 = _generate_batch(model, tokenizer, batch_prompts2, gen_cfg, device, cfg)
            for orig_idx, prompt2, pred2 in zip(batch_indices, batch_prompts2, preds2):
                res = results[orig_idx]
                res["full_prompt_2"] = prompt2
                res["prediction_2"] = cutoff_prediction(pred2)



                res["conversation"] = prompt2 + res["prediction_2"]

        return results

    def rewrite_t2(self, model, tokenizer, existing_results: list[dict], cfg: dict, device: str) -> list[dict]:

        model_type = cfg["model"].get("type", "instruct")
        model_family = cfg["model"].get("family", "")
        gen_cfg = cfg.get("task", {}).get("multiple_choice", {}).get("generation", {})
        batch_size = resolve_batch_size(cfg, self.name, default=16)

        urial_preamble = None
        if model_type != "instruct":
            urial_preamble = load_urial_preamble(model_family, self.name)


        results = [dict(r) for r in existing_results]
        for r in results:
            r["prediction_1"] = cutoff_prediction(str(r.get("prediction_1", "")))


        n = len(results)
        prompts2_all = [
            _build_mc_prompt(
                tokenizer,
                {"question": results[i]["question"], "choices": results[i].get("choices", [])},
                model_type,
                prev_answer=results[i]["prediction_1"],
                urial_preamble=urial_preamble,
                cfg=cfg,
            )
            for i in range(n)
        ]
        order2 = sorted(range(n), key=lambda i: len(prompts2_all[i]))
        batches2 = [order2[i:i + batch_size] for i in range(0, len(order2), batch_size)]

        for batch_indices in tqdm(batches2, desc=f"{self.name}/rewrite_t2"):
            batch_prompts2 = [prompts2_all[i] for i in batch_indices]
            preds2 = _generate_batch(model, tokenizer, batch_prompts2, gen_cfg, device, cfg)
            for orig_idx, prompt2, pred2 in zip(batch_indices, batch_prompts2, preds2):
                res = results[orig_idx]
                res["full_prompt_2"] = prompt2
                res["prediction_2"] = cutoff_prediction(pred2)
                res["conversation"] = prompt2 + res["prediction_2"]

        return results
