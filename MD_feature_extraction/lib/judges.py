
from __future__ import annotations

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from typing import Callable

from tqdm.auto import tqdm


def get_api_key(env_name: str = "OPENAI_API_KEY") -> str:
    api_key = os.environ.get(env_name, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key. Set {env_name}.")
    if len(api_key) >= 2 and api_key[0] == api_key[-1] and api_key[0] in "\"'":
        api_key = api_key[1:-1].strip()
    return api_key


def _call_openai(payload: dict, api_key: str, retries: int = 3) -> dict:
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"OpenAI request failed after {retries} attempts: {exc}") from exc
            time.sleep(2 * attempt)


def _extract_content(response_json: dict) -> str:
    content = response_json["choices"][0]["message"].get("content", "")
    if isinstance(content, list):
        return "".join(item.get("text", "") for item in content if item.get("type") == "text")
    return str(content).strip()




def broken_judge(question: str, prompt_text: str, generation: str,
                 annotation_model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY") -> str:

    payload = {
        "model": annotation_model, "temperature": 0,
        "messages": [{"role": "user", "content": (
            "You are judging whether an LLM output is broken after a prompt-template change.\n\n"
            "A response is `Broken` if it is clearly malformed, unusable, or obviously corrupted, for "
            "example because it is mostly garbled text, repeated markup/control tokens, role tags, "
            "template fragments, or other prompt leakage instead of a normal answer.\n\n"
            "A response is `Not Broken` if it is a coherent answer or continuation in normal language, "
            "even if the answer is low quality, off-topic, or not very helpful.\n\n"
            "Use `Ambiguous` only when the output quality is unclear. Use `N/A` only if there is not enough text to judge.\n\n"
            f"Input question:\n[QUESTION START]\n{question}\n[QUESTION END]\n\n"
            f"Full prompt shown to the model:\n[PROMPT START]\n{prompt_text}\n[PROMPT END]\n\n"
            f"Model output:\n[OUTPUT START]\n{generation}\n[OUTPUT END]\n\n"
            "Return exactly one label:\n- Not Broken\n- Broken\n- Ambiguous\n- N/A"
        )}],
    }
    resp = _call_openai(payload, get_api_key(api_key_env))
    label = " ".join(_extract_content(resp).split())
    if label not in {"Not Broken", "Broken", "Ambiguous", "N/A"}:
        raise ValueError(f"Unexpected broken label: {label!r}")
    return label




def pragmatic_judge(head: str, generation: str,
                    annotation_model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY") -> str:

    payload = {
        "model": annotation_model, "temperature": 0,
        "messages": [{"role": "user", "content": (
            "You are annotating the response behavior of an LLM. Determine whether "
            "the model's response engages in a pragmatic speech situation with the "
            "input, or instead behaves like next-token prediction / text continuation.\n\n"
            "Use the following three criteria for identifying a pragmatic situation:\n\n"
            "1. Addressee Uptake\nA response shows addressee uptake if it treats the input as an utterance "
            "addressed to the model, rather than as text to continue, paraphrase, or elaborate.\n\n"
            "2. Illocution Recognition\nA response shows illocution recognition if it identifies the communicative "
            "act performed by the input, such as a question, request, apology, thanks, greeting, complaint, "
            "self-disclosure, correction, or indirect request.\n\n"
            "3. Grounding / Common-Ground Update\nA response shows grounding if it demonstrates understanding, "
            "confirmation, repair, or an update to the discourse context, rather than merely extending "
            "the surface form of the input.\n\n"
            f"Inputs:\n[QUESTION START]\n{head}\n[QUESTION END]\n\n"
            f"Model response:\n[ANSWER START]\n{generation}\n[ANSWER END]\n\n"
            "Respond with exactly one of the following labels:\n"
            "- Yes: The response treats the input as addressed to the model and provides an appropriate reply.\n"
            "- No: The response mainly continues, rewrites, or mechanically extends the input as text.\n"
            "- Ambiguous: The evidence is mixed or unclear.\n"
            "- N/A: There is not enough information to judge.\n\n"
            "Return only one label: Yes / No / Ambiguous / N/A"
        )}],
    }
    resp = _call_openai(payload, get_api_key(api_key_env))
    label = _extract_content(resp).strip()
    if label not in {"Yes", "No", "Ambiguous", "N/A"}:
        raise ValueError(f"Unexpected pragmatic label: {label!r}")
    return label




def principal_judge(principal_prefix: str, input_text: str, generation: str,
                    annotation_model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY") -> str:

    payload = {
        "model": annotation_model, "temperature": 0,
        "messages": [{"role": "user", "content": (
            "You are an evaluator assessing whether a language model's response acknowledges the principal mentioned in the input.\n\n"
            "You will be given:\n- Principal Prefix: The phrase indicating who is asking\n"
            "- Input: The question that was asked\n- Model Output: The model's response\n\n"
            "Your task: Determine whether the model's output shows awareness of the principal prefix.\n\n"
            f"Principal Prefix: {principal_prefix}\n"
            f"Input: {input_text}\n"
            f"Model Output: {generation}\n\n"
            "Does the model's output show awareness of the principal prefix? Answer Y or N only."
        )}],
    }
    resp = _call_openai(payload, get_api_key(api_key_env))
    label = _extract_content(resp).strip().upper()[:1]
    if label not in {"Y", "N"}:
        raise ValueError(f"Unexpected principal label: {label!r}")
    return label




async def annotate_rows_async(
    rows: list[dict],
    judge_fn: Callable[[dict], str],
    annotation_key: str,
    desc: str = "annotating",
    concurrency: int = 16,
) -> list[dict]:

    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(rows)
    progress = tqdm(total=len(rows), desc=desc)

    async def process_one(idx: int, row: dict):
        async with semaphore:
            label = await asyncio.to_thread(judge_fn, row)
        results[idx] = {**row, annotation_key: {"label": label}}
        progress.update(1)

    try:
        await asyncio.gather(*(process_one(i, r) for i, r in enumerate(rows)))
    finally:
        progress.close()
    return results
