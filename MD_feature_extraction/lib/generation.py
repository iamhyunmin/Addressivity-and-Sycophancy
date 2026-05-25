
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_name: str, device: str = "cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=None,
    ).to(device)
    model.eval()
    return model, tokenizer


def render_prompt(tokenizer, input_text: str, instruct_prompt: str = "",
                  model_type: str = "instruct", prompt_set: str = "pragmatic") -> str:

    if model_type == "instruct":
        messages = []
        if instruct_prompt:
            messages.append({"role": "system", "content": instruct_prompt})
        messages.append({"role": "user", "content": input_text})
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:

            combined = f"{instruct_prompt}\n\n{input_text}" if instruct_prompt else input_text
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": combined}],
                tokenize=False, add_generation_prompt=True,
            )
    else:

        if prompt_set == "pragmatic_qa":

            return f"{instruct_prompt}\n\nQuestion: {input_text}\nAnswer:" if instruct_prompt else f"Question: {input_text}\nAnswer:"
        else:

            return f"{instruct_prompt}\n\n{input_text}" if instruct_prompt else input_text



def render_chat_prompt(tokenizer, input_text: str, instruct_prompt: str = "") -> str:
    return render_prompt(tokenizer, input_text, instruct_prompt, model_type="instruct")


def clean_generated_text(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text.startswith("model\n"):
        text = text[len("model\n"):].strip()
    for marker in (
        "<end_of_turn>", "<eos>", "<bos>", "<start_of_turn>",
        "<|eot_id|>", "<|start_header_id|>", "\nmodel\n",
    ):
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    if "\n\n" in text:
        text = text.split("\n\n", 1)[0].strip()
    return text


def generate_batch(
    model, tokenizer, prompts: list[str], device: str,
    max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0,
    model_type: str = "instruct",
) -> list[str]:

    add_special = model_type != "instruct"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=add_special)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    padded_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)

    results = []
    for row_ids in output_ids:
        text = tokenizer.decode(row_ids[padded_len:], skip_special_tokens=True).strip()
        results.append(clean_generated_text(text))
    return results
