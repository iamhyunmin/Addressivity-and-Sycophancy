
from __future__ import annotations



NEG_TEMPLATES = [
    {"name": "question_answer", "template": "Question:\n{question}\n\nAnswer:"},
    {"name": "instruction_response", "template": "Instruction:\n{question}\nResponse:"},
    {"name": "alpaca_style", "template": (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n\n{question}\n\n### Response:"
    )},
    {"name": "vicuna_style", "template": (
        "A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        "### Human: {question}\n\n### Assistant:"
    )},
    {"name": "inst_style", "template": "<s>[INST] {question} [/INST]"},
]



PRAGMATIC_PROMPTS = [
    "You are receiving a message. Reply to it.",
    "This is a conversation between a user and you.",
    "You are a respondent in a conversation.",
    "The following is a message directed at you. Respond.",
    "Address what the user has said.",
]

GENERATION_PROMPTS = [
    "Given the input sentence, predict what sentence would come next.",
    "Given the input below, output the next sentence.",
    "Produce a continuation of the passage below.",
    "Produce a plausible next sentence for following text .",
    "Generate the next sentence.",
]




