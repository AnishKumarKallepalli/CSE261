"""
Zero-shot emotion classification with Qwen (instruction-tuned).
Runs on Reddit validation and Twitter test; saves metrics and predictions to outputs/results/qwen_zero_shot.json.
"""

import os
import re
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM

import config
from data_load import get_cached


LABELS_STR = ", ".join(config.SIX_LABELS)
PROMPT_TEMPLATE = (
    "Classify the emotion of the following text. Choose exactly one: {labels}. "
    "Reply with only that one word.\n\nText: {text}"
)


def _response_to_label_id(response: str) -> int:
    """Map model output to label index in SIX_LABELS. Default 0 if unparseable."""
    if not response:
        return 0
    s = response.strip().lower()
    # Take first line or first token
    s = s.split("\n")[0].strip()
    s = s.split()[0] if s else ""
    # Remove trailing punctuation
    s = re.sub(r"[.,;:!?]+$", "", s)
    return config.SIX_LABEL2ID.get(s, 0)


def _predict_batch(texts, tokenizer, model, device, batch_size=4, max_new_tokens=15):
    """Run zero-shot Qwen on a list of texts; returns list of label indices."""
    model.eval()
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_inputs = []
        for text in batch_texts:
            content = PROMPT_TEMPLATE.format(labels=LABELS_STR, text=text)
            messages = [{"role": "user", "content": content}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_inputs.append(prompt)
        enc = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True,
        )
        target_device = next(model.parameters()).device
        enc = {k: v.to(target_device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # Decode only the generated part (after input length)
        input_len = enc["input_ids"].shape[1]
        for j in range(len(batch_texts)):
            gen_ids = out[j, input_len:]
            decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_preds.append(_response_to_label_id(decoded))
    return np.array(all_preds, dtype=np.int64)


def run_qwen_zero_shot():
    """Run zero-shot Qwen on Reddit validation and Twitter test; save metrics and predictions."""
    reddit, twitter = get_cached()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = getattr(config, "QWEN_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

    print("Loading Qwen model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    results = {"model": model_name, "label_names": config.SIX_LABELS}

    # Reddit validation
    X_val = reddit["validation"]["texts"]
    y_val = np.array(reddit["validation"]["labels"])
    print("Running zero-shot on Reddit validation...")
    pred_val = _predict_batch(X_val, tokenizer, model, device)
    results["qwen_reddit_val"] = {
        "accuracy": float(accuracy_score(y_val, pred_val)),
        "macro_f1": float(f1_score(y_val, pred_val, average="macro")),
    }
    results["qwen_reddit_val_preds"] = pred_val.tolist()

    # Twitter test
    X_te = twitter["test"]["texts"]
    y_te = np.array(twitter["test"]["labels"])
    print("Running zero-shot on Twitter test...")
    pred_te = _predict_batch(X_te, tokenizer, model, device)
    results["qwen_twitter_test"] = {
        "accuracy": float(accuracy_score(y_te, pred_te)),
        "macro_f1": float(f1_score(y_te, pred_te, average="macro")),
    }
    results["qwen_twitter_test_preds"] = pred_te.tolist()

    out_path = os.path.join(config.RESULTS_DIR, "qwen_zero_shot.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved", out_path)
    return results


if __name__ == "__main__":
    run_qwen_zero_shot()
