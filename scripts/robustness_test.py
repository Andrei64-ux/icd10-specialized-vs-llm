"""
Testare de robustete (O1, cerinta explicita): cat de mult scade performanta
modelului specializat (si, pe un esantion, a LLM-ului generalist) cand textul
clinic e zgomotos (typo-uri) sau incomplet (trunchiat / propozitii lipsa).
"""
import re
import random

import numpy as np
import requests
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from build_dataset import MIN_CODE_FREQ, build_label_space, load_split
from llm_baseline import OLLAMA_URL, MODEL as LLM_MODEL, PROMPT_TEMPLATE, parse_codes

MODEL_DIR = "icd10_model"
THRESHOLD = 0.5
LLM_SAMPLE_SIZE = 60  # subset pentru LLM, din motive de timp (fiecare apel ~3s pe GPU)


def add_typos(text, prob=0.05, seed=0):
    rng = random.Random(seed)
    chars = list(text)
    i = 0
    while i < len(chars):
        if chars[i].isalpha() and rng.random() < prob:
            op = rng.choice(["delete", "swap", "dup"])
            if op == "delete":
                del chars[i]
                continue
            elif op == "swap" and i < len(chars) - 1:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif op == "dup":
                chars.insert(i, chars[i])
                i += 1
        i += 1
    return "".join(chars)


def truncate_text(text, keep_ratio=0.5):
    return text[: int(len(text) * keep_ratio)]


def remove_sentences(text, keep_ratio=0.6, seed=0):
    rng = random.Random(seed)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) <= 1:
        return text
    n_keep = max(1, int(len(sentences) * keep_ratio))
    kept_idx = sorted(rng.sample(range(len(sentences)), min(n_keep, len(sentences))))
    return " ".join(sentences[i] for i in kept_idx)


PERTURBATIONS = {
    "curat (fara perturbare)": lambda t: t,
    "typo-uri (5% caractere)": lambda t: add_typos(t, prob=0.05),
    "trunchiat (50% din text)": lambda t: truncate_text(t, keep_ratio=0.5),
    "incomplet (60% din propozitii)": lambda t: remove_sentences(t, keep_ratio=0.6),
}


def run_specialized_model(texts_by_case, case_ids, label_to_idx, tokenizer, model, transform):
    labels = sorted(label_to_idx, key=label_to_idx.get)
    y_pred = np.zeros((len(case_ids), len(labels)), dtype=int)
    with torch.no_grad():
        for i, case_id in enumerate(case_ids):
            text = transform(texts_by_case[case_id])
            enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            probs = torch.sigmoid(model(**enc).logits).squeeze(0).numpy()
            y_pred[i] = (probs >= THRESHOLD).astype(int)
    return y_pred


def run_llm(texts_by_case, case_ids, label_to_idx, transform):
    labels = sorted(label_to_idx, key=label_to_idx.get)
    y_pred = np.zeros((len(case_ids), len(labels)), dtype=int)
    for i, case_id in enumerate(case_ids):
        text = transform(texts_by_case[case_id])
        prompt = PROMPT_TEMPLATE.format(text=text)
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": LLM_MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.1, "num_predict": 150}},
                timeout=120,
            )
            raw = resp.json()["response"]
        except Exception:
            continue
        for code in parse_codes(raw):
            if code in label_to_idx:
                y_pred[i, label_to_idx[code]] = 1
        print(f"  LLM [{i+1}/{len(case_ids)}]", flush=True)
    return y_pred


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    _, train_codes = load_split("train")
    label_to_idx = build_label_space(train_codes, MIN_CODE_FREQ)
    labels = sorted(label_to_idx, key=label_to_idx.get)

    texts_by_case, dev_codes = load_split("dev")
    case_ids = sorted(texts_by_case)

    y_true = np.zeros((len(case_ids), len(labels)), dtype=int)
    for i, case_id in enumerate(case_ids):
        for code in dev_codes.get(case_id, []):
            if code in label_to_idx:
                y_true[i, label_to_idx[code]] = 1

    llm_case_ids = case_ids[:LLM_SAMPLE_SIZE]
    y_true_llm_subset = y_true[:LLM_SAMPLE_SIZE]

    print("=== Model specializat (ES) — tot setul de dev (250 cazuri) ===")
    spec_results = {}
    for name, fn in PERTURBATIONS.items():
        y_pred = run_specialized_model(texts_by_case, case_ids, label_to_idx, tokenizer, model, fn)
        f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        spec_results[name] = f1
        print(f"{name:35s} F1_micro={f1:.3f}", flush=True)

    print(f"\n=== LLM generalist (mistral) — esantion {LLM_SAMPLE_SIZE} cazuri ===")
    llm_results = {}
    for name, fn in PERTURBATIONS.items():
        print(f"-- {name} --", flush=True)
        y_pred = run_llm(texts_by_case, llm_case_ids, label_to_idx, fn)
        f1 = f1_score(y_true_llm_subset, y_pred, average="micro", zero_division=0)
        llm_results[name] = f1
        print(f"{name:35s} F1_micro={f1:.3f}", flush=True)

    print("\n=== Rezumat degradare (fata de text curat) ===")
    clean_spec = spec_results["curat (fara perturbare)"]
    clean_llm = llm_results["curat (fara perturbare)"]
    for name in PERTURBATIONS:
        if name == "curat (fara perturbare)":
            continue
        d_spec = spec_results[name] - clean_spec
        d_llm = llm_results[name] - clean_llm
        print(f"{name:35s} specializat: {d_spec:+.3f}   LLM: {d_llm:+.3f}")


if __name__ == "__main__":
    main()
