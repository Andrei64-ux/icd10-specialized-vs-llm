"""
Masurare comparativa de latenta: model specializat (local, CPU) vs. LLM generalist
(mistral, prin Ollama, GPU) - completeaza ultima componenta netestata a ipotezei O1
("la un cost de inferenta redus").
"""
import time

import numpy as np
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from build_dataset import MIN_CODE_FREQ, build_label_space, load_split
from llm_baseline import OLLAMA_URL, MODEL as LLM_MODEL, PROMPT_TEMPLATE

MODEL_DIR = "icd10_model"
N_SAMPLES = 30


def measure_specialized(texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # warm-up (exclus din masuratoare) - elimina costul de incarcare lazy a kernelurilor
    enc = tokenizer(texts[0], truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        model(**enc)

    times = []
    for text in texts:
        t0 = time.perf_counter()
        enc = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
            torch.sigmoid(logits)
        times.append(time.perf_counter() - t0)
    return times


def measure_llm(texts):
    times = []
    for text in texts:
        prompt = PROMPT_TEMPLATE.format(text=text)
        t0 = time.perf_counter()
        try:
            requests.post(
                OLLAMA_URL,
                json={"model": LLM_MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.1, "num_predict": 150}},
                timeout=120,
            )
        except Exception:
            continue
        times.append(time.perf_counter() - t0)
    return times


def summarize(name, times):
    arr = np.array(times)
    print(f"{name}:")
    print(f"  n={len(arr)}  medie={arr.mean()*1000:.1f}ms  mediana={np.median(arr)*1000:.1f}ms  "
          f"p95={np.percentile(arr, 95)*1000:.1f}ms  min={arr.min()*1000:.1f}ms  max={arr.max()*1000:.1f}ms")
    return arr.mean()


def main():
    _, train_codes = load_split("train")
    build_label_space(train_codes, MIN_CODE_FREQ)  # doar pentru consistenta cu restul pipeline-ului
    texts_by_case, _ = load_split("dev")
    case_ids = sorted(texts_by_case)[:N_SAMPLES]
    texts = [texts_by_case[c] for c in case_ids]

    print(f"Masurare pe {len(texts)} cazuri (text spaniol, acelasi set folosit si la comparatia F1)\n")

    print("=== Model specializat (CPU, local, in-process) ===")
    spec_times = measure_specialized(texts)
    spec_mean = summarize("Model specializat", spec_times)

    print("\n=== LLM generalist (mistral, Ollama, GPU) ===")
    llm_times = measure_llm(texts)
    llm_mean = summarize("LLM generalist", llm_times)

    print(f"\n=== Rezumat ===")
    print(f"Model specializat: {spec_mean*1000:.1f}ms/cerere in medie")
    print(f"LLM generalist:    {llm_mean*1000:.1f}ms/cerere in medie")
    print(f"Raport de viteza:  modelul specializat e de {llm_mean/spec_mean:.1f}x mai rapid")
    print(f"\nNota: modelul specializat ruleaza pe CPU, LLM-ul ruleaza pe GPU (RTX 5050) -"
          f" comparatia e, daca ceva, defavorabila modelului specializat din perspectiva hardware.")


if __name__ == "__main__":
    main()
