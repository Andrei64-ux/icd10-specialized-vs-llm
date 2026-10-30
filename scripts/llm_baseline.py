"""
Baseline LLM generalist (Ollama/mistral, prompting) pentru codificare ICD-10,
pe acelasi set de date (CodiEsp dev, spaniola) si acelasi spatiu de etichete
(769 coduri, prag>=2) folosit de modelul specializat - comparatie corecta,
nu doar cifre din literatura.
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import requests

from build_dataset import MIN_CODE_FREQ, build_label_space, load_split

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL = "mistral"
OUT_PATH = Path(__file__).resolve().parent / "llm_baseline_results.json"
CODE_PATTERN = re.compile(r"\b([a-z]\d{2}(?:\.\d+)?)\b", re.IGNORECASE)

PROMPT_TEMPLATE = """Eres un medico codificador clinico experto en ICD-10.
Basandote EXCLUSIVAMENTE en el siguiente texto clinico, identifica los codigos ICD-10 de diagnostico (CIE-10-CM) relevantes.

Responde ESTRICTAMENTE con una lista de codigos separados por comas, en minusculas (ej: r50.9, i10, j44.9).
No incluyas explicaciones, nombres de enfermedades ni ningun otro texto.

Texto clinico:
\"\"\"{text}\"\"\"

Codigos ICD-10:"""


def query_llm(text):
    prompt = PROMPT_TEMPLATE.format(text=text)
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 150},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


def parse_codes(raw_response):
    return sorted(set(m.group(1).lower() for m in CODE_PATTERN.finditer(raw_response)))


def main():
    _, train_codes = load_split("train")
    label_to_idx = build_label_space(train_codes, MIN_CODE_FREQ)
    labels = sorted(label_to_idx, key=label_to_idx.get)

    texts, dev_codes = load_split("dev")
    case_ids = sorted(texts)

    y_true = np.zeros((len(case_ids), len(labels)), dtype=int)
    y_pred = np.zeros((len(case_ids), len(labels)), dtype=int)
    raw_responses = {}
    out_of_space_codes = set()

    t_start = time.time()
    for i, case_id in enumerate(case_ids):
        for code in dev_codes.get(case_id, []):
            if code in label_to_idx:
                y_true[i, label_to_idx[code]] = 1

        try:
            raw = query_llm(texts[case_id])
        except Exception as e:
            print(f"[{i+1}/{len(case_ids)}] {case_id}: EROARE ({e})", flush=True)
            raw_responses[case_id] = f"EROARE: {e}"
            continue

        raw_responses[case_id] = raw
        codes = parse_codes(raw)
        for code in codes:
            if code in label_to_idx:
                y_pred[i, label_to_idx[code]] = 1
            else:
                out_of_space_codes.add(code)

        elapsed = time.time() - t_start
        avg = elapsed / (i + 1)
        remaining = avg * (len(case_ids) - i - 1)
        print(
            f"[{i+1}/{len(case_ids)}] {case_id}: {len(codes)} coduri | "
            f"timp mediu={avg:.1f}s | ramas estimat={remaining/60:.1f}min",
            flush=True,
        )

    result = {
        "labels": labels,
        "case_ids": case_ids,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "raw_responses": raw_responses,
        "out_of_space_codes": sorted(out_of_space_codes),
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nGata. Rezultate salvate in {OUT_PATH}", flush=True)
    print(f"Coduri prezise de LLM in afara spatiului de 769 (ignorate la evaluare): "
          f"{len(out_of_space_codes)}", flush=True)


if __name__ == "__main__":
    main()
