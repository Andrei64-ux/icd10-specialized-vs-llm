"""
Exporta predictiile per-caz ale celor trei modele ICD-10 specializate (ES,
RO fara adaptare, RO cu adaptare MLM) in acelasi format JSON folosit deja
pentru LLM (llm_baseline_results.json) - necesar pentru sectiunea "Data and
code availability" a articolului (predictiile per-caz ale TUTUROR modelelor,
nu doar ale LLM-ului).
"""
import json
from pathlib import Path

from stats_compare import get_predictions

BASE = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "predictions"
OUT_DIR.mkdir(exist_ok=True)

MODELS = [
    ("icd10_specialized_es", "icd10_model", BASE / "train_dev" / "train_dev"),
    ("icd10_specialized_ro_no_adapt", "icd10_model_ro", BASE / "train_dev_ro"),
    ("icd10_specialized_ro_mlm", "icd10_model_ro_mlm", BASE / "train_dev_ro"),
]


def main():
    for name, model_dir, corpus_dir in MODELS:
        print(f"Predictii pentru {name}...")
        case_ids, y_true, y_pred = get_predictions(model_dir, corpus_dir)
        labels = json.loads((Path(model_dir) / "labels.json").read_text(encoding="utf-8"))

        out = {
            "model": name,
            "model_dir": model_dir,
            "corpus_dir": str(corpus_dir),
            "labels": labels,
            "case_ids": case_ids,
            "y_true": y_true.astype(int).tolist(),
            "y_pred": y_pred.astype(int).tolist(),
        }
        out_path = OUT_DIR / f"{name}_predictions.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f)
        print(f"  salvat: {out_path} ({len(case_ids)} cazuri, {len(labels)} etichete)")


if __name__ == "__main__":
    main()
