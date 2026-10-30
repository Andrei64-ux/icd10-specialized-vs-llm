"""
Regenereaza si salveaza pe disc textele perturbate (typo-uri, trunchiere,
propozitii incomplete) pentru toate cele 250 de cazuri dev, folosind EXACT
aceleasi functii si seed-uri din robustness_test.py - necesar pentru sectiunea
"Data and code availability" (textele perturbate, nu doar codul care le
genereaza). Deterministe: rularea din nou produce byte-identic acelasi output.
"""
import json
from pathlib import Path

from build_dataset import MIN_CODE_FREQ, build_label_space, load_split
from robustness_test import PERTURBATIONS

OUT_PATH = Path(__file__).resolve().parent / "predictions" / "perturbed_texts_dev.json"


def main():
    _, train_codes = load_split("train")
    build_label_space(train_codes, MIN_CODE_FREQ)
    texts_by_case, _ = load_split("dev")
    case_ids = sorted(texts_by_case)

    out = {}
    for case_id in case_ids:
        original = texts_by_case[case_id]
        out[case_id] = {name: fn(original) for name, fn in PERTURBATIONS.items()}

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Salvat: {OUT_PATH} ({len(case_ids)} cazuri x {len(PERTURBATIONS)} variante)")


if __name__ == "__main__":
    main()
