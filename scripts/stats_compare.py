"""
Harness de comparatie statistica intre doua modele evaluate pe DEV, cu perechi
potrivite pe caz (matched pairs) - pregatit pentru comparatia reala cu un LLM
generalist, validat aici comparand modelele ES (spaniola) vs RO (tradus).

Metodologie:
- McNemar (micro-averaged): fiecare (caz, cod) e un "trial" independent - se
  aplica McNemar pe tabelul agregat de discordanta (A corect/B gresit vs
  A gresit/B corect). Simplificare comuna pentru multi-label, documentata
  explicit ca atare (nu sunt independente in sens strict, dar e practica
  standard cand nu e fezabil un test per-eticheta separat).
- Bootstrap CI: reesantionare pe CAZURI (nu pe perechi caz-cod), 1000 replici,
  interval de incredere 95% pentru diferenta F1_micro(A) - F1_micro(B).
"""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from build_dataset import load_split

THRESHOLD = 0.5
N_BOOTSTRAP = 1000


def get_predictions(model_dir, corpus_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    labels = json.loads((Path(model_dir) / "labels.json").read_text(encoding="utf-8"))
    label_to_idx = {c: i for i, c in enumerate(labels)}

    texts, codes_by_case = load_split_from(corpus_dir, "dev")
    case_ids = sorted(texts)

    y_true = np.zeros((len(case_ids), len(labels)))
    y_pred = np.zeros((len(case_ids), len(labels)))

    with torch.no_grad():
        for i, case_id in enumerate(case_ids):
            for code in codes_by_case.get(case_id, []):
                if code in label_to_idx:
                    y_true[i, label_to_idx[code]] = 1
            enc = tokenizer(texts[case_id], truncation=True, max_length=512, return_tensors="pt")
            probs = torch.sigmoid(model(**enc).logits).squeeze(0).numpy()
            y_pred[i] = (probs >= THRESHOLD).astype(int)

    return case_ids, y_true, y_pred


def load_split_from(corpus_dir, split):
    import csv
    base = Path(corpus_dir)
    text_dir = base / split / "text_files"
    ann_path = base / split / f"{split}_annotations_task1_processed.tsv"

    texts = {f.stem: f.read_text(encoding="utf-8") for f in text_dir.glob("*.txt")}
    codes_by_case = {}
    with open(ann_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            case_id, code = row[0], row[2].split(" ")[0].strip().lower()
            codes_by_case.setdefault(case_id, set()).add(code)
    return texts, {k: sorted(v) for k, v in codes_by_case.items()}


def mcnemar_micro(y_true_a, y_pred_a, y_true_b, y_pred_b, restrict_to_relevant=False):
    """McNemar pe perechi caz-cod, pe corectitudine bruta (predictie == adevar).

    NOTA IMPORTANTA (corectata dupa verificare empirica): pentru etichete binare,
    perechile 'concordante' (ambele modele corecte SAU ambele gresite) au automat
    predictii identice intre cele doua modele - deci filtrarea la 'doar perechi
    relevante' (unde adevarul sau vreo predictie e pozitiva) NU schimba b/c fata de
    varianta bruta; parametrul e pastrat doar pentru transparenta, nu schimba rezultatul.

    Motivul real al discrepantei fata de bootstrap CI pe F1: b si c pun in aceeasi
    categorie doua tipuri diferite de dezacord - cazuri rare unde adevarul e pozitiv
    (echivalent unui fals negativ ratat de un model) si cazuri foarte numeroase unde
    adevarul e negativ (echivalent unui fals pozitiv). Cele doua tipuri afecteaza F1
    diferit (prin recall vs. precizie), dar corectitudinea bruta le trateaza identic -
    un McNemar 'balansat' (b~=c) nu implica un impact balansat asupra F1. Pentru
    comparatii pe date extrem de dezechilibrate, bootstrap CI pe metrica de interes
    (F1) e mai informativ decat McNemar pe corectitudine bruta; McNemar ramane util
    ca test complementar, nu ca test primar."""
    if restrict_to_relevant:
        relevant = (y_true_a == 1) | (y_pred_a == 1) | (y_pred_b == 1)
        y_true_a, y_pred_a, y_pred_b = y_true_a[relevant], y_pred_a[relevant], y_pred_b[relevant]
        y_true_b = y_true_a  # etichetele adevarate sunt identice intre ES si RO (aceleasi cazuri)

    correct_a = (y_pred_a == y_true_a)
    correct_b = (y_pred_b == y_true_b)

    b = int(np.sum(correct_a & ~correct_b))  # A corect, B gresit
    c = int(np.sum(~correct_a & correct_b))  # A gresit, B corect
    n_pairs = int(correct_a.size)

    if b + c == 0:
        return {"n_pairs": n_pairs, "b": b, "c": c, "statistic": 0.0, "p_value": 1.0}

    statistic = (abs(b - c) - 1) ** 2 / (b + c)  # cu corectie de continuitate
    p_value = 1 - chi2.cdf(statistic, df=1)
    return {"n_pairs": n_pairs, "b": b, "c": c, "statistic": statistic, "p_value": p_value}


def bootstrap_ci_f1_diff(y_true_a, y_pred_a, y_true_b, y_pred_b, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.RandomState(seed)
    n = y_true_a.shape[0]
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        f1_a = f1_score(y_true_a[idx], y_pred_a[idx], average="micro", zero_division=0)
        f1_b = f1_score(y_true_b[idx], y_pred_b[idx], average="micro", zero_division=0)
        diffs.append(f1_a - f1_b)
    diffs = np.array(diffs)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lower": float(np.percentile(diffs, 2.5)),
        "ci_upper": float(np.percentile(diffs, 97.5)),
    }


def main():
    print("Se incarca modelul ES si predictiile pe dev (spaniola)...")
    _, y_true_es, y_pred_es = get_predictions(
        "icd10_model",
        Path(__file__).resolve().parents[1] / "train_dev" / "train_dev",
    )

    print("Se incarca modelul RO si predictiile pe dev (tradus)...")
    _, y_true_ro, y_pred_ro = get_predictions(
        "icd10_model_ro",
        Path(__file__).resolve().parents[1] / "train_dev_ro",
    )

    f1_es = f1_score(y_true_es, y_pred_es, average="micro", zero_division=0)
    f1_ro = f1_score(y_true_ro, y_pred_ro, average="micro", zero_division=0)
    print(f"\nF1_micro ES={f1_es:.3f}  RO={f1_ro:.3f}  diferenta={f1_es - f1_ro:.3f}")

    print("\n=== McNemar pe corectitudine bruta (perechi caz-cod) ===")
    mc = mcnemar_micro(y_true_es, y_pred_es, y_true_ro, y_pred_ro)
    print(f"n perechi total = {mc['n_pairs']}, discordante (b+c) = {mc['b'] + mc['c']}")
    print(f"b (ES corect, RO gresit) = {mc['b']}, c (ES gresit, RO corect) = {mc['c']}")
    print(f"statistica = {mc['statistic']:.2f}, p-value = {mc['p_value']:.6f}")
    print("=> diferenta SEMNIFICATIVA statistic" if mc["p_value"] < 0.05 else "=> diferenta NESEMNIFICATIVA statistic")
    print("NOTA: acest test conflă erorile de tip fals-negativ (adevar=1) cu cele de")
    print("tip fals-pozitiv (adevar=0) intr-un singur b/c - vezi bootstrap CI mai jos")
    print("pentru semnificatia reala pe metrica de interes (F1).")

    print("\n=== Bootstrap CI 95% pentru F1_micro(ES) - F1_micro(RO) ===")
    boot = bootstrap_ci_f1_diff(y_true_es, y_pred_es, y_true_ro, y_pred_ro)
    print(f"diferenta medie = {boot['mean_diff']:.3f}, CI 95% = [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")
    contains_zero = boot["ci_lower"] <= 0 <= boot["ci_upper"]
    print("=> intervalul include 0 (nesemnificativ)" if contains_zero else "=> intervalul NU include 0 (semnificativ)")


if __name__ == "__main__":
    main()
