"""
Comparatie finala: model specializat (fine-tuned, spaniola) vs. LLM generalist
(mistral, prompting), pe acelasi set de date (CodiEsp dev) si acelasi spatiu
de etichete (769 coduri) - testul statistic cerut explicit de O1.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from stats_compare import bootstrap_ci_f1_diff, get_predictions, mcnemar_micro

LLM_RESULTS_PATH = Path(__file__).resolve().parent / "llm_baseline_results.json"


def main():
    with open(LLM_RESULTS_PATH, encoding="utf-8") as f:
        llm_data = json.load(f)

    y_true_llm = np.array(llm_data["y_true"])
    y_pred_llm = np.array(llm_data["y_pred"])

    print("Se incarca modelul specializat (ES) si predictiile pe dev...")
    _, y_true_es, y_pred_es = get_predictions(
        "icd10_model",
        Path(__file__).resolve().parents[1] / "train_dev" / "train_dev",
    )

    assert y_true_es.shape == y_true_llm.shape, "Spatiile de etichete nu se potrivesc!"
    assert np.array_equal(y_true_es, y_true_llm), "Adevarul de referinta difera intre cele doua evaluari!"

    f1_es = f1_score(y_true_es, y_pred_es, average="micro", zero_division=0)
    f1_llm = f1_score(y_true_llm, y_pred_llm, average="micro", zero_division=0)
    f1_macro_es = f1_score(y_true_es, y_pred_es, average="macro", zero_division=0)
    f1_macro_llm = f1_score(y_true_llm, y_pred_llm, average="macro", zero_division=0)
    prec_llm = precision_score(y_true_llm, y_pred_llm, average="micro", zero_division=0)
    rec_llm = recall_score(y_true_llm, y_pred_llm, average="micro", zero_division=0)

    print("\n=== Rezultate brute ===")
    print(f"Model specializat (ES): F1_micro={f1_es:.3f}  F1_macro={f1_macro_es:.3f}")
    print(f"LLM generalist (mistral): F1_micro={f1_llm:.3f}  F1_macro={f1_macro_llm:.3f}  "
          f"precizie={prec_llm:.3f}  recall={rec_llm:.3f}")
    print(f"Diferenta F1_micro (specializat - LLM) = {f1_es - f1_llm:+.3f}")
    print(f"Coduri prezise de LLM in afara spatiului de 769 (ignorate): "
          f"{len(llm_data['out_of_space_codes'])}")

    print("\n=== McNemar pe corectitudine bruta (perechi caz-cod) ===")
    mc = mcnemar_micro(y_true_es, y_pred_es, y_true_llm, y_pred_llm)
    print(f"n perechi total = {mc['n_pairs']}, discordante (b+c) = {mc['b'] + mc['c']}")
    print(f"b (specializat corect, LLM gresit) = {mc['b']}, c (specializat gresit, LLM corect) = {mc['c']}")
    print(f"statistica = {mc['statistic']:.2f}, p-value = {mc['p_value']:.6f}")
    print("=> diferenta SEMNIFICATIVA statistic" if mc["p_value"] < 0.05 else "=> diferenta NESEMNIFICATIVA statistic")

    print("\n=== Bootstrap CI 95% pentru F1_micro(specializat) - F1_micro(LLM) ===")
    boot = bootstrap_ci_f1_diff(y_true_es, y_pred_es, y_true_llm, y_pred_llm)
    print(f"diferenta medie = {boot['mean_diff']:.3f}, CI 95% = [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]")
    contains_zero = boot["ci_lower"] <= 0 <= boot["ci_upper"]
    print("=> intervalul include 0 (nesemnificativ)" if contains_zero else "=> intervalul NU include 0 (semnificativ)")


if __name__ == "__main__":
    main()
