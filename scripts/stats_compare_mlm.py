"""
Completeaza comparatiile statistice lipsa, cerute de reviewer-ul articolului:
ES vs RO(cu adaptare MLM) si RO(fara adaptare) vs RO(cu adaptare), folosind
exact aceeasi metodologie (bootstrap CI pe F1_micro, resesantionare pe caz)
deja validata in stats_compare.py.
"""
from pathlib import Path

from sklearn.metrics import f1_score

from stats_compare import bootstrap_ci_f1_diff, get_predictions, mcnemar_micro

BASE = Path(__file__).resolve().parents[1]


def report(name_a, dir_a, corpus_a, name_b, dir_b, corpus_b):
    print(f"\n=== {name_a} vs {name_b} ===")
    _, y_true_a, y_pred_a = get_predictions(dir_a, corpus_a)
    _, y_true_b, y_pred_b = get_predictions(dir_b, corpus_b)

    f1_a = f1_score(y_true_a, y_pred_a, average="micro", zero_division=0)
    f1_b = f1_score(y_true_b, y_pred_b, average="micro", zero_division=0)
    print(f"F1_micro {name_a}={f1_a:.4f}  {name_b}={f1_b:.4f}  diferenta={f1_a - f1_b:.4f}")

    mc = mcnemar_micro(y_true_a, y_pred_a, y_true_b, y_pred_b)
    print(f"McNemar: b={mc['b']}, c={mc['c']}, statistica={mc['statistic']:.2f}, p={mc['p_value']:.6f}")

    boot = bootstrap_ci_f1_diff(y_true_a, y_pred_a, y_true_b, y_pred_b)
    print(f"Bootstrap CI 95% diferenta F1_micro({name_a}-{name_b}): "
          f"medie={boot['mean_diff']:.4f}, CI=[{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")
    return f1_a, f1_b, boot


def main():
    es_dir = "icd10_model"
    es_corpus = BASE / "train_dev" / "train_dev"
    ro_dir = "icd10_model_ro"
    ro_mlm_dir = "icd10_model_ro_mlm"
    ro_corpus = BASE / "train_dev_ro"

    report("ES", es_dir, es_corpus, "RO_MLM", ro_mlm_dir, ro_corpus)
    report("RO_noMLM", ro_dir, ro_corpus, "RO_MLM", ro_mlm_dir, ro_corpus)


if __name__ == "__main__":
    main()
