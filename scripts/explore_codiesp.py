import csv
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "train_dev" / "train_dev"


def load_split(split):
    text_dir = BASE / split / "text_files"
    ann_path = BASE / split / f"{split}_annotations_task1_processed.tsv"

    texts = {}
    for txt_file in text_dir.glob("*.txt"):
        texts[txt_file.stem] = txt_file.read_text(encoding="utf-8")

    codes_by_case = {}
    with open(ann_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            case_id, _, code, _term = row[0], row[1], row[2], row[3]
            code = code.split(" ")[0].strip().lower()
            codes_by_case.setdefault(case_id, set()).add(code)

    return texts, {case_id: sorted(codes) for case_id, codes in codes_by_case.items()}


def summarize(split):
    texts, codes_by_case = load_split(split)

    print(f"\n=== {split} ===")
    print(f"Cazuri cu text: {len(texts)}")
    print(f"Cazuri cu coduri adnotate: {len(codes_by_case)}")

    missing = set(texts) - set(codes_by_case)
    if missing:
        print(f"Cazuri fara niciun cod adnotat: {len(missing)} (ex: {list(missing)[:3]})")

    codes_per_case = [len(v) for v in codes_by_case.values()]
    print(f"Coduri per caz: min={min(codes_per_case)}, max={max(codes_per_case)}, "
          f"medie={sum(codes_per_case) / len(codes_per_case):.1f}")

    all_codes = Counter(c for codes in codes_by_case.values() for c in codes)
    print(f"Coduri unice: {len(all_codes)}")
    print("Top 10 cele mai frecvente coduri:")
    for code, count in all_codes.most_common(10):
        print(f"  {code}: {count}")

    lengths = [len(t) for t in texts.values()]
    print(f"Lungime text (caractere): min={min(lengths)}, max={max(lengths)}, "
          f"medie={sum(lengths) / len(lengths):.0f}")

    rare_codes = sum(1 for _, count in all_codes.items() if count == 1)
    print(f"Coduri care apar o singura data (risc pentru split train/test): {rare_codes}")


if __name__ == "__main__":
    summarize("train")
    summarize("dev")
