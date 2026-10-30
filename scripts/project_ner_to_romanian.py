"""
Proiecteaza adnotarile NER (CodiEsp task3, offset-uri pe textul spaniol original)
pe corpusul deja tradus in romana (train_dev_ro), fara adnotare manuala noua.

Metoda ("annotation projection"): pentru fiecare entitate, se traduce DOAR textul
entitatii (nu documentul intreg) cu acelasi model deja folosit pentru traducerea
corpusului, apoi se cauta acea fraza in documentul romanesc corespunzator. Daca se
gaseste, offset-ul se proiecteaza; daca nu, entitatea se pierde (raportat explicit,
nu ascuns) - traducerea documentului intreg si traducerea izolata a unei fraze nu
produc mereu acelasi text, din cauza contextului.
"""
import csv
import os
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

# Cai implicite: functioneaza local (relative la acest fisier) SAU pe Colab, daca se
# seteaza variabilele de mediu NER_SRC_BASE / NER_RO_BASE (vezi README_colab.md).
SRC_BASE = Path(os.environ.get(
    "NER_SRC_BASE", str(Path(__file__).resolve().parents[1] / "train_dev" / "train_dev")
))
RO_BASE = Path(os.environ.get(
    "NER_RO_BASE", str(Path(__file__).resolve().parents[1] / "train_dev_ro")
))
MODEL_NAME = "Helsinki-NLP/opus-mt-es-ro"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_task3(split):
    ann_path = SRC_BASE / split / f"{split}_annotations_task3_processed.tsv"
    entities_by_case = {}
    with open(ann_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            case_id, entity_type, code, _term, offsets = row[0], row[1], row[2], row[3], row[4]
            subspans = [tuple(int(x) for x in part.split()) for part in offsets.split(";")]
            entities_by_case.setdefault(case_id, []).append((subspans, entity_type, code))
    return entities_by_case


def translate_batch(texts, tokenizer, model, batch_size=64):
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        batch = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
        translated = model.generate(**batch, max_new_tokens=64)
        results.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))
        print(f"  tradus {min(i + batch_size, len(texts))}/{len(texts)} entitati", flush=True)
    return results


def project_split(split, tokenizer, model):
    entities_by_case = load_task3(split)
    es_text_dir = SRC_BASE / split / "text_files"
    ro_text_dir = RO_BASE / split / "text_files"

    # pasul 1: aduna toate entitatile si textele lor spaniole, fara I/O repetat
    flat_entities = []
    es_span_texts = []
    for case_id, entities in entities_by_case.items():
        es_text = (es_text_dir / f"{case_id}.txt").read_text(encoding="utf-8")
        for subspans, entity_type, code in entities:
            es_span_text = " ".join(es_text[s:e] for s, e in subspans)
            flat_entities.append((case_id, entity_type, code))
            es_span_texts.append(es_span_text)

    # pasul 2: traducere in batch-uri (nu cate una)
    print(f"[{split}] traduc {len(es_span_texts)} entitati in batch-uri...")
    ro_span_texts = translate_batch(es_span_texts, tokenizer, model)

    # pasul 3: cautare in documentul romanesc corespunzator
    ro_text_cache = {}
    stats = {"total": 0, "projected": 0}
    out_rows = []
    for (case_id, entity_type, code), ro_span_text in zip(flat_entities, ro_span_texts):
        stats["total"] += 1
        if case_id not in ro_text_cache:
            ro_text_cache[case_id] = (ro_text_dir / f"{case_id}.txt").read_text(encoding="utf-8")
        ro_text = ro_text_cache[case_id]

        pos_real = ro_text.lower().find(ro_span_text.lower())
        if pos_real == -1:
            continue

        end_real = pos_real + len(ro_span_text)
        out_rows.append([case_id, entity_type, code, ro_span_text, f"{pos_real} {end_real}"])
        stats["projected"] += 1

    return out_rows, stats


def main():
    print(f"Device: {DEVICE}", flush=True)
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

    for split in ("train", "dev"):
        rows, stats = project_split(split, tokenizer, model)
        pct = 100 * stats["projected"] / stats["total"] if stats["total"] else 0
        print(f"[{split}] proiectate {stats['projected']}/{stats['total']} entitati ({pct:.1f}%)")

        out_path = RO_BASE / split / f"{split}_annotations_task3_processed.tsv"
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows(rows)
        print(f"[{split}] salvat in {out_path}")


if __name__ == "__main__":
    main()
