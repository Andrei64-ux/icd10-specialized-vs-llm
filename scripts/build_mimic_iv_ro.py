"""
Pregateste un al doilea corpus romanesc tradus, de data asta din date clinice
REALE in engleza (MIMIC-IV-Note + diagnoses_icd), ca replicare independenta a
metodologiei deja aplicate pe CodiEsp (spaniola -> romana). NU inlocuieste
nevoia unui corpus romanesc nativ (tot text tradus automat ramane) - ofera o
a doua sursa reala, independenta, pentru aceeasi metoda de adaptare lingvistica.

PRECONDITIE (de facut personal, nu poate fi delegat): acces credentializat la
MIMIC-IV si MIMIC-IV-Note pe PhysioNet (cont + curs CITI "Data or Specimens
Only Research" + DUA semnat separat pentru fiecare dataset). Vezi
https://physionet.org/content/mimic-iv-note/ si https://mimic.mit.edu/docs/faq/

Structura asteptata pe disc (dupa descarcare, necesita cont PhysioNet):
  MIMIC_IV_DIR/hosp/diagnoses_icd.csv.gz   (coloane: subject_id, hadm_id, seq_num,
                                             icd_code, icd_version)
  MIMIC_IV_NOTE_DIR/note/discharge.csv.gz  (coloane: note_id, subject_id,
                                             hadm_id, note_type, text, ...)

Aceasta aliniere (discharge summary <-> coduri ICD la nivel de internare) e
analoaga cu CodiEsp: un document clinic <-> mai multe coduri ICD-10, NU
adnotare cu offset de caracter (CodiEsp task3/NER nu are echivalent direct
in MIMIC-IV fara adnotare suplimentara).
"""
import gzip
import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

MIMIC_IV_DIR = os.environ.get("MIMIC_IV_DIR", "./mimic-iv")
MIMIC_IV_NOTE_DIR = os.environ.get("MIMIC_IV_NOTE_DIR", "./mimic-iv-note")
OUT_DIR = Path(os.environ.get("OUT_DIR", "./mimic_iv_ro"))
MIN_CODE_FREQ = 2
MAX_CASES = int(os.environ.get("MAX_CASES", "750"))  # analog CodiEsp: 500 train + 250 dev
TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-en-ro"
CHUNK_CHARS = 400  # acelasi plafon folosit la traducerea CodiEsp (sentence-bounded)


def load_icd10_diagnoses(mimic_iv_dir):
    """Pastreaza DOAR internarile codificate ICD-10-CM (icd_version==10) -
    MIMIC-IV acopera si perioada ICD-9 (pre-oct-2015), exact discrepanta
    semnalata in limitari pentru CodiEsp/MIMIC-III."""
    path = Path(mimic_iv_dir) / "hosp" / "diagnoses_icd.csv.gz"
    df = pd.read_csv(path, compression="gzip", dtype=str)
    df = df[df["icd_version"] == "10"]
    codes_by_hadm = defaultdict(list)
    for hadm_id, code in zip(df["hadm_id"], df["icd_code"]):
        codes_by_hadm[hadm_id].append(code.lower())
    return codes_by_hadm


def load_discharge_notes(mimic_iv_note_dir, hadm_ids_wanted):
    path = Path(mimic_iv_note_dir) / "note" / "discharge.csv.gz"
    notes_by_hadm = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for chunk in pd.read_csv(f, chunksize=5000, dtype=str):
            chunk = chunk[chunk["hadm_id"].isin(hadm_ids_wanted)]
            for hadm_id, text in zip(chunk["hadm_id"], chunk["text"]):
                if hadm_id not in notes_by_hadm:  # un discharge summary per internare
                    notes_by_hadm[hadm_id] = text
            if len(notes_by_hadm) >= len(hadm_ids_wanted):
                break
    return notes_by_hadm


def build_label_space(codes_by_hadm, min_freq):
    freq = defaultdict(int)
    for codes in codes_by_hadm.values():
        for c in set(codes):  # deduplicat per caz - vezi bug-ul corectat in O1 (Faza 2)
            freq[c] += 1
    return {c for c, n in freq.items() if n >= min_freq}


def chunk_text(text, max_chars=CHUNK_CHARS):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) > max_chars and current:
            chunks.append(current.strip())
            current = s
        else:
            current += " " + s
    if current.strip():
        chunks.append(current.strip())
    return chunks


def translate_corpus(notes_by_hadm):
    from transformers import MarianMTModel, MarianTokenizer

    tokenizer = MarianTokenizer.from_pretrained(TRANSLATE_MODEL)
    model = MarianMTModel.from_pretrained(TRANSLATE_MODEL)

    translated = {}
    for i, (hadm_id, text) in enumerate(notes_by_hadm.items()):
        chunks = chunk_text(text)
        encoded = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=512)
        generated = model.generate(**encoded, max_length=512)
        ro_chunks = tokenizer.batch_decode(generated, skip_special_tokens=True)
        translated[hadm_id] = " ".join(ro_chunks)
        if (i + 1) % 25 == 0:
            print(f"  tradus {i + 1}/{len(notes_by_hadm)}", flush=True)
    return translated


def main():
    print("Incarcare diagnoses_icd.csv.gz (doar icd_version=10)...")
    codes_by_hadm = load_icd10_diagnoses(MIMIC_IV_DIR)
    print(f"Internari cu cel putin un cod ICD-10: {len(codes_by_hadm)}")

    label_space = build_label_space(codes_by_hadm, MIN_CODE_FREQ)
    print(f"Spatiu de etichete (prag>={MIN_CODE_FREQ}): {len(label_space)} coduri")

    hadm_ids = list(codes_by_hadm)[:MAX_CASES]
    print(f"Incarcare discharge summaries pentru {len(hadm_ids)} internari...")
    notes_by_hadm = load_discharge_notes(MIMIC_IV_NOTE_DIR, set(hadm_ids))
    print(f"Note gasite: {len(notes_by_hadm)}")

    print("Traducere en->ro (MarianMT, acelasi model folosit pentru descrierile ICD-10)...")
    translated = translate_corpus(notes_by_hadm)

    OUT_DIR.mkdir(exist_ok=True)
    n_train = int(len(translated) * 2 / 3)
    hadm_ids_translated = list(translated)
    splits = {"train": hadm_ids_translated[:n_train], "dev": hadm_ids_translated[n_train:]}

    for split, ids in splits.items():
        split_dir = OUT_DIR / split / "text_files"
        split_dir.mkdir(parents=True, exist_ok=True)
        ann_path = OUT_DIR / split / f"{split}_annotations_task1_processed.tsv"
        with open(ann_path, "w", encoding="utf-8") as f:
            for hadm_id in ids:
                (split_dir / f"{hadm_id}.txt").write_text(translated[hadm_id], encoding="utf-8")
                for code in set(codes_by_hadm[hadm_id]) & label_space:
                    f.write(f"{hadm_id}\tDIAGNOSTICO\t{code}\t-\n")
        print(f"[{split}] {len(ids)} cazuri salvate in {split_dir.parent}")

    print(f"\nGata: corpus MIMIC-IV-RO in {OUT_DIR} (format compatibil cu build_dataset.py, "
          f"CORPUS_DIR={OUT_DIR})")


if __name__ == "__main__":
    main()
