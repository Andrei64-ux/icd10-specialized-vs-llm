import json
from pathlib import Path

import simple_icd_10_cm as icd10cm
from transformers import MarianMTModel, MarianTokenizer

LABELS_PATH = Path(__file__).resolve().parent / "icd10_model" / "labels.json"
OUT_PATH = Path(__file__).resolve().parent / "icd10_ro_descriptions.json"
MODEL_NAME = "Helsinki-NLP/opus-mt-en-ro"


def main():
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

    descriptions = {}
    for code in labels:
        try:
            descriptions[code] = icd10cm.get_description(code.upper())
        except Exception:
            descriptions[code] = None

    codes_with_desc = [c for c, d in descriptions.items() if d]
    texts = [descriptions[c] for c in codes_with_desc]
    print(f"{len(codes_with_desc)}/{len(labels)} coduri au descriere engleza gasita")

    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)

    result = {}
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_codes = codes_with_desc[i:i + batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        translated = model.generate(**enc, max_new_tokens=128)
        out = tokenizer.batch_decode(translated, skip_special_tokens=True)
        for code, ro_text in zip(batch_codes, out):
            result[code] = ro_text
        if (i // batch_size) % 5 == 0:
            print(f"{min(i + batch_size, len(texts))}/{len(texts)} traduse")

    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Gata: {len(result)} descrieri salvate in {OUT_PATH}")


if __name__ == "__main__":
    main()
