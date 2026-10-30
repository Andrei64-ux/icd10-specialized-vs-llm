import csv
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Implicit: corpusul original CodiEsp (spaniola). Pentru varianta tradusa in romana,
# se suprascriu prin variabile de mediu (vezi translate_to_romanian.py / train_icd10.py).
BASE = Path(os.environ.get(
    "CORPUS_DIR",
    str(Path(__file__).resolve().parents[1] / "train_dev" / "train_dev"),
))
MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "PlanTL-GOB-ES/bsc-bio-ehr-es")
MIN_CODE_FREQ = 2


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
            case_id, code = row[0], row[2].split(" ")[0].strip().lower()
            codes_by_case.setdefault(case_id, set()).add(code)

    return texts, codes_by_case


def build_label_space(train_codes, min_freq):
    freq = {}
    for codes in train_codes.values():
        for c in codes:
            freq[c] = freq.get(c, 0) + 1
    labels = sorted(c for c, n in freq.items() if n >= min_freq)
    return {label: i for i, label in enumerate(labels)}


class CodiEspDataset(Dataset):
    def __init__(self, split, label_to_idx, tokenizer, max_length=512):
        self.texts, self.codes_by_case = load_split(split)
        self.case_ids = sorted(self.texts)
        self.label_to_idx = label_to_idx
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        encoding = self.tokenizer(
            self.texts[case_id],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        labels = torch.zeros(len(self.label_to_idx))
        for code in self.codes_by_case.get(case_id, []):
            if code in self.label_to_idx:
                labels[self.label_to_idx[code]] = 1.0
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _, train_codes = load_split("train")
    label_to_idx = build_label_space(train_codes, MIN_CODE_FREQ)
    print(f"Spatiu de etichete: {len(label_to_idx)} coduri (prag>={MIN_CODE_FREQ})")

    train_ds = CodiEspDataset("train", label_to_idx, tokenizer)
    dev_ds = CodiEspDataset("dev", label_to_idx, tokenizer)
    print(f"train: {len(train_ds)} cazuri | dev: {len(dev_ds)} cazuri")

    sample = train_ds[0]
    print(
        f"input_ids shape: {tuple(sample['input_ids'].shape)}, "
        f"labels shape: {tuple(sample['labels'].shape)}, "
        f"coduri active in exemplu: {int(sample['labels'].sum())}"
    )
