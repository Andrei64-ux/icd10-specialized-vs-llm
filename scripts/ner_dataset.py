import csv
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

# Implicit: corpusul original CodiEsp (spaniola). Pentru varianta romana
# (proiectata), se suprascrie prin variabila de mediu CORPUS_DIR.
BASE = Path(os.environ.get(
    "CORPUS_DIR",
    str(Path(__file__).resolve().parents[1] / "train_dev" / "train_dev"),
))

LABEL_LIST = ["O", "B-DIAGNOSTICO", "I-DIAGNOSTICO", "B-PROCEDIMIENTO", "I-PROCEDIMIENTO"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}


def load_ner_split(split):
    text_dir = BASE / split / "text_files"
    ann_path = BASE / split / f"{split}_annotations_task3_processed.tsv"

    texts = {f.stem: f.read_text(encoding="utf-8") for f in text_dir.glob("*.txt")}

    spans_by_case = {}
    with open(ann_path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            case_id, entity_type, _code, _term, offsets = row[0], row[1], row[2], row[3], row[4]
            # o entitate poate avea mai multe sub-spanuri discontinue, separate prin ';'
            # (ex. "2787 2801;2810 2823" pentru un termen cu cuvinte intercalate)
            subspans = [tuple(int(x) for x in part.split()) for part in offsets.split(";")]
            spans_by_case.setdefault(case_id, []).append((subspans, entity_type))

    return texts, spans_by_case


def spans_to_bio(offset_mapping, entities):
    """offset_mapping: lista de (char_start, char_end) per token (de la tokenizer fast).
    entities: lista de (subspans, entity_type), subspans = lista de (start, end) in caractere.
    Returneaza lista de id-uri de eticheta BIO, una per token."""
    labels = [LABEL_TO_ID["O"]] * len(offset_mapping)

    for subspans, entity_type in entities:
        first_token = True
        for start, end in subspans:
            for i, (tok_start, tok_end) in enumerate(offset_mapping):
                if tok_start == tok_end:  # token special (CLS/SEP/PAD), offset (0,0)
                    continue
                if tok_start >= start and tok_end <= end:
                    tag = f"{'B' if first_token else 'I'}-{entity_type}"
                    labels[i] = LABEL_TO_ID[tag]
                    first_token = False

    return labels


class NerDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=512):
        self.texts, self.spans_by_case = load_ner_split(split)
        self.case_ids = sorted(self.texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        text = self.texts[case_id]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
        )
        labels = spans_to_bio(encoding["offset_mapping"], self.spans_by_case.get(case_id, []))
        # tokenii speciali (offset (0,0)) primesc -100, ignorati de CrossEntropyLoss
        labels = [
            -100 if (s == 0 and e == 0) else lab
            for (s, e), lab in zip(encoding["offset_mapping"], labels)
        ]

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(labels),
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("PlanTL-GOB-ES/bsc-bio-ehr-es")
    ds = NerDataset("train", tokenizer)
    print(f"train: {len(ds)} cazuri")
    sample = ds[0]
    nontrivial = sum(1 for l in sample["labels"].tolist() if l > 0)
    print(f"exemplu: {nontrivial} tokeni cu eticheta non-O (din {sample['labels'].shape[0]} tokeni)")
