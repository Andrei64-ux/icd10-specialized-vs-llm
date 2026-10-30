import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

CORPUS_PATH = Path(__file__).resolve().parents[1] / "monero" / "MoNERo" / "MoNERo.txt"

LABEL_LIST = ["O", "B-ANAT", "I-ANAT", "B-CHEM", "I-CHEM", "B-DISO", "I-DISO", "B-PROC", "I-PROC"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}


def load_sentences():
    """CoNLL: token \t lema \t eticheta-BIO \t POS, propozitii separate prin linie goala."""
    sentences = []
    current = []
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if current:
                sentences.append(current)
                current = []
            continue
        parts = line.split("\t")
        token, tag = parts[0], parts[2]
        # corpusul are cateva (4 din ~155k) linii cu eticheta goala - tratate ca 'O'.
        tag = tag.upper() if tag else "O"
        current.append((token, tag))
    if current:
        sentences.append(current)
    return sentences


def train_dev_split(sentences, dev_ratio=0.2, seed=42):
    rng = random.Random(seed)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    n_dev = int(len(shuffled) * dev_ratio)
    return shuffled[n_dev:], shuffled[:n_dev]


class MoneroDataset(Dataset):
    def __init__(self, sentences, tokenizer, max_length=256):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = [t for t, _ in self.sentences[idx]]
        tags = [tag for _, tag in self.sentences[idx]]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        word_ids = encoding.word_ids()
        labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != prev_word_id:
                labels.append(LABEL_TO_ID[tags[word_id]])
            else:
                labels.append(-100)  # doar primul sub-token al unui cuvant conteaza
            prev_word_id = word_id

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(labels),
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    sentences = load_sentences()
    train_sents, dev_sents = train_dev_split(sentences)
    print(f"Total propozitii: {len(sentences)} | train: {len(train_sents)} | dev: {len(dev_sents)}")

    n_entities = sum(1 for s in sentences for _, tag in s if tag.startswith("B-"))
    print(f"Total entitati (B-*): {n_entities}")

    tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-cased-v1")
    ds = MoneroDataset(train_sents, tokenizer)
    sample = ds[0]
    nontrivial = sum(1 for l in sample["labels"].tolist() if l > 0)
    print(f"exemplu: {nontrivial} tokeni cu eticheta non-O (din {sample['labels'].shape[0]} tokeni)")
