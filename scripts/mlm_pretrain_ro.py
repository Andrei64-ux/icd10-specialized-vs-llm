import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader

BASE_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"
CORPUS_DIR = Path(__file__).resolve().parents[1] / "train_dev_ro" / "train" / "text_files"
OUTPUT_DIR = os.environ.get("MLM_OUTPUT_DIR", str(Path(__file__).resolve().parent / "ro_clinical_mlm_base"))
MAX_LENGTH = 512
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MlmTextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=MAX_LENGTH, padding="max_length"
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


def main():
    texts = [f.read_text(encoding="utf-8") for f in sorted(CORPUS_DIR.glob("*.txt"))]
    print(f"Corpus de domain adaptation: {len(texts)} documente (doar train, dev ramane netins)")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL).to(DEVICE)

    dataset = MlmTextDataset(texts, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
        avg_loss = total_loss / len(loader)
        print(f"epoca {epoch}: MLM loss={avg_loss:.4f}")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model adaptat de domeniu salvat in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
