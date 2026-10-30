import os

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ner_dataset import LABEL_LIST, NerDataset

MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "PlanTL-GOB-ES/bsc-bio-ehr-es")
BATCH_SIZE = 4
EPOCHS = 8
LEARNING_RATE = 2e-5
HEAD_LEARNING_RATE = 1e-3
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "ner_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def flatten_valid(preds, labels):
    """Elimina pozitiile -100 (tokeni speciali/padding) inainte de calculul metricilor."""
    preds_flat, labels_flat = [], []
    for p_seq, l_seq in zip(preds, labels):
        for p, l in zip(p_seq, l_seq):
            if l != -100:
                preds_flat.append(p)
                labels_flat.append(l)
    return preds_flat, labels_flat


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].tolist())

    preds_flat, labels_flat = flatten_valid(all_preds, all_labels)
    label_ids = list(range(len(LABEL_LIST)))
    return {
        "f1_micro": f1_score(labels_flat, preds_flat, labels=label_ids, average="micro", zero_division=0),
        "f1_macro": f1_score(labels_flat, preds_flat, labels=label_ids, average="macro", zero_division=0),
        "f1_entities": f1_score(
            labels_flat, preds_flat,
            labels=[LABEL_LIST.index(l) for l in LABEL_LIST if l != "O"],
            average="micro", zero_division=0,
        ),
        "precision_entities": precision_score(
            labels_flat, preds_flat,
            labels=[LABEL_LIST.index(l) for l in LABEL_LIST if l != "O"],
            average="micro", zero_division=0,
        ),
        "recall_entities": recall_score(
            labels_flat, preds_flat,
            labels=[LABEL_LIST.index(l) for l in LABEL_LIST if l != "O"],
            average="micro", zero_division=0,
        ),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = NerDataset("train", tokenizer)
    dev_ds = NerDataset("dev", tokenizer)
    print(f"train: {len(train_ds)} cazuri | dev: {len(dev_ds)} cazuri")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_LIST)
    ).to(DEVICE)

    head_param_ids = {id(p) for p in model.classifier.parameters()}
    base_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": LEARNING_RATE},
            {"params": model.classifier.parameters(), "lr": HEAD_LEARNING_RATE},
        ]
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(train_loader)
        m = evaluate(model, dev_loader)
        print(
            f"epoca {epoch}: loss={avg_loss:.4f} | F1_micro={m['f1_micro']:.3f} "
            f"F1_macro={m['f1_macro']:.3f} F1_entitati={m['f1_entities']:.3f} "
            f"precizie_entitati={m['precision_entities']:.3f} recall_entitati={m['recall_entities']:.3f}"
        )

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model NER salvat in ./{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
