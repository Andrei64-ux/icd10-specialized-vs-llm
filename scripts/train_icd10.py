import json
import os

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from build_dataset import MIN_CODE_FREQ, MODEL_NAME, CodiEspDataset, build_label_space, load_split

BATCH_SIZE = 4
EPOCHS = 8
LEARNING_RATE = 2e-5
HEAD_LEARNING_RATE = 1e-3
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "icd10_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Praguri candidate: 0.5 e nepotrivit cand doar ~1-2% din etichete sunt pozitive per exemplu.
CANDIDATE_THRESHOLDS = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05]


def compute_pos_weight(train_ds, num_labels, cap=50.0):
    pos_counts = torch.zeros(num_labels)
    for i in range(len(train_ds)):
        pos_counts += train_ds[i]["labels"]
    pos_counts = pos_counts.clamp(min=1.0)
    neg_counts = len(train_ds) - pos_counts
    return (neg_counts / pos_counts).clamp(max=cap)


def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            all_probs.append(torch.sigmoid(logits).cpu())
            all_labels.append(batch["labels"].int())
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels).numpy()

    best = {"threshold": None, "f1_micro": -1.0}
    for t in CANDIDATE_THRESHOLDS:
        preds = (probs >= t).int().numpy()
        f1 = f1_score(labels, preds, average="micro", zero_division=0)
        if f1 > best["f1_micro"]:
            best = {
                "threshold": t,
                "f1_micro": f1,
                "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
                "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
                "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
            }
    return best


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _, train_codes = load_split("train")
    label_to_idx = build_label_space(train_codes, MIN_CODE_FREQ)
    print(f"Spatiu de etichete: {len(label_to_idx)} coduri (prag>={MIN_CODE_FREQ})")

    train_ds = CodiEspDataset("train", label_to_idx, tokenizer)
    dev_ds = CodiEspDataset("dev", label_to_idx, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_to_idx),
        problem_type="multi_label_classification",
    ).to(DEVICE)

    # Capul de clasificare e nou-initializat si are nevoie de o rata de invatare mult mai mare
    # decat encoder-ul deja pre-antrenat, altfel converge prea incet in cateva epoci.
    head_param_ids = {id(p) for p in model.classifier.parameters()}
    base_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": LEARNING_RATE},
            {"params": model.classifier.parameters(), "lr": HEAD_LEARNING_RATE},
        ]
    )
    pos_weight = compute_pos_weight(train_ds, len(label_to_idx)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        best = evaluate(model, dev_loader)
        print(
            f"epoca {epoch}: loss={avg_loss:.4f} | prag_optim={best['threshold']} "
            f"F1_micro={best['f1_micro']:.3f} F1_macro={best['f1_macro']:.3f} "
            f"precizie_micro={best['precision_micro']:.3f} recall_micro={best['recall_micro']:.3f}"
        )

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    labels_sorted = sorted(label_to_idx, key=label_to_idx.get)
    with open(f"{OUTPUT_DIR}/labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_sorted, f, ensure_ascii=False, indent=2)
    print(f"Model salvat in ./{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
