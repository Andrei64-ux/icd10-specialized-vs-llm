import os

import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from monero_dataset import LABEL_LIST, MoneroDataset, load_sentences, train_dev_split

MODEL_DIR = os.environ.get("MODEL_DIR", "monero_ner_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()

    sentences = load_sentences()
    _, dev_sents = train_dev_split(sentences)
    dev_ds = MoneroDataset(dev_sents, tokenizer)
    loader = DataLoader(dev_ds, batch_size=4)

    all_pred_tags, all_true_tags = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            labels = batch["labels"].tolist()

            for pred_seq, label_seq in zip(preds, labels):
                pred_tags, true_tags = [], []
                for p, l in zip(pred_seq, label_seq):
                    if l == -100:
                        continue
                    pred_tags.append(LABEL_LIST[p])
                    true_tags.append(LABEL_LIST[l])
                all_pred_tags.append(pred_tags)
                all_true_tags.append(true_tags)

    print("=== Evaluare entity-level (span exact, seqeval) — MoNERo (română nativă) ===")
    print(f"F1 (micro)      = {f1_score(all_true_tags, all_pred_tags):.3f}")
    print(f"Precizie (micro) = {precision_score(all_true_tags, all_pred_tags):.3f}")
    print(f"Recall (micro)   = {recall_score(all_true_tags, all_pred_tags):.3f}")
    print()
    print(classification_report(all_true_tags, all_pred_tags, digits=3))


if __name__ == "__main__":
    main()
