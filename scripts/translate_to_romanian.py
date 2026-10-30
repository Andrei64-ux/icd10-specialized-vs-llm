import re
import shutil
from pathlib import Path

from transformers import MarianMTModel, MarianTokenizer

SRC_BASE = Path(__file__).resolve().parents[1] / "train_dev" / "train_dev"
DST_BASE = Path(__file__).resolve().parents[1] / "train_dev_ro"
MODEL_NAME = "Helsinki-NLP/opus-mt-es-ro"
CHUNK_MAX_LEN = 400


def split_chunks(text, max_len=CHUNK_MAX_LEN):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) <= max_len:
            cur += (" " if cur else "") + s
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks


def translate_split(split, tokenizer, model):
    src_dir = SRC_BASE / split / "text_files"
    dst_dir = DST_BASE / split / "text_files"
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.txt"))
    for i, f in enumerate(files, 1):
        out_path = dst_dir / f.name
        if out_path.exists():
            continue
        text = f.read_text(encoding="utf-8")
        chunks = split_chunks(text)
        batch = tokenizer(chunks, return_tensors="pt", padding=True, truncation=True, max_length=CHUNK_MAX_LEN)
        translated = model.generate(**batch, max_new_tokens=CHUNK_MAX_LEN)
        out = tokenizer.batch_decode(translated, skip_special_tokens=True)
        out_path.write_text(" ".join(out), encoding="utf-8")
        if i % 25 == 0 or i == len(files):
            print(f"[{split}] {i}/{len(files)} documente traduse")

    # Codurile ICD-10 nu se traduc - copiem adnotarile neschimbate.
    for tsv in (SRC_BASE / split).glob(f"{split}_annotations_task*_processed.tsv"):
        shutil.copy(tsv, DST_BASE / split / tsv.name)


def main():
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)

    for split in ("train", "dev"):
        translate_split(split, tokenizer, model)

    print(f"Gata. Corpus tradus in {DST_BASE}")


if __name__ == "__main__":
    main()
