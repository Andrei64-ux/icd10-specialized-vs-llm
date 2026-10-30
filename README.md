# Specialized clinical language models vs. a general-purpose LLM for ICD-10 coding

Code, logs and per-case predictions for the study *"Specialized Clinical Language
Models versus a General-Purpose Large Language Model for Automated ICD-10 Coding,
with Adaptation to Romanian"* (U.P.B. Scientific Bulletin, Series C, 2026).

This repository does **not** redistribute the CodiEsp or MoNERo corpora. Both are
freely available from their original sources (see the paper's references [5] and
[14]); `predictions/case_id_splits.json` lists the exact official case identifiers
(500 train / 250 dev) used in this study, which is the full official release of
both splits — no subsampling was done.

## Repository structure

```
scripts/      all Python scripts used for data preparation, training, evaluation,
              statistical analysis and the FastAPI microservice integration
logs/         raw console logs of the actual training/evaluation runs reported
              in the paper (per-epoch metrics, exact command output)
predictions/  per-case predictions of every model, in a common JSON format
              (case_ids, labels, y_true, y_pred), plus the LLM's raw text
              responses, the perturbed evaluation texts, and the case ID lists
requirements.txt   exact package versions used (Python 3.12)
```

### `scripts/` — what each file does

| Script | Purpose |
|---|---|
| `build_dataset.py` | Loads CodiEsp splits, builds the label space (`MIN_CODE_FREQ=2`) |
| `train_icd10.py` | Trains the multi-label ICD-10 classifier (discriminative LR, capped pos_weight) |
| `translate_to_romanian.py` | Translates CodiEsp to Romanian (OPUS-MT, es-ro) |
| `mlm_pretrain_ro.py` | Domain-adaptive MLM pretraining on the translated corpus |
| `llm_baseline.py` | Zero-shot Mistral 7B / Ollama baseline, prompt + code parsing |
| `stats_compare.py` | Bootstrap CI + McNemar's test, matched-pair comparison harness |
| `compare_vs_llm.py` | Specialized model vs. LLM, full statistical comparison |
| `stats_compare_mlm.py` | ES vs. RO(MLM) and RO(no-MLM) vs. RO(MLM) comparisons |
| `robustness_test.py` | Typo / truncation / sentence-drop perturbations, both models |
| `latency_test.py` | Single-request latency measurement, both models |
| `ner_dataset.py`, `train_ner.py` | CodiEsp NER (Spanish), BIO tagging via offset mapping |
| `project_ner_to_romanian.py` | Annotation projection (isolated entity translation + search) |
| `monero_dataset.py`, `train_monero_ner.py` | Native Romanian NER on MoNERo |
| `eval_ner_entities.py`, `eval_monero_entities.py` | Entity-level (seqeval) evaluation |
| `export_predictions.py`, `export_perturbed_texts.py` | Regenerate the files in `predictions/` |

### `predictions/` — format

Each `*_predictions.json` has the same shape:
```json
{"labels": [...], "case_ids": [...], "y_true": [[0,1,...], ...], "y_pred": [[0,1,...], ...]}
```
`llm_mistral7b_predictions.json` additionally has `raw_responses` (the raw text
returned by Mistral 7B for each case, before regex code extraction). The exact
zero-shot prompt template is in `scripts/llm_baseline.py` (`PROMPT_TEMPLATE`).

`perturbed_texts_dev.json` contains, for each of the 250 dev cases, the four
text variants scored in Table 3 (clean, typos, truncated, incomplete) —
regenerated deterministically from the fixed seeds in `robustness_test.py`;
re-running `export_perturbed_texts.py` reproduces this file byte-for-byte.

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/train_icd10.py                     # -> icd10_model/
python scripts/translate_to_romanian.py            # -> translated corpus
python scripts/mlm_pretrain_ro.py                  # -> ro_clinical_mlm_base/
OUTPUT_DIR=icd10_model_ro CORPUS_DIR=<translated> python scripts/train_icd10.py
python scripts/llm_baseline.py                     # requires a local Ollama + mistral
python scripts/stats_compare.py
python scripts/stats_compare_mlm.py
python scripts/robustness_test.py
python scripts/latency_test.py
```

Trained model weights (~0.5 GB each) are not included in this repository; they
are fully reproducible from the scripts above and the documented hyperparameters
(Section 2.2 of the paper). No random seed was fixed for model training (see the
corrected statement in the paper); exact numerical reproduction of a single run
is therefore not expected, but the reported effect sizes and statistical tests
were stable across the reported runs.

## Known limitation reflected in this release

The batch size (4) and threshold grid used by `train_icd10.py`
(`CANDIDATE_THRESHOLDS = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05]`) are
hardcoded constants in the script, matching exactly what is reported in the
paper — no values were adjusted after the fact for this release.

## License

Code in this repository is released under the MIT License (see `LICENSE`).
This does not cover the CodiEsp or MoNERo corpora, which retain their own
original licenses/terms of use.

## Citation

If you use this code, please cite the paper (full reference to be added once
published; see U.P.B. Scientific Bulletin, Series C, 2026).
