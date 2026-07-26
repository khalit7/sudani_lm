# sudani_lm

A language model for **Sudanese Arabic**, trained from scratch: tokenizer, pretraining and dialect finetuning, with no pretrained checkpoint anywhere in the pipeline.

Standard Arabic models handle Sudanese dialect poorly: it is under-represented in every public corpus, and most of it exists as informal chat rather than written text. This repo is an end-to-end attempt at the problem, built to understand each stage rather than to call a library.

## Approach

1. **Tokenizer** — trained on the Arabic corpus rather than reused, so dialect spelling variation is not shredded into single characters.
2. **Pretraining** — a decoder-only transformer (4 layers, 512-dim, 8 heads, 1,024-token context) on Arabic text. Adam, warmup-cosine schedule with 5% warmup, effective batch size 128. The aim is to train different models/architectures and experiment with them.
3. **Dialect finetuning** — Sudanese chat data. *Note: the training data is my own private WhatsApp history. It is not in this repository and never will be; only the code and configs are public.*

## What's in here

```text
configs/     pretraining.yaml, arabic_ift.yaml — every run is a config, not a code edit
src/         models/, dataset/, trainer.py, evaluator.py, factory.py
tokenizers/  trained tokenizers
train.py     entry point for pretraining and finetuning
inference.py generation from a checkpoint
```

Adding a model or a dataset means adding a class and pointing a config at it; the training loop does not change.

## Evaluation during training

Validation loss every 500 steps, MMLU on the same cadence, and generation samples every 1,000 steps across a temperature sweep (0 → 2) so degeneration and incoherence are visible while the run is still going. Gradient norms are logged every step.

## Results so far

TODO: Add this

## Running it

```bash
uv sync
uv run train.py --config configs/pretraining.yaml
uv run inference.py --checkpoint <path>
```

## Status

Work in progress ...
