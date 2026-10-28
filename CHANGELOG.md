# Changelog

## Unreleased (October 2026)

- **Fixed: long-input bypass.** `/scan` accepted 20,000 characters but only scanned the first 512 tokens, so harmless padding followed by an injection was allowed. Prompts are now scanned in overlapping 512-token windows and blocked if any window fires (#1).
- Tier-1 CI: tests for every row of the API security table, fixed evaluation sets in `eval/` with data-integrity tests, ruff, Dependabot, upper-bounded requirements (#2).
- New 100-prompt benign false-positive test set (#3).
- Tier-2 CI: real-model evaluation and regression gate, with a ProtectAI DeBERTa baseline.
- Tidy-up: `test_shield.py` renamed to `demo.py`; `train_shield.py` uses fp16 only on CUDA; README run steps; this changelog and `SECURITY.md`.
- MIT licence for the code (23 Oct 2026).

## v3 (published 30 July 2026)

- 300 combinatorial contrastive attacks (17 verbs × 30 system references × 15 qualifiers × 14 actions) restored recall after v2 without bringing back the false positive.
- Clean 60-attack holdout with disjoint vocabulary: 58/60.

## v2 (before 30 July 2026)

- 500 benign self-corrections plus 96 contrastive attacks fixed the high-confidence false positive on "Ignore my earlier question about Python…", but recall dropped by one attack on the validation set.

## v1 (before 30 July 2026)

- Fine-tuned `Llama-Prompt-Guard-2-86M` on 12,858 samples from three public datasets; FastAPI service with server-side threshold, API key, fail-closed errors and hashed logs.
