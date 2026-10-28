# Evaluation sets

Fixed copies of every set the model is judged on, so the numbers in the main README can be reproduced and CI can check them. **Never append any of these to `train.jsonl`.**

| File | Rows | Source | Status |
|---|---|---|---|
| `holdout_v3.jsonl` | 60 attacks | `generate_hard_negatives_v3.py` (seed 43), holdout vocabulary only | **Clean test set.** Never used for training or model selection. The 58/60 headline comes from this file. |
| `holdout_v2.jsonl` | 60 benign, 20 attacks | `generate_hard_negatives.py` (seed 42), holdout templates only | Never trained on. Not reported in earlier results. |
| `benign_fpr.jsonl` | 100 benign | hand-written on 28 Oct 2026 for this repo, in six categories (security education, developer "ignore/override" wording, self-corrections, quoted content, non-English, ordinary) | **Clean false-positive test set.** Written after training; never used for model selection. |
| `ood_validation.jsonl` | 27 attacks, 26 benign | the hand-written lists in `validate_shield.py` | **Validation set.** It steered three rounds of development, so its score is optimistic. |

`tests/test_eval_data.py` checks that both holdouts match their generators byte for byte and that each holdout's vocabulary is disjoint from the training vocabulary, so these files cannot drift from the code that defines them. It also checks that the benign set is unique, contains no prompt from the other sets, and none from the generated training data. (The public-dataset part of `train.jsonl` is not in the repo, so overlap with it is checked by the leakage step in `validate_shield.py`, not in CI.)

Limits worth knowing: both holdouts are template-generated, so they test vocabulary generalisation within a few sentence shapes, not the full space of real attacks. The v3 holdout has no benign prompts, so it measures recall only; `benign_fpr.jsonl` covers the other side. Once its results start steering changes to the model, it becomes a validation set too, and a fresh one is needed.
