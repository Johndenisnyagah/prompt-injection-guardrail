---
license: llama4
base_model: meta-llama/Llama-Prompt-Guard-2-86M
pipeline_tag: text-classification
language:
  - en
tags:
  - prompt-injection
  - llm-security
  - guardrails
  - ai-safety
  - security
  - llama
---

# Llama-Prompt-Guard-2-86M-injection-guardrail

**Built with Llama.**

A binary classifier that flags prompt-injection attacks before they reach an LLM. It is a fine-tune of Meta's [`Llama-Prompt-Guard-2-86M`](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) on 12,858 samples from three public datasets plus targeted synthetic data. Label `0` = benign, `1` = injection.

Code, evaluation sets, tests and the FastAPI service: [github.com/Johndenisnyagah/prompt-injection-guardrail](https://github.com/Johndenisnyagah/prompt-injection-guardrail).

## Results

Measured in CI on sets that never influenced training or model selection ([full results](https://github.com/Johndenisnyagah/prompt-injection-guardrail/blob/main/docs/evaluation-results.md)):

| Set | Fine-tuned | Stock Prompt Guard 2 (base) | ProtectAI DeBERTa v2 |
|---|---|---|---|
| Clean holdout, 60 attacks (disjoint vocabulary) | **58/60** | 9/60 | 60/60 |
| Clean benign set, 100 prompts (false positives) | 8/100 | 1/100 | 17/100 |
| v2 holdout, 20 attacks / 60 benign | 20/20 · 7/60 false positives | 14/20 · 11/60 | 20/20 · 38/60 |
| Attack after 702 harmless tokens, windowed scanning | **53/60** | 2/60 | 30/60 |

Fine-tuning takes the clean holdout from 9/60 to 58/60, at the cost of more false positives on the clean benign set (1/100 → 8/100).

A 53-prompt hand-written set (53/53) and the in-distribution test split (recall 0.980, n=2,112) were used during development, so they are not generalisation evidence.

Latency: about 20–30 ms per prompt on an RTX 4080 laptop GPU, about 150 ms on a CPU-only CI runner.

## Use

The model sees 512 tokens. For longer inputs, split into overlapping 512-token windows and block if any window fires, as Meta recommends for Prompt Guard 2. Truncating instead lets an attacker hide an injection after harmless padding. The GitHub repository's `app.py` implements windowing.

## Limitations

- **False positives on users instructing the assistant** about its own output ("Answer only with code from now on"): 8/100 on the clean benign set.
- **Weaker on attacks phrased as questions or statements** rather than commands. Both holdout misses are of this kind.
- **Long inputs dilute short attacks:** 53/60 when an attack shares its window with harmless text.
- **Poorly calibrated:** output is close to a binary switch, so there is no useful human-review band.
- **English-dominant:** non-English coverage was not systematically evaluated.
- **One layer, not a defence on its own.** Pair it with least-privilege tool access, context isolation and output validation (OWASP LLM01).

## Licence

This model is a derivative of Llama Prompt Guard 2 and is distributed under the **Llama 4 Community License Agreement** ([`LICENSE`](LICENSE) in this repository; also at [developer.meta.com/ai/llama4/license](https://developer.meta.com/ai/llama4/license/)). See [`NOTICE`](NOTICE). Use must also follow Meta's [Acceptable Use Policy](https://developer.meta.com/ai/llama4/use-policy/). The code in the GitHub repository is MIT-licensed separately.
