# Security policy

## Reporting a vulnerability

Please do not open a public issue. Report privately through this repository's **Security** tab → **Report a vulnerability**. Include what you found, how to reproduce it, and the commit you tested.

This is a one-person project maintained alongside study, so responses are best effort. I aim to acknowledge reports within 14 days and keep you updated. With your agreement, you are credited in the advisory.

## In scope

- The API in `app.py`: authentication, the server-side threshold, fail-closed behaviour, decision logging, and input handling.
- **Detection bypasses** that defeat the guardrail through the API in a way the README does not already list as a limitation, for example a new encoding, input shape or length trick. Known weaknesses (question- or statement-phrased attacks, thin non-English coverage) are documented in the README and are not new reports.
- Mistakes in the evaluation code or sets that make a published number wrong.

## Out of scope

- Vulnerabilities in PyTorch, transformers, FastAPI or other dependencies; report those upstream.
- The base model `meta-llama/Llama-Prompt-Guard-2-86M`; report those to Meta.
- Deployments that ignore the README's guidance, such as binding to `0.0.0.0` without a reason or running without other defences. A classifier is one layer, not a complete defence.
