"""Test doubles that stand in for the real tokenizer and model.

They let the API's security behaviour be tested in CI without downloading
weights. The fake tokenizer is word-level, so one word is one token, which
makes window boundaries easy to reason about. The fake model "detects" an
injection only if the trigger word is inside the window it was given, which
is exactly the property the long-input tests need.
"""

import torch

TRIGGER = "OVERRIDE_SYSTEM"
CLS_ID, SEP_ID, PAD_ID = 1, 2, 0


class _Encoding(dict):
    def to(self, device):
        return _Encoding({k: v.to(device) for k, v in self.items()})


class FakeTokenizer:
    """Mimics the parts of a Hugging Face tokenizer that app.py uses."""

    pad_token_id = PAD_ID

    def __init__(self):
        self.vocab = {"[PAD]": PAD_ID, "[CLS]": CLS_ID, "[SEP]": SEP_ID}

    def _id(self, word):
        if word not in self.vocab:
            self.vocab[word] = len(self.vocab)
        return self.vocab[word]

    def __call__(self, text, add_special_tokens=True, truncation=False,
                 max_length=None, return_tensors=None, padding=False, **kwargs):
        if isinstance(text, list):  # batch call, as Hugging Face allows
            rows = [self(t, add_special_tokens, truncation, max_length)["input_ids"]
                    for t in text]
            return self.pad({"input_ids": rows}) if return_tensors else {"input_ids": rows}
        ids = [self._id(w) for w in text.split()]
        if truncation and max_length is not None:
            room = max_length - (2 if add_special_tokens else 0)
            ids = ids[:room]
        if add_special_tokens:
            ids = [CLS_ID] + ids + [SEP_ID]
        if return_tensors:
            return self.pad({"input_ids": [ids]})
        return {"input_ids": ids}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, ids):
        return [CLS_ID] + list(ids) + [SEP_ID]

    def pad(self, features, return_tensors="pt"):
        rows = features["input_ids"]
        width = max(len(r) for r in rows)
        ids = [r + [PAD_ID] * (width - len(r)) for r in rows]
        mask = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
        return _Encoding({
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(mask),
        })


class _Output:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """Returns a confident 'injection' score when the trigger is in view."""

    def __init__(self, tokenizer, fail=False):
        self.tokenizer = tokenizer
        self.fail = fail
        self.window_lengths = []

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        if self.fail:
            raise RuntimeError("simulated inference failure")
        trigger_id = self.tokenizer._id(TRIGGER)
        logits = []
        for row, mask in zip(input_ids.tolist(), attention_mask.tolist()):
            self.window_lengths.append(sum(mask))
            seen = trigger_id in row
            logits.append([-5.0, 5.0] if seen else [5.0, -5.0])
        return _Output(torch.tensor(logits))
