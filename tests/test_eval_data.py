"""The evaluation sets must stay what the README says they are.

1. Each committed holdout file is exactly what its seeded generator produces.
2. Each holdout's vocabulary is disjoint from the training vocabulary.
3. No holdout prompt appears in the generated training data.
4. The committed validation set matches the lists in validate_shield.py.
These need no model, so they always run.
"""

import ast
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """Run both generators in a scratch directory and return their outputs."""
    out = tmp_path_factory.mktemp("gen")
    import os
    cwd = os.getcwd()
    os.chdir(out)
    try:
        v2 = runpy.run_path(str(ROOT / "generate_hard_negatives.py"), run_name="__main__")
        v3 = runpy.run_path(str(ROOT / "generate_hard_negatives_v3.py"), run_name="__main__")
    finally:
        os.chdir(cwd)
    return out, v2, v3


@pytest.mark.parametrize("committed,produced", [
    ("holdout_v3.jsonl", "hard_negatives_v3_holdout.jsonl"),
    ("holdout_v2.jsonl", "hard_negatives_holdout.jsonl"),
])
def test_holdout_matches_its_generator(generated, committed, produced):
    out, _, _ = generated
    assert (EVAL / committed).read_bytes() == (out / produced).read_bytes()


def _lower(items):
    return {s.lower() for s in items}


@pytest.mark.parametrize("train_name,hold_name", [
    ("VERBS", "H_VERBS"),
    ("SYSTEM_REFS", "H_SYSTEM_REFS"),
    ("QUALIFIERS", "H_QUALIFIERS"),
    ("ACTIONS", "H_ACTIONS"),
])
def test_v3_holdout_vocabulary_is_disjoint(generated, train_name, hold_name):
    _, _, v3 = generated
    train, hold = _lower(v3[train_name]), _lower(v3[hold_name])
    assert train.isdisjoint(hold)


def test_v3_holdout_system_references_never_appear_in_training_vocabulary(generated):
    _, _, v3 = generated
    train_text = " ".join(_lower(v3["SYSTEM_REFS"] + v3["QUALIFIERS"] + v3["ACTIONS"]))
    leaks = [ref for ref in _lower(v3["H_SYSTEM_REFS"]) if ref in train_text]
    assert not leaks


def test_no_holdout_prompt_is_in_generated_training_data(generated):
    out, _, _ = generated
    train = {r["text"].lower() for name in ("hard_negatives_train.jsonl",
                                            "hard_negatives_v3_train.jsonl")
             for r in load_jsonl(out / name)}
    hold = {r["text"].lower() for name in ("holdout_v2.jsonl", "holdout_v3.jsonl")
            for r in load_jsonl(EVAL / name)}
    assert not (train & hold)


def test_holdout_sizes_and_labels():
    v3 = load_jsonl(EVAL / "holdout_v3.jsonl")
    assert len(v3) == 60 and {r["label"] for r in v3} == {1}
    v2 = load_jsonl(EVAL / "holdout_v2.jsonl")
    assert sum(r["label"] == 0 for r in v2) == 60
    assert sum(r["label"] == 1 for r in v2) == 20


def test_validation_set_matches_validate_shield():
    tree = ast.parse((ROOT / "validate_shield.py").read_text(encoding="utf-8"))
    lists = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in ("OOD_ATTACKS", "OOD_BENIGN")}
    expected = ([{"text": t, "label": 1} for t in lists["OOD_ATTACKS"]]
                + [{"text": t, "label": 0} for t in lists["OOD_BENIGN"]])
    assert load_jsonl(EVAL / "ood_validation.jsonl") == expected
