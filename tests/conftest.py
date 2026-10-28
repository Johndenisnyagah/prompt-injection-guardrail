import os
import sys
from pathlib import Path

import pytest

# app.py reads its config at import time, so set a key before importing it.
os.environ.setdefault("GUARD_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import app as guard_app  # noqa: E402
from tests.fakes import FakeModel, FakeTokenizer  # noqa: E402

API_KEY = "test-key"


@pytest.fixture
def fake_state(monkeypatch):
    """Install the fake tokenizer/model in place of the real ones."""
    tokenizer = FakeTokenizer()
    model = FakeModel(tokenizer)
    monkeypatch.setattr(guard_app, "API_KEY", API_KEY)
    monkeypatch.setitem(guard_app.STATE, "tokenizer", tokenizer)
    monkeypatch.setitem(guard_app.STATE, "model", model)
    monkeypatch.setitem(guard_app.STATE, "device", "cpu")
    monkeypatch.setitem(guard_app.STATE, "ready", True)
    return tokenizer, model


@pytest.fixture
def client(fake_state):
    # No `with` block: the lifespan (which loads real weights) does not run.
    return TestClient(guard_app.app)


@pytest.fixture
def headers():
    return {"X-API-Key": API_KEY}
