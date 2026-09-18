"""Harness tests: cite, refuse, sessions, and generator gating."""

from pathlib import Path

from agent.chunk import load_kb
from agent.harness import Harness

KB = load_kb(Path(__file__).resolve().parents[1] / "kb")


def test_cites_when_sources_exist():
    h = Harness(KB)
    out = h.ask("What does the Kafka task mesh do?")
    assert out["refused"] is False
    assert out["citations"]
    assert out["session_id"]
    assert "kafka" in out["answer"].lower()


def test_refuses_when_retrieval_empty():
    h = Harness(KB)
    out = h.ask("Who won the 2014 World Cup?")
    assert out["refused"] is True
    assert out["citations"] == []
    assert "sources" in out["answer"].lower()


def test_followup_keeps_session():
    h = Harness(KB)
    first = h.ask("What does the Kafka task mesh do?")
    second = h.ask("How does a worker claim a job?", session_id=first["session_id"])
    assert second["session_id"] == first["session_id"]
    assert second["refused"] is False


def test_generate_is_not_used_on_empty_hits(monkeypatch):
    h = Harness(KB)
    called = {"n": 0}

    def boom(*_args, **_kwargs):
        called["n"] += 1
        raise AssertionError("generator must not run")

    monkeypatch.setattr("agent.harness.generate", boom)
    out = h.ask("Who won the 2014 World Cup?")
    assert out["refused"] is True
    assert called["n"] == 0
