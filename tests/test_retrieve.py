from pathlib import Path

from agent.chunk import load_kb
from agent.harness import Harness
from agent.retrieve import lexical

KB = load_kb(Path(__file__).resolve().parents[1] / "kb")


def test_kafka_question_hits():
    hits = lexical("What does the Kafka task mesh do?", KB)
    assert hits
    assert any("kafka" in h.doc for h in hits)


def test_off_topic_is_empty():
    hits = lexical("Who won the 2014 World Cup?", KB)
    assert hits == []
