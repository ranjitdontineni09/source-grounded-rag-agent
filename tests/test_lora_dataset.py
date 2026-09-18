import json
from pathlib import Path


def test_dataset_is_cite_or_refuse():
    path = Path(__file__).resolve().parents[1] / "lora" / "dataset.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    required = {"instruction", "context", "question", "output"}
    refuses = sum(1 for r in rows if r["output"].strip().upper() == "REFUSE")
    answers = [r for r in rows if r["output"].strip().upper() != "REFUSE"]
    assert len(rows) >= 8
    assert refuses >= 2
    for row in rows:
        assert required <= row.keys()
    for row in answers:
        assert row["context"].strip()
