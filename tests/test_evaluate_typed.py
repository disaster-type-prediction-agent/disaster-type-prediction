"""The typed-evaluation report, built from recorded results (no Gemini calls)."""
import json

from src import evaluate_typed as ev


def record(case_id, truth, direct, typed, read_wrong=(), ok=True):
    models = ev.MODELS
    return {
        "case_id": case_id, "true_type": truth, "language": "Thai", "message": "msg", "gemini_calls": 3,
        "direct": {m: direct for m in models},
        "typed": None if typed is None else {m: typed for m in models},
        "read": {"correct": 6 - len(read_wrong), "wrong": list(read_wrong), "detail": []},
        "explanation_ok": ok,
    }


def test_summary_compares_direct_and_typed(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "OUT_DIR", tmp_path)
    monkeypatch.setattr(ev, "RESULTS", tmp_path / "results.jsonl")
    rows = [
        record("C1", "Storm", "Storm", "Storm"),
        record("C2", "Flood", "Flood", "Drought", read_wrong=["Start month"]),
        record("C3", "Drought", "Flood", "Flood", ok=False),
        record("C4", "Flood", "Flood", None),  # the agent never ran the models
    ]
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ev.summarize()
    report = (tmp_path / "summary.md").read_text()
    assert "| Random Forest | 75.0% | 25.0% |" in report  # direct 3/4 right, typed 1/4
    assert "**3/4**" in report                          # 3 of 4 reached the models
    assert "Explanations matching the model output: **2/3**" in report
    assert "C2 (Flood → Drought), misread: Start month" in report
    assert (tmp_path / "summary.csv").read_text().startswith("model,")
