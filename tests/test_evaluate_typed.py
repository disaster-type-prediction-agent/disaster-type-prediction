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


def test_simulated_user_says_nothing_was_measured_only_when_the_record_says_so():
    assert ev.follow_up_reply({"magnitude_scale": "Unknown"}, "English") == ev.NOTHING_MEASURED["English"]
    assert ev.follow_up_reply({"magnitude_scale": "Kph"}, "Thai") == ev.NO_MORE["Thai"]
    assert ev.follow_up_reply({"magnitude_scale": "Km2"}, "English") == ev.NO_MORE["English"]


def test_burmese_simulated_user_has_both_replies():
    assert ev.follow_up_reply({"magnitude_scale": "Unknown"}, "Burmese") == ev.NOTHING_MEASURED["Burmese"]
    assert ev.follow_up_reply({"magnitude_scale": "Kph"}, "Burmese") == ev.NO_MORE["Burmese"]


def test_summary_names_the_languages_used(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "OUT_DIR", tmp_path)
    monkeypatch.setattr(ev, "RESULTS", tmp_path / "results.jsonl")
    rows = [dict(record("C1", "Storm", "Storm", "Storm"), language="Burmese"),
            dict(record("C2", "Flood", "Flood", "Flood"), language="Burmese")]
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    ev.summarize()
    assert "all written in Burmese" in (tmp_path / "summary.md").read_text()


def test_writer_rejects_burmese_hazard_names_but_not_unread_scales():
    from src.llm_tasks import DISASTER_WORDS, _carries_scale

    leaked = "မုန်တိုင်းကြောင့် ရန်ကုန်မှာ နိုဝင်ဘာလ ၃ ရက်ကြာ လေတိုက်ခဲ့တယ်"  # names the storm
    assert any(w in leaked for w in DISASTER_WORDS)
    # A Burmese message the code cannot read: the scale check is left to the agent's quote rule.
    unmeasured = {"magnitude_scale": "Kph", "magnitude_value": None}
    assert _carries_scale(unmeasured, "ရန်ကုန်မှာ နိုဝင်ဘာလ ၃ ရက်ကြာ လေတိုက်နှုန်း တိုင်းတာထားပါတယ်")
    # English and Thai are still checked by the code.
    assert not _carries_scale(unmeasured, "Something happened in Yangon in November for 3 days.")
