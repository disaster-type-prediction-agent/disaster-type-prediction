"""Free questions update the scoreboard, and Gemini is scored next to the model (Gemini calls are mocked).

These use the fixed flow (agent mode off); tests/test_gemini_agent.py covers agent mode."""
from unittest import mock

import pytest

from src.agent import DisasterAuditorLLM


def fake_stream(self, prompt):
    yield "Mocked answer."


def route_for(event_truth=None, intent="new_event"):
    event = {"country": "Bangladesh", "region": None, "start_month": 7, "duration_days": 10.0,
             "magnitude_value": 5000.0, "magnitude_scale": "Km2"}
    return {"intent": intent, "disaster_type": None, "region": None, "only_hard": False,
            "only_disagreement": False, "event": event if intent == "new_event" else None,
            "stated_true_type": event_truth}


@pytest.fixture
def chat(app):
    """The app with a (fake) Gemini key; routing, guessing and the chat reply are all mocked."""
    patches = [
        mock.patch.object(DisasterAuditorLLM, "_stream_gemini", fake_stream),
        mock.patch("src.llm_tasks.llm_guess", return_value={"predicted_class": "Flood", "confidence": 0.8, "reason": "Km2."}),
    ]
    for p in patches:
        p.start()
    at = app({"GEMINI_API_KEY": "AIzaSyFAKE"})
    at.session_state["gemini_agent"] = False  # these tests cover the fixed route-then-answer flow

    def ask(text, route):
        with mock.patch("src.llm_tasks.route_question", return_value=route):
            at.chat_input[0].set_value(text).run()
        assert not at.exception, at.exception
        return at

    yield at, ask
    for p in patches:
        p.stop()


def records(at):
    return at.session_state["stats"]["records"]


def test_described_event_waits_for_confirmation_then_scores(chat):
    at, ask = chat
    ask("A flood-like event in Bangladesh, July, 10 days, 5000 km2", route_for())

    audit = at.session_state["last_audit"]
    case = audit["case"]
    assert case["case_id"] == "EVENT-1" and case["source"] == "user"
    assert case["summary"]["region"] == "Asia", "region filled in from the country"
    assert "llm" not in audit, "the LLM baseline is off by default: only the ML model predicts"
    assert records(at) == [], "no score until the user says what it really was"

    idx = next(i for i, m in enumerate(at.session_state["messages"]) if m.get("audit"))
    at.selectbox(key=f"truth_EVENT-1_{idx}").set_value("Flood").run()  # picking a type enables Confirm
    at.button(key=f"confirm_EVENT-1_{idx}").click().run()
    assert not at.exception
    [rec] = records(at)
    assert rec["source"] == "Your event" and rec["true_type"] == "Flood" and rec["gemini_prediction"] is None


def test_stated_true_type_scores_immediately(chat):
    at, ask = chat
    ask("Storm in Bangladesh ... it was actually a Storm", route_for(event_truth="Storm"))
    [rec] = records(at)
    assert rec["true_type"] == "Storm" and rec["model_prediction"]


def test_test_bank_request_scores_the_model(chat):
    at, ask = chat
    ask("give me a random case", route_for(intent="test_bank_case"))
    [rec] = records(at)
    assert rec["source"] == "Test bank" and rec["gemini_prediction"] is None
    assert at.session_state["stats"]["records"] and not at.exception


def test_llm_baseline_is_opt_in_and_scored_beside_the_model(chat):
    at, ask = chat
    at.session_state["llm_baseline"] = True
    ask("give me a random case", route_for(intent="test_bank_case"))
    [rec] = records(at)
    assert rec["gemini_prediction"] == "Flood"


def test_plain_question_changes_nothing_and_cites_sources(chat):
    at, ask = chat
    ask("why is drought hard?", route_for(intent="question"))
    assert records(at) == []
    last = at.session_state["messages"][-1]
    assert last["content"] == "Mocked answer." and "general knowledge" in last["sources"]


def test_without_a_key_keywords_still_route_test_cases(app):
    at = app()
    at.chat_input[0].set_value("random case please").run()
    assert not at.exception
    [rec] = at.session_state["stats"]["records"]
    assert rec["source"] == "Test bank" and rec["gemini_prediction"] is None


def test_compare_after_a_described_event_compares_that_event(chat):
    at, ask = chat
    ask("10 days of rain in Bangladesh in July, 5000 km2", route_for())
    at.chat_input[0].set_value("/compare").run()
    assert not at.exception, at.exception

    audit = at.session_state["last_audit"]
    assert audit["case"]["case_id"] == "EVENT-1", "the user's event, not a test-bank case"
    assert set(audit["models"]) == {"Random Forest", "LightGBM", "XGBoost"}
    assert at.session_state["messages"][-2]["content"] == "`/compare` EVENT-1"


def test_compare_with_no_event_still_finds_a_disagreement_case(chat):
    at, _ = chat
    at.chat_input[0].set_value("/compare").run()
    assert not at.exception, at.exception
    assert at.session_state["last_audit"]["case"]["case_id"].startswith("CASE-")
