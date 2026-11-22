"""Too little to predict from: the app asks the user first (once), decided in code, then the models run."""
from unittest import mock

from google.genai import types

from src.auditor_agent import SessionCases
from src.event_features import MEASURE_QUESTION, event_from_case, missing_details
from tests.test_gemini_agent import FakeClient, call, chunk, run

FULL = {"country": "Thailand", "start_month": 11, "duration_days": 3, "magnitude_value": 120, "magnitude_scale": "Kph"}


def test_what_counts_as_too_little():
    assert missing_details(FULL) == []
    assert missing_details({"country": "Argentina", "start_month": 1, "magnitude_scale": "°C"}) == []  # no figure is fine
    assert missing_details({**FULL, "magnitude_scale": None, "magnitude_value": None}) == [MEASURE_QUESTION]
    assert missing_details({**FULL, "country": None}) == ["where it happened (country)"]
    assert missing_details({**FULL, "country": None, "region": "Asia"}) == []
    assert missing_details({**FULL, "start_month": None, "duration_days": None}) == ["the month it started", "how long it lasted"]
    assert missing_details({}) == ["where it happened (country)", MEASURE_QUESTION, "the month it started", "how long it lasted"]


def test_every_test_bank_case_that_names_a_scale_is_enough(registry):
    """Recorded cases with a scale never trigger a question; those without one (Landslide, many Epidemics) do."""
    for case in registry.cases[:400]:
        event = event_from_case(case)
        has_scale = event["magnitude_scale"] != "Unknown"
        assert (missing_details(event) == []) == has_scale, case["case_id"]


def classify(tools, session, message, **args):
    session.message = message
    return tools.execute("classify_event", args, ["Random Forest"], session)


def test_asks_once_then_runs_on_the_next_message(tools):
    session = SessionCases()
    result, summary, audit, _ = classify(tools, session, "something happened in Congo in September", country="Congo", start_month=9)
    assert result["status"] == "need_more_details" and audit is None and not session.cases
    assert MEASURE_QUESTION in result["missing"] and summary == "Asked for more details"

    # Same message: the LLM calling again in the same turn doesn't get past the question.
    result, _, audit, _ = classify(tools, session, "something happened in Congo in September", country="Congo", start_month=9)
    assert audit is None

    # The user's next message, even "don't know", runs the models on what was given.
    result, _, audit, _ = classify(tools, session, "I don't know", country="Congo", start_month=9)
    assert audit and list(session.cases) == ["EVENT-1"] and "need_more_details" not in str(result.get("status"))
    assert session.asked_details_on is None

    # A new thin event later is asked about again.
    result, _, audit, _ = classify(tools, session, "and one in Peru?", country="Peru")
    assert audit is None and result["status"] == "need_more_details"


def test_enough_detail_runs_straight_away(tools):
    result, _, audit, _ = classify(tools, SessionCases(), "storm text", **FULL)
    assert audit and "status" not in result


def test_gemini_asks_and_the_answer_completes_the_event(tools):
    session, history = SessionCases(), []
    session.message = "Something happened in Congo in September."
    events, _, _ = run(tools, [[chunk(call("classify_event", country="Congo", start_month=9))],
                               [chunk(types.Part(text="Was anything measured, like area or wind speed?"))]],
                       history=history, session=session)
    assert not session.cases and [e for e in events if e[0] == "tool"][0][1]["audit"] is None

    session.message = "Nothing was measured."
    events, _, _ = run(tools, [[chunk(call("classify_event", country="Congo", start_month=9, nothing_measured=True))],
                               [chunk(types.Part(text="The model says ..."))]],
                       question=session.message, history=history, session=session)
    tool = [e for e in events if e[0] == "tool"][0][1]
    case = session.cases["EVENT-1"]
    assert tool["audit"] and case["event"]["magnitude_scale"] == "Unknown"
    assert "Magnitude scale" not in case["filled"], "'nothing was measured' is an answer, not a default"
    assert "Magnitude value" not in case["filled"]


def test_dont_know_keeps_the_scale_flagged_as_a_default(tools):
    session = SessionCases()
    classify(tools, session, "Congo, September", country="Congo", start_month=9)
    _, _, audit, _ = classify(tools, session, "I don't know", country="Congo", start_month=9)
    assert "Magnitude scale" in audit["case"]["filled"]


def test_nothing_measured_in_the_first_message_needs_no_question(tools):
    _, _, audit, _ = classify(tools, SessionCases(), "Congo, September, nothing was measured",
                              country="Congo", start_month=9, nothing_measured=True)
    assert audit is not None


def test_typed_test_survives_the_question_and_is_scored(app, registry):
    """A Typed test whose case has no scale (a Landslide): the app asks, the user answers, the test still counts."""
    case = next(c for c in registry.cases if c["ground_truth_class"] == "Landslide")
    event = event_from_case(case)
    read = {k: v for k, v in event.items() if v not in (None, "Unknown")}
    at = app({"GEMINI_API_KEY": "AIzaSyFAKE"})
    ask = [[chunk(call("classify_event", **read))], [chunk(types.Part(text="Was anything measured?"))]]
    with mock.patch("src.llm_tasks.write_scenario", return_value="A message without the type"), \
         mock.patch("src.models.registry.ModelRegistry.get_random_case", return_value=case), \
         mock.patch("src.agent.make_gemini_client", return_value=FakeClient(ask)):
        next(b for b in at.button if b.label == "Typed test").click().run()
    assert not at.exception, at.exception
    assert not at.session_state["stats"]["records"] and at.session_state["typed_test"] is not None

    answer = [[chunk(call("classify_event", **read))], [chunk(types.Part(text="The model says ..."))]]
    with mock.patch("src.agent.make_gemini_client", return_value=FakeClient(answer)):
        at.chat_input[0].set_value("I don't know any other details").run()
    assert not at.exception, at.exception
    [rec] = at.session_state["stats"]["records"]
    assert rec["source"] == "Typed test" and rec["true_type"] == "Landslide"


def test_a_named_quantity_beats_nothing_measured(tools):
    """Typed eval: Gemini read 'wind speed was recorded, no figure' as nothing_measured; the scale must win."""
    msg = "In the Philippines, wind speed was recorded this November for 2 days, but no figure was given."
    result, _, audit, _ = classify(tools, SessionCases(), msg, country="Philippines", start_month=11,
                                   duration_days=2, nothing_measured=True)
    assert audit and audit["case"]["summary"]["magnitude_scale"] == "Kph"

    # The LLM naming the scale itself also beats the flag.
    result, _, audit, _ = classify(tools, SessionCases(), "storm, winds measured, no number", country="Philippines",
                                   start_month=11, duration_days=2, magnitude_scale="Kph", nothing_measured=True)
    assert audit and audit["case"]["summary"]["magnitude_scale"] == "Kph"


def test_nothing_measured_still_counts_when_no_quantity_is_named(tools):
    msg = "A slope collapsed in Peru in March for 1 day; nothing was measured."
    result, _, audit, _ = classify(tools, SessionCases(), msg, country="Peru", start_month=3, duration_days=1,
                                   nothing_measured=True)
    assert audit and audit["case"]["summary"]["magnitude_scale"] == "Unknown"


def test_the_answer_turn_still_reads_the_scale_from_the_first_message(tools):
    session = SessionCases()
    first = "In Liberia, the affected area was reported, no figure."
    result, _, audit, _ = classify(tools, session, first, country="Liberia")
    assert result["status"] == "need_more_details"
    result, _, audit, _ = classify(tools, session, "I don't know any other details; that's all I have.",
                                   country="Liberia", nothing_measured=True)
    assert audit and audit["case"]["summary"]["magnitude_scale"] == "Km2"
