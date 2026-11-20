"""Gemini agent mode: Gemini picks the tools via function calling (the Gemini API is faked)."""
from unittest import mock

import pytest
from google.genai import errors, types

from src.auditor_agent import GEMINI_HISTORY_CONTENTS, GeminiAuditorAgent, SessionCases


def chunk(*parts):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))])


def call(name, **args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args, id=f"call-{name}"))


class FakeClient:
    """Plays back one scripted reply per request; an Exception in the script is raised instead."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.models = self

    def generate_content_stream(self, model, contents, config):
        self.requests.append((model, list(contents)))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, tuple):  # (chunks before failing, error)
            yield from step[0]
            raise step[1]
        yield from step


def run(tools, script, question="q", history=None, session=None):
    client = FakeClient(script)
    history = [] if history is None else history
    with mock.patch("src.agent.make_gemini_client", return_value=client), \
         mock.patch("src.agent.gemini_models", return_value=["model-a", "model-b"]):
        events = list(GeminiAuditorAgent(tools, "AIzaSyFAKE").run_turn(history, question, ["Random Forest"], session))
    return events, history, client


def server_error(code=503):
    return errors.ServerError(code, {"error": {"code": code, "message": "high demand", "status": "UNAVAILABLE"}})


def test_classifies_a_described_event_then_answers(tools):
    session = SessionCases()
    # Gemini sends numbers as floats (11.0): they must still pass validation as integers.
    events, history, client = run(tools, [
        [chunk(call("classify_event", country="thailand", start_month=11.0, magnitude_value=120.0, magnitude_scale="Kph"))],
        [chunk(types.Part(text="It's a ")), chunk(types.Part(text="Storm."))],
    ], session=session)

    [tool] = [p for k, p in events if k == "tool"]
    assert not tool["is_error"] and tool["audit"]["case"]["case_id"] == "EVENT-1"
    assert tool["audit"]["case"]["summary"]["country"] == "Thailand" and tool["audit"]["case"]["filled"] == ["Event duration"]
    assert "".join(p for k, p in events if k == "text") == "It's a Storm."
    assert "EVENT-1" in session.cases
    # user, model(call), user(function response), model(answer); the response carries the call id
    assert [c.role for c in history] == ["user", "model", "user", "model"]
    response = history[2].parts[0].function_response
    assert response.id == "call-classify_event" and response.response["filled_in"] == ["Event duration"]


def test_follow_up_finds_the_event_from_the_session(tools):
    session = SessionCases()
    run(tools, [[chunk(call("classify_event", country="Thailand", magnitude_value=120.0, magnitude_scale="Kph"))],
                [chunk(types.Part(text="Storm."))]], session=session)
    events, _, _ = run(tools, [[chunk(call("explain_prediction", case_id="EVENT-1", model="Random Forest", target_class="Flood"))],
                               [chunk(types.Part(text="Kph rules out Flood."))]], session=session)
    [tool] = [p for k, p in events if k == "tool"]
    assert not tool["is_error"] and "Flood" in tool["summary"]


def test_invalid_tool_input_is_reported_back_not_run(tools):
    events, history, _ = run(tools, [[chunk(call("classify_event", start_month=13.0))], [chunk(types.Part(text="Sorry."))]])
    [tool] = [p for k, p in events if k == "tool"]
    assert tool["is_error"] and "INVALID_JSON" in history[2].parts[0].function_response.response


def test_falls_back_to_the_next_model_and_retracts_partial_text(tools):
    events, history, client = run(tools, [
        ([chunk(types.Part(text="Half an ans"))], server_error()),  # model-a fails mid-answer
        [chunk(types.Part(text="Full answer."))],                    # model-b answers
    ])
    assert ("retract", len("Half an ans")) in events
    assert [m for m, _ in client.requests] == ["model-a", "model-b"]
    assert [c.role for c in history] == ["user", "model"] and history[1].parts[0].text == "Full answer."


def test_quota_error_on_every_model_is_explained(tools):
    events, history, _ = run(tools, [server_error(429), server_error(429)])
    assert events[-1][0] == "error" and "quota" in events[-1][1]
    assert [c.role for c in history] == ["user"], "a failed round leaves nothing half-written"


def test_long_history_is_trimmed_at_a_plain_user_message(tools):
    history = []
    for i in range(30):  # 30 x (user question, model call, user tool result, model answer) = 120 contents
        history += [types.Content(role="user", parts=[types.Part(text=f"q{i}")]),
                    types.Content(role="model", parts=[call("get_significance")]),
                    types.Content(role="user", parts=[types.Part.from_function_response(name="get_significance", response={})]),
                    types.Content(role="model", parts=[types.Part(text="a")])]
    GeminiAuditorAgent._trim(history)
    assert len(history) <= GEMINI_HISTORY_CONTENTS
    assert history[0].role == "user" and history[0].parts[0].text


def test_app_agent_mode_shows_the_event_card_and_scores_it(app):
    with mock.patch("src.llm_tasks.llm_guess", return_value={"predicted_class": "Storm", "confidence": 0.9, "reason": "Kph."}):
        at = app({"GEMINI_API_KEY": "AIzaSyFAKE"})
        script = [[chunk(call("classify_event", country="Thailand", magnitude_value=120.0, magnitude_scale="Kph",
                              stated_true_type="Storm"))],
                  [chunk(types.Part(text="Storm, as you said."))]]
        with mock.patch("src.agent.make_gemini_client", return_value=FakeClient(script)):
            at.chat_input[0].set_value("A storm hit Thailand, 120 kph winds; it was a storm").run()
    assert not at.exception, at.exception
    last = at.session_state["messages"][-1]
    assert last["content"] == "Storm, as you said." and "classify_event" in last["sources"]
    [rec] = at.session_state["stats"]["records"]
    assert rec["case_id"] == "EVENT-1" and rec["true_type"] == "Storm" and rec["gemini_prediction"] == "Storm"
