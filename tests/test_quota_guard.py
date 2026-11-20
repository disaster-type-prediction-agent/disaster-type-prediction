"""Quota guard: every Gemini request is counted, and the shared key has a per-session and per-day budget."""
from types import SimpleNamespace
from unittest import mock

import pytest
import streamlit as st
from google.genai import errors, types

from src import agent
from tests.test_gemini_agent import FakeClient, call, chunk, server_error

SHARED = "AIzaSySHARED"


@pytest.fixture(autouse=True)
def fresh_shared_counter():
    """The day counter and cooldown live in st.cache_resource (process-wide): start and end each test clean."""
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def raw_client(script):
    return SimpleNamespace(models=FakeClient(script))


def test_every_request_and_quota_error_reaches_the_hook():
    seen = []
    agent.set_usage_hook(seen.append)
    try:
        with mock.patch("src.agent._make_raw_client", return_value=raw_client([[chunk(types.Part(text="hi"))], server_error(429)])):
            client = agent.make_gemini_client("k")
            assert "".join(c.candidates[0].content.parts[0].text for c in client.models.generate_content_stream(model="m", contents="q")) == "hi"
            with pytest.raises(errors.APIError):
                list(client.models.generate_content_stream(model="m", contents="q"))
    finally:
        agent.set_usage_hook(None)
    assert seen == ["call", "call", "quota"]


def ask(at, text):
    at.chat_input[0].set_value(text).run()
    assert not at.exception, at.exception
    return at.session_state["messages"][-1]


def test_agent_turn_is_counted_against_the_session(app):
    at = app({"GEMINI_API_KEY": SHARED})
    script = [[chunk(call("get_significance"))], [chunk(types.Part(text="No real difference."))]]
    with mock.patch("src.agent._make_raw_client", return_value=raw_client(script)):
        ask(at, "is RF better than XGBoost?")
    assert at.session_state["gemini_calls"] == 2


def test_session_limit_switches_to_built_in_answers(app):
    at = app({"GEMINI_API_KEY": SHARED})
    at.session_state["gemini_calls"] = 40
    with mock.patch("src.agent._make_raw_client", side_effect=AssertionError("Gemini must not be called")):
        last = ask(at, "give me a random case please")
    assert "built-in summary" in last["content"] and "Gemini limit" in last["sources"]
    assert at.session_state["stats"]["records"], "the ML models still run and score the case"
    assert any("40/40" in c.value for c in at.sidebar.caption)


def test_quota_error_pauses_gemini_and_still_answers(app):
    at = app({"GEMINI_API_KEY": SHARED})
    errors_only = [server_error(429)] * 3  # every model in the fallback list is out of quota
    with mock.patch("src.agent._make_raw_client", return_value=raw_client(errors_only)):
        last = ask(at, "hello")
    assert "quota" in last["content"] and "built-in summary" in last["content"]
    with mock.patch("src.agent._make_raw_client", side_effect=AssertionError("paused: no calls")):
        last = ask(at, "hello again")
    assert "busy right now" in last["content"]


def test_your_own_key_is_not_limited(app):
    at = app({"GEMINI_API_KEY": SHARED})
    at.session_state["gemini_api_key"] = "AIzaSyMINE"
    at.session_state["gemini_calls"] = 999
    script = [[chunk(types.Part(text="Answer from Gemini."))]]
    with mock.patch("src.agent._make_raw_client", return_value=raw_client(script)):
        last = ask(at, "what is Macro-F1?")
    assert last["content"] == "Answer from Gemini."


def test_counted_client_keeps_the_real_client_alive():
    """genai.Client closes its connection when garbage-collected; the wrapper must hold on to it."""
    import gc
    import weakref

    class Raw:
        models = FakeClient([])

    with mock.patch("src.agent._make_raw_client", return_value=Raw()):
        client = agent.make_gemini_client("k")
    ref = weakref.ref(client._client)
    gc.collect()
    assert ref() is not None
