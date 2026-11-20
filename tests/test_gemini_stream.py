"""DisasterAuditorLLM's Gemini path with a mocked google-genai client: model order, fallback and RESTART."""
from types import SimpleNamespace as NS
from unittest import mock

from google import genai

from src.agent import DEFAULT_GEMINI_MODELS, RESTART, DisasterAuditorLLM, gemini_models


def stream_with(behaviours):
    """Patch generate_content_stream; behaviours[i] decides what the i-th call does."""
    calls = []

    def fake(self, *, model, contents, **kw):
        calls.append(model)
        return behaviours[len(calls) - 1]()

    return mock.patch.object(genai.models.Models, "generate_content_stream", fake), calls


def ok(*chunks):
    return lambda: iter([NS(text=c) for c in chunks])


def fails_immediately():
    def go():
        raise RuntimeError("503 UNAVAILABLE")
    return go


def fails_after(*chunks):
    def go():
        def gen():
            for c in chunks:
                yield NS(text=c)
            raise RuntimeError("503 UNAVAILABLE")
        return gen()
    return go


def final_text(pieces):
    reply = ""
    for p in pieces:
        reply = "" if p is RESTART else reply + p
    return reply


def test_first_model_answers(clean_env):
    patch, calls = stream_with([ok("Hello ", "team.")])
    with patch:
        pieces = list(DisasterAuditorLLM("gemini", "k").chat_stream("q"))
    assert final_text(pieces) == "Hello team." and calls == DEFAULT_GEMINI_MODELS[:1]


def test_failure_before_any_text_falls_back_silently(clean_env):
    patch, calls = stream_with([fails_immediately(), ok("Backup answer.")])
    with patch:
        pieces = list(DisasterAuditorLLM("gemini", "k").chat_stream("q"))
    assert RESTART not in pieces
    assert final_text(pieces) == "Backup answer." and calls == DEFAULT_GEMINI_MODELS[:2]


def test_failure_mid_answer_restarts_cleanly(clean_env):
    patch, _ = stream_with([fails_after("Partial "), ok("Clean answer.")])
    with patch:
        pieces = list(DisasterAuditorLLM("gemini", "k").chat_stream("q"))
    assert RESTART in pieces
    assert final_text(pieces) == "Clean answer."


def test_all_models_failing_reports_an_error(clean_env):
    patch, calls = stream_with([fails_immediately()] * len(DEFAULT_GEMINI_MODELS))
    with patch:
        reply = final_text(DisasterAuditorLLM("gemini", "k").chat_stream("q"))
    assert "Gemini API Error" in reply and len(calls) == len(DEFAULT_GEMINI_MODELS)


def test_gemini_model_setting_goes_first(clean_env):
    clean_env.setenv("GEMINI_MODEL", "gemini-x, gemini-3.5-flash")
    assert gemini_models()[:2] == ["gemini-x", "gemini-3.5-flash"]
    assert len(gemini_models()) == len(set(gemini_models())), "no duplicates"


def test_no_key_uses_the_local_engine(clean_env):
    reply = "".join(DisasterAuditorLLM("gemini", "").chat_stream("hello"))
    assert "Disaster Auditor AI" in reply
