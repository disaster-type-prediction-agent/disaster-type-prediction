"""The Streamlit app, run headlessly with AppTest.

Each test runs from an empty temp folder (with models/ linked in), so the developer's real
.streamlit/secrets.toml never leaks into the result.
"""
import pytest


def test_app_starts_without_errors(app):
    at = app()
    assert not at.exception
    assert at.session_state["layout_mode"] == "Chat"


@pytest.mark.parametrize(
    "secrets, env, provider, gemini, claude",
    [
        ({"GEMINI_API_KEY": "AIzaSyFAKE"}, {}, "Google Gemini", True, False),
        ({"ANTHROPIC_API_KEY": "sk-ant-FAKE"}, {}, "Anthropic Claude", False, True),
        ({"GEMINI_API_KEY": "AIzaSyFAKE", "ANTHROPIC_API_KEY": "sk-ant-FAKE", "LLM_PROVIDER": "claude"}, {},
         "Anthropic Claude", True, True),
        ({}, {"GEMINI_API_KEY": "AIzaSyENV"}, "Google Gemini", True, False),
        ({}, {}, "Google Gemini", False, False),
    ],
    ids=["gemini-secret", "claude-only", "both+LLM_PROVIDER", "env-var", "nothing"],
)
def test_keys_load_automatically(app, secrets, env, provider, gemini, claude):
    at = app(secrets, env)
    assert not at.exception
    assert at.session_state["api_provider"] == provider
    assert bool(at.session_state["gemini_api_key"]) is gemini
    assert bool(at.session_state["claude_api_key"]) is claude


@pytest.mark.parametrize("view", ["Chat + details", "Benchmarks"])
def test_every_view_renders(app, view):
    at = app()
    at.session_state["layout_mode"] = view
    at.run()
    assert not at.exception
