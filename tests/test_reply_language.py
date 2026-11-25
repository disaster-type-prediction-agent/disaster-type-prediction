"""Replies follow the language of the latest question, not of earlier turns."""
from unittest import mock

import pytest
from google.genai import types

from src.agent import DisasterAuditorLLM, language_note, reply_language
from tests.test_gemini_agent import FakeClient, chunk


@pytest.mark.parametrize("text, lang", [
    ("Why not Flood?", "English"),
    ("ทำไมไม่ใช่ Flood", "Thai"),
    ("พายุเข้าไทย ลม 120 kph", "Thai"),
    ("Flood in เชียงใหม่ in July, 10 days, 5000 km2, what does the model say?", "English"),
    ("RF vs XGBoost", "English"),
    ("1234", "English"),
    ("Inondation au Bangladesh en juillet, 10 jours. Que prédit le modèle ?", "other"),
    ("Banjir di Indonesia bulan Januari selama 5 hari", "other"),
    ("Inundación en México en septiembre", "other"),
    ("日本で3月にマグニチュード7.2の地震", "other"),
    ("中国七月发生洪水", "other"),
    ("Наводнение в России в мае", "other"),
])
def test_reply_language(text, lang):
    assert reply_language(text) == lang


def test_fixed_flow_prompt_ends_with_the_note():
    prompt = DisasterAuditorLLM("gemini", "k")._build_prompt(
        "What is Macro-F1?", [{"role": "user", "content": "พายุเข้าไทยไหม"}], None)
    assert prompt.endswith(language_note("What is Macro-F1?")) and "English" in prompt.splitlines()[-1]


def test_agent_question_carries_the_note_after_thai_turns(app):
    at = app({"GEMINI_API_KEY": "AIzaSyFAKE"})
    client = FakeClient([[chunk(types.Part(text="ตอบ"))], [chunk(types.Part(text="Answer"))]])
    with mock.patch("src.agent.make_gemini_client", return_value=client):
        at.chat_input[0].set_value("พายุคืออะไร").run()
        at.chat_input[0].set_value("And in English?").run()
    assert not at.exception
    sent = [c for c in client.requests[-1][1] if c.role == "user"][-1].parts[0].text
    assert sent.rstrip().endswith(language_note("And in English?")) and "Reply language: English" in sent


def test_other_languages_answer_in_kind_with_english_class_names():
    note = language_note("日本で地震がありました")
    assert "language of the latest user message" in note and "(Earthquake)" in note


def test_agent_turn_labels_the_ui_context_apart_from_the_users_words():
    """Run together, Gemini read "selected models: Random Forest" into the message and dropped its country."""
    from src.agent import agent_turn_text
    text = agent_turn_text("In the Philippines, wind speed was recorded.", ["selected models: Random Forest"])
    head, rest = text.split("\n\nUser message:\n", 1)
    assert head == "[UI context, app state, not the user's words: selected models: Random Forest]"
    assert rest.startswith("In the Philippines, wind speed was recorded.\n\n[Reply language: English.")
    assert agent_turn_text("hi", []).startswith("User message:\nhi")


def test_the_app_sends_the_labelled_turn(app):
    from types import SimpleNamespace
    from unittest import mock
    from google.genai import types
    from tests.test_gemini_agent import FakeClient, chunk

    at = app({"GEMINI_API_KEY": "AIzaSyTEST"})
    fake = FakeClient([[chunk(types.Part(text="Which month did it start?"))]])
    with mock.patch("src.agent._make_raw_client", return_value=SimpleNamespace(models=fake)):
        at.chat_input[0].set_value("In the Philippines, wind speed was recorded.").run()
    sent = fake.requests[0][1][-1].parts[0].text
    assert sent.startswith("[UI context, app state, not the user's words: selected models:")
    assert "\n\nUser message:\nIn the Philippines, wind speed was recorded.\n\n[Reply language" in sent
