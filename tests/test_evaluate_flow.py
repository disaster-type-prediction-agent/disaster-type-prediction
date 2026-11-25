"""The flow eval's fixed conversations and its scoring, offline (Gemini scripted)."""
from unittest import mock

import pytest
from google.genai import types

from src.evaluate_flow import CASES, run_case
from src.event_features import code_reads, names_a_type, observed_scale
from tests.test_gemini_agent import FakeClient, call, chunk

BY_ID = {c["id"]: c for c in CASES}


def test_the_conversations_never_name_a_type_and_cover_every_answer():
    for case in CASES:
        assert not names_a_type(case["message"]) and not names_a_type(case["answer"]), case["id"]
    answers = {(c["expect"]["scale"] is not None, bool(c["expect"].get("ruled_out"))) for c in CASES if c["proposes"]}
    assert answers == {(True, False), (False, True), (False, False)}  # yes, no, not sure
    assert len({c["language"] for c in CASES}) >= 8


@pytest.mark.parametrize("case", [c for c in CASES if c["proposes"]], ids=lambda c: c["id"])
def test_the_code_reads_thai_and_english_and_leaves_the_rest_to_quotes(case):
    readable = case["language"] in ("Thai", "English")
    assert code_reads(case["message"]) == readable
    if readable:  # the observation the app should propose is one the code itself finds
        assert observed_scale(case["message"])[0] == case["proposes"]


def scripted(case, first_args, answer_args, tools, registry):
    script = [[chunk(call("classify_event", **first_args))],
              [chunk(call("classify_event", **answer_args))], [chunk(types.Part(text="The models say ..."))]]
    calls = {"n": 0}
    with mock.patch("src.agent.make_gemini_client", return_value=FakeClient(script)), \
         mock.patch("src.agent.gemini_models", return_value=["model-a"]):
        return run_case(case, "AIzaSyFAKE", tools, registry, calls)


def test_thai_yes_is_scored_right(tools, registry):
    case = BY_ID["th-shake-yes"]
    args = dict(country="Thailand", start_month=5, duration_days=1)
    record = scripted(case, args, args, tools, registry)
    assert record["proposed"] == "Moment Magnitude" and record["question"].startswith("จากที่เล่าว่า")
    assert all(v is not False for v in record["checks"].values()), record["checks"]
    assert record["event"]["magnitude_scale"] == "Moment Magnitude"


def test_thai_no_is_scored_right_and_a_wrong_reading_fails(tools, registry):
    case = BY_ID["th-shake-no"]
    args = dict(country="Thailand", start_month=5, duration_days=1)
    record = scripted(case, args, args, tools, registry)
    assert record["ruled_out"] == ["Moment Magnitude"] and record["checks"]["ruled_out_ok"]

    wrong_month = dict(args, start_month=6)
    record = scripted(case, args, wrong_month, tools, registry)
    assert record["checks"]["month_ok"] is False
