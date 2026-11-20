"""ClaudeAuditorAgent's tool-use loop, driven by a scripted fake Anthropic client."""
from types import SimpleNamespace as NS
from unittest import mock

from src.auditor_agent import MAX_TOOL_ROUNDS, ClaudeAuditorAgent


def tool_use(i, name, inp):
    return NS(type="tool_use", id=f"toolu_{i}", name=name, input=inp)


def text(s):
    return NS(type="text", text=s)


class FakeStream:
    """One streamed response: text deltas, an optional unparseable-JSON failure, then the final message."""

    def __init__(self, script):
        self.script = script

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for t in self.script.get("text", []):
            yield NS(type="text", text=t)
        if self.script.get("raise"):
            raise ValueError("unparseable tool input")

    def get_final_message(self):
        return NS(stop_reason=self.script["stop"], content=self.script["content"])


def run(tools, scripts, history=None):
    queue = list(scripts)
    client = mock.MagicMock()
    client.beta.messages.stream.side_effect = lambda **kw: FakeStream(queue.pop(0))
    history = [] if history is None else history
    with mock.patch("anthropic.Anthropic", return_value=client):
        events = list(ClaudeAuditorAgent(tools, "test-key").run_turn(history, "question", ["Random Forest"]))
    return events, history, client


def assert_history_valid(history):
    """Every assistant tool_use is answered by a tool_result in the next message (the API 400s otherwise)."""
    for i, m in enumerate(history):
        if m["role"] == "assistant" and isinstance(m["content"], list):
            ids = {b.id for b in m["content"] if b.type == "tool_use"}
            if ids:
                answered = {r["tool_use_id"] for r in history[i + 1]["content"] if r.get("type") == "tool_result"}
                assert ids <= answered, f"unanswered tool calls at message {i}"


def test_tool_round_then_answer(tools):
    events, history, _ = run(tools, [
        {"stop": "tool_use", "content": [tool_use(1, "get_model_metrics", {}), tool_use(2, "get_significance", {})]},
        {"stop": "end_turn", "text": ["Light", "GBM"], "content": [text("LightGBM")]},
    ])
    assert [k for k, _ in events] == ["tool", "tool", "text", "text"]
    assert len(history[2]["content"]) == 2, "parallel results must go back in ONE user message"
    assert_history_valid(history)


def test_case_tools_surface_an_audit_card(tools):
    events, _, _ = run(tools, [
        {"stop": "tool_use", "content": [tool_use(1, "find_case", {"disaster_type": "Storm"})]},
        {"stop": "end_turn", "content": [text("ok")]},
    ])
    tool_event = events[0][1]
    assert tool_event["audit"] is not None and tool_event["models"] == ["Random Forest"]


def test_unparseable_json_retracts_partial_text_and_retries(tools):
    events, _, client = run(tools, [
        {"stop": "tool_use", "text": ["abc"], "raise": True, "content": []},
        {"stop": "end_turn", "text": ["ok"], "content": [text("ok")]},
    ])
    assert events == [("text", "abc"), ("retract", 3), ("text", "ok")]
    assert client.beta.messages.stream.call_count == 2


def test_invalid_tool_input_returns_an_error_result(tools):
    events, history, _ = run(tools, [
        {"stop": "tool_use", "content": [tool_use(1, "find_case", {"region": "Mars"})]},
        {"stop": "end_turn", "content": [text("sorry")]},
    ])
    assert events[0][1]["is_error"]
    result = history[2]["content"][0]
    assert result["is_error"] and "INVALID_JSON" in result["content"]


def test_refusal_never_runs_the_cut_off_tool(tools):
    events, history, _ = run(tools, [{"stop": "refusal", "content": [tool_use(1, "find_case", {})]}])
    assert events[-1][0] == "error"
    assert not any(kind == "tool" for kind, _ in events)
    assert_history_valid(history)


def test_truncated_tool_call_is_not_run(tools):
    events, history, _ = run(tools, [{"stop": "max_tokens", "content": [tool_use(1, "find_case", {})]}])
    assert events[-1][0] == "error" and not any(kind == "tool" for kind, _ in events)
    assert_history_valid(history)


def test_interrupted_turn_is_repaired_on_the_next_one(tools):
    history = [{"role": "user", "content": "q0"},
               {"role": "assistant", "content": [tool_use(5, "get_significance", {})]}]
    _, history, _ = run(tools, [{"stop": "end_turn", "content": [text("hi")]}], history=history)
    assert_history_valid(history)
    assert history[2]["content"][0]["tool_use_id"] == "toolu_5"


def test_tool_budget_stops_the_loop(tools):
    looping = {"stop": "tool_use", "content": [tool_use(9, "get_significance", {})]}
    events, history, client = run(tools, [looping] * (MAX_TOOL_ROUNDS + 1))
    assert events[-1][0] == "error"
    assert client.beta.messages.stream.call_count == MAX_TOOL_ROUNDS + 1
    assert_history_valid(history)


def test_request_shape(tools):
    _, _, client = run(tools, [{"stop": "end_turn", "content": [text("hi")]}])
    kwargs = client.beta.messages.stream.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["fallbacks"] == "default"
    assert kwargs["cache_control"] == {"type": "ephemeral"}
