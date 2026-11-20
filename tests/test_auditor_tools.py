"""The six tools Claude can call, run against the real model bank."""
import pytest


def test_every_tool_runs(tools, registry):
    case_id = registry.cases[0]["case_id"]
    calls = [
        ("find_case", {"only_hard": True}),
        ("get_case", {"case_id": case_id.lower()}),  # ids are matched case-insensitively
        ("explain_prediction", {"case_id": case_id, "model": "LightGBM", "target_class": "Flood", "top_k": 3}),
        ("get_model_metrics", {}),
        ("get_significance", {}),
        ("get_error_breakdown", {"disaster_type": "Storm"}),
    ]
    for name, args in calls:
        assert tools.validate(name, args) is None, name
        result, summary, _, _ = tools.execute(name, args, ["Random Forest"])
        assert "error" not in result, (name, result)
        assert summary


def test_case_tools_return_a_card(tools):
    result, _, audit, models = tools.execute("find_case", {"disaster_type": "Flood"}, ["Random Forest"])
    assert result["ground_truth"] == "Flood"
    assert audit is not None and models == ["Random Forest"]


def test_misclassified_by_adds_that_model_and_really_missed(tools):
    _, _, audit, models = tools.execute(
        "find_case", {"misclassified_by": "XGBoost", "models": ["LightGBM"]}, ["Random Forest"]
    )
    assert "XGBoost" in models
    assert not audit["models"]["XGBoost"]["is_correct"]


def test_unknown_case_is_an_error_not_a_crash(tools):
    result, summary, audit, _ = tools.execute("get_case", {"case_id": "CASE-99999"}, ["Random Forest"])
    assert "error" in result and audit is None and "not found" in summary


def test_explain_defaults_to_the_predicted_class(tools, registry):
    case = registry.cases[0]
    result, _, _, _ = tools.execute("explain_prediction", {"case_id": case["case_id"], "model": "XGBoost"}, [])
    assert result["explained_class"] == case["evaluations"]["XGBoost"]["predicted_class"]
    assert len(result["pushes_toward"]) <= 5


def test_error_breakdown_counts_add_up(tools):
    result, _, _, _ = tools.execute("get_error_breakdown", {}, [])
    per_class = result["per_class"]
    assert sum(c["test_cases"] for c in per_class.values()) == result["total_cases"]
    for entry in per_class.values():
        for m in entry["models"].values():
            assert 0 <= m["errors"] <= entry["test_cases"]


@pytest.mark.parametrize(
    "name, args",
    [
        ("find_case", {"region": "Mars"}),
        ("find_case", {"only_hard": "yes"}),
        ("find_case", {"models": []}),
        ("get_case", {}),
        ("explain_prediction", {"case_id": "CASE-0001", "model": "RF"}),
        ("explain_prediction", {"case_id": "CASE-0001", "model": "XGBoost", "top_k": 50}),
        ("explain_prediction", {"case_id": "CASE-0001", "model": "XGBoost", "top_k": True}),
        ("get_model_metrics", {"model": "SVM"}),
        ("no_such_tool", {}),
        ("find_case", "not an object"),
    ],
)
def test_invalid_inputs_are_rejected(tools, name, args):
    assert tools.validate(name, args) is not None


def test_tool_schemas_match_the_registry(tools, registry):
    defs = {d["name"]: d for d in tools.definitions()}
    assert set(defs) == {"find_case", "get_case", "classify_event", "explain_prediction", "get_model_metrics",
                         "get_significance", "get_error_breakdown"}
    assert defs["explain_prediction"]["input_schema"]["properties"]["model"]["enum"] == registry.available_models
    assert all(d["eager_input_streaming"] for d in defs.values())
