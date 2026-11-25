"""SHAP summed into the 6 attributes, filled-in defaults flagged, and the context Gemini gets."""
import pytest

from src.agent import DisasterAuditorLLM
from src.event_features import filled_attributes, make_event_case

PARTIAL = {"country": "Thailand", "region": "Asia", "magnitude_value": 120, "magnitude_scale": "Kph"}  # no month, no duration


@pytest.fixture(scope="module")
def partial_audit(registry):
    case = make_event_case(PARTIAL, registry.feature_names, "EVENT-1")
    return registry.audit_case(case, registry.available_models)


@pytest.mark.parametrize("model", ["Random Forest", "LightGBM", "XGBoost"])
def test_attribute_shap_adds_up_to_the_feature_shap(registry, model):
    case = registry.cases[7]
    shap = registry.get_model(model).explain(case["features"], top_k=len(registry.feature_names))
    rows = shap["by_attribute"]
    assert [r["attribute"] for r in rows] and len(rows) == 6
    assert abs(sum(r["shap"] for r in rows) - sum(c["shap_value"] for c in shap["top_all"])) < 1e-2  # SHAP is additive
    assert abs(sum(abs(r["share"]) for r in rows) - 100) < 1  # shares are % of the total


def test_explanations_never_cite_absent_one_hot_features(registry):
    model = registry.get_model("Random Forest")
    for case in registry.cases[:50]:
        for sentence in model.explain(case["features"])["human_explanations"]:
            assert "=0.0" not in sentence and "Feature '" not in sentence, sentence


def test_filled_attributes_are_flagged(partial_audit):
    assert filled_attributes(PARTIAL) == ["Start month", "Event duration"]
    assert filled_attributes({**PARTIAL, "start_month": 11, "duration_days": 3}) == []
    for m in partial_audit["models"].values():
        rows = {r["attribute"]: r for r in m["shap"]["by_attribute"]}
        assert rows["Start month"]["filled"] and rows["Event duration"]["filled"]
        assert not rows["Magnitude scale"]["filled"]
        for sentence in m["shap"]["human_explanations"]:
            if sentence.startswith(("Start month", "Event duration")):
                assert "wasn't given" in sentence


def test_shares_compare_across_models_unlike_raw_shap(partial_audit):
    """Raw SHAP for the same attribute differs ~20x between models (units differ); shares don't."""
    kph = {m: next(r for r in d["shap"]["by_attribute"] if r["attribute"] == "Magnitude scale")
           for m, d in partial_audit["models"].items()}
    assert kph["LightGBM"]["shap"] / kph["Random Forest"]["shap"] > 5
    assert all(r["share"] > 80 for r in kph.values())


def test_gemini_context_has_every_attribute_and_shap_share(partial_audit):
    context = DisasterAuditorLLM._case_context(partial_audit)
    for needle in ["Country: Thailand", "Magnitude value: 120", "Event duration: 0 days", "Start month: Jan (1)",
                   "NOT GIVEN", "SHAP by attribute", "Magnitude scale = Kph: +", "top probabilities"]:
        assert needle in context, needle


def test_why_not_the_true_type_on_a_misclassified_case(registry):
    case = next(c for c in registry.cases if not c["evaluations"]["Random Forest"]["is_correct"])
    m = registry.audit_case(case, ["Random Forest"])["models"]["Random Forest"]
    contrast = m["contrast"]
    assert contrast["class"] == case["ground_truth_class"] != m["prediction"]["predicted_class"]
    assert contrast["reason"] == "true type" and len(contrast["by_attribute"]) == 6
    assert contrast["human_explanations"] and all(contrast["class"] in s for s in contrast["human_explanations"])


def test_why_not_the_runner_up_on_a_correct_case(registry):
    case = next(c for c in registry.cases if c["evaluations"]["Random Forest"]["is_correct"])
    m = registry.audit_case(case, ["Random Forest"])["models"]["Random Forest"]
    probs = sorted(m["prediction"]["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
    assert m["contrast"]["reason"] == "runner-up" and m["contrast"]["class"] == probs[1][0]


def test_confirming_a_different_true_type_switches_the_contrast(registry):
    audit = registry.audit_case(make_event_case(PARTIAL, registry.feature_names, "EVENT-9"), ["Random Forest"])
    m = audit["models"]["Random Forest"]
    assert m["prediction"]["predicted_class"] == "Storm" and m["contrast"]["reason"] == "runner-up"
    audit["case"]["ground_truth_class"], audit["case"]["ground_truth_id"] = "Flood", registry.class_names.index("Flood")
    registry.refresh_contrast(audit)
    assert m["contrast"]["class"] == "Flood" and m["contrast"]["reason"] == "true type"


def test_gemini_context_includes_why_not(partial_audit):
    assert "Why not " in DisasterAuditorLLM._case_context(partial_audit)
