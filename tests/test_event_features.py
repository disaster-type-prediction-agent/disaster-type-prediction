"""Encoding a described event into the models' 245 features, and country matching."""
from src.event_features import (
    RAW_ATTRIBUTES, attribute_columns, build_features, event_from_case, make_event_case,
)
from src.llm_tasks import match_country


def test_round_trip_reproduces_the_test_bank_features(registry):
    """Recover each case's 6 attributes, re-encode them: the features must come back exactly."""
    names = registry.metadata["feature_names"]
    for case in registry.cases[:500]:
        rebuilt = build_features(event_from_case(case), names)
        worst = max(abs(rebuilt[k] - case["features"][k]) for k in names)
        assert worst < 1e-4, (case["case_id"], worst)


def test_event_case_predicts_like_a_test_case(registry):
    case = registry.cases[3]
    names = registry.metadata["feature_names"]
    event_case = make_event_case(event_from_case(case), names, "EVENT-1")
    model = registry.get_model("Random Forest")
    assert model.predict(event_case["features"])["predicted_class"] == case["evaluations"]["Random Forest"]["predicted_class"]


def test_missing_values_follow_the_notebook_rules(registry):
    names = registry.metadata["feature_names"]
    f = build_features({"country": "Atlantis"}, names)
    assert f["start_month"] == 1.0            # missing month -> 1
    assert f["event_duration_log_norm"] == 0  # missing duration -> 0
    assert f["magnitude_missing"] == 1.0 and f["magnitude_scaled"] == 0
    assert f["magnitude_scale_Unknown"] == 1.0
    assert not any(v for k, v in f.items() if k.startswith(("country_", "region_")))  # unknown country: no one-hot


def test_values_are_scaled_like_training(registry):
    names = registry.metadata["feature_names"]
    f = build_features({"duration_days": 3287, "magnitude_value": 13025874, "start_month": 7,
                        "magnitude_scale": "Kph", "region": "Asia", "country": "Thailand"}, names)
    assert abs(f["event_duration_log_norm"] - 1.0) < 1e-9  # the longest event in the data -> 1
    assert abs(f["magnitude_scaled"] - 1.0) < 1e-9
    assert f["start_month"] == 7 and f["magnitude_scale_Kph"] == 1 and f["region_Asia"] == 1 and f["country_Thailand"] == 1


def test_true_type_is_stored_when_given(registry):
    names = registry.metadata["feature_names"]
    unknown = make_event_case({}, names, "EVENT-1", truth=None, class_names=registry.class_names)
    known = make_event_case({}, names, "EVENT-2", truth="Storm", class_names=registry.class_names)
    assert unknown["ground_truth_class"] is None and unknown["ground_truth_id"] is None
    assert known["ground_truth_class"] == "Storm" and known["ground_truth_id"] == registry.class_names.index("Storm")


def test_six_attributes_cover_all_245_features(registry):
    names = registry.metadata["feature_names"]
    covered = [c for _, cols, _ in RAW_ATTRIBUTES for c in attribute_columns(cols, names)]
    assert len(RAW_ATTRIBUTES) == 6 and len(names) == 245
    assert sorted(covered) == sorted(names), "each feature belongs to exactly one attribute"


def test_country_matching(registry):
    countries = [n[len("country_"):] for n in registry.metadata["feature_names"] if n.startswith("country_")]
    assert match_country("thailand", countries) == "Thailand"
    assert match_country("USA", countries) == "United States of America"
    assert match_country("Vietnam", countries) == "Viet Nam"
    assert match_country("Turkey", countries) == "Türkiye"
    assert match_country("Iran", countries) == "Iran (Islamic Republic of)"
    assert match_country("Bangladsh", countries) == "Bangladesh"  # small typo
    assert match_country("Atlantis", countries) is None
    assert match_country(None, countries) is None


def test_compare_events_tolerates_rounding_but_not_wrong_values():
    from src.event_features import compare_events
    typed = {"country": "Thailand", "region": "Asia", "start_month": 11, "duration_days": 3,
             "magnitude_value": 1234.0, "magnitude_scale": "Km2"}
    assert compare_events(typed, {**typed, "magnitude_value": 1200, "duration_days": 4})["correct"] == 6
    result = compare_events(typed, {**typed, "start_month": 10, "magnitude_scale": None})
    assert result["correct"] == 4 and {w["attribute"] for w in result["wrong"]} == {"Start month", "Magnitude scale"}
    assert compare_events({"magnitude_scale": "Unknown"}, {"magnitude_scale": None})["correct"] == 6
