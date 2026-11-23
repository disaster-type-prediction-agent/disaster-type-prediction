"""A scale the user never stated is not "nothing was measured": the models are averaged over likely scales."""
from src.auditor_agent import SessionCases
from src.event_features import make_event_case
from src.scale_prior import ScalePrior

THAI_FLOOD = {"country": "Thailand", "region": "Asia", "start_month": 9, "duration_days": 30}


def test_prior_weights_are_a_distribution_led_by_local_scales(registry):
    w = registry.scale_prior.weights("Japan", "Asia", 3)
    assert abs(sum(w.values()) - 1) < 1e-9 and min(w.values()) >= 0.01
    assert next(iter(w)) == "Moment Magnitude"  # Japan in March: earthquakes dominate the records
    assert next(iter(registry.scale_prior.weights("Thailand", "Asia", 9))) in ("Km2", "Kph")
    assert abs(sum(registry.scale_prior.weights().values()) - 1) < 1e-9  # nothing known: the global mix


def test_prior_backs_off_for_an_unseen_country():
    prior = ScalePrior({"country": {}, "region": {"Asia": {"9": {"Km2": 8, "Kph": 2}}}})
    w = prior.weights("Atlantis", "Asia", 9)
    assert w["Km2"] > w["Kph"]


def test_not_given_is_averaged_but_unknown_is_not(registry):
    not_given = make_event_case(THAI_FLOOD, registry.feature_names, "EVENT-1")
    unknown = make_event_case({**THAI_FLOOD, "magnitude_scale": "Unknown"}, registry.feature_names, "EVENT-2")
    assert not_given["scale_not_given"] and not unknown["scale_not_given"]
    assert not_given["summary"]["magnitude_scale"] == "not given"

    averaged = registry.audit_case(not_given, ["Random Forest"])["models"]["Random Forest"]["prediction"]
    plain = registry.audit_case(unknown, ["Random Forest"])["models"]["Random Forest"]["prediction"]
    assert averaged["scale_mix"] and "scale_mix" not in plain
    assert abs(sum(averaged["probabilities"].values()) - 1) < 1e-2
    top2 = sorted(averaged["probabilities"], key=averaged["probabilities"].get)[-2:]
    assert set(top2) == {"Flood", "Storm"}  # was Epidemic when the gap was filled with "Unknown"
    assert plain["predicted_class"] in ("Epidemic", "Landslide")  # "nothing measured" keeps its meaning


def test_agent_runs_averaged_after_dont_know_but_unknown_after_nothing_measured(tools):
    session = SessionCases()
    session.message = "a disaster in Thailand in September for a month"
    result, _, audit, _ = tools.execute("classify_event", {"country": "Thailand", "start_month": 9, "duration_days": 30},
                                        ["Random Forest"], session)
    assert result["status"] == "need_more_details"

    session.message = "I don't know"
    result, _, audit, _ = tools.execute("classify_event", {"country": "Thailand", "start_month": 9, "duration_days": 30},
                                        ["Random Forest"], session)
    assert audit["case"]["scale_not_given"]
    assert result["attributes"]["Magnitude scale"].startswith("not given (models averaged")

    session = SessionCases()
    session.message = "Thailand, September, a month long, nothing was measured"
    _, _, audit, _ = tools.execute("classify_event", {"country": "Thailand", "start_month": 9, "duration_days": 30,
                                                      "nothing_measured": True}, ["Random Forest"], session)
    assert not audit["case"]["scale_not_given"] and audit["case"]["event"]["magnitude_scale"] == "Unknown"


def test_explain_prediction_matches_the_averaged_prediction(tools, registry):
    session = SessionCases()
    case = make_event_case(THAI_FLOOD, registry.feature_names, session.next_id())
    session.add(case)
    result, _, _, _ = tools.execute("explain_prediction", {"case_id": case["case_id"], "model": "Random Forest"},
                                    ["Random Forest"], session)
    averaged = registry.predict_case(case, "Random Forest")[0]["predicted_class"]
    assert result["model_predicted"] == averaged == result["explained_class"]
