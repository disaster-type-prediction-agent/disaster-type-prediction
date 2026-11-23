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


def test_answer_keeps_the_country_from_the_first_message(tools):
    """Live app: "ญี่ปุ่น ... มีนาคม 1 วัน", then "ไม่ทราบครับ" came back as region Asia with no country."""
    session = SessionCases()
    session.message = "มีเหตุการณ์เกิดขึ้นที่ญี่ปุ่นเดือนมีนาคม กินเวลา 1 วัน"
    result, _, audit, _ = tools.execute("classify_event", {"country": "Japan", "start_month": 3, "duration_days": 1},
                                        ["Random Forest"], session)
    assert result["status"] == "need_more_details"

    session.message = "ไม่ทราบครับ"
    result, _, audit, _ = tools.execute("classify_event", {"region": "Asia", "start_month": 3},
                                        ["Random Forest"], session)
    event = audit["case"]["event"]
    assert event["country"] == "Japan" and event["duration_days"] == 1
    assert "country" in result["note"] and session.asked_event is None
    assert audit["models"]["Random Forest"]["prediction"]["predicted_class"] == "Earthquake"


def test_a_new_place_in_the_answer_is_not_overwritten(tools):
    session = SessionCases()
    session.message = "something in Japan in March"
    tools.execute("classify_event", {"country": "Japan", "start_month": 3}, ["Random Forest"], session)
    session.message = "sorry, it was in Chile, I don't know what was measured"
    _, _, audit, _ = tools.execute("classify_event", {"country": "Chile", "start_month": 3}, ["Random Forest"], session)
    assert audit["case"]["event"]["country"] == "Chile"

    session = SessionCases()
    session.message = "something in Japan in March"
    tools.execute("classify_event", {"country": "Japan", "start_month": 3}, ["Random Forest"], session)
    session.message = "actually somewhere in Europe"
    _, _, audit, _ = tools.execute("classify_event", {"region": "Europe", "start_month": 3}, ["Random Forest"], session)
    assert audit["case"]["event"]["country"] is None and audit["case"]["event"]["region"] == "Europe"


def test_nothing_measured_needs_the_users_words():
    from src.event_features import says_nothing_measured
    for said in ["ไม่มีการวัดค่าใด ๆ ไว้", "ไม่ได้วัดอะไรเลย", "โดยไม่มีการบันทึกขนาดความรุนแรง",
                 "Nothing was measured for it", "no measurement was taken", "it wasn't measured"]:
        assert says_nothing_measured(said), said
    for not_said in ["ไม่ทราบครับ", "I don't know", "ไม่มีการบันทึกระยะเวลา", "the duration wasn't recorded", None]:
        assert not says_nothing_measured(not_said), not_said


def test_thai_country_names_are_read():
    import json
    from src.event_features import country_from_text, countries_from
    names = countries_from(json.load(open("models/metadata.json"))["feature_names"])
    assert country_from_text("มีเหตุการณ์เกิดขึ้นที่ญี่ปุ่นเดือนมีนาคม", names) == "Japan"
    assert country_from_text("เกิดที่ซูดานใต้", names) == "South Sudan"
    assert country_from_text("ที่สาธารณรัฐประชาธิปไตยคองโก", names) == "Democratic Republic of the Congo"
    assert country_from_text("เกิดเหตุที่ฮ่องกง", names) == "China, Hong Kong Special Administrative Region"
    assert country_from_text("ลาวาไหลลงมา", names) is None


def test_live_app_conversation_now_gives_japan_and_averages(tools):
    """The exact tool calls Gemini made in the live app: region only, then nothing_measured on 'ไม่ทราบครับ'."""
    session = SessionCases()
    session.message = "มีเหตุการณ์เกิดขึ้นที่ญี่ปุ่นเดือนมีนาคม กินเวลา 1 วัน"
    result, _, _, _ = tools.execute("classify_event", {"region": "Asia", "start_month": 3}, ["Random Forest"], session)
    assert result["status"] == "need_more_details" and result["given_so_far"]["country"] == "Japan"

    session.message = "ไม่ทราบครับ"
    result, summary, audit, _ = tools.execute("classify_event", {"region": "Asia", "start_month": 3, "nothing_measured": True},
                                              ["Random Forest"], session)
    assert summary.endswith("(Japan)")  # the step label shows the country the app settled on, not Gemini's argument
    case = audit["case"]
    assert case["event"]["country"] == "Japan" and case["scale_not_given"]
    assert "nothing_measured was ignored" in result["note"]
    assert audit["models"]["Random Forest"]["prediction"]["predicted_class"] == "Earthquake"
