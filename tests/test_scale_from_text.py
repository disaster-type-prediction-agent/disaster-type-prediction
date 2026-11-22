"""A message that names what was measured but gives no figure still sets the magnitude scale.

In the typed evaluation this was the main cause of typed messages going wrong: the reader left the
scale empty, the models saw "Unknown", and e.g. an Extreme Temperature case became Landslide."""
from types import SimpleNamespace
from unittest import mock

import pytest

from src import llm_tasks
from src.auditor_agent import SessionCases
from src.event_features import event_from_case, scale_from_text


@pytest.mark.parametrize("text, scale", [
    ("In Argentina, temperatures were recently reported for January without any specific figures.", "°C"),
    ("In Indonesia, a vaccination drive was reported in May.", "Vaccinated"),
    ("อุณหภูมิสูงถึง 45 องศา", "°C"),
    ("ลมแรง 120 กม./ชม. ที่ภาคใต้", "Kph"),
    ("ขนาด 6.2 ริกเตอร์ ในเดือนมีนาคม", "Moment Magnitude"),
    ("the affected area was reported, no figure", "Km2"),
    ("พื้นที่ประสบภัยกว้างมาก", "Km2"),
    ("In August, in Portugal, lasting 4 days, though the affected area was not recorded.", "Km2"),  # not "gust"
    ("Vaccinations were reported", "Vaccinated"),
    ("about 40 degrees", "°C"),
])
def test_names_the_measured_quantity(text, scale):
    assert scale_from_text(text) == scale


@pytest.mark.parametrize("text", [
    "พื้นที่ภาคใต้ของไทย เดือนพฤศจิกายน",          # "area" as a place, not a measurement
    "In the area around Bangkok, it lasted 3 days",
    "Something was measured but no number was recorded",
    "wind speed and temperature were both reported",  # two quantities: don't guess
    "a massacre in August in Triumph county",          # "acre", "gust", "mph" inside other words
    "",
    None,
])
def test_no_scale_when_unclear(text):
    assert scale_from_text(text) is None


def classify(tools, case, message):
    event = event_from_case(case)
    args = {k: event[k] for k in ("country", "start_month", "duration_days") if event.get(k) is not None}
    session = SessionCases()
    session.message, session.asked_details_on = message, "an earlier question"  # already asked: run regardless
    result, _, audit, _ = tools.execute("classify_event", args, ["Random Forest"], session)
    return result, audit


def test_classify_event_recovers_the_scale_and_the_right_answer(tools, registry):
    case = registry.get_case("CASE-1799")  # Extreme Temperature, °C, no figure recorded
    assert case["ground_truth_class"] == "Extreme Temperature"
    message = "In Argentina, temperatures were recently reported for January without any specific figures."

    result, audit = classify(tools, case, message)
    assert audit["case"]["event"]["magnitude_scale"] == "°C"
    assert "read from the message" in result["note"]
    assert audit["models"]["Random Forest"]["prediction"]["predicted_class"] == "Extreme Temperature"

    _, without = classify(tools, case, "")  # what happened before: the scale stayed Unknown
    assert without["models"]["Random Forest"]["prediction"]["predicted_class"] != "Extreme Temperature"


def test_the_llm_reading_wins_over_the_keywords(tools, registry):
    session = SessionCases()
    session.message = "temperatures were reported"
    _, _, audit, _ = tools.execute("classify_event", {"country": "Thailand", "magnitude_scale": "Kph", "start_month": 11},
                                   ["Random Forest"], session)
    assert audit["case"]["event"]["magnitude_scale"] == "Kph"


def test_routed_event_gets_the_scale_too():
    """Agent mode off: the router's event goes through the same safety net."""
    route = {"intent": "new_event", "event": {"country": "Argentina", "magnitude_scale": None}}
    with mock.patch("src.llm_tasks._generate_json", return_value=SimpleNamespace(model_dump=lambda: route)):
        out = llm_tasks.route_question("k", "Argentina, January, temperatures reported", ["Storm"], ["Argentina"])
    assert out["event"]["magnitude_scale"] == "°C"


def test_writer_must_name_the_quantity_when_no_figure_was_recorded():
    event = {"country": "Algeria", "start_month": 2, "magnitude_scale": "Moment Magnitude", "magnitude_value": None}
    vague = SimpleNamespace(message="In Algeria in February, the size was measured but no number was recorded.")
    named = SimpleNamespace(message="In Algeria in February; its magnitude was measured but not recorded.")
    with mock.patch("src.llm_tasks._generate_json", side_effect=[vague, named]) as gen:
        assert llm_tasks.write_scenario("k", event, "English") == named.message
    assert gen.call_count == 2


def test_unknown_magnitude_names_no_scale():
    from src.event_features import scale_from_text
    assert scale_from_text("an intense event struck with unknown magnitude") is None
    assert scale_from_text("the magnitude was not recorded") is None
    assert scale_from_text("a magnitude 6.1 earthquake") == "Moment Magnitude"
