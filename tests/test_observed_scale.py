"""What the user saw or felt ("บ้านสั่น") is proposed as a scale, and only the user's yes puts it into the models."""
import re

from src.auditor_agent import SessionCases
from src.event_features import OBSERVATION_WORDS, TYPE_WORDS, confirms, duration_from_text, observed_scale

EXAMPLE = "ที่ไทยในปี 2014 อยู่ๆ บ้านสั่นเป็นเวลา 10 กว่านาที ทำให้บ้านพังนิดหน่อย"


def test_observations_are_never_disaster_names():
    for scale, patterns in OBSERVATION_WORDS.items():
        joined = re.compile("|".join(patterns))
        for words in TYPE_WORDS.values():
            for word in words:
                assert not joined.fullmatch(word), (scale, word)
    assert observed_scale("แผ่นดินไหวที่เชียงราย") is None   # the type's name is not an observation
    assert observed_scale("an earthquake in Chile") is None


def test_what_counts_as_an_observation():
    assert observed_scale(EXAMPLE) == ("Moment Magnitude", "บ้านสั่น")
    assert observed_scale("strong winds tore roofs off in Manila")[0] == "Kph"
    assert observed_scale("อากาศร้อนมากจนนอนไม่ได้")[0] == "°C"
    assert observed_scale("ท่วมหลายหมู่บ้าน")[0] == "Km2"
    assert observed_scale("หนาวสั่นทั้งคืน") is None                    # shivering is not the ground shaking
    assert observed_scale("the ground was shaking and strong winds") is None  # two quantities: unclear
    assert observed_scale("a slope collapsed and houses were destroyed") is None
    assert duration_from_text("บ้านสั่นเป็นเวลา 10 กว่านาที") == 0.01


def test_yes_and_no():
    for yes in ["ใช่ครับ", "ใช่ ประมาณเดือนพฤษภา", "ใช่ แต่ไม่รู้เดือน", "น่าจะใช่ครับ", "ถูกต้องครับ", "yes, in May", "correct"]:
        assert confirms(yes), yes
    for no in ["ไม่ใช่", "ไม่แน่ใจครับ", "ไม่ทราบครับ", "ไม่น่าจะใช่", "no", "I don't know", "not sure",
               "I don't know any other details; that's all I have.", None]:
        assert not confirms(no), no


def classify(tools, session, message, **args):
    session.message = message
    return tools.execute("classify_event", args, ["Random Forest"], session)


def test_the_users_example_is_asked_then_confirmed(tools):
    session = SessionCases()
    result, summary, audit, _ = classify(tools, session, EXAMPLE, country="Thailand", duration_days=0.007)
    assert audit is None and result["status"] == "need_more_details" and summary == "Asked to confirm what was observed"
    assert result["observed"]["words"] == "บ้านสั่น" and "magnitude" in result["observed"]["measured_quantity"]
    assert "the month it started" in result["missing"]  # asked anyway: a missing month would become January
    assert "Earthquake" not in result["instruction"]

    result, _, audit, _ = classify(tools, session, "ใช่ครับ ประมาณเดือนพฤษภาคม", country="Thailand", start_month=5)
    case = audit["case"]
    assert case["event"]["magnitude_scale"] == "Moment Magnitude" and not case["scale_not_given"]
    assert case["scale_observed"] == {"scale": "Moment Magnitude", "words": "บ้านสั่น"}
    assert "confirmed by the user" in result["attributes"]["Magnitude scale"]
    assert audit["models"]["Random Forest"]["prediction"]["predicted_class"] == "Earthquake"
    assert session.proposed_scale is None


def test_not_sure_keeps_the_scale_not_given(tools):
    session = SessionCases()
    classify(tools, session, EXAMPLE, country="Thailand")
    result, _, audit, _ = classify(tools, session, "ไม่แน่ใจครับ", country="Thailand")
    case = audit["case"]
    assert case["scale_not_given"] and "scale_observed" not in case
    assert "did not confirm" in result["note"]


def test_nothing_measured_answer_beats_the_proposal(tools):
    session = SessionCases()
    classify(tools, session, "Something in Peru in March; the house shook for a minute", country="Peru", start_month=3)
    _, _, audit, _ = classify(tools, session, "Nothing was measured for it.", country="Peru", start_month=3,
                              nothing_measured=True)
    assert audit["case"]["event"]["magnitude_scale"] == "Unknown"


def test_the_llm_may_not_turn_an_observation_into_a_scale(tools):
    """Gemini passing Moment Magnitude for 'บ้านสั่น' on its own is set aside and the user is asked instead."""
    session = SessionCases()
    result, _, audit, _ = classify(tools, session, EXAMPLE, country="Thailand", start_month=5,
                                   magnitude_scale="Moment Magnitude")
    assert audit is None and result["observed"]["words"] == "บ้านสั่น"

    # After the user's yes, the same argument stands.
    _, _, audit, _ = classify(tools, session, "yes", country="Thailand", start_month=5, magnitude_scale="Moment Magnitude")
    assert audit["case"]["scale_observed"]["scale"] == "Moment Magnitude"


def test_named_quantity_needs_no_confirmation(tools):
    msg = "ที่ไทย บ้านสั่น ขนาด 6.1 ริกเตอร์ เดือนพฤษภาคม"
    result, _, audit, _ = classify(tools, SessionCases(), msg, country="Thailand", start_month=5, magnitude_value=6.1)
    assert audit and audit["case"]["event"]["magnitude_scale"] == "Moment Magnitude"
    assert "scale_observed" not in audit["case"]


def test_a_month_the_user_never_named_is_dropped(tools):
    """Live Gemini passed start_month=1 for the example, which names a year but no month, so nobody asked for it."""
    from src.event_features import mentions_month
    assert not mentions_month(EXAMPLE) and not mentions_month("ไม่แน่ใจครับ")
    for said in ["in September", "เดือนพฤษภาคม", "ช่วง ก.ย.", "เดือน 9", "on 9/2011", "this Nov.", "ปลายมีนาคม"]:
        assert mentions_month(said), said

    session = SessionCases()
    result, _, audit, _ = classify(tools, session, EXAMPLE, country="Thailand", start_month=1, duration_days=0.01)
    assert audit is None and "the month it started" in result["missing"] and "start_month" not in result["given_so_far"]


def test_the_confirmation_question_is_written_by_the_app():
    from src.event_features import confirm_question, names_a_type
    th = confirm_question("Moment Magnitude", "บ้านสั่น", ["the month it started"], EXAMPLE)
    assert "บ้านสั่น" in th and "แรงสั่นสะเทือน" in th and "เดือนไหน" in th and not names_a_type(th)
    en = confirm_question("Kph", "strong winds", [], "strong winds in Manila")
    assert "wind speed" in en and not names_a_type(en)
    for scale in ("Moment Magnitude", "Kph", "°C", "Km2"):
        for text in (EXAMPLE, "in English"):
            assert not names_a_type(confirm_question(scale, "x", ["where it happened (country)"], text)), scale
    assert names_a_type("เป็นการวัดขนาดของแผ่นดินไหวใช่ไหมครับ") and not names_a_type("ใช่ไหมครับ")


def test_a_reply_naming_the_type_is_replaced_by_the_apps_question(tools):
    """Live app: Gemini asked "เป็นการวัดขนาดของแผ่นดินไหวใช่ไหมครับ", naming the type before the models ran."""
    from google.genai import types
    from tests.test_gemini_agent import call, chunk, run

    session = SessionCases()
    session.message = EXAMPLE
    events, history, _ = run(tools, [[chunk(call("classify_event", country="Thailand", duration_days=0.007))],
                                     [chunk(types.Part(text="เป็นการวัดขนาดของแผ่นดินไหวใช่ไหมครับ"))]],
                             question=EXAMPLE, session=session)
    text = ""
    for kind, payload in events:
        text = text + payload if kind == "text" else text[:-payload] if kind == "retract" else text
    tool = next(p for k, p in events if k == "tool")
    assert text == tool["result"]["question"] and not tool["result"]["question"].count("แผ่นดินไหว")
    assert history[-1].parts[0].text == text  # the transcript holds what the user saw

    # A reply that names no type is left as Gemini wrote it.
    session = SessionCases()
    session.message = EXAMPLE
    events, _, _ = run(tools, [[chunk(call("classify_event", country="Thailand"))],
                               [chunk(types.Part(text="บ้านสั่นนี้ วัดเป็นแรงสั่นสะเทือนใช่ไหมครับ"))]],
                       question=EXAMPLE, session=session)
    assert "".join(p for k, p in events if k == "text") == "บ้านสั่นนี้ วัดเป็นแรงสั่นสะเทือนใช่ไหมครับ"


def test_the_question_is_asked_even_when_gemini_fails(tools):
    """Live run: Gemini timed out right after classify_event asked, and the user saw only an error."""
    from tests.test_gemini_agent import call, chunk, run, server_error

    session = SessionCases()
    session.message = EXAMPLE
    events, history, _ = run(tools, [[chunk(call("classify_event", country="Thailand"))], server_error(), server_error()],
                             question=EXAMPLE, session=session)
    tool = next(p for k, p in events if k == "tool")
    assert "".join(p for k, p in events if k == "text") == tool["result"]["question"]
    assert not [p for k, p in events if k == "error"] and history[-1].role == "model"
