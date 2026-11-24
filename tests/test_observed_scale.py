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


def test_the_app_shows_its_question_without_asking_gemini_again(tools):
    """Live app: asked to copy the question, Gemini named the type ("ขนาดของแผ่นดินไหว"), then reworded it and dropped
    the averaging line. The app now shows its own question and ends the turn: no second Gemini request."""
    from tests.test_gemini_agent import call, chunk, run

    session = SessionCases()
    session.message = EXAMPLE
    # A second scripted reply that would name the type, if the app ever asked Gemini for one.
    from google.genai import types
    events, history, client = run(tools, [[chunk(call("classify_event", country="Thailand", duration_days=0.007))],
                                          [chunk(types.Part(text="เป็นการวัดขนาดของแผ่นดินไหวใช่ไหมครับ"))]],
                                  question=EXAMPLE, session=session)
    tool = next(p for k, p in events if k == "tool")
    question = tool["result"]["question"]
    assert len(client.requests) == 1
    assert "".join(p for k, p in events if k == "text") == question
    assert "แผ่นดินไหว" not in question and "ระบบจะเฉลี่ย" in question
    assert not [p for k, p in events if k == "error"]
    assert [c.role for c in history] == ["user", "model", "user", "model"] and history[-1].parts[0].text == question

    # The answer goes to Gemini as usual, and the models run.
    session.message = "ไม่แน่ใจครับ"
    events, _, client = run(tools, [[chunk(call("classify_event", country="Thailand"))],
                                    [chunk(types.Part(text="Random Forest predicted Drought."))]],
                            question="ไม่แน่ใจครับ", history=history, session=session)
    assert len(client.requests) == 2 and next(p for k, p in events if k == "tool")["audit"]["case"]["scale_not_given"]


def test_the_source_line_says_the_app_wrote_the_question():
    """Live app: under the app's own question the line still read "...and Gemini's general knowledge"."""
    from src.auditor_agent import agent_sources
    from src.event_features import confirm_question
    question = confirm_question("Moment Magnitude", "บ้านสั่น", ["the month it started"], EXAMPLE)
    step = {"detail": '`classify_event` {"country": "Thailand"}', "is_error": False,
            "result": {"status": "need_more_details", "question": question}}
    line = agent_sources([step], question, "Gemini")
    assert "written by the app" in line and "general knowledge" not in line

    answered = {**step, "result": {"status": "classified"}}
    assert agent_sources([answered], "Random Forest predicted Drought.", "Gemini").startswith("Based on 1 tool result")
    assert agent_sources([], "hi", "Gemini").startswith("No tools used")


def test_no_and_not_sure_are_different_answers():
    from src.event_features import denies
    for no in ["ไม่ใช่ครับ", "ไม่ใช่", "no", "Nope, it wasn't", "wrong", "ไม่ถูกครับ"]:
        assert denies(no), no
    for not_no in ["ไม่แน่ใจครับ", "ไม่ทราบ", "no idea", "I don't know", "not sure", "no, I'm not sure", "ใช่ครับ", None]:
        assert not denies(not_no), not_no


def test_no_leaves_the_rejected_scale_out_of_the_averaging(tools, registry):
    """Live app: after "ไม่ใช่ครับ" the mix still held Moment Magnitude 3%, the scale the user had just rejected."""
    session = SessionCases()
    classify(tools, session, EXAMPLE, country="Thailand")
    result, _, audit, _ = classify(tools, session, "ไม่ใช่ครับ", country="Thailand")
    case = audit["case"]
    assert case["scale_not_given"] and case["scale_ruled_out"] == ["Moment Magnitude"]
    mix = audit["models"]["Random Forest"]["prediction"]["scale_mix"]
    assert "Moment Magnitude" not in [m["scale"] for m in mix] and abs(sum(m["weight"] for m in mix) - 1) < 0.01
    assert "left out of the averaging" in result["note"]
    assert "Moment Magnitude ruled out by the user" in result["attributes"]["Magnitude scale"]

    # "ไม่แน่ใจ" rules nothing out: Moment Magnitude stays a possibility.
    session = SessionCases()
    classify(tools, session, EXAMPLE, country="Thailand")
    _, _, audit, _ = classify(tools, session, "ไม่แน่ใจครับ", country="Thailand")
    assert "scale_ruled_out" not in audit["case"]
    assert "Moment Magnitude" in [m["scale"] for m in audit["models"]["Random Forest"]["prediction"]["scale_mix"]]


ENGLISH = "In Thailand in 2014 my house suddenly shook for more than 10 minutes and was slightly damaged"


def test_english_observation_with_an_adverb_is_asked(tools):
    """Live app: "my house suddenly shook" was not seen as an observation, so Gemini's Moment Magnitude ran unasked."""
    assert observed_scale(ENGLISH) == ("Moment Magnitude", "house suddenly shook")
    assert observed_scale("the ground was violently shaking")[0] == "Moment Magnitude"
    assert observed_scale("my house really shook all night")[0] == "Moment Magnitude"
    assert observed_scale("the house was shaking")[0] == "Moment Magnitude"
    assert observed_scale("he shook the house keys") is None

    session = SessionCases()
    result, _, audit, _ = classify(tools, session, ENGLISH, country="Thailand", duration_days=10,
                                   magnitude_scale="Moment Magnitude")
    assert audit is None and result["observed"]["words"] == "house suddenly shook"
    assert "wind" not in result["question"] and "size of the ground shaking" in result["question"]


def test_a_duration_left_in_minutes_is_corrected(tools):
    """Live app: Gemini passed duration_days=10 for "more than 10 minutes"."""
    session = SessionCases()
    classify(tools, session, ENGLISH, country="Thailand", duration_days=10)
    result, _, audit, _ = classify(tools, session, "yes, it was in May", country="Thailand", start_month=5,
                                   duration_days=10)
    assert audit["case"]["event"]["duration_days"] == 0.01
    assert "replaced by 0.01 days" in result["note"]

    # A figure close to the user's (0.007 for "10 กว่านาที") is left as the LLM read it.
    session = SessionCases()
    classify(tools, session, EXAMPLE, country="Thailand", duration_days=0.007)
    _, _, audit, _ = classify(tools, session, "ใช่ครับ เดือนพฤษภาคม", country="Thailand", start_month=5,
                              duration_days=0.007)
    assert audit["case"]["event"]["duration_days"] == 0.007


JA = "2014年にタイで、突然家が10分以上揺れて、家が少し壊れました"


def test_quotes_back_values_in_languages_the_code_cannot_read(tools):
    """Live Gemini: in Japanese it passed Moment Magnitude for "家が揺れて" unasked, and "5月" was dropped as no month."""
    from src.event_features import code_reads, quoted
    assert not code_reads(JA) and not code_reads("Sí, fue en mayo") and code_reads("Nothing was measured")
    assert quoted("揺れて", JA) and quoted("tháng Năm", "Đúng, vào tháng năm") and not quoted("5月", JA)
    assert not quoted("1", "in 2014") and not quoted("", JA)

    session = SessionCases()
    # No quote for the scale or the month: both dropped; the duration with its quote stays.
    result, _, audit, _ = classify(tools, session, JA, country="Thailand", magnitude_scale="Moment Magnitude",
                                   start_month=1, duration_days=0.007, duration_quote="10分以上")
    assert audit is None and "observed" not in result and "start_month" not in result["given_so_far"]
    assert result["given_so_far"]["duration_days"] == 0.007

    # With the user's words quoted, the scale is an observation the user must confirm, as "บ้านสั่น" is.
    session = SessionCases()
    result, _, audit, _ = classify(tools, session, JA, country="Thailand", magnitude_scale="Moment Magnitude",
                                   scale_quote="家が10分以上揺れて")
    assert audit is None and result["observed"]["words"] == "家が10分以上揺れて"
    _, _, audit, _ = classify(tools, session, "はい、5月でした", country="Thailand", magnitude_scale="Moment Magnitude",
                              scale_quote="家が10分以上揺れて", answer="yes", answer_quote="はい",
                              start_month=5, start_month_quote="5月")
    case = audit["case"]
    assert case["event"]["magnitude_scale"] == "Moment Magnitude" and case["event"]["start_month"] == 5
    assert case["scale_observed"]["words"] == "家が10分以上揺れて"


def test_a_made_up_answer_quote_does_not_confirm(tools):
    session = SessionCases()
    classify(tools, session, JA, country="Thailand", magnitude_scale="Moment Magnitude", scale_quote="揺れて")
    _, _, audit, _ = classify(tools, session, "いいえ", country="Thailand", answer="yes", answer_quote="はい")
    assert audit["case"]["scale_not_given"] and "scale_observed" not in audit["case"]

    session = SessionCases()
    classify(tools, session, JA, country="Thailand", magnitude_scale="Moment Magnitude", scale_quote="揺れて")
    _, _, audit, _ = classify(tools, session, "いいえ", country="Thailand", answer="no", answer_quote="いいえ")
    assert audit["case"]["scale_ruled_out"] == ["Moment Magnitude"]


def test_quotes_must_mean_the_value_not_just_be_there(tools):
    """Live Gemini: month quoted as "2014年"/"Năm 2014", magnitude_value 7 with no figure, 10 days for "十多分钟"."""
    session = SessionCases()
    result, _, audit, _ = classify(tools, session, JA, country="Thailand", start_month=5, start_month_quote="2014年",
                                   magnitude_scale="Moment Magnitude", magnitude_value=7, scale_quote="揺れて")
    assert audit is None and "start_month" not in result["given_so_far"]
    assert "magnitude_value" not in result["given_so_far"]

    vi = "Năm 2014 ở Thái Lan, nhà tôi đột nhiên rung lắc hơn 10 phút"
    result, _, _, _ = classify(tools, SessionCases(), vi, country="Thailand", start_month=1, start_month_quote="Năm 2014")
    assert "start_month" not in result["given_so_far"]

    zh = "2014年在泰国，房子突然摇晃了十多分钟，房子有点损坏"
    result, _, _, _ = classify(tools, SessionCases(), zh, country="Thailand", duration_days=10,
                               duration_amount=10, duration_unit="minutes", duration_quote="十多分钟")
    assert result["given_so_far"]["duration_days"] == round(10 / 1440, 4)

    # A figure the user wrote stays.
    result, _, audit, _ = classify(tools, SessionCases(), "2014年5月、タイでマグニチュード6.3", country="Thailand",
                                   start_month=5, start_month_quote="5月", magnitude_value=6.3,
                                   magnitude_scale="Moment Magnitude", scale_quote="マグニチュード6.3")
    assert result["given_so_far"].get("magnitude_value") == 6.3


def test_a_figure_needs_the_measurement_words_and_nothing_measured_needs_readable_words(tools):
    """Live Gemini: magnitude_value 10 from "10 minutos"; nothing_measured quoted as "家が少し壊れました"."""
    es = "En Tailandia en 2014 mi casa de repente tembló durante más de 10 minutos y sufrió daños leves"
    session = SessionCases()
    result, _, _, _ = classify(tools, session, es, country="Thailand", magnitude_scale="Moment Magnitude",
                               magnitude_value=10, scale_quote="tembló", duration_amount=10, duration_unit="minutes",
                               duration_quote="más de 10 minutos")
    assert "magnitude_value" not in result["given_so_far"] and result["given_so_far"]["duration_days"] > 0
    # The answer turn: the duration checked on the first message is kept without a new quote.
    _, _, audit, _ = classify(tools, session, "Sí, fue en mayo", country="Thailand", magnitude_scale="Moment Magnitude",
                              scale_quote="tembló", answer="yes", answer_quote="Sí", start_month=5,
                              start_month_quote="mayo", duration_amount=10, duration_unit="minutes")
    assert audit["case"]["event"]["duration_days"] == round(10 / 1440, 4)
    assert audit["case"]["event"]["magnitude_value"] is None

    result, _, audit, _ = classify(tools, SessionCases(), JA + " 5月", country="Thailand", start_month=5,
                                   start_month_quote="5月", duration_amount=10, duration_unit="minutes",
                                   duration_quote="10分以上", nothing_measured=True,
                                   nothing_measured_quote="家が少し壊れました")
    # Not taken as "nothing measured": the scale is still missing, so the app asks instead of running on Unknown.
    assert audit is None and "magnitude_scale" not in result["given_so_far"]
    assert "nothing_measured was ignored" in result.get("note", "") or result["status"] == "need_more_details"


def test_typed_eval_regressions_after_the_quote_checks(tools):
    """Typed eval rerun: "1,675 วัน" was read as 675 days; "ไม่มีการระบุตัวเลข" came back as magnitude_value 0."""
    msg = "เกิดเหตุการณ์ในประเทศมอริเตาเนีย ทวีปแอฟริกา ตั้งแต่เดือนมิถุนายน เป็นเวลานานถึง 1,675 วัน โดยมีการบันทึกพื้นที่ได้รับผลกระทบด้วย"
    assert duration_from_text(msg) == 1675
    _, _, audit, _ = classify(tools, SessionCases(), msg, country="Mauritania", start_month=6, duration_days=1675,
                              magnitude_scale="Km2")
    assert audit["case"]["event"]["duration_days"] == 1675

    msg = "เกิดเหตุการณ์ขึ้นที่ประเทศเอสวาตินีในทวีปแอฟริกาเมื่อเดือนกรกฎาคมที่ผ่านมา โดยมีการบันทึกพื้นที่ได้รับผลกระทบแต่ไม่มีการระบุตัวเลข"
    result, _, audit, _ = classify(tools, SessionCases(), msg, country="Eswatini", start_month=7,
                                   magnitude_scale="Km2", magnitude_value=0)
    assert (audit or {}).get("case", {}).get("event", {}).get("magnitude_value") is None
    assert "magnitude_value 0 was dropped" in result.get("note", "")
    # A zero the user wrote stays.
    _, _, audit, _ = classify(tools, SessionCases(), "Peru in March, 0 people vaccinated, it lasted 3 days",
                              country="Peru", start_month=3, duration_days=3, magnitude_scale="Vaccinated",
                              magnitude_value=0)
    assert audit["case"]["event"]["magnitude_value"] == 0
