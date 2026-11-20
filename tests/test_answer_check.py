"""The explanation check: an LLM answer must agree with what the ML model actually said."""
import pytest

from src.answer_check import check_answer, mentions_class
from src.event_features import make_event_case


@pytest.fixture(scope="module")
def audit(registry):
    case = make_event_case({"country": "Thailand", "region": "Asia", "magnitude_value": 120, "magnitude_scale": "Kph"},
                           registry.feature_names, "EVENT-1")
    return registry.audit_case(case, ["Random Forest"])


def kph_share(audit):
    return next(r["share"] for r in audit["models"]["Random Forest"]["shap"]["by_attribute"] if r["attribute"] == "Magnitude scale")


def test_faithful_answer_passes(audit):
    conf = audit["models"]["Random Forest"]["prediction"]["confidence"] * 100
    text = f"โมเดลทายว่าเป็นพายุ ({conf:.1f}%) เพราะหน่วย Kph มีผล {kph_share(audit):.0f}% ของหลักฐาน"
    assert check_answer(text, [audit]) == {"ok": True, "issues": []}


def test_wrong_class_is_flagged(audit):
    result = check_answer("The model predicted Flood.", [audit])
    assert not result["ok"] and "Storm" in result["issues"][0]


def test_invented_percentage_is_flagged(audit):
    result = check_answer("Storm, with 73.4% confidence.", [audit])
    assert not result["ok"] and "73.4%" in result["issues"][0]


def test_quoted_benchmark_figures_are_allowed(audit, registry):
    text = "Storm. Overall the Random Forest scores 91.7% accuracy."
    assert check_answer(text, [audit], [registry.leaderboard])["ok"]


def test_metrics_only_answer_is_checked_on_figures(registry):
    assert check_answer("Accuracy is 91.7%.", [], [registry.leaderboard])["ok"]
    assert not check_answer("Accuracy is 97.2%.", [], [registry.leaderboard])["ok"]
    assert check_answer("No numbers here.", [], [registry.leaderboard]) is None


def test_thai_class_names_count():
    assert mentions_class("น่าจะเป็นอุทกภัย", "Flood") and mentions_class("ไฟป่า", "Wildfire")
