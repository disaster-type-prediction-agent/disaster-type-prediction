"""Does an LLM answer say what the ML model actually said? Checked in code, with no extra LLM call.

The models make every prediction; the LLM only explains them. The real risk is an explanation that
drifts from the model: naming a different class, or quoting a percentage the model never produced.
"""
import re
from typing import Any, Dict, Iterable, List, Optional, Set

# Thai words the LLM may use instead of the English class name.
CLASS_WORDS = {
    "Drought": ["ภัยแล้ง", "แล้ง"],
    "Earthquake": ["แผ่นดินไหว"],
    "Epidemic": ["โรคระบาด"],
    "Extreme Temperature": ["อุณหภูมิสุดขั้ว", "คลื่นความร้อน", "ร้อนจัด", "หนาวจัด", "อุณหภูมิ"],
    "Flood": ["น้ำท่วม", "อุทกภัย"],
    "Landslide": ["ดินถล่ม", "ดินโคลนถล่ม"],
    "Storm": ["พายุ"],
    "Wildfire": ["ไฟป่า"],
}
PERCENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
# The level of a confidence interval ("95% CI") is a convention, not a figure the model produced.
# "73.4% confidence" alone is the model's own confidence and is still checked.
CONFIDENCE_LEVEL = re.compile(
    r"\d+(?:[.,]\d+)?\s*%\s*(?:CI\b|C\.I\.|confidence\s+(?:interval|level|band))"
    r"|ช่วงความเชื่อมั่น(?:ที่)?\s*\d+(?:[.,]\d+)?\s*%",
    re.IGNORECASE,
)
NUMBER = re.compile(r"\d+(?:\.\d+)?")
# A bound (">97%", "over 99%", "มากกว่า 96%") summarises the figures instead of quoting one: it is right when
# a model figure lies on the stated side of it ("all three above 97%" with 97.7, 99.9, 99.9).
_NUM = r"(\d+(?:[.,]\d+)?)\s*%"
LOWER_BOUND = re.compile(r"(?:>=?|≥|\b(?:over|above|more than|at least|greater than|exceeding)\s+|มากกว่า\s*|เกิน\s*|สูงกว่า\s*)" + _NUM,
                         re.IGNORECASE)
UPPER_BOUND = re.compile(r"(?:<=?|≤|\b(?:under|below|less than|at most)\s+|น้อยกว่า\s*|ต่ำกว่า\s*|ไม่ถึง\s*)" + _NUM,
                         re.IGNORECASE)
TOLERANCE = 0.6  # percentage points: covers rounding 88.8 -> 89


def mentions_class(text: str, cls: str) -> bool:
    lower = text.lower()
    return cls.lower() in lower or any(w in text for w in CLASS_WORDS.get(cls, []))


def numbers_in(obj: Any, out: Optional[Set[float]] = None) -> Set[float]:
    """Every number in a nested result, plus fractions as percentages (0.888 -> 88.8)."""
    out = set() if out is None else out
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.add(abs(float(obj)))
        if abs(obj) <= 1:
            out.add(abs(float(obj)) * 100)
    elif isinstance(obj, str):
        out.update(float(n) for n in NUMBER.findall(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            numbers_in(v, out)
    return out


def check_answer(text: str, audits: Iterable[Dict[str, Any]], sources: Iterable[Any] = (),
                 name_prediction: bool = True) -> Optional[Dict[str, Any]]:
    """Compare an answer with the model output it was based on.

    audits: case audits the answer is about. With name_prediction, each one's first-model prediction
            must be named (an answer about a newly classified case has to say what the model said).
    sources: anything else the answer may legitimately quote (tool results, benchmark figures, prompts).
    Returns None when there is nothing to check against, else {"ok", "issues"}.
    """
    audits = [a for a in audits if a]
    sources = [s for s in sources if s is not None]
    if not text.strip() or not (audits or sources):
        return None
    issues: List[str] = []
    if name_prediction:
        for audit in audits:
            model, result = next(iter(audit["models"].items()))
            predicted = result["prediction"]["predicted_class"]
            if not mentions_class(text, predicted):
                issues.append(f"doesn't name {model}'s prediction ({predicted}) for {audit['case']['case_id']}")

    allowed = numbers_in([{k: v for k, v in a.items() if k != "case"} for a in audits])
    for a in audits:
        numbers_in(a["case"].get("summary"), allowed)
        numbers_in(a["case"].get("event"), allowed)
    numbers_in(sources, allowed)
    body = CONFIDENCE_LEVEL.sub(" ", text)
    percents = {n for n in allowed if n <= 100}
    wrong_bounds = sorted(
        [f">{p:g}%" for p in (float(x.replace(",", ".")) for x in LOWER_BOUND.findall(body))
         if not any(n >= p - TOLERANCE for n in percents)]
        + [f"<{p:g}%" for p in (float(x.replace(",", ".")) for x in UPPER_BOUND.findall(body))
           if not any(n <= p + TOLERANCE for n in percents)])
    if wrong_bounds:
        issues.append("states " + ", ".join(wrong_bounds) + ", which no model figure matches")
    body = UPPER_BOUND.sub(" ", LOWER_BOUND.sub(" ", body))
    quoted = {float(p.replace(",", ".")) for p in PERCENT.findall(body)}
    unknown = sorted(p for p in quoted if not any(abs(p - n) <= TOLERANCE for n in allowed))
    if unknown:
        issues.append("quotes " + ", ".join(f"{p:g}%" for p in unknown) + " not found in the model output")
    if not audits and not quoted and not wrong_bounds:
        return None  # no case and no figures: nothing the model said to compare against
    return {"ok": not issues, "issues": issues}
