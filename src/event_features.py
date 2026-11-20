"""Turn an event a user describes (country, month, duration, magnitude ...) into the models' 245 features.

The models were trained on 6 raw attributes, one-hot / scaled into 245 columns
(notebooks/02_data_EDA.ipynb):

    country          230 one-hot columns   country_<name>
    magnitude_scale    6 one-hot columns   magnitude_scale_<scale>   (missing -> "Unknown")
    region             5 one-hot columns   region_<region>
    start_month        1 number            1-12 (missing -> 1)
    event_duration     1 number            MinMax(log1p(days)), negatives clipped to 0, missing -> 0
    magnitude          2 numbers           MinMax(log1p(value)) + a missing flag (missing -> 0)

The two MinMax scalers were fitted on the full raw EM-DAT export before Volcanic-activity rows were
dropped. Their minimum is 0 (both inputs are clipped/filled to >= 0) and their maxima are below;
re-running the notebook's steps on the raw file with these constants reproduces every distinct
value in data/processed/disaster_processed_feature_selected.csv exactly.
"""
import calendar
import math
from typing import Any, Dict, Iterable, List, Optional

DURATION_LOG_MAX = 8.098034756176071   # log1p(3,287 days), the longest event in the raw data
MAGNITUDE_LOG_MAX = 16.38244832185021  # log1p(13,025,874), the largest magnitude in the raw data

MAGNITUDE_SCALES = ["Km2", "Kph", "Moment Magnitude", "Vaccinated", "°C", "Unknown"]
REGIONS = ["Africa", "Americas", "Asia", "Europe", "Oceania"]

# (attribute, its feature columns: exact names, or prefixes ending in "_", encoding)
RAW_ATTRIBUTES = [
    ("Country", ("country_",), "One-hot, one column per country"),
    ("Magnitude scale", ("magnitude_scale_",), "One-hot: Km2, Kph, Moment Magnitude, Vaccinated, °C, Unknown"),
    ("Region", ("region_",), "One-hot: Africa, Americas, Asia, Europe, Oceania"),
    ("Start month", ("start_month",), "Number, 1-12"),
    ("Event duration", ("event_duration_log_norm",), "log1p(days), scaled 0-1"),
    ("Magnitude value", ("magnitude_scaled", "magnitude_missing"), "log1p(value) scaled 0-1, plus a missing flag"),
]


def attribute_columns(columns, feature_names: Iterable[str]) -> List[str]:
    return [f for f in feature_names
            if any(f.startswith(c) if c.endswith("_") else f == c for c in columns)]


def countries_from(feature_names: Iterable[str]) -> List[str]:
    return [f[len("country_"):] for f in feature_names if f.startswith("country_")]


def _scale(value: Optional[float], log_max: float) -> float:
    if value is None or value != value:  # None or NaN
        return 0.0
    return min(math.log1p(max(float(value), 0.0)) / log_max, 1.0)


def unscale(norm: float, log_max: float) -> float:
    """Inverse of the MinMax(log1p(x)) transform: back to days or magnitude units."""
    return math.expm1(float(norm) * log_max)


def build_features(event: Dict[str, Any], feature_names: List[str]) -> Dict[str, float]:
    """event keys: country, region, start_month, duration_days, magnitude_value, magnitude_scale (any may be None)."""
    features = {name: 0.0 for name in feature_names}

    month = event.get("start_month")
    features["start_month"] = float(month) if month and 1 <= int(month) <= 12 else 1.0
    features["event_duration_log_norm"] = _scale(event.get("duration_days"), DURATION_LOG_MAX)

    magnitude = event.get("magnitude_value")
    features["magnitude_missing"] = 1.0 if magnitude is None else 0.0
    features["magnitude_scaled"] = _scale(magnitude, MAGNITUDE_LOG_MAX)

    scale = event.get("magnitude_scale") or "Unknown"
    if f"magnitude_scale_{scale}" in features:
        features[f"magnitude_scale_{scale}"] = 1.0
    if event.get("region") and f"region_{event['region']}" in features:
        features[f"region_{event['region']}"] = 1.0
    if event.get("country") and f"country_{event['country']}" in features:
        features[f"country_{event['country']}"] = 1.0
    return features


def event_from_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """The same raw attributes for a test-bank case, recovered from its features (used to brief the LLM)."""
    f, s = case["features"], case["summary"]
    missing = bool(f.get("magnitude_missing"))
    return {
        "country": s.get("country"),
        "region": s.get("region"),
        "start_month": int(f.get("start_month") or 1),
        "duration_days": round(unscale(f.get("event_duration_log_norm", 0.0), DURATION_LOG_MAX)),
        "magnitude_value": None if missing else round(unscale(f.get("magnitude_scaled", 0.0), MAGNITUDE_LOG_MAX), 2),
        "magnitude_scale": s.get("magnitude_scale") or "Unknown",
    }


def filled_attributes(event: Dict[str, Any]) -> List[str]:
    """Attributes the user didn't give: build_features filled them with the training default,
    so the models see a value (e.g. month 1 = January) that nobody actually said."""
    month = event.get("start_month")
    missing = {
        "Country": not event.get("country"),
        "Region": not event.get("region"),
        "Start month": not (month and 1 <= int(month) <= 12),
        "Event duration": event.get("duration_days") is None,
        "Magnitude scale": not event.get("magnitude_scale"),
        "Magnitude value": event.get("magnitude_value") is None,
    }
    return [name for name, _, _ in RAW_ATTRIBUTES if missing[name]]


def _one_hot_value(features: Dict[str, float], prefix: str) -> str:
    return next((k[len(prefix):] for k, v in features.items() if k.startswith(prefix) and v), "none")


def attribute_values(features: Dict[str, float]) -> Dict[str, str]:
    """What each of the 6 attributes is for this record, read back from its 245 features."""
    days = unscale(features.get("event_duration_log_norm", 0.0), DURATION_LOG_MAX)
    raw = unscale(features.get("magnitude_scaled", 0.0), MAGNITUDE_LOG_MAX)
    magnitude = ("not recorded" if features.get("magnitude_missing")
                 else f"{raw:,.0f}" if raw >= 100 else f"{raw:,.3g}")
    return {
        "Country": _one_hot_value(features, "country_"),
        "Magnitude scale": _one_hot_value(features, "magnitude_scale_"),
        "Region": _one_hot_value(features, "region_"),
        "Start month": f"{calendar.month_abbr[min(max(int(features.get('start_month') or 1), 1), 12)]} ({int(features.get('start_month') or 1)})",
        "Event duration": f"{days:,.0f} days",
        "Magnitude value": magnitude,
    }


def attribute_shap(feature_names: List[str], features: Dict[str, float], shap_values) -> List[Dict[str, Any]]:
    """SHAP summed back into the 6 attributes, largest effect first.

    SHAP values add up, so the 230 country columns sum to one 'Country' effect: "not Mexico, not
    Peru, ..." folds into the attribute's value instead of showing up as separate absent-feature
    reasons. `share` is the signed % of the record's total attribution; tree models report SHAP in
    different units (probability for Random Forest, log-odds for the boosters), so compare
    shares across models, never raw values.
    """
    per_feature = dict(zip(feature_names, (float(v) for v in shap_values)))
    values = attribute_values(features)
    rows = [{
        "attribute": name,
        "value": values[name],
        "shap": sum(per_feature[c] for c in attribute_columns(cols, feature_names)),
    } for name, cols, _ in RAW_ATTRIBUTES]
    total = sum(abs(r["shap"]) for r in rows) or 1.0
    for r in rows:
        r["share"] = round(100 * r["shap"] / total, 1)
        r["shap"] = round(r["shap"], 4)
        r["filled"] = False
    return sorted(rows, key=lambda r: abs(r["shap"]), reverse=True)


def explain_attributes(rows: List[Dict[str, Any]], target_class: str, top_k: int = 3) -> List[str]:
    """Plain-language reasons, one per attribute; filled-in defaults are labelled, not given as evidence."""
    sentences = []
    # Attributes under 1% of the evidence are noise; always keep the strongest one.
    shown = [r for i, r in enumerate(rows) if i == 0 or abs(r["share"]) >= 1][:top_k]
    for r in shown:
        verb = "pushed toward" if r["shap"] > 0 else "pulled away from"
        if r["filled"]:
            sentences.append(f"{r['attribute']} wasn't given; the default ({r['value']}) {verb} "
                             f"{target_class} ({r['share']:+.0f}% of the evidence) — treat this as an artefact, not a reason")
        else:
            sentences.append(f"{r['attribute']} = {r['value']} {verb} {target_class} ({r['share']:+.0f}% of the evidence)")
    return sentences


def explain_contrast(predicted: Dict[str, Any], contrast: Dict[str, Any], top_k: int = 2) -> List[str]:
    """'Why X and not Y?' in plain language: what held Y back, and what evidence X and Y share."""
    pred_class, other = predicted["target_class"], contrast["class"]
    pred_rows = {r["attribute"]: r for r in predicted["by_attribute"]}

    def label(r):
        return f"{r['attribute']} = {r['value']}" + (" (not given, default)" if r["filled"] else "")

    sentences = []
    against = [r for r in contrast["by_attribute"] if r["share"] <= -1][:top_k]
    if against:
        sentences.append(f"What held back {other}: " + ", ".join(f"{label(r)} ({r['share']:+.0f}%)" for r in against))
    shared = [r for r in contrast["by_attribute"] if r["share"] >= 10 and pred_rows[r["attribute"]]["share"] >= 10]
    for r in shared[:1]:
        sentences.append(f"{label(r)} supports both {pred_class} ({pred_rows[r['attribute']]['share']:+.0f}%) "
                         f"and {other} ({r['share']:+.0f}%), so it can't tell them apart")
    if not sentences:
        sentences.append(f"No single attribute stands out against {other}; the models simply found {pred_class} more typical")
    return sentences


def mark_filled(shap_info: Dict[str, Any], filled: Iterable[str]) -> Dict[str, Any]:
    """Flag the attributes the user didn't give and rebuild the plain-language reasons."""
    filled = set(filled)
    for r in shap_info["by_attribute"]:
        r["filled"] = r["attribute"] in filled
    shap_info["human_explanations"] = explain_attributes(shap_info["by_attribute"], shap_info["target_class"])
    return shap_info


def describe_event(event: Dict[str, Any]) -> str:
    """Plain-text description of the 6 attributes, the only information the LLM is given (same as the models)."""
    mag = event.get("magnitude_value")
    return "\n".join([
        f"- Country: {event.get('country') or 'unknown'}",
        f"- Region: {event.get('region') or 'unknown'}",
        f"- Start month: {event.get('start_month') or 'unknown'}",
        f"- Duration: {event['duration_days']} days" if event.get("duration_days") is not None else "- Duration: unknown",
        f"- Magnitude: {mag} ({event.get('magnitude_scale') or 'Unknown'})" if mag is not None
        else f"- Magnitude: not recorded (scale: {event.get('magnitude_scale') or 'Unknown'})",
    ])


def make_event_case(event: Dict[str, Any], feature_names: List[str], case_id: str,
                    truth: Optional[str] = None, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """A case dict shaped like the registry's test-bank cases, for an event the user described."""
    truth_id = class_names.index(truth) if truth and class_names and truth in class_names else None
    return {
        "case_id": case_id,
        "source": "user",
        "ground_truth_class": truth if truth_id is not None else None,
        "ground_truth_id": truth_id,
        "event": event,
        "filled": filled_attributes(event),
        "summary": {
            "country": event.get("country") or "Unknown",
            "region": event.get("region") or "Unknown",
            "start_month": event.get("start_month") or 1,
            "magnitude_scale": event.get("magnitude_scale") or "Unknown",
            "magnitude_missing": event.get("magnitude_value") is None,
        },
        "features": build_features(event, feature_names),
        "evaluations": {},
        "is_disagreement": False,
    }
