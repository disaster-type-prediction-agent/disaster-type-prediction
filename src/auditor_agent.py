"""Tool-using auditor agents (Claude or Gemini): the LLM decides which registry tools to call and answers from their results."""
import json
from collections import Counter
from typing import Any, Dict, Generator, List, Optional, Tuple

from src.event_features import (MAGNITUDE_SCALES, REGIONS, case_attribute_values, make_event_case, mark_filled,
                                missing_details, says_nothing_measured, scale_from_text, observed_scale, confirms, SCALE_QUANTITY, mentions_month, country_from_text, duration_from_text, type_stated_in_text)

MODEL = "claude-opus-5-5"
MAX_TOOL_ROUNDS = 6
MAX_JSON_RETRIES = 2

SYSTEM_PROMPT = """You are the Disaster Auditor, an expert in natural-disaster classification, ML model evaluation, and explainable AI (SHAP).

You audit an 8-class disaster-type classifier (Drought, Earthquake, Epidemic, Extreme Temperature, Flood, Landslide, Storm, Wildfire) trained on EM-DAT records. Three leak-free SMOTETomek pipelines are compared: Random Forest, LightGBM and XGBoost, evaluated on a stratified 20% held-out test bank. Classes are imbalanced (Flood and Storm dominate), so Macro-F1 is the headline metric, not accuracy.

How to work:
- Use the tools for every fact about the models, metrics, cases or SHAP values. Never state a metric, count, confidence or SHAP value you did not get from a tool in this conversation.
- When the user wants to see, test or find a case, call find_case (or get_case for a specific id). The app shows each case you load as a card, so summarize what matters instead of repeating every field.
- When the user describes a specific event of their own (a place, month, duration, or size such as wind speed or area), call classify_event with only the details they actually gave; pass stated_true_type only if they say what it really was. Then tell them which attributes were missing (filled with training defaults) and that adding them may change the prediction.
- If classify_event returns need_more_details, the models did not run: ask the user for those details (one short question) and stop; never predict or hint at a type yourself. On their next message, call classify_event again with everything they've said so far (it runs then, even if they don't know the rest; if they say nothing was measured, pass nothing_measured=true).
- "Nothing was measured" and "not said" are different. Pass nothing_measured=true only when the user says nothing was measured; then the models see the Unknown scale, as for recorded events without a measurement. If the user just doesn't say or doesn't know, leave both out: the app averages the models over the scales usually recorded for that place and month, and the case shows that mix. Never set magnitude_scale from the disaster the user describes (an earthquake, a flood), only from a quantity they say was measured. Nor from what they saw or felt (the house shaking, strong wind, extreme heat): leave it out, and the app asks them to confirm the quantity; if classify_event returns `observed`, ask exactly that.
- To revisit a case already loaded or classified (another model, why not X, more detail), use get_case or explain_prediction with its id (EVENT-n or CASE-n). Call classify_event again only when the user changes or adds event details; otherwise you'd create a duplicate event.
- Write for people: say "not given (default used)" and "% of the evidence", never raw field names like filled_in or share_pct.
- If the question is about particular models, pass them in `models`; otherwise leave it out to use the models selected in the UI.
- To explain a prediction, call explain_prediction. To explain why a model did NOT pick a class, call it with target_class set to that class. Case results already include `why_not` (SHAP toward the true type when the model was wrong, else the runner-up).
- SHAP is reported per raw attribute as `share_pct` (signed % of that model's total evidence). Compare models by share_pct, never by raw SHAP. An attribute with `filled_in: true` was not given by the user: never cite it as a reason.
- Anything not from a tool (disaster science, history) is general knowledge: say so briefly.
- For "which model is better" questions, combine get_model_metrics with get_significance; differences inside the McNemar noise band are not real wins.
- Ground explanations in disaster physics and seasonality (monsoon floods vs. storms, magnitude scales that only exist for some hazards, event duration patterns).
- Each user turn may start with a [UI context, app state, ...] line describing the case open on screen and the selected models; the user's own words follow "User message:". Treat the context as background, not as the question, and read event details only from the user's words.
- Reply in the language of the user's LATEST message, whatever language it is, as its [Reply language] note says, not the language of earlier turns (polite ครับ/ค่ะ in Thai). In English replies write each disaster class in English only (Landslide, never 滑坡 (Landslide)). In any other language, write the class in that language with its English name in brackets, e.g. in Thai ดินถล่ม (Landslide). Be concise and use short markdown.
- Stay on disaster classification, these models and their evaluation. Politely decline unrelated requests."""


def _slim_attr(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {"attribute": row["attribute"], "value": row["value"], "share_pct": row["share"]}
    if row.get("filled"):
        out["filled_in"] = True
    return out


def _enum_or_none(value, allowed):
    return value is None or value in allowed


def _keep_earlier_details(event: Dict[str, Any], earlier: Dict[str, Any]) -> List[str]:
    """Fill what the answer to our question left out from the event the first message described.
    Values in the answer win. A region that differs from the earlier one means the place changed: the
    earlier country is then not kept. Returns the attributes filled in."""
    kept = []
    if not event.get("country") and earlier.get("country") and event.get("region") in (None, earlier.get("region")):
        event["country"], event["region"] = earlier["country"], earlier.get("region")
        kept.append("country")
    for key, label in (("region", "region"), ("start_month", "month"), ("duration_days", "duration"),
                       ("magnitude_value", "magnitude value")):
        if event.get(key) is None and earlier.get(key) is not None:
            event[key] = earlier[key]
            kept.append(label)
    if event.get("magnitude_scale") in (None, "Unknown") and earlier.get("magnitude_scale") not in (None, "", "Unknown"):
        event["magnitude_scale"] = earlier["magnitude_scale"]
        kept.append("magnitude scale")
    return kept


def _close_dangling_tool_calls(history: List[Dict[str, Any]], reason: str) -> None:
    """If the transcript ends on an assistant turn with unanswered tool calls, append error results for them.

    Appends rather than edits, so the history stays append-only (required to keep thinking blocks valid).
    """
    if not history or history[-1]["role"] != "assistant":
        return
    content = history[-1]["content"]
    ids = [b.id for b in content if getattr(b, "type", None) == "tool_use"] if isinstance(content, list) else []
    if ids:
        history.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": i, "is_error": True, "content": reason} for i in ids
        ]})


class SessionCases:
    """Events the user described in this chat (EVENT-1, EVENT-2, ...), so tools can find them again.

    Lives in the user's session, not in AuditorTools, which is shared by every session."""

    def __init__(self):
        self.cases: Dict[str, Dict[str, Any]] = {}
        self.counter = 0
        self.message = ""  # the user's latest message as typed (no UI context), for checks done in code
        self.asked_details_on: Optional[str] = None  # the message on which classify_event asked for more details
        self.asked_event: Optional[Dict[str, Any]] = None  # what that message gave, kept for the user's answer
        self.proposed_scale: Optional[Tuple[str, str]] = None  # (scale, words): what the user observed, to confirm

    def next_id(self) -> str:
        self.counter += 1
        return f"EVENT-{self.counter}"

    def add(self, case: Dict[str, Any]) -> None:
        self.cases[case["case_id"]] = case


class AuditorTools:
    """Tool definitions and executors backed by the ModelRegistry."""

    def __init__(self, registry):
        self.registry = registry
        self.classes = list(registry.class_names)
        self.models = list(registry.available_models)
        self.regions = sorted({c["summary"]["region"] for c in registry.cases if c["summary"]["region"] != "Unknown"})
        self.countries = [f[len("country_"):] for f in registry.feature_names if f.startswith("country_")]
        self.country_regions = registry.country_regions()
        self._error_breakdown = self._build_error_breakdown()

    # ---------------------------------------------------------------- schemas
    def definitions(self) -> List[Dict[str, Any]]:
        models_param = {
            "type": "array",
            "items": {"type": "string", "enum": self.models},
            "description": "Models to evaluate. Omit to use the models currently selected in the UI.",
        }
        tools = [
            {
                "name": "find_case",
                "description": (
                    "Pick a random held-out test case matching the filters, run the chosen models on it, and show it "
                    "to the user as a case card. Returns ground truth, location, timing, each model's prediction, "
                    "confidence, correctness and top SHAP signals. Use whenever the user wants to see or test a case."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "disaster_type": {"type": "string", "enum": self.classes, "description": "Ground-truth class to sample from."},
                        "region": {"type": "string", "enum": self.regions},
                        "only_hard": {"type": "boolean", "description": "Only cases at least one model misclassified."},
                        "only_disagreement": {"type": "boolean", "description": "Only cases where the three models disagree."},
                        "misclassified_by": {"type": "string", "enum": self.models, "description": "Only cases this model got wrong."},
                        "models": models_param,
                    },
                },
            },
            {
                "name": "get_case",
                "description": "Load a specific case by id (a test case like CASE-0172, or a user event like EVENT-1), run the chosen models on it, and show it as a case card.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string", "description": "Case id such as CASE-0172."},
                        "models": models_param,
                    },
                    "required": ["case_id"],
                },
            },
            {
                "name": "classify_event",
                "description": (
                    "Run the models on an event the user describes (not from the test bank) and show it as a case card. "
                    "Pass ONLY values the user explicitly stated. Never estimate, assume or invent a value (no typical "
                    "durations or sizes): omit it, and the app fills the training default and reports it as filled_in. "
                    "Returns predictions, SHAP per attribute and why_not. If too little is given, it returns "
                    "need_more_details instead and runs nothing: ask the user for those details."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "country": {"type": "string", "description": "Country in English, e.g. Thailand."},
                        "region": {"type": "string", "enum": REGIONS, "description": "Only if no country is given."},
                        "start_month": {"type": "integer", "minimum": 1, "maximum": 12},
                        "duration_days": {"type": "number", "minimum": 0, "description": "Only if the user said how long it lasted; convert hours/weeks to days."},
                        "magnitude_value": {"type": "number", "minimum": 0,
                                            "description": "Wind speed in kph, area in km2, Richter/moment magnitude, temperature °C, or people vaccinated."},
                        "magnitude_scale": {"type": "string", "enum": MAGNITUDE_SCALES, "description": "What was measured, even when no figure is given (e.g. 'affected area was reported' -> Km2): area Km2, wind speed Kph, Richter/moment magnitude, temperature °C, people vaccinated."},
                        "nothing_measured": {"type": "boolean", "description": (
                            "Only if the user said nothing at all was measured for the event (not just that they "
                            "don't know). If they name what was measured but give no figure ('wind speed was "
                            "recorded, no number'), that is magnitude_scale (Kph), not nothing_measured.")},
                        "stated_true_type": {"type": "string", "enum": self.classes,
                                             "description": "Only if the user says what the event actually was."},
                        "models": models_param,
                    },
                },
            },
            {
                "name": "explain_prediction",
                "description": (
                    "SHAP attributions for one model on one case: the features pushing toward and away from a class. "
                    "By default explains the model's predicted class; set target_class to ask why it did or didn't pick another class."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string"},
                        "model": {"type": "string", "enum": self.models},
                        "target_class": {"type": "string", "enum": self.classes},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["case_id", "model"],
                },
            },
            {
                "name": "get_model_metrics",
                "description": "Test-set metrics: Macro-F1 with bootstrap 95% CI, accuracy, weighted-F1, and per-class F1 and recall. Omit model for all three.",
                "input_schema": {
                    "type": "object",
                    "properties": {"model": {"type": "string", "enum": self.models}},
                },
            },
            {
                "name": "get_significance",
                "description": "Paired McNemar tests between each pair of models on the test set (p-values and whether the difference is significant).",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_error_breakdown",
                "description": (
                    "Per-class error analysis over the whole test bank: case counts, each model's errors per class, its most common "
                    "confusions, and how many cases the models disagree on. Use for 'where does X fail' or 'which classes get confused' questions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "disaster_type": {"type": "string", "enum": self.classes, "description": "Limit to one ground-truth class."},
                    },
                },
            },
        ]
        # Streamed requests with client tools: let inputs stream as generated (we validate before running).
        for t in tools:
            t["eager_input_streaming"] = True
        return tools

    # ------------------------------------------------------------- validation
    def validate(self, name: str, args: Any) -> Optional[str]:
        """Return an error message if the (possibly partially parsed) input is invalid, else None."""
        if not isinstance(args, dict):
            return "input must be an object"

        def bool_ok(k):
            return k not in args or isinstance(args[k], bool)

        def models_ok():
            m = args.get("models")
            return m is None or (isinstance(m, list) and m and all(x in self.models for x in m))

        if name == "find_case":
            ok = (_enum_or_none(args.get("disaster_type"), self.classes)
                  and _enum_or_none(args.get("region"), self.regions)
                  and _enum_or_none(args.get("misclassified_by"), self.models)
                  and bool_ok("only_hard") and bool_ok("only_disagreement") and models_ok())
        elif name == "classify_event":
            month, numbers = args.get("start_month"), [args.get(k) for k in ("duration_days", "magnitude_value")]
            ok = (_enum_or_none(args.get("region"), REGIONS)
                  and _enum_or_none(args.get("magnitude_scale"), MAGNITUDE_SCALES)
                  and _enum_or_none(args.get("stated_true_type"), self.classes)
                  and bool_ok("nothing_measured")
                  and (args.get("country") is None or isinstance(args["country"], str))
                  and (month is None or (isinstance(month, int) and not isinstance(month, bool) and 1 <= month <= 12))
                  and all(n is None or (isinstance(n, (int, float)) and not isinstance(n, bool) and n >= 0) for n in numbers)
                  and models_ok())
        elif name == "get_case":
            ok = isinstance(args.get("case_id"), str) and models_ok()
        elif name == "explain_prediction":
            top_k = args.get("top_k", 5)
            ok = (isinstance(args.get("case_id"), str) and args.get("model") in self.models
                  and _enum_or_none(args.get("target_class"), self.classes)
                  and isinstance(top_k, int) and not isinstance(top_k, bool) and 1 <= top_k <= 10)
        elif name == "get_model_metrics":
            ok = _enum_or_none(args.get("model"), self.models)
        elif name == "get_significance":
            ok = True
        elif name == "get_error_breakdown":
            ok = _enum_or_none(args.get("disaster_type"), self.classes)
        else:
            return f"unknown tool '{name}'"
        return None if ok else "input does not match the tool schema"

    # -------------------------------------------------------------- execution
    def _lookup(self, case_id: str, session: Optional[SessionCases]) -> Optional[Dict[str, Any]]:
        case_id = case_id.strip().upper()
        return (session.cases.get(case_id) if session else None) or self.registry.get_case(case_id)

    def execute(self, name: str, args: Dict[str, Any], default_models: List[str],
                session: Optional[SessionCases] = None) -> Tuple[Dict[str, Any], str, Optional[Dict], List[str]]:
        """Run a tool. Returns (result for the LLM, one-line UI summary, audit to show as a card or None, models audited)."""
        if name == "classify_event":
            from src.llm_tasks import match_country

            models = list(args.get("models") or default_models)
            country = match_country(args.get("country"), self.countries)
            event = {
                "country": country,
                "region": args.get("region") or self.country_regions.get(country),
                "start_month": args.get("start_month"),
                "duration_days": args.get("duration_days"),
                "magnitude_value": args.get("magnitude_value"),
                "magnitude_scale": args.get("magnitude_scale"),
            }
            session = session or SessionCases()
            notes = []
            nothing_measured = bool(args.get("nothing_measured")) and args.get("magnitude_value") is None
            # The event's own words: on the answer to our question, the message that described it too.
            text = session.message
            if session.asked_details_on and session.asked_details_on != session.message:
                text = f"{session.asked_details_on}\n{session.message}"
            answering = bool(session.asked_details_on and session.asked_details_on != session.message)
            if answering and session.asked_event:
                # The user is answering our question. The LLM sometimes re-sends only part of the first message
                # (in the app, "ญี่ปุ่น ... มีนาคม" came back as region Asia with no country): keep what it gave.
                kept = _keep_earlier_details(event, session.asked_event)
                if kept:
                    notes.append(f"Kept from the user's first message: {', '.join(kept)}.")
            # Safety nets for values the LLM sometimes leaves out although the message states them (in the
            # app, "In the Philippines, ..." came back without a country; "over 2 days" without a duration).
            if not event["country"]:  # a region alone is less: the country named in the message wins
                found = country_from_text(text, self.countries)
                # A region the LLM gave that doesn't fit the country found means the place changed ("actually in Europe").
                if found and event["region"] in (None, self.country_regions.get(found)):
                    event["country"], event["region"] = found, self.country_regions.get(found)
                    notes.append(f"Country {found} was read from the message by the app.")
            if event["duration_days"] is None:
                days = duration_from_text(text)
                if days is not None:
                    event["duration_days"] = days
                    notes.append(f"Duration {days:g} days was read from the message by the app.")
            if event["start_month"] and text and not mentions_month(text):
                # The LLM filled a month the user never gave (in the app, start_month=1 for "ที่ไทยในปี 2014 ...").
                # Left out, the app asks for it and, if still unknown, reports January as a filled default.
                event["start_month"] = None
                notes.append("start_month was dropped: the user's words name no month.")
            if nothing_measured and not says_nothing_measured(text):
                # The LLM read "I don't know" as "nothing measured" (seen in the app on "ไม่ทราบครับ"). That would
                # feed the models the Unknown scale; not knowing is averaged over likely scales instead.
                nothing_measured = False
                notes.append("nothing_measured was ignored: the user didn't say nothing was measured.")
            # What the user saw or felt ("บ้านสั่น") points at a measured quantity but is not one: the app proposes
            # the scale and only the user's yes puts it into the models. The LLM may not do this on its own.
            said_scale = scale_from_text(text)
            seen = observed_scale(text)
            proposed = session.proposed_scale if answering else None
            confirmed = bool(proposed) and confirms(session.message)
            observed = None
            if (event["magnitude_scale"] not in (None, "Unknown") and not said_scale and seen
                    and seen[0] == event["magnitude_scale"]):
                if confirmed and proposed[0] == event["magnitude_scale"]:
                    observed = proposed
                else:
                    notes.append(f"Magnitude scale {event['magnitude_scale']} was set aside: it came from what the "
                                 f"user observed (\"{seen[1]}\"), which the user has not confirmed.")
                    event["magnitude_scale"] = None
            if event["magnitude_scale"] in (None, "Unknown"):
                event["magnitude_scale"] = None
                scale = said_scale
                if not scale and confirmed and not says_nothing_measured(session.message):
                    event["magnitude_scale"], observed = proposed[0], proposed
                    notes.append(f"Magnitude scale {proposed[0]} comes from what the user observed "
                                 f"(\"{proposed[1]}\"), confirmed by the user.")
                elif proposed and not scale:
                    notes.append(f"The user did not confirm {proposed[0]} from what they observed; "
                                 "the scale stays not given.")
                if scale:
                    # The message names what was measured. "Wind speed was recorded, no figure" is a scale with no
                    # value, not "nothing measured", whatever the LLM passed: the more specific evidence wins.
                    event["magnitude_scale"] = scale
                    notes.append(f"Magnitude scale {scale} was read from the message by the app.")
                elif nothing_measured and not event["magnitude_scale"]:
                    event["magnitude_scale"] = "Unknown"  # a real answer, like the recorded cases with no measurement
            nothing_measured = nothing_measured and event["magnitude_scale"] == "Unknown"
            missing = missing_details(event, nothing_measured)
            if missing and not answering:
                # Too little to predict from: ask first. Decided here in code, not left to the LLM. Asked once:
                # whatever the user's next message adds, the models run on it (no loop of questions).
                session.asked_details_on = session.message
                session.asked_event = dict(event)
                if not event.get("start_month") and "the month it started" not in missing:
                    missing.append("the month it started")  # asking anyway: a missing month becomes January
                given = {k: v for k, v in event.items() if v not in (None, "", "Unknown")}
                session.proposed_scale = seen if (seen and not event["magnitude_scale"]) else None
                if session.proposed_scale:
                    scale, words = session.proposed_scale
                    return ({"status": "need_more_details", "missing": missing, "given_so_far": given,
                             "observed": {"words": words, "measured_quantity": SCALE_QUANTITY[scale]},
                             "instruction": (
                                 f"The models were NOT run. The user described \"{words}\", which is what "
                                 f"{SCALE_QUANTITY[scale]} measures. In one short message in their language, ask them to "
                                 f"confirm that the quantity measured was {SCALE_QUANTITY[scale]} (yes / no / not sure), "
                                 "and for the other missing details listed. Name only the quantity, never a disaster "
                                 "type, and do not guess the type. Say that if they are not sure, the app will average "
                                 "over what is usually recorded there.")},
                            "Asked to confirm what was observed", None, [])
                return ({"status": "need_more_details", "missing": missing, "given_so_far": given,
                         "instruction": (
                             "The models were NOT run. Ask the user, in one short question in their language, for the "
                             "missing details listed, with an example or two. For what was measured, offer the choices "
                             "(affected area, wind speed, magnitude, temperature, people vaccinated, or nothing was "
                             "measured), never one that hints at a disaster type. Say they can answer that they don't "
                             "know, and the app will then run the models with default values (an unknown measurement is "
                             "averaged over what is usually recorded there). Do not guess the disaster type.")},
                        "Asked for more details", None, [])
            session.asked_details_on = None
            session.asked_event = None
            session.proposed_scale = None
            truth = args.get("stated_true_type")
            if truth and not type_stated_in_text(truth, text):
                # The LLM guessed the type; the user never said it. A guess must not score the model.
                # Not naming the type here: Gemini read the note back as "the type you expected" in the app.
                notes.append("stated_true_type was dropped: the user never said what the event was. "
                             "Do not name or hint at a type the user did not say.")
                truth = None
            case = make_event_case(event, self.registry.feature_names, session.next_id(),
                                   truth=truth, class_names=self.classes)
            if observed and event["magnitude_scale"] == observed[0]:
                case["scale_observed"] = {"scale": observed[0], "words": observed[1]}
            session.add(case)
            audit = self.registry.audit_case(case, models)
            result = self._case_payload(audit)
            if args.get("country") and not country:
                notes.append(f"Country '{args['country']}' is not one the models know; it was left out.")
            if notes:
                result["note"] = " ".join(notes)
            return result, f"Classified {case['case_id']} ({event['country'] or 'no country'})", audit, models

        if name == "find_case":
            models = list(args.get("models") or default_models)
            # A case picked because a model got it wrong must show that model's prediction.
            if args.get("misclassified_by") and args["misclassified_by"] not in models:
                models.append(args["misclassified_by"])
            case = self.registry.get_random_case(
                disaster_type=args.get("disaster_type"),
                only_hard=args.get("only_hard", False),
                only_disagreement=args.get("only_disagreement", False),
                region=args.get("region"),
                misclassified_by=args.get("misclassified_by"),
            )
            filters = ", ".join(
                [args[k] for k in ("disaster_type", "region") if args.get(k)]
                + (["hard"] if args.get("only_hard") else [])
                + (["disagreement"] if args.get("only_disagreement") else [])
                + ([f"missed by {args['misclassified_by']}"] if args.get("misclassified_by") else [])
            ) or "any case"
            if not case:
                return {"error": "No test case matches these filters."}, f"No case found ({filters})", None, models
            audit = self.registry.audit_case(case, models)
            return self._case_payload(audit), f"Found {case['case_id']} ({filters})", audit, models

        if name == "get_case":
            models = args.get("models") or default_models
            case = self._lookup(args["case_id"], session)
            if not case:
                return {"error": f"Case '{args['case_id']}' not found."}, f"Case {args['case_id']} not found", None, models
            audit = self.registry.audit_case(case, models)
            return self._case_payload(audit), f"Loaded {case['case_id']}", audit, models

        if name == "explain_prediction":
            case = self._lookup(args["case_id"], session)
            if not case:
                return {"error": f"Case '{args['case_id']}' not found."}, f"Case {args['case_id']} not found", None, []
            model = self.registry.get_model(args["model"])
            prediction = case["evaluations"].get(args["model"])
            features = case["features"]
            if not prediction:
                prediction, features = self.registry.predict_case(case, args["model"])
            target = args.get("target_class")
            target_idx = self.classes.index(target) if target else prediction["predicted_id"]
            shap = model.explain(features, target_class_idx=target_idx, top_k=args.get("top_k", 5))
            mark_filled(shap, case.get("filled", []))
            slim = lambda rows: [{"feature": r["feature"], "value": r["value"], "shap": r["shap_value"]} for r in rows]
            result = {
                "case_id": case["case_id"],
                "model": args["model"],
                "explained_class": shap["target_class"],
                "model_predicted": prediction["predicted_class"],
                "ground_truth": case["ground_truth_class"],
                "pushes_toward": slim(shap["top_positive"]),
                "pushes_away": slim(shap["top_negative"]),
                "by_attribute": [_slim_attr(r) for r in shap["by_attribute"]],  # the 6 raw attributes, % of total evidence
            }
            return result, f"SHAP for {args['model']} on {case['case_id']} → {shap['target_class']}", None, []

        if name == "get_model_metrics":
            lb = self.registry.leaderboard
            data = {args["model"]: lb[args["model"]]} if args.get("model") else lb
            return data, f"Read metrics ({args.get('model') or 'all models'})", None, []

        if name == "get_significance":
            return self.registry.mcnemar, "Read McNemar significance tests", None, []

        if name == "get_error_breakdown":
            cls = args.get("disaster_type")
            data = {cls: self._error_breakdown["per_class"][cls]} if cls else self._error_breakdown["per_class"]
            result = {"per_class": data, "disagreement_cases": self._error_breakdown["disagreement_cases"],
                      "total_cases": self._error_breakdown["total_cases"]}
            return result, f"Read error breakdown ({cls or 'all classes'})", None, []

        return {"error": f"unknown tool '{name}'"}, f"Unknown tool {name}", None, []

    # ---------------------------------------------------------------- helpers
    def _case_payload(self, audit: Dict[str, Any]) -> Dict[str, Any]:
        case = audit["case"]
        models = {}
        for m_name, m in audit["models"].items():
            p = m["prediction"]
            top = sorted(p.get("probabilities", {}).items(), key=lambda kv: kv[1], reverse=True)[:3]
            entry = {
                "predicted": p["predicted_class"],
                "confidence": p["confidence"],
                "correct": m["is_correct"],  # None = the user hasn't said what it really was
                "top_probabilities": {c: v for c, v in top},
                "shap_by_attribute": [_slim_attr(r) for r in m["shap"]["by_attribute"]],
            }
            if m.get("contrast"):
                c = m["contrast"]
                entry["why_not"] = {"class": c["class"], "because": c["reason"], "probability": c["probability"],
                                    "shap_by_attribute": [_slim_attr(r) for r in c["by_attribute"]],
                                    "summary": c["human_explanations"]}
            models[m_name] = entry
        return {
            "case_id": case["case_id"],
            "source": "user-described event" if case.get("source") == "user" else "held-out test bank",
            "ground_truth": case["ground_truth_class"],
            "attributes": case_attribute_values(case, next(iter(audit["models"].values()), {}).get("prediction")),
            "filled_in": case.get("filled", []),
            "models": models,
            "models_disagree_in_full_bank": case.get("is_disagreement", False),
        }

    def _build_error_breakdown(self) -> Dict[str, Any]:
        per_class = {}
        for cls in self.classes:
            cases = [c for c in self.registry.cases if c["ground_truth_class"] == cls]
            entry = {"test_cases": len(cases), "models": {}}
            for m in self.models:
                wrong = [c["evaluations"][m]["predicted_class"] for c in cases if not c["evaluations"][m]["is_correct"]]
                entry["models"][m] = {
                    "errors": len(wrong),
                    "recall": round(1 - len(wrong) / len(cases), 4) if cases else None,
                    "top_confusions": [{"predicted_as": k, "count": v} for k, v in Counter(wrong).most_common(3)],
                }
            entry["disagreement_cases"] = sum(1 for c in cases if c.get("is_disagreement"))
            per_class[cls] = entry
        return {
            "per_class": per_class,
            "disagreement_cases": sum(1 for c in self.registry.cases if c.get("is_disagreement")),
            "total_cases": len(self.registry.cases),
        }


def run_tools(tools: AuditorTools, calls: List[Tuple[str, Any]], default_models: List[str],
              session: Optional[SessionCases]) -> Tuple[List[Tuple[Dict[str, Any], bool]], List[Dict[str, Any]]]:
    """Validate and run each (name, input) call. Returns ([(result, is_error)], [UI step events]).

    Every call gets a result, even a rejected or failing one, so the LLM's transcript always stays valid."""
    results, events = [], []
    for name, args in calls:
        problem = tools.validate(name, args)
        if problem:
            results.append(({"INVALID_JSON": json.dumps(args, default=str), "reason": problem}, True))
            events.append({"name": name, "input": args, "summary": f"Rejected invalid input ({problem})",
                           "is_error": True, "audit": None, "models": []})
            continue
        try:
            result, summary, audit, models = tools.execute(name, args, default_models, session)
            is_error = "error" in result
        except Exception as e:  # keep the loop alive; let the LLM see what failed
            result, summary, audit, models, is_error = {"error": str(e)}, f"{name} failed: {e}", None, [], True
        results.append((result, is_error))
        events.append({"name": name, "input": args, "summary": summary, "is_error": is_error, "audit": audit,
                       "models": models, "result": result})
    return results, events


class ClaudeAuditorAgent:
    """Streaming tool-use loop. `history` is the Messages API transcript and is only ever appended to."""

    def __init__(self, tools: AuditorTools, api_key: str, workspace_id: Optional[str] = None):
        self.tools = tools
        self.api_key = api_key
        self.workspace_id = (workspace_id or "").strip() or None

    def run_turn(self, history: List[Dict[str, Any]], user_text: str, default_models: List[str],
                 session: Optional[SessionCases] = None) -> Generator[Tuple[str, Any], None, None]:
        """Yields ("text", str) for answer tokens, ("retract", n) to drop the last n streamed characters
        when a failed attempt is retried, ("tool", dict) per tool call, and ("error", str) on failure."""
        import anthropic

        # Organization-level keys (not scoped to a workspace) must name the workspace on every request.
        headers = {"anthropic-workspace-id": self.workspace_id} if self.workspace_id else None
        client = anthropic.Anthropic(api_key=self.api_key, default_headers=headers)
        tool_defs = self.tools.definitions()
        _close_dangling_tool_calls(history, "Not run: the previous turn was interrupted.")
        history.append({"role": "user", "content": user_text})

        json_retries = 0
        rounds = 0
        while True:
            streamed = 0
            try:
                with client.beta.messages.stream(
                    model=MODEL,
                    max_tokens=16000,
                    system=SYSTEM_PROMPT,
                    tools=tool_defs,
                    messages=history,
                    # Agentic tool use: medium (this model's default) balances tool judgment and latency.
                    output_config={"effort": "medium"},
                    # System prompt and tools are identical every call; cache them across the loop.
                    cache_control={"type": "ephemeral"},
                    # On a safety-classifier decline, retry server-side on a fallback model.
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                ) as stream:
                    for event in stream:
                        if event.type == "text":
                            streamed += len(event.text)
                            yield "text", event.text
                    response = stream.get_final_message()
                json_retries = 0
            except ValueError:
                # Tool input JSON the SDK couldn't parse at all; no tool_use id to answer, so re-issue the turn.
                # Take back the text this failed attempt already streamed so the retry doesn't duplicate it.
                if streamed:
                    yield "retract", streamed
                json_retries += 1
                if json_retries > MAX_JSON_RETRIES:
                    yield "error", "Claude produced an unreadable tool call several times. Please rephrase and try again."
                    return
                continue
            except anthropic.AuthenticationError:
                yield "error", "Invalid Anthropic API key. Please check the key in the sidebar settings."
                return
            except anthropic.RateLimitError:
                yield "error", "Rate limited by the Anthropic API. Please wait a moment and try again."
                return
            except anthropic.APIStatusError as e:
                if "anthropic-workspace-id" in str(e.message):
                    yield "error", (
                        "This API key belongs to your organization but not to a specific workspace. "
                        "Enter the **Workspace ID** under Settings in the sidebar, or create a key inside a workspace in the Claude Console."
                    )
                else:
                    yield "error", f"Claude API error ({e.status_code}): {e.message}"
                return
            except anthropic.APIConnectionError:
                yield "error", "Could not reach the Anthropic API. Check your connection."
                return

            # Append the assistant turn unchanged (thinking blocks included) to keep history append-only.
            history.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "pause_turn":
                continue
            if response.stop_reason == "refusal":
                # A refusal can cut a tool call off mid-input: never run it, but answer it so history stays valid.
                _close_dangling_tool_calls(history, "Not run: the response was declined.")
                yield "error", "Claude declined to answer this request."
                return

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                if response.stop_reason == "max_tokens":
                    yield "text", "\n\n_(Response truncated.)_"
                return

            if response.stop_reason == "max_tokens":
                # A truncated tool input parses as a valid partial object: don't run it.
                history.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                     "content": "Tool input was truncated; nothing was run."} for b in tool_uses
                ]})
                yield "error", "Claude's response was cut off mid tool call. Please try a narrower question."
                return

            rounds += 1
            if rounds > MAX_TOOL_ROUNDS:
                # Claude ignored the "answer now" note; close out the pending calls and stop.
                history.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                     "content": "Tool budget exhausted; nothing was run."} for b in tool_uses
                ]})
                yield "error", "Stopped after too many tool calls for one question. Try asking something narrower."
                return

            # Run every tool before yielding anything: Streamlit can stop this generator at any yield
            # (a rerun), and an assistant tool_use left without its tool_result would 400 every later turn.
            outcomes, events = run_tools(self.tools, [(b.name, b.input) for b in tool_uses], default_models, session)
            results = [{"type": "tool_result", "tool_use_id": b.id, "is_error": is_error,
                        "content": json.dumps(result, ensure_ascii=False, default=str)}
                       for b, (result, is_error) in zip(tool_uses, outcomes)]
            if rounds == MAX_TOOL_ROUNDS:
                results.append({"type": "text", "text": "Tool budget for this question is used up. Answer now with what you have."})
            # All results for this turn go back in one user message.
            history.append({"role": "user", "content": results})
            for ev in events:
                yield "tool", ev


def _plain(value: Any) -> Any:
    """Gemini sends JSON numbers as floats (5.0); give integers back as int so validation sees 5."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


GEMINI_HISTORY_CONTENTS = 40  # keep the transcript (with its tool results) from growing without bound


class GeminiAuditorAgent:
    """The same tool loop on Gemini function calling (free tier). Gemini picks the tools; we run them.

    `history` is a list of google-genai Content objects. Each model turn is appended exactly as it
    came back (thought signatures included), which Gemini requires for multi-step function calling.
    A round is only appended once it fully succeeded, so a failed or fallback attempt leaves no trace."""

    def __init__(self, tools: AuditorTools, api_key: str):
        self.tools = tools
        self.api_key = api_key

    def _declarations(self):
        from google.genai import types

        return [types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t["name"], description=t["description"], parameters_json_schema=t["input_schema"])
            for t in self.tools.definitions()
        ])]

    @staticmethod
    def _trim(history: List[Any]) -> None:
        """Drop the oldest turns, cutting only at a plain user message so no tool call loses its result."""
        if len(history) <= GEMINI_HISTORY_CONTENTS:
            return
        for i in range(len(history) - GEMINI_HISTORY_CONTENTS, len(history)):
            c = history[i]
            if c.role == "user" and all(getattr(p, "function_response", None) is None for p in c.parts or []):
                del history[:i]
                return

    def run_turn(self, history: List[Any], user_text: str, default_models: List[str],
                 session: Optional[SessionCases] = None) -> Generator[Tuple[str, Any], None, None]:
        """Same events as ClaudeAuditorAgent.run_turn: ("text"), ("retract", n), ("tool", dict), ("error")."""
        from google.genai import errors, types
        from src.agent import gemini_models, make_gemini_client

        client = make_gemini_client(self.api_key)
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=self._declarations(),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # we run the tools
        )
        self._trim(history)
        history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
        models = gemini_models()

        for rounds in range(1, MAX_TOOL_ROUNDS + 2):
            parts, last_err = None, None
            for model_name in models:  # fall back to the next model on 503/429 (each has its own quota)
                streamed, got = 0, []
                try:
                    for chunk in client.models.generate_content_stream(model=model_name, contents=history, config=config):
                        for part in (chunk.candidates[0].content.parts if chunk.candidates and chunk.candidates[0].content else None) or []:
                            got.append(part)
                            if part.text and not part.thought:
                                streamed += len(part.text)
                                yield "text", part.text
                    parts = got
                    break
                except errors.APIError as e:
                    last_err = e
                    if streamed:
                        yield "retract", streamed  # this attempt's partial text: the next model starts over
                    if e.code in (400, 401, 403) and "key" in str(e).lower():
                        yield "error", "Invalid Gemini API key. Please check the key in the sidebar settings."
                        return
                except Exception as e:  # timeouts, connection errors
                    last_err = e
                    if streamed:
                        yield "retract", streamed
            if parts is None:
                if getattr(last_err, "code", None) == 429:
                    yield "error", "Gemini's free quota is used up for now. Wait a minute, or switch off agent mode in Settings."
                else:
                    yield "error", f"Gemini didn't answer ({last_err}). Please try again."
                return

            history.append(types.Content(role="model", parts=parts))
            calls = [p.function_call for p in parts if p.function_call]
            if not calls:
                return
            if rounds > MAX_TOOL_ROUNDS:
                # The budget note was ignored: answer the calls without running them, so the transcript stays valid.
                history.append(types.Content(role="user", parts=[
                    types.Part.from_function_response(name=c.name, response={"error": "Tool budget exhausted; nothing was run."})
                    for c in calls]))
                yield "error", "Stopped after too many tool calls for one question. Try asking something narrower."
                return

            outcomes, events = run_tools(self.tools, [(c.name, _plain(dict(c.args or {}))) for c in calls], default_models, session)
            responses = []
            for call, (result, _) in zip(calls, outcomes):
                response = types.Part.from_function_response(
                    name=call.name, response=json.loads(json.dumps(result, ensure_ascii=False, default=str)))
                if call.id:
                    response.function_response.id = call.id
                responses.append(response)
            if rounds == MAX_TOOL_ROUNDS:
                responses.append(types.Part(text="Tool budget for this question is used up. Answer now with what you have."))
            history.append(types.Content(role="user", parts=responses))
            for ev in events:
                yield "tool", ev
