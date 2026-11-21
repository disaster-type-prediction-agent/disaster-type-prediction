"""Gemini structured-output tasks used beside the chat.

1. route_question: what does a free question want? A test-bank case (with filters), a prediction for an
   event the user describes (with its 6 attributes extracted), or just an answer.
2. llm_guess: the LLM's own disaster-type prediction from the SAME 6 attributes the models get, so its
   accuracy can be scored against ground truth next to the models.

Both use JSON output constrained by a schema (enums for classes, countries, scales), so the app never
has to parse free text. They return None on any failure; callers fall back to keyword rules or skip.
"""
from difflib import get_close_matches
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.agent import gemini_models, make_gemini_client
from src.event_features import MAGNITUDE_SCALES, REGIONS, describe_event, scale_from_text


@lru_cache(maxsize=4)
def _schemas(classes: tuple, countries: tuple):
    # The country stays free text: a 230-value enum makes Gemini reject the schema (400 INVALID_ARGUMENT).
    # match_country() maps the answer onto the models' country list in code instead.
    ClassName = Literal[classes]
    Region = Literal[tuple(REGIONS)]
    Scale = Literal[tuple(MAGNITUDE_SCALES)]

    class Event(BaseModel):
        country: Optional[str] = Field(description="Country where it happened, in English; null if not given")
        region: Optional[Region] = Field(description="Continent-level region; null if unknown")
        start_month: Optional[int] = Field(description="Month it started, 1-12; null if not given")
        duration_days: Optional[float] = Field(description="How many days it lasted; null if not given")
        magnitude_value: Optional[float] = Field(description="Size of the event as a number (area km2, wind kph, Richter/Moment magnitude, temperature °C, people vaccinated); null if not given")
        magnitude_scale: Optional[Scale] = Field(description="What was measured, even when no figure is given: area -> Km2, wind speed -> Kph, magnitude/Richter -> Moment Magnitude, temperature -> °C, people vaccinated -> Vaccinated; null if the message says nothing about it")

    class Route(BaseModel):
        intent: Literal["test_bank_case", "new_event", "question"] = Field(
            description="test_bank_case: user wants to test/audit/see a case from the dataset. "
                        "new_event: the message describes a specific event with at least one concrete detail "
                        "(a place, month, duration or size such as wind speed or area), even when it also asks why, "
                        "what the model would predict, or what type it is; the models must run on it to answer. "
                        "question: no specific event is described (explanations of the models, metrics, general questions).")
        disaster_type: Optional[ClassName] = Field(description="For test_bank_case: the disaster type to sample, if asked for")
        region: Optional[Region] = Field(description="For test_bank_case: region to sample from, if asked for")
        only_hard: bool = Field(description="For test_bank_case: user wants a hard/misclassified case")
        only_disagreement: bool = Field(description="For test_bank_case: user wants a case the models disagree on")
        event: Optional[Event] = Field(description="For new_event: the event's attributes")
        stated_true_type: Optional[ClassName] = Field(description="For new_event: the disaster type the user says it actually was, if they say so")

    class Guess(BaseModel):
        predicted_class: ClassName
        confidence: float = Field(description="0-1")
        reason: str = Field(description="One short sentence")

    return Route, Guess


def match_country(name: Optional[str], countries: List[str]) -> Optional[str]:
    """Map a free-text country onto the models' country names ("USA" -> "United States of America")."""
    if not name:
        return None
    lowered = {c.lower(): c for c in countries}
    key = name.strip().lower()
    if key in lowered:
        return lowered[key]
    aliases = {"usa": "united states of america", "us": "united states of america", "united states": "united states of america",
               "uk": "united kingdom of great britain and northern ireland", "united kingdom": "united kingdom of great britain and northern ireland",
               "russia": "russian federation", "vietnam": "viet nam", "south korea": "republic of korea", "korea": "republic of korea",
               "north korea": "democratic people's republic of korea", "laos": "lao people's democratic republic",
               "iran": "iran (islamic republic of)", "bolivia": "bolivia (plurinational state of)", "syria": "syrian arab republic",
               "tanzania": "united republic of tanzania", "venezuela": "venezuela (bolivarian republic of)", "turkey": "türkiye"}
    if aliases.get(key) in lowered:
        return lowered[aliases[key]]
    starts = [c for c in countries if c.lower().startswith(key)]
    if len(starts) == 1:
        return starts[0]
    close = get_close_matches(key, list(lowered), n=1, cutoff=0.85)
    return lowered[close[0]] if close else None


def _generate_json(api_key: str, prompt: str, schema, temperature: float = 0) -> Optional[BaseModel]:
    from google.genai import types

    client = make_gemini_client(api_key)
    config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=schema, temperature=temperature)
    for model in gemini_models():
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            parsed = response.parsed
            if isinstance(parsed, schema):
                return parsed
        except Exception:
            continue  # busy / quota / bad output: try the next model
    return None


def route_question(api_key: str, question: str, classes: List[str], countries: List[str]) -> Optional[Dict[str, Any]]:
    Route, _ = _schemas(tuple(classes), tuple(countries))
    prompt = (
        "You route messages in a disaster-classification auditing app. Classes: " + ", ".join(classes) + ".\n"
        "Classify the user's message and fill in the fields. Only extract values the message actually states; "
        "use null otherwise. Convert units: hours->days, mph->kph, miles2->km2. The message may be in Thai or English.\n\n"
        f"Message:\n{question}"
    )
    result = _generate_json(api_key, prompt, Route)
    if not result:
        return None
    route = result.model_dump()
    if route.get("event"):
        route["event"]["country"] = match_country(route["event"].get("country"), countries)
        if route["event"].get("magnitude_scale") in (None, "Unknown"):  # the reader left it out
            route["event"]["magnitude_scale"] = scale_from_text(question)  # None if not named: a flagged default
    return route


def llm_guess(api_key: str, event: Dict[str, Any], classes: List[str], countries: List[str]) -> Optional[Dict[str, Any]]:
    _, Guess = _schemas(tuple(classes), tuple(countries))
    prompt = (
        "Classify a historical natural-disaster record from the EM-DAT database into exactly one of: "
        + ", ".join(classes) + ".\n"
        "You get only these pre-event attributes (the same ones the ML models get):\n"
        f"{describe_event(event)}\n\n"
        "Hints: the magnitude unit usually depends on the hazard (Km2 = affected area, Kph = wind speed, "
        "Moment Magnitude = earthquakes, °C = temperature, Vaccinated = epidemics). Answer with your best guess."
    )
    result = _generate_json(api_key, prompt, Guess)
    if not result:
        return None
    guess = result.model_dump()
    guess["confidence"] = max(0.0, min(1.0, float(guess["confidence"])))
    return guess


class _Scenario(BaseModel):
    message: str = Field(description="The chat message the person would type")


DISASTER_WORDS = ["drought", "earthquake", "epidemic", "heat wave", "heatwave", "cold wave", "flood",
                  "landslide", "storm", "typhoon", "hurricane", "cyclone", "wildfire", "bushfire", "fire",
                  "ภัยแล้ง", "แผ่นดินไหว", "โรคระบาด", "คลื่นความร้อน", "น้ำท่วม", "อุทกภัย", "ดินถล่ม", "พายุ", "ไต้ฝุ่น", "ไฟป่า"]


MEASURED_AS = {"Km2": "the affected area (km²)", "Kph": "the wind speed (km/h)",
               "Moment Magnitude": "the magnitude (on the moment magnitude scale)",
               "°C": "the temperature (°C)", "Vaccinated": "the number of people vaccinated"}


def _unmeasured_hint(event: Dict[str, Any]) -> str:
    """A unit without a value still tells the models a lot (Km2 vs Kph), so the message must carry it."""
    scale = event.get("magnitude_scale")
    if event.get("magnitude_value") is None and scale in MEASURED_AS:
        return (f"\n- Size: what was measured is {MEASURED_AS[scale]}, but no figure was recorded. "
                "Name that quantity in the message (e.g. 'the affected area was reported' or 'temperatures were "
                "recorded') without giving a number; don't just say 'it was measured'.")
    return ""


def _carries_scale(event: Dict[str, Any], message: str) -> bool:
    """A case measured without a figure must name the quantity, or the reader can't recover the scale."""
    if event.get("magnitude_value") is not None or event.get("magnitude_scale") not in MEASURED_AS:
        return True
    return scale_from_text(message) == event["magnitude_scale"]


def write_scenario(api_key: str, event: Dict[str, Any], language: str = "Thai") -> Optional[str]:
    """Turn a test-bank case's 6 attributes into a message a person might type, WITHOUT the disaster type.

    The writer never sees the true type, so it cannot leak it; messages that still name a hazard are rejected."""
    prompt = (
        "Write ONE short message (1-2 sentences) that a person might type into a disaster-classification chat app, "
        f"in {language}, describing the event below so the app can classify it.\n"
        "Rules: include every known detail in natural everyday wording (you may round numbers and use units like "
        "km/h, km², degrees, magnitude, people vaccinated, days or weeks); leave out anything marked unknown or not "
        "recorded; add no other details; NEVER name or hint at the type of disaster (no words like storm, flood, "
        "earthquake, drought, fire, epidemic, heat wave, landslide).\n\n"
        f"{describe_event(event)}"
        + _unmeasured_hint(event)
    )
    for _ in range(3):  # retry if the message names a hazard, or drops the quantity measured without a figure
        result = _generate_json(api_key, prompt, _Scenario, temperature=0.9)  # varied wording, like real users
        if result and result.message.strip() and not any(w in result.message.lower() for w in DISASTER_WORDS) \
                and _carries_scale(event, result.message):
            return result.message.strip()
    return None
