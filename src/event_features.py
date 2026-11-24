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
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

DURATION_LOG_MAX = 8.098034756176071   # log1p(3,287 days), the longest event in the raw data
MAGNITUDE_LOG_MAX = 16.38244832185021  # log1p(13,025,874), the largest magnitude in the raw data

MAGNITUDE_SCALES = ["Km2", "Kph", "Moment Magnitude", "Vaccinated", "°C", "Unknown"]
REGIONS = ["Africa", "Americas", "Asia", "Europe", "Oceania"]

# Words for WHAT WAS MEASURED (never disaster names), English and Thai. A message can name the quantity
# without a figure ("temperatures were reported"); the scale alone still tells the models a lot.
SCALE_WORDS = {
    "Km2": ["km2", "km²", "sq km", "sq. km", "square kilomet", "square mile", "hectare", "acre", "affected area",
            "area affected", "area was affected", "burned area", "burnt area", "area burned", "area burnt",
            "ตร.กม", "ตารางกิโลเมตร", "ตารางไมล์", "เฮกตาร์", "พื้นที่ได้รับผลกระทบ", "พื้นที่ที่ได้รับผลกระทบ",
            "พื้นที่ประสบภัย", "พื้นที่เสียหาย", "ขนาดพื้นที่"],
    "Kph": ["km/h", "kph", "kmh", "km per hour", "kilometres per hour", "kilometers per hour", "mph", "miles per hour",
            "wind speed", "windspeed", "gust", "กม./ชม", "กม/ชม", "กิโลเมตรต่อชั่วโมง", "ความเร็วลม"],
    "Moment Magnitude": ["magnitude", "richter", "ริกเตอร์", "แมกนิจูด", "แม็กนิจูด", "แมกนิจูด"],
    "°C": ["°c", "°f", "celsius", "fahrenheit", "degree", "temperature", "องศา", "อุณหภูมิ", "เซลเซียส"],
    "Vaccinated": ["vaccin", "วัคซีน"],
}


def scale_from_text(text: Optional[str]) -> Optional[str]:
    """The magnitude scale a message names, or None if it names none or more than one.

    A safety net for the LLM reader, which sometimes leaves the scale empty when no figure is given."""
    # "unknown magnitude" names no quantity: "magnitude" there is just "size", not the Richter/moment scale.
    low = _NO_MAGNITUDE.sub(" ", (text or "").lower())
    found = [scale for scale, pattern in _SCALE_PATTERNS.items() if pattern.search(low)]
    return found[0] if len(found) == 1 else None



# Common short names the models' country list spells differently. Ambiguous ones ("us", "korea") are left out.
COUNTRY_ALIASES = {"usa": "United States of America", "united states": "United States of America",
                   "uk": "United Kingdom of Great Britain and Northern Ireland",
                   "united kingdom": "United Kingdom of Great Britain and Northern Ireland",
                   "britain": "United Kingdom of Great Britain and Northern Ireland", "russia": "Russian Federation",
                   "vietnam": "Viet Nam", "south korea": "Republic of Korea",
                   "north korea": "Democratic People's Republic of Korea", "laos": "Lao People's Democratic Republic",
                   "syria": "Syrian Arab Republic", "tanzania": "United Republic of Tanzania", "turkey": "Türkiye",
                   "hong kong": "China, Hong Kong Special Administrative Region", "dr congo": "Democratic Republic of the Congo"}

# Thai names, for the same safety net: Gemini sometimes reads "ญี่ปุ่น" as region Asia and drops the country.
# Longer names win (see country_from_text), so "ซูดานใต้" is not also read as "ซูดาน". Bare "ลาว" is left out:
# it is also the start of "ลาวา" (lava).
THAI_COUNTRY_NAMES = {
    "ญี่ปุ่น": "Japan", "จีน": "China", "อินเดีย": "India", "อินโดนีเซีย": "Indonesia", "ฟิลิปปินส์": "Philippines",
    "ไทย": "Thailand", "เวียดนาม": "Viet Nam", "เมียนมา": "Myanmar", "เมียนมาร์": "Myanmar", "พม่า": "Myanmar",
    "กัมพูชา": "Cambodia", "ประเทศลาว": "Lao People's Democratic Republic", "สปป.ลาว": "Lao People's Democratic Republic",
    "สปป. ลาว": "Lao People's Democratic Republic", "มาเลเซีย": "Malaysia", "สิงคโปร์": "Singapore",
    "บังกลาเทศ": "Bangladesh", "บังคลาเทศ": "Bangladesh", "ปากีสถาน": "Pakistan", "เนปาล": "Nepal",
    "ศรีลังกา": "Sri Lanka", "อัฟกานิสถาน": "Afghanistan", "อิหร่าน": "Iran (Islamic Republic of)", "อิรัก": "Iraq",
    "ตุรกี": "Türkiye", "ทูร์เคีย": "Türkiye", "ซาอุดีอาระเบีย": "Saudi Arabia", "เยเมน": "Yemen",
    "เกาหลีใต้": "Republic of Korea", "เกาหลีเหนือ": "Democratic People's Republic of Korea", "มองโกเลีย": "Mongolia",
    "ไต้หวัน": "Taiwan (Province of China)", "ฮ่องกง": "China, Hong Kong Special Administrative Region",
    "ติมอร์-เลสเต": "Timor-Leste", "คาซัคสถาน": "Kazakhstan", "ซีเรีย": "Syrian Arab Republic", "อิสราเอล": "Israel",
    "จอร์แดน": "Jordan", "สหรัฐอเมริกา": "United States of America", "สหรัฐฯ": "United States of America",
    "อเมริกา": "United States of America", "แคนาดา": "Canada", "เม็กซิโก": "Mexico", "บราซิล": "Brazil",
    "อาร์เจนตินา": "Argentina", "ชิลี": "Chile", "เปรู": "Peru", "โคลอมเบีย": "Colombia", "เอกวาดอร์": "Ecuador",
    "โบลิเวีย": "Bolivia (Plurinational State of)", "เวเนซุเอลา": "Venezuela (Bolivarian Republic of)", "คิวบา": "Cuba",
    "เฮติ": "Haiti", "กัวเตมาลา": "Guatemala", "ฮอนดูรัส": "Honduras", "ออสเตรเลีย": "Australia",
    "นิวซีแลนด์": "New Zealand", "ปาปัวนิวกินี": "Papua New Guinea", "ฟิจิ": "Fiji",
    "สหราชอาณาจักร": "United Kingdom of Great Britain and Northern Ireland", "อังกฤษ": "United Kingdom of Great Britain and Northern Ireland",
    "ฝรั่งเศส": "France", "เยอรมนี": "Germany", "เยอรมัน": "Germany", "อิตาลี": "Italy", "สเปน": "Spain",
    "โปรตุเกส": "Portugal", "กรีซ": "Greece", "รัสเซีย": "Russian Federation", "ยูเครน": "Ukraine", "โปแลนด์": "Poland",
    "โรมาเนีย": "Romania", "เนเธอร์แลนด์": "Netherlands (Kingdom of the)", "สวิตเซอร์แลนด์": "Switzerland",
    "อียิปต์": "Egypt", "ไนจีเรีย": "Nigeria", "เคนยา": "Kenya", "เอธิโอเปีย": "Ethiopia", "โซมาเลีย": "Somalia",
    "ซูดานใต้": "South Sudan", "ซูดาน": "Sudan", "แอฟริกาใต้": "South Africa",
    "สาธารณรัฐประชาธิปไตยคองโก": "Democratic Republic of the Congo", "คองโก": "Congo", "ยูกันดา": "Uganda",
    "แทนซาเนีย": "United Republic of Tanzania", "โมซัมบิก": "Mozambique", "มาดากัสการ์": "Madagascar",
    "มาลาวี": "Malawi", "ซิมบับเว": "Zimbabwe", "แซมเบีย": "Zambia", "ไนเจอร์": "Niger", "มาลี": "Mali",
    "ชาด": "Chad", "กานา": "Ghana", "แคเมอรูน": "Cameroon", "แองโกลา": "Angola", "โมร็อกโก": "Morocco",
    "แอลจีเรีย": "Algeria", "บุรุนดี": "Burundi", "รวันดา": "Rwanda",
}


def country_from_text(text: Optional[str], countries: List[str]) -> Optional[str]:
    """The one country a message names (English, or the Thai names above), or None (none, several, or not in the models' list).

    A safety net for the LLM reader. Longer names are matched first and taken out of the text, so "Congo"
    inside "Democratic Republic of the Congo" or "Guinea" inside "Papua New Guinea" is not counted again."""
    low = (text or "").lower()
    names = {}
    for c in countries:
        names[c.lower()] = c
        short = c.split(" (")[0].lower()  # "Iran (Islamic Republic of)" -> "iran"
        names.setdefault(short, c)
    for alias, c in {**COUNTRY_ALIASES, **THAI_COUNTRY_NAMES}.items():
        if c in countries:
            names.setdefault(alias, c)
    found = set()
    for name in sorted(names, key=len, reverse=True):
        pattern = r"(?<![a-z])" + re.escape(name) + r"(?![a-z])"
        if re.search(pattern, low):
            found.add(names[name])
            low = re.sub(pattern, " ", low)
    return found.pop() if len(found) == 1 else None


_DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*(?:กว่า\s*)?"
                       r"(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|นาที|ชั่วโมง|วัน|สัปดาห์|อาทิตย์|เดือน)(?![a-z])")
_DAYS_PER = {"minute": 1 / 1440, "min": 1 / 1440, "hour": 1 / 24, "hr": 1 / 24, "day": 1, "week": 7, "month": 30,
             "นาที": 1 / 1440, "ชั่วโมง": 1 / 24, "วัน": 1, "สัปดาห์": 7, "อาทิตย์": 7, "เดือน": 30}


def duration_from_text(text: Optional[str]) -> Optional[float]:
    """How long the event lasted, in days, when the message gives exactly one length ("over 2 days", "3 สัปดาห์").

    None when it gives none or several (which one is the duration is then the LLM's call, not ours)."""
    found = {(float(n), unit.rstrip("s") if unit.isascii() else unit) for n, unit in _DURATION.findall((text or "").lower())}
    if len(found) != 1:
        return None
    n, unit = found.pop()
    return round(n * _DAYS_PER[unit], 2)


# A month the message names: English names and abbreviations, Thai names (มกรา... with or without คม/ยน/พันธ์)
# and abbreviations (ม.ค.), "month 5"/"เดือน 5", and numeric dates. Used to drop a month the LLM made up: in the
# app, Gemini passed start_month=1 for "ที่ไทยในปี 2014 ..." which names no month, so the app never asked for it.
_MONTH = re.compile(
    r"(?<![a-z])(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?![a-z])"
    r"|มกรา|กุมภา|มีนา|เมษา|พฤษภา|มิถุนา|กรกฎา|สิงหา|กันยา|ตุลา|พฤศจิกา|ธันวา"
    r"|ม\.ค|ก\.พ|มี\.ค|เม\.ย|พ\.ค|มิ\.ย|ก\.ค|ส\.ค|ก\.ย|ต\.ค|พ\.ย|ธ\.ค"
    r"|(?:month|เดือน(?:ที่)?)\s*(?:0?[1-9]|1[0-2])(?!\d)|(?<!\d)(?:0?[1-9]|1[0-2])[/-](?:19|20)?\d{2}(?!\d)"
    r"|(?:19|20)\d{2}-(?:0[1-9]|1[0-2])(?!\d)")


def mentions_month(text: Optional[str]) -> bool:
    """Whether the message names a month at all (which one is the LLM's reading; this only checks there is one)."""
    return bool(_MONTH.search((text or "").lower()))


# Words that SAY what an event was (the disaster type itself), English and Thai. Unlike the class words the
# answer check accepts, measurement words ("temperature", "อุณหภูมิ") are not here: naming what was measured
# is not saying what the event was.
TYPE_WORDS = {
    "Drought": ["drought", "dry spell", "ภัยแล้ง", "แล้ง"],
    "Earthquake": ["earthquake", "quake", "tremor", "แผ่นดินไหว"],
    "Epidemic": ["epidemic", "outbreak", "pandemic", "โรคระบาด", "การระบาด"],
    "Extreme Temperature": ["extreme temperature", "heatwave", "heat wave", "cold wave", "cold snap", "คลื่นความร้อน",
                            "คลื่นความหนาว", "ร้อนจัด", "หนาวจัด", "อุณหภูมิสุดขั้ว"],
    "Flood": ["flood", "ท่วม", "อุทกภัย"],
    "Landslide": ["landslide", "mudslide", "rockslide", "ดินถล่ม", "ดินโคลนถล่ม", "ดินสไลด์"],
    "Storm": ["storm", "typhoon", "cyclone", "hurricane", "tornado", "พายุ", "ไต้ฝุ่น"],
    "Wildfire": ["wildfire", "bushfire", "forest fire", "ไฟป่า"],
}


def type_stated_in_text(cls: Optional[str], text: Optional[str]) -> bool:
    """Whether the user's own words say the event was this type ("it was a storm", "น้ำท่วม").

    stated_true_type scores the model on "Your event", so the LLM may pass it only when the user said it:
    reading "wind speed was recorded" as a Storm is a guess, and it marked the model "Correct" on its own say-so."""
    low = (text or "").lower()
    return bool(cls) and any((re.search(r"(?<![a-z])" + re.escape(w), low) if w.isascii() else w in low)
                             for w in TYPE_WORDS.get(cls, []))

# The user saying nothing was measured, as opposed to not knowing. "No figure was recorded" is NOT one:
# it means a quantity was measured without a number (that is a scale with no value). Thai has no spaces, so it matches anywhere.
_NOTHING_MEASURED = re.compile(
    r"nothing (?:was |is |has been )?(?:measured|recorded)|nothing at all was (?:measured|recorded)"
    r"|no measurements? (?:was |were )?(?:taken|recorded|made|given)|no measurements?\b"
    r"|(?:was|were)(?:n't| not) measured|not measured|unmeasured"
    r"|(?:magnitude|size|measurement)s? (?:was |were )?(?:not|n't) (?:recorded|measured|given)"
    r"|no (?:recorded |known )?(?:magnitude|size|measurement)s?\b(?! (?:of|for) )"
    r"|without (?:a |any )?(?:recorded |known )?(?:magnitude|measurement)s?\b"
    r"|ไม่มีการวัด|ไม่ได้วัด|ไม่ได้มีการวัด|ไม่มีค่าที่วัด|ไม่มีข้อมูลการวัด"
    r"|ไม่(?:มี|ได้|ได้มี)(?:การ)?บันทึก(?:ขนาด|ค่า|ความรุนแรง|การวัด)|ไม่ได้บันทึก(?:ขนาด|ค่า|ความรุนแรง)"
    r"|ไม่มี\S{0,20}ขนาด\S{0,10}ที่บันทึก")


def says_nothing_measured(text: Optional[str]) -> bool:
    """Whether the user's own words say nothing was measured ("ไม่มีการวัด", "nothing was recorded").

    nothing_measured feeds the models the Unknown scale, which reads as Landslide or Epidemic. In the app,
    Gemini passed it for a bare "ไม่ทราบครับ" (I don't know), so it counts only when the words say so."""
    return bool(_NOTHING_MEASURED.search((text or "").lower()))


# What a person can SEE or FEEL of the quantity a scale measures, without a number or the name of the quantity:
# "บ้านสั่น" is the shaking a seismometer records (Moment Magnitude), "ลมแรง" the wind an anemometer records.
# Never disaster names (TYPE_WORDS; a test checks it), and never effects that any hazard could cause
# ("houses collapsed", "crops died"). The app only PROPOSES the scale and the user must confirm it, so the
# value the models see comes from the user, not from this table or the LLM. Vaccinated has no observation.
OBSERVATION_WORDS = {
    "Moment Magnitude": [
        r"(?:ground|earth|house|houses|home|building|buildings|walls?|floor|windows?)\s+(?:was\s+|were\s+|started\s+|began\s+)?"
        r"(?:shook|shaking|shake|swayed|swaying|trembled|trembling)",
        r"(?:shaking|trembling) (?:of the )?(?:ground|earth|house|building)s?",
        r"(?:บ้าน|พื้น|ตึก|อาคาร|แผ่นดิน|ผนัง|หน้าต่าง)(?:ก็)?(?:สั่น|โยก)", r"สั่นสะเทือน", r"สั่นไหว"],
    "Kph": [
        r"(?:strong|high|heavy|violent|fierce|powerful) winds?", r"winds? (?:blew|tore|howled|ripped)",
        r"(?:roofs?|trees?) (?:were |was )?(?:blown|torn) (?:off|down|away)",
        r"ลมแรง", r"ลมพัด", r"ลมกระโชก", r"ลมกรรโชก", r"หลังคา\S{0,6}ปลิว", r"ลมหอบ"],
    "°C": [
        r"(?:very|extremely|unbearably|scorching|blistering) (?:hot|cold)", r"scorching", r"bitter(?:ly)? cold",
        r"freezing (?:cold|weather|temperatures?)", r"(?:extreme|intense) (?:heat|cold)",
        r"ร้อนมาก", r"ร้อนจนทน", r"อากาศร้อน", r"หนาวมาก", r"หนาวเหน็บ", r"อากาศหนาว", r"หนาวจนทน"],
    "Km2": [
        r"(?:large|wide|vast|huge|big) (?:area|region|stretch)", r"(?:many|several|dozens of|hundreds of) "
        r"(?:villages|districts|provinces|towns)", r"(?:whole|entire) (?:province|district|region|valley)",
        r"หลาย(?:หมู่บ้าน|ตำบล|อำเภอ|จังหวัด)", r"ทั้ง(?:จังหวัด|อำเภอ|ตำบล|หมู่บ้าน)", r"พื้นที่กว้าง", r"เป็นบริเวณกว้าง"],
}
_OBSERVATION_PATTERNS = {scale: re.compile("|".join(words)) for scale, words in OBSERVATION_WORDS.items()}

# What the quantity is, for the question the app asks (in the LLM's words to the user).
SCALE_QUANTITY = {"Moment Magnitude": "the size of the ground shaking (Richter/moment magnitude)",
                  "Kph": "wind speed", "°C": "temperature", "Km2": "the affected area", "Vaccinated": "people vaccinated"}


def observed_scale(text: Optional[str]) -> Optional[Tuple[str, str]]:
    """(scale, the words matched) when the message describes seeing or feeling exactly one measured quantity.

    For a message that names no scale ("บ้านสั่นอยู่ 10 นาที"): the app asks the user to confirm the scale.
    None when it describes none, or quantities of several scales (then which one was measured is unclear)."""
    low = (text or "").lower()
    found = {scale: m.group(0) for scale, pattern in _OBSERVATION_PATTERNS.items() if (m := pattern.search(low))}
    return next(iter(found.items())) if len(found) == 1 else None


# The user's answer to "was it X that was measured?". The first yes or no word decides: "ใช่ แต่ไม่รู้เดือน"
# confirms, "ไม่แน่ใจ" does not. Thai "ไม่ใช่" contains "ใช่", hence the look-behind.
_YES_NO = re.compile(
    r"(?P<no>ไม่ใช่|ไม่แน่ใจ|ไม่ทราบ|ไม่รู้|ไม่ถูก|ไม่น่า|(?<![a-z])(?:no|nope|not sure|unsure|don'?t know|do not know|wrong|not really)(?![a-z]))"
    r"|(?P<yes>(?<!ไม่)ใช่|ถูกต้อง|(?<!ไม่)ถูก(?:แล้ว|ครับ|ค่ะ)|ประมาณนั้น|น่าจะ(?:ใช่|ถูก)"
    r"|(?<![a-z])(?:yes|yeah|yep|yup|correct|right|exactly|sure|that's it|i think so)(?![a-z]))")


def confirms(text: Optional[str]) -> bool:
    """Whether an answer confirms what the app proposed: its first yes/no word is a yes."""
    m = _YES_NO.search((text or "").lower())
    return bool(m and m.group("yes"))


_NO_MAGNITUDE = re.compile(r"\b(?:unknown|no|without an?|unrecorded|unspecified) magnitude\b"
                           r"|\bmagnitude (?:was |is )?(?:unknown|not (?:recorded|given|known|reported))\b")

# English words must start a word ("gust" is not in "August", "acre" not in "massacre"); a word may go on
# ("vaccin" -> vaccination, "degree" -> degrees). Thai has no spaces between words, so it matches anywhere.
_SCALE_PATTERNS = {
    scale: re.compile("|".join((r"(?<![a-z])" if w[0].isascii() and w[0].isalpha() else "") + re.escape(w)
                               for w in words))
    for scale, words in SCALE_WORDS.items()
}

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


MEASURE_QUESTION = "what was measured (affected area, wind speed, magnitude, temperature, people vaccinated) or that nothing was"


def missing_details(event: Dict[str, Any], nothing_measured: bool = False) -> List[str]:
    """The details to ask for before predicting, or [] if the event is described well enough.

    Asked when the place or the measured quantity is missing, or both the month and the duration are.
    "What was measured" matters most: no scale is itself evidence (every Landslide and half the Epidemics
    in the data have none), so a scale left out by mistake reads as "nothing was measured".
    nothing_measured: the user said so, which answers the question (the scale really is Unknown)."""
    where = bool(event.get("country") or event.get("region"))
    scale = nothing_measured or event.get("magnitude_scale") not in (None, "", "Unknown")
    month = bool(event.get("start_month"))
    duration = event.get("duration_days") is not None
    if where and scale and (month or duration):
        return []
    asks = [("where it happened (country)", where), (MEASURE_QUESTION, scale),
            ("the month it started", month), ("how long it lasted", duration)]
    return [text for text, given in asks if not given]


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
        # Scale "Unknown" given on purpose means nothing was measured: no value is then the real answer too.
        "Magnitude value": event.get("magnitude_value") is None and event.get("magnitude_scale") != "Unknown",
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


def case_attribute_values(case: Dict[str, Any], prediction: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """attribute_values for a case, saying "not given" for a scale the user never stated (not "Unknown",
    which means nothing was measured), with the scales the models were averaged over."""
    values = attribute_values(case.get("features", {}))
    if case.get("scale_not_given"):
        mix = (prediction or {}).get("scale_mix")
        values["Magnitude scale"] = "not given" + (
            " (models averaged over the scales usually recorded there: "
            + ", ".join(f"{m['scale']} {m['weight']:.0%}" for m in mix) + ")" if mix else "")
    seen = case.get("scale_observed")
    if seen:
        values["Magnitude scale"] += f" (from what the user observed, \"{seen['words']}\", confirmed by the user)"
    return values


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
        else f"- Magnitude: not recorded (scale: {event.get('magnitude_scale') or 'not given'})",
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
            "magnitude_scale": event.get("magnitude_scale") or "not given",
            "magnitude_missing": event.get("magnitude_value") is None,
        },
        "features": build_features(event, feature_names),
        # No scale given is not "nothing was measured" (scale Unknown): the registry then averages the models
        # over the scales usually recorded for that place and month instead of feeding them "Unknown".
        "scale_not_given": not event.get("magnitude_scale"),
        "evaluations": {},
        "is_disagreement": False,
    }


def compare_events(original: Dict[str, Any], extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Did the LLM read a typed description back into the right 6 attributes?

    Numbers may be rounded in the sentence, so they match within 10% (durations within a day)."""
    def same_number(a, b, abs_tol=0.0):
        if a is None or b is None:
            return a is None and b is None
        return abs(float(a) - float(b)) <= max(abs_tol, 0.10 * abs(float(a)))

    def norm(v):
        return None if v in (None, "", "Unknown") else v

    # Compare what the models would see: a missing month is encoded as 1 and a missing duration as 0,
    # so "not mentioned" and "January" / "0 days" are the same input, not a misread.
    def month(e):
        return e.get("start_month") or 1

    def days(e):
        return e.get("duration_days") or 0

    checks = {
        "Country": norm(original.get("country")) == norm(extracted.get("country")),
        "Region": norm(original.get("region")) == norm(extracted.get("region")),
        "Start month": month(original) == month(extracted),
        "Event duration": same_number(days(original), days(extracted), abs_tol=1.0),
        "Magnitude scale": norm(original.get("magnitude_scale")) == norm(extracted.get("magnitude_scale")),
        "Magnitude value": same_number(original.get("magnitude_value"), extracted.get("magnitude_value")),
    }
    wrong = [name for name, ok in checks.items() if not ok]
    return {
        "correct": len(checks) - len(wrong),
        "total": len(checks),
        "wrong": [{"attribute": n, "typed": _event_value(original, n), "read": _event_value(extracted, n)} for n in wrong],
    }


_EVENT_KEYS = {"Country": "country", "Region": "region", "Start month": "start_month", "Event duration": "duration_days",
               "Magnitude scale": "magnitude_scale", "Magnitude value": "magnitude_value"}


def _event_value(event: Dict[str, Any], attribute: str) -> Any:
    return event.get(_EVENT_KEYS[attribute])
