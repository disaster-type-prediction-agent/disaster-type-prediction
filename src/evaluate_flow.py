"""Flow evaluation: the confirmation question and languages other than Thai and English, measured as rates.

The typed eval (src/evaluate_typed.py) never sends a message that describes what the user saw ("บ้านสั่น") or
one in another language, so those paths were only tried by hand, once each. Gemini answers differently from run
to run, so one try says little. Here a fixed set of hand-written conversations is run several times each:

  1. The user describes the event (an observation with no number, or a named quantity, in one of 8 languages).
  2. The app should ask to confirm the quantity, in its own words (no Gemini rewording, no type named).
  3. The simulated user answers: yes / no / not sure, sometimes with an extra detail.
  4. What reached the models is checked against what the conversation said:
     confirmed scale used, denied scale left out of the averaging, "not sure" changes nothing, and
     country, month, duration and figure read correctly.

The messages are fixed (no writer LLM), so a change in the rates comes from the code or from Gemini's variance.
The ML models make every prediction; the intended type of each conversation is recorded for reference only.

    python -m src.evaluate_flow --repeats 3      # run (needs GEMINI_API_KEY; ~3 Gemini calls per run)
    python -m src.evaluate_flow --summarize      # write reports/flow_eval/summary.md from the results
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from src.evaluate_typed import MODELS, load_key

OUT_DIR = Path("reports/flow_eval")
RESULTS = OUT_DIR / "results.jsonl"
MM, KPH, DEG, KM2 = "Moment Magnitude", "Kph", "°C", "Km2"

# Each conversation: what the user writes first, how they answer the app's question, and what the models
# should then get. "proposes": the scale the app should ask to confirm (None: no question expected).
# "expect": the final event; "scale": the scale used (None = not given, averaged), "ruled_out": left out.
CASES = [
    # Thai: the code reads it (regex guards).
    dict(id="th-shake-yes", language="Thai", intended="Earthquake", proposes=MM,
         message="ที่ประเทศไทย เดือนพฤษภาคม บ้านสั่นอยู่นานประมาณ 1 วัน", answer="ใช่ครับ",
         expect=dict(country="Thailand", start_month=5, duration_days=1, scale=MM)),
    dict(id="th-shake-no", language="Thai", intended="Earthquake", proposes=MM,
         message="ที่ประเทศไทย เดือนพฤษภาคม บ้านสั่นอยู่นานประมาณ 1 วัน", answer="ไม่ใช่ครับ",
         expect=dict(country="Thailand", start_month=5, duration_days=1, scale=None, ruled_out=MM)),
    dict(id="th-shake-unsure", language="Thai", intended="Earthquake", proposes=MM,
         message="ที่ประเทศไทย เดือนพฤษภาคม บ้านสั่นอยู่นานประมาณ 1 วัน", answer="ไม่แน่ใจครับ",
         expect=dict(country="Thailand", start_month=5, duration_days=1, scale=None)),
    dict(id="th-shake-yes-figure", language="Thai", intended="Earthquake", proposes=MM,
         message="ที่ประเทศไทย บ้านสั่นอยู่นานประมาณ 1 วัน", answer="ใช่ครับ วัดได้ 6.3 ริกเตอร์ เดือนพฤษภาคม",
         expect=dict(country="Thailand", start_month=5, duration_days=1, scale=MM, value=6.3)),
    dict(id="th-wind-yes", language="Thai", intended="Storm", proposes=KPH,
         message="ที่ฟิลิปปินส์ เดือนพฤศจิกายน ลมแรงมากอยู่ 2 วัน", answer="ใช่ครับ",
         expect=dict(country="Philippines", start_month=11, duration_days=2, scale=KPH)),
    dict(id="th-heat-no", language="Thai", intended="Extreme Temperature", proposes=DEG,
         message="ที่อินเดีย เดือนพฤษภาคม อากาศร้อนมากนาน 2 สัปดาห์", answer="ไม่ใช่ครับ",
         expect=dict(country="India", start_month=5, duration_days=14, scale=None, ruled_out=DEG)),
    # English: the code reads it.
    dict(id="en-shake-yes", language="English", intended="Earthquake", proposes=MM,
         message="In Nepal in April my house shook for about a day.", answer="Yes.",
         expect=dict(country="Nepal", start_month=4, duration_days=1, scale=MM)),
    dict(id="en-wind-no", language="English", intended="Storm", proposes=KPH,
         message="Strong winds in the Philippines in November for 2 days.", answer="No, that's not it.",
         expect=dict(country="Philippines", start_month=11, duration_days=2, scale=None, ruled_out=KPH)),
    dict(id="en-area-yes", language="English", intended="Flood", proposes=KM2,
         message="Many villages in Bangladesh were affected in July for 3 weeks.", answer="Yes, that's right.",
         expect=dict(country="Bangladesh", start_month=7, duration_days=21, scale=KM2)),
    # Other languages: Gemini must quote the user's words (quoted()), the code checks the quotes.
    dict(id="ja-shake-yes", language="Japanese", intended="Earthquake", proposes=MM,
         message="2014年5月にタイで、家が1日ほど揺れました。", answer="はい、そうです。",
         expect=dict(country="Thailand", start_month=5, duration_days=1, scale=MM)),
    dict(id="es-shake-no", language="Spanish", intended="Earthquake", proposes=MM,
         message="En mayo, en Chile, mi casa tembló durante un día.", answer="No, no es eso.",
         expect=dict(country="Chile", start_month=5, duration_days=1, scale=None, ruled_out=MM)),
    dict(id="zh-wind-yes", language="Chinese", intended="Storm", proposes=KPH,
         message="2019年11月在菲律宾，大风刮了两天。", answer="是的。",
         expect=dict(country="Philippines", start_month=11, duration_days=2, scale=KPH)),
    dict(id="vi-heat-yes", language="Vietnamese", intended="Extreme Temperature", proposes=DEG,
         message="Tháng 5 ở Ấn Độ, trời nóng khủng khiếp suốt hai tuần.", answer="Đúng vậy.",
         expect=dict(country="India", start_month=5, duration_days=14, scale=DEG)),
    dict(id="my-shake-yes", language="Burmese", intended="Earthquake", proposes=MM,
         message="မေလမှာ မြန်မာနိုင်ငံမှာ အိမ်တွေ တစ်ရက်လောက် လှုပ်ခဲ့တယ်။", answer="ဟုတ်ကဲ့။",
         expect=dict(country="Myanmar", start_month=5, duration_days=1, scale=MM)),
    dict(id="id-area-unsure", language="Indonesian", intended="Flood", proposes=KM2,
         message="Banyak desa di Indonesia terdampak pada bulan Januari selama seminggu.", answer="Tidak tahu.",
         expect=dict(country="Indonesia", start_month=1, duration_days=7, scale=None)),
    # A named quantity with its figure, in a language the code can't read: the value must survive the quote
    # check (asked to confirm or used directly; the user says yes if asked).
    dict(id="ja-named-figure", language="Japanese", intended="Earthquake", proposes=None,
         message="2011年3月、日本でマグニチュード9.0の揺れが1日続きました。", answer="はい。",
         expect=dict(country="Japan", start_month=3, duration_days=1, scale=MM, value=9.0)),
    dict(id="es-named-figure", language="Spanish", intended="Storm", proposes=None,
         message="En México, en octubre, vientos de 200 km/h durante 2 días.", answer="Sí.",
         expect=dict(country="Mexico", start_month=10, duration_days=2, scale=KPH, value=200)),
]
CHECKS = ["asked_as_expected", "question_clean", "scale_ok", "ruled_out_ok", "country_ok", "month_ok",
          "duration_ok", "value_ok"]


def score(case, asked, proposed, question_text, event, ruled_out):
    """Compare one run with what the conversation said. Each check is True/False (None: not applicable)."""
    from src.event_features import names_a_type

    exp = case["expect"]
    if case["proposes"]:
        asked_ok = proposed == case["proposes"]
    else:
        asked_ok = proposed is None  # a confirmation of the named scale itself is fine; another scale is not
        asked_ok = asked_ok or proposed == exp["scale"]
    out = {"asked_as_expected": asked_ok,
           # Shown before the models ran: must not name or hint at a type (the app writes its own question).
           "question_clean": None if not asked else not names_a_type(question_text)}
    if event is None:
        return {**out, **{k: False for k in CHECKS if k not in out}}
    scale = event.get("magnitude_scale") if event.get("magnitude_scale") not in ("", "Unknown") else None
    dur = event.get("duration_days")
    out.update({
        "scale_ok": scale == exp["scale"],
        "ruled_out_ok": (ruled_out or []) == ([exp["ruled_out"]] if exp.get("ruled_out") else []),
        "country_ok": event.get("country") == exp["country"],
        "month_ok": event.get("start_month") == exp["start_month"],
        "duration_ok": dur is not None and abs(dur - exp["duration_days"]) <= max(1, 0.1 * exp["duration_days"]),
        "value_ok": None if "value" not in exp else (
            event.get("magnitude_value") is not None and abs(event["magnitude_value"] - exp["value"]) <= 0.1 * exp["value"]),
    })
    return out


def run_case(case, key, tools, registry, calls):
    from src.agent import agent_turn_text
    from src.auditor_agent import GeminiAuditorAgent, SessionCases

    agent, history, session = GeminiAuditorAgent(tools, key), [], SessionCases()
    start, errors = calls["n"], []

    def turn(user_text):
        session.message = user_text
        text = ""
        for kind, payload in agent.run_turn(history, agent_turn_text(user_text, [f"selected models: {', '.join(MODELS)}"]),
                                            MODELS, session):
            if kind == "text":
                text += payload
            elif kind == "retract":
                text = text[:-payload]
            elif kind == "error":
                errors.append(str(payload))
        return text

    first = turn(case["message"])
    asked = bool(not session.cases and session.asked_details_on)
    proposed = session.proposed_scale[0] if (asked and session.proposed_scale) else None
    reply = turn(case["answer"]) if asked else None
    typed = next(iter(session.cases.values()), None)
    event = typed["event"] if typed else None
    record = {"case": case["id"], "language": case["language"], "asked": asked, "proposed": proposed,
              "question": first if asked else None, "reply": reply if asked else first,
              "event": event, "ruled_out": typed.get("scale_ruled_out") if typed else None,
              "agent_errors": errors, "gemini_calls": calls["n"] - start}
    record["checks"] = score(case, asked, proposed, first if asked else "", event, record["ruled_out"])
    if typed:
        audit = registry.audit_case(typed, MODELS)
        record["predicted"] = {m: audit["models"][m]["prediction"]["predicted_class"] for m in MODELS}
    return record


def run(repeats, pause, only=None):
    from src import agent
    from src.auditor_agent import AuditorTools
    from src.models.registry import ModelRegistry

    key = load_key()
    registry = ModelRegistry()
    tools = AuditorTools(registry)
    calls = {"n": 0, "quota": 0}
    agent.set_usage_hook(lambda kind: calls.__setitem__("n" if kind == "call" else "quota",
                                                        calls["n" if kind == "call" else "quota"] + 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = defaultdict(int)
    if RESULTS.exists():
        for r in map(json.loads, filter(str.strip, RESULTS.open())):
            done[r["case"]] += 1
    todo = [c for c in CASES if not only or c["id"] in only for _ in range(max(0, repeats - done[c["id"]]))]
    print(f"{len(CASES)} conversations x {repeats} runs; {len(todo)} runs to go", flush=True)
    quota_start = calls["quota"]
    for n, case in enumerate(todo, 1):
        for attempt in range(3):
            quota_before = calls["quota"]
            try:
                record = run_case(case, key, tools, registry, calls)
            except Exception as e:  # keep going; the run is retried on the next start
                record = {"case": case["id"], "error": repr(e)}
            failed = record.get("error") or (record.get("event") is None and record.get("agent_errors"))
            if failed and calls["quota"] > quota_before and attempt < 2:
                print("  quota hit; waiting 65s", flush=True)
                time.sleep(65)
                continue
            break
        if record.get("error"):
            print(f"[{n}/{len(todo)}] {case['id']} ERROR {record['error']}", flush=True)
        elif record.get("event") is None and record.get("agent_errors"):
            # Gemini timed out or ran out of quota before the models ran: that measures the free key, not the
            # flow, so the run is not recorded and is redone on the next start.
            print(f"[{n}/{len(todo)}] {case['id']} SKIPPED (Gemini error: {record['agent_errors'][-1][:60]})", flush=True)
            if calls["quota"] - quota_start >= 6:
                print("free quota used up; stopping. Run again later to finish the remaining runs.", flush=True)
                break
        else:
            with RESULTS.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            bad = [k for k, v in record["checks"].items() if v is False]
            print(f"[{n}/{len(todo)}] {case['id']:<20} asked {record['proposed'] or record['asked']!s:<17} "
                  f"{'OK' if not bad else 'FAIL ' + ', '.join(bad)}  (calls so far {calls['n']})", flush=True)
        time.sleep(pause)
    print(f"done; Gemini calls this run: {calls['n']}, quota errors: {calls['quota']}", flush=True)


def summarize():
    records = [json.loads(line) for line in RESULTS.open() if line.strip()]
    by_case = defaultdict(list)
    for r in records:
        by_case[r["case"]].append(r)
    n = len(records)

    def rate(rs, check):
        vals = [r["checks"].get(check) for r in rs if r["checks"].get(check) is not None]
        return f"{sum(vals)}/{len(vals)}" if vals else "—"

    def all_ok(r):
        return all(v is not False for v in r["checks"].values())

    lines = ["# Flow evaluation: confirmation question and other languages\n",
             f"{len(by_case)} of {len(CASES)} fixed conversations run, in {len({r['language'] for r in records})} "
             f"of {len({c['language'] for c in CASES})} languages, {n} runs in all (each conversation run several times: Gemini varies from run to run). "
             "The ML models make every prediction; these checks are about what reaches them.\n",
             f"**Runs fully right: {sum(map(all_ok, records))}/{n} ({sum(map(all_ok, records)) / max(n, 1):.0%})**\n",
             "## Per check (all runs)\n", "| Check | Passed |", "|---|---|"]
    labels = {"asked_as_expected": "Asked to confirm the expected quantity (or nothing, for a named one)",
              "question_clean": "Question shown before the models ran names no type",
              "scale_ok": "Scale given to the models matches the answer",
              "ruled_out_ok": "\"No\" left that scale out of the averaging (and only then)",
              "country_ok": "Country", "month_ok": "Start month", "duration_ok": "Duration",
              "value_ok": "Figure (where the user gave one)"}
    for c in CHECKS:
        lines.append(f"| {labels[c]} | {rate(records, c)} |")

    lines += ["", "## Per conversation\n",
              "| Conversation | Language | Runs fully right | Asked as expected | Scale | Ruled out | Country | Month | Duration | Figure | RF prediction (intended) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in CASES:
        rs = by_case.get(case["id"], [])
        if not rs:
            continue
        preds = defaultdict(int)
        for r in rs:
            preds[(r.get("predicted") or {}).get("Random Forest", "(none)")] += 1
        pred = ", ".join(f"{p} ×{k}" for p, k in sorted(preds.items(), key=lambda x: -x[1]))
        lines.append(f"| {case['id']} | {case['language']} | {sum(map(all_ok, rs))}/{len(rs)} | "
                     + " | ".join(rate(rs, c) for c in ["asked_as_expected", "scale_ok", "ruled_out_ok", "country_ok",
                                                         "month_ok", "duration_ok", "value_ok"])
                     + f" | {pred} ({case['intended']}) |")

    lang = defaultdict(list)
    for r in records:
        lang[r["language"]].append(r)
    lines += ["", "## Per language\n", "| Language | Runs fully right |", "|---|---|"]
    for name, rs in lang.items():
        lines.append(f"| {name} | {sum(map(all_ok, rs))}/{len(rs)} |")

    fails = [r for r in records if not all_ok(r)]
    lines += ["", f"## Runs with a failed check: {len(fails)}\n"]
    for r in fails:
        bad = ", ".join(k for k, v in r["checks"].items() if v is False)
        ev = r.get("event") or {}
        got = {k: ev.get(k) for k in ("country", "start_month", "duration_days", "magnitude_scale", "magnitude_value")}
        lines.append(f"- {r['case']}: {bad} · asked {r['proposed'] or r['asked']} · got {got}"
                     + (f" · ruled out {r['ruled_out']}" if r.get("ruled_out") else ""))
    lines.append(f"\nGemini requests per run: {sum(r.get('gemini_calls', 0) for r in records) / max(n, 1):.1f} on average.")
    lines.append("Small sample: each rate is out of a few runs per conversation; treat one or two runs as noise.")
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--pause", type=float, default=6.0, help="seconds between runs (free-tier rate limits)")
    ap.add_argument("--only", help="comma-separated conversation ids to run")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    only = {c.strip() for c in args.only.split(",")} if args.only else None
    summarize() if args.summarize else run(args.repeats, args.pause, only)
