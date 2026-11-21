"""Typed-path evaluation: how well does the system classify an event when a person types it?

For a stratified sample of held-out test cases (same number per class):
  1. Gemini writes the case as a chat message WITHOUT its disaster type (the writer never sees it).
  2. The Gemini agent reads the message, as in the app, and calls classify_event.
  3. The three ML models predict from what the agent read; SHAP explains; the answer is checked.
Each case is compared with the same case's recorded features (the "direct" path), so the gap
between the two is the cost of typing. The ML models are the only predictors throughout.

Resumable: results are appended to a JSONL file and finished cases are skipped on restart.

    python -m src.evaluate_typed --per-class 10            # run (needs GEMINI_API_KEY)
    python -m src.evaluate_typed --summarize               # write the report from the results
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

OUT_DIR = Path("reports/typed_eval")
RESULTS = OUT_DIR / "results.jsonl"
MODELS = ["Random Forest", "LightGBM", "XGBoost"]
NO_MORE = {"Thai": "ไม่ทราบรายละเอียดอื่นแล้ว เท่าที่บอกไปคือทั้งหมด", "English": "I don't know any other details; that's all I have."}
ATTRIBUTES = ["Country", "Region", "Start month", "Event duration", "Magnitude scale", "Magnitude value"]


def load_key():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key and Path(".streamlit/secrets.toml").exists():
        import tomllib

        key = str(tomllib.load(open(".streamlit/secrets.toml", "rb")).get("GEMINI_API_KEY", "")).strip()
    if not key:
        sys.exit("Set GEMINI_API_KEY (environment or .streamlit/secrets.toml).")
    return key


def sample_cases(registry, per_class, seed):
    rng = random.Random(seed)
    picked = []
    for cls in registry.class_names:
        pool = [c for c in registry.cases if c["ground_truth_class"] == cls]
        picked += rng.sample(pool, min(per_class, len(pool)))
    return picked


def run_case(case, index, key, tools, registry, calls):
    from src.answer_check import check_answer
    from src.auditor_agent import GeminiAuditorAgent, SessionCases
    from src.event_features import compare_events, event_from_case
    from src.llm_tasks import write_scenario

    language = "Thai" if index % 2 == 0 else "English"
    event = event_from_case(case)
    start = calls["n"]
    record = {
        "case_id": case["case_id"], "true_type": case["ground_truth_class"], "language": language,
        "direct": {m: case["evaluations"][m]["predicted_class"] for m in MODELS},
    }
    message = write_scenario(key, event, language)
    if not message:
        record["error"] = "writer failed"
        return record
    record["message"] = message

    agent, history, session = GeminiAuditorAgent(tools, key), [], SessionCases()
    tool_names, error, asked_back = [], None, None

    def turn(user_text):
        nonlocal error
        session.message = user_text
        text = ""
        for kind, payload in agent.run_turn(history, user_text, MODELS, session):
            if kind == "text":
                text += payload
            elif kind == "retract":
                text = text[:-payload]
            elif kind == "tool":
                tool_names.append(payload["name"])
            elif kind == "error":
                error = payload
        return text

    text = turn(message)
    if not session.cases and session.asked_details_on:
        # The app asked for more details. The message already holds everything the case records, so answer
        # the way a user who knows nothing more would, and the models then run on what was given.
        asked_back = text
        text = turn(NO_MORE[language])
    record.update({"tools": tool_names, "answer": text, "asked_back": asked_back, "agent_error": error,
                   "gemini_calls": calls["n"] - start})

    typed_case = next(iter(session.cases.values()), None)
    if typed_case is None:
        record["typed"] = None  # the agent never ran the models: a failure of the typed path
        return record
    read = compare_events(event, typed_case["event"])
    record["read"] = {"correct": read["correct"], "wrong": [w["attribute"] for w in read["wrong"]],
                      "detail": read["wrong"]}
    audit = registry.audit_case(typed_case, MODELS)
    record["typed"] = {m: audit["models"][m]["prediction"]["predicted_class"] for m in MODELS}
    record["typed_confidence"] = {m: audit["models"][m]["prediction"]["confidence"] for m in MODELS}
    check = check_answer(text, [audit], [registry.leaderboard, registry.mcnemar])
    record["explanation_ok"] = None if check is None else check["ok"]
    record["explanation_issues"] = check["issues"] if check else []
    return record


def drop_results(case_ids):
    """Remove these cases from the results so the next run redoes them (e.g. after a fix that affects them)."""
    keep = [line for line in RESULTS.open() if line.strip() and json.loads(line)["case_id"] not in case_ids]
    RESULTS.write_text("".join(keep))


def run(per_class, seed, pause):
    from src import agent
    from src.auditor_agent import AuditorTools
    from src.models.registry import ModelRegistry

    key = load_key()
    registry = ModelRegistry()
    tools = AuditorTools(registry)
    calls = {"n": 0, "quota": 0}

    def hook(kind):
        calls["n" if kind == "call" else "quota"] += 1

    agent.set_usage_hook(hook)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if RESULTS.exists():
        done = {json.loads(line)["case_id"] for line in RESULTS.open() if line.strip()}

    cases = sample_cases(registry, per_class, seed)
    todo = [(i, c) for i, c in enumerate(cases) if c["case_id"] not in done]
    print(f"{len(cases)} cases sampled, {len(done)} already done, {len(todo)} to run", flush=True)
    for n, (i, case) in enumerate(todo, 1):
        for attempt in range(3):
            quota_before = calls["quota"]
            try:
                record = run_case(case, i, key, tools, registry, calls)
            except Exception as e:  # keep going; the case is retried on the next run
                record = {"case_id": case["case_id"], "error": repr(e)}
            failed = record.get("error") or record.get("typed") is None
            if failed and calls["quota"] > quota_before and attempt < 2:
                print("  quota hit; waiting 65s", flush=True)
                time.sleep(65)
                continue
            break
        if record.get("error"):
            print(f"[{n}/{len(todo)}] {case['case_id']} ERROR {record['error']}", flush=True)
        else:
            with RESULTS.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            typed = (record.get("typed") or {}).get("Random Forest")
            print(f"[{n}/{len(todo)}] {case['case_id']} {record['true_type']:>19} -> RF typed {typed} "
                  f"(direct {record['direct']['Random Forest']}), read {record.get('read', {}).get('correct')}/6, "
                  f"calls so far {calls['n']}", flush=True)
        time.sleep(pause)
    print(f"done; Gemini calls this run: {calls['n']}, quota errors: {calls['quota']}", flush=True)


def summarize():
    from sklearn.metrics import f1_score

    records = [json.loads(line) for line in RESULTS.open() if line.strip()]
    n = len(records)
    classes = sorted({r["true_type"] for r in records})
    lines = [f"# Typed-path evaluation\n",
             f"{n} held-out test cases, {n // max(len(classes), 1)} per class, half written in Thai and half in English. "
             "Gemini wrote each case as a chat message without its disaster type; the Gemini agent read it back "
             "and called the ML models, exactly as in the app. The ML models make every prediction.\n"]

    def scores(path, model):
        pairs = [(r["true_type"], (r[path] or {}).get(model) or "(no prediction)") for r in records]
        y, p = zip(*pairs)
        return sum(a == b for a, b in pairs) / n, f1_score(y, p, labels=classes, average="macro", zero_division=0)

    lines += ["## Direct features vs typed message (same cases)\n",
              "| Model | Direct accuracy | Typed accuracy | Direct Macro-F1 | Typed Macro-F1 | Macro-F1 change |",
              "|---|---|---|---|---|---|"]
    rows = []
    for m in MODELS:
        da, df = scores("direct", m)
        ta, tf = scores("typed", m)
        rows.append({"model": m, "direct_accuracy": da, "typed_accuracy": ta, "direct_macro_f1": df, "typed_macro_f1": tf})
        lines.append(f"| {m} | {da:.1%} | {ta:.1%} | {df:.3f} | {tf:.3f} | {tf - df:+.3f} |")

    ran = [r for r in records if r.get("typed")]
    lines += ["", "## The LLM as interpreter\n",
              f"- Ran the models via classify_event: **{len(ran)}/{n}** ({len(ran) / n:.0%})",
              f"- All 6 attributes read correctly: **{sum(r['read']['correct'] == 6 for r in ran)}/{len(ran)}**",
              f"- Attributes read correctly overall: **{sum(r['read']['correct'] for r in ran) / (6 * len(ran)):.1%}**"
              if ran else "- (no cases ran)"]
    if any("asked_back" in r for r in records):  # recorded since the app started asking back
        asked = [r for r in records if r.get("asked_back")]
        lines.append(f"- Asked the user for more details first (too little given): **{len(asked)}/{n}**")
    checked = [r for r in ran if r.get("explanation_ok") is not None]
    if checked:
        lines.append(f"- Explanations matching the model output: **{sum(r['explanation_ok'] for r in checked)}/{len(checked)}**")
    calls = [r.get("gemini_calls", 0) for r in records]
    lines.append(f"- Gemini requests per case: {sum(calls) / n:.1f} on average (writer + agent rounds + fallbacks)")

    wrong = Counter(a for r in ran for a in r["read"]["wrong"])
    lines += ["", "| Attribute | Read correctly |", "|---|---|"]
    for a in ATTRIBUTES:
        lines.append(f"| {a} | {(len(ran) - wrong[a]) / len(ran):.0%} |" if ran else f"| {a} | — |")

    lines += ["", "## Per class (Random Forest)\n", "| Class | Cases | Direct correct | Typed correct |", "|---|---|---|---|"]
    per = defaultdict(lambda: [0, 0, 0])
    for r in records:
        p = per[r["true_type"]]
        p[0] += 1
        p[1] += r["direct"]["Random Forest"] == r["true_type"]
        p[2] += (r.get("typed") or {}).get("Random Forest") == r["true_type"]
    for cls in classes:
        c, d, t = per[cls]
        lines.append(f"| {cls} | {c} | {d}/{c} | {t}/{c} |")

    # Where typing changed the answer: was it a misread, or the model?
    flips = [r for r in ran if r["direct"]["Random Forest"] == r["true_type"] != r["typed"]["Random Forest"]]
    lines += ["", f"## Cases right on direct features but wrong when typed (Random Forest): {len(flips)}\n"]
    for r in flips:
        lines.append(f"- {r['case_id']} ({r['true_type']} → {r['typed']['Random Forest']}), misread: "
                     f"{', '.join(r['read']['wrong']) or 'nothing'} · “{r['message']}”")
    lines += ["", "Small sample: treat differences of a few cases as noise. The full-test-set figures "
              "(3,495 cases) on the Leaderboard tab remain the reference for the models themselves."]

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    import csv

    with (OUT_DIR / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pause", type=float, default=6.0, help="seconds between cases (free-tier rate limits)")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--redo", help="comma-separated case ids to drop from the results and run again")
    args = ap.parse_args()
    if args.redo:
        drop_results({c.strip().upper() for c in args.redo.split(",") if c.strip()})
    summarize() if args.summarize else run(args.per_class, args.seed, args.pause)
