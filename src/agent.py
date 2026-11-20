import os
import json
import threading
import time
from typing import Any, Callable, Dict, Generator, List, Optional

SYSTEM_PROMPT = """You are the "Disaster Auditor Agent", an expert AI specializing in natural disaster classification, machine learning evaluation, and Explainable AI (SHAP).

APPLICATION CONTEXT & DOMAIN KNOWLEDGE:
- You audit a 8-class natural disaster type classifier built on the EM-DAT historical benchmark dataset.
- The 8 disaster categories are: Drought, Earthquake, Epidemic, Extreme Temperature, Flood, Landslide, Storm, Wildfire.
- Test bank size: 3,495 historical events evaluated on a 20% stratified holdout set.
- Class imbalance is ~12:1 (Floods and Storms dominate; Droughts and Wildfires are minority classes).
- Primary evaluation metric is Macro-F1 (unweighted mean of class F1-scores), NOT raw accuracy.

THE 3 LEAK-FREE MODELS COMPARED:
1. Random Forest (SMOTETomek Pipeline): Test Macro-F1 = 0.8518, Accuracy = 91.70%, Recall = 86.94% (Best minority class recall on Wildfire & Landslide).
2. LightGBM (SMOTETomek Pipeline): Test Macro-F1 = 0.8489, Accuracy = 91.85%, Recall = 86.32% (Fastest inference, leaf-wise gradient boosting).
3. XGBoost (SMOTETomek Pipeline): Test Macro-F1 = 0.8480, Accuracy = 91.47%, Recall = 86.62% (Regularized depth-wise gradient boosting).
- Disagreement rate: The 3 models disagree on 251 out of 3,495 cases (7.2%), mostly on boundary confusions like Flood vs. Storm during monsoon seasons or Drought vs. Extreme Temperature.

MODEL INPUTS (245 features from 6 pre-event attributes):
- Country (230 one-hot columns), magnitude scale (6 one-hot: Km2, Kph, Moment Magnitude, Vaccinated, °C, Unknown), region (5 one-hot), start month (1-12), event duration (log1p of days scaled 0-1), magnitude value (log1p scaled 0-1, plus a missing flag).
- No impact data (deaths, damage) is used, to avoid leakage. The magnitude scale is the strongest signal because each hazard is measured in its own unit.

WHERE YOUR ANSWERS COME FROM:
- Facts about the models and metrics: only the figures above.
- Facts about a case: only the CURRENT AUDIT CASE CONTEXT below, if present.
- Anything else (disaster science, history, general ML): your own general knowledge. When an answer relies on it, say so briefly (e.g. "From general knowledge, not the app's data: ..."). Never invent metrics, case details or SHAP values.

GUIDELINES:
- Answer user questions informatively, clearly, and authoritatively.
- Be concise: lead with the answer, then at most 3-5 short bullets or ~150 words, unless the user asks for more detail.
- Support both Thai and English naturally: if the user asks in Thai, reply in fluent, polite Thai (ครับ/ค่ะ); if in English, reply in English.
- If an active audit case is provided in the context, refer to its specific metrics, ground truth, predicted class, confidence, and SHAP features.
- READING SHAP: use the "SHAP by attribute" % shares in the context. Compare models by these shares only (raw SHAP units differ: probability for Random Forest, log-odds for LightGBM/XGBoost). An attribute marked as a filled-in default was not given by the user: never present it as a reason for the prediction; instead say the prediction may change once the user supplies it, and invite them to.
- WHY NOT ANOTHER CLASS: when a model is WRONG, use its "Why not <true type>?" lines: name what held the true type back and what evidence both classes share. Use the runner-up lines when asked why the model didn't pick something else.
- If asked about how models work or error diagnostics, explain the physical and seasonal disaster dynamics (e.g. why missing magnitude scales sway models toward floods, or how event duration influences epidemics vs. earthquakes)."""

# Fast, current models in order of preference (measured on this project: ~1-2s to the first word).
# Override with GEMINI_MODEL in .streamlit/secrets.toml or the environment, e.g. "gemini-3.5-flash"
# or a comma-separated list.
DEFAULT_GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash"]
GEMINI_TIMEOUT_MS = 30_000
HISTORY_TURNS = 10  # earlier chat turns sent with each question

# Yielded by chat_stream when a model failed mid-answer and the next one starts over: drop the text so far.
RESTART = object()


def gemini_models() -> List[str]:
    configured = [m.strip() for m in os.getenv("GEMINI_MODEL", "").split(",") if m.strip()]
    return configured + [m for m in DEFAULT_GEMINI_MODELS if m not in configured]


# Usage hook: the app sets one per script run (thread-local, so sessions don't mix) to count calls
# against its quota guard. Called with "call" for every request and "quota" when Gemini answers 429.
_usage = threading.local()


def set_usage_hook(hook: Optional[Callable[[str], None]]) -> None:
    _usage.hook = hook


def _report(kind: str) -> None:
    hook = getattr(_usage, "hook", None)
    if hook:
        hook(kind)


def _is_quota_error(e: Exception) -> bool:
    return getattr(e, "code", None) == 429


class _CountedModels:
    """client.models with every request reported to the usage hook (the app's quota guard)."""

    def __init__(self, models):
        self._models = models

    def generate_content(self, **kwargs):
        _report("call")
        try:
            return self._models.generate_content(**kwargs)
        except Exception as e:
            if _is_quota_error(e):
                _report("quota")
            raise

    def generate_content_stream(self, **kwargs):
        _report("call")
        try:
            yield from self._models.generate_content_stream(**kwargs)
        except Exception as e:
            if _is_quota_error(e):
                _report("quota")
            raise


class _CountedClient:
    def __init__(self, client):
        # Keep the real client alive: genai.Client closes its HTTP connection when garbage-collected,
        # and holding only client.models would let that happen mid-request.
        self._client = client
        self.models = _CountedModels(client.models)


def make_gemini_client(api_key: str):
    """Every Gemini request in the app goes through this client, so every one is counted."""
    return _CountedClient(_make_raw_client(api_key))


def _make_raw_client(api_key: str):
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            # Give up on a stalled model after 30s instead of waiting minutes, and don't let the
            # SDK silently retry an overloaded model: callers fall back down gemini_models() instead.
            timeout=GEMINI_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


class DisasterAuditorLLM:
    """Plain chat via Google Gemini, or a local summary when no key is set.

    Claude runs as a tool-using agent instead; see src/auditor_agent.py.
    """

    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None):
        self.provider = provider.lower()
        env_var = "GEMINI_API_KEY" if self.provider == "gemini" else "ANTHROPIC_API_KEY"
        self.api_key = (api_key or "").strip() or os.getenv(env_var)

    def chat_stream(self, user_message: str, history: List[Dict[str, str]] = None, current_case: Optional[Dict] = None) -> Generator[str, None, None]:
        """Stream conversational response token-by-token."""
        prompt = self._build_prompt(user_message, history or [], current_case)

        if self.provider == "gemini" and self.api_key:
            yield from self._stream_gemini(prompt)
        else:
            yield from self._stream_local_fallback(user_message, current_case)

    @staticmethod
    def _case_context(current_case: Optional[Dict]) -> str:
        """Everything the models saw and said about the case on screen, so Gemini explains from data, not guesses."""
        if not current_case:
            return ""
        from src.event_features import attribute_values

        case = current_case.get("case", {})
        filled = set(case.get("filled", []))
        source = "an event the user described" if case.get("source") == "user" else "the held-out test bank"
        truth = case.get("ground_truth_class") or "not confirmed by the user yet"
        lines = [f"\n\nCURRENT AUDIT CASE CONTEXT ({source}):", f"- Case ID: {case.get('case_id')}", f"- Ground truth: {truth}",
                 "- The 6 attributes the models saw:"]
        for name, value in attribute_values(case.get("features", {})).items():
            note = "  [NOT GIVEN by the user: training default filled in]" if name in filled else ""
            lines.append(f"    {name}: {value}{note}")

        for m_name, m in current_case.get("models", {}).items():
            pred = m["prediction"]
            correct = "unverified" if m["is_correct"] is None else ("correct" if m["is_correct"] else "WRONG")
            top = sorted(pred.get("probabilities", {}).items(), key=lambda kv: kv[1], reverse=True)[:3]
            lines.append(f"- {m_name}: predicted {pred['predicted_class']} ({pred['confidence'] * 100:.1f}%, {correct}); "
                         f"top probabilities: " + ", ".join(f"{c} {v * 100:.0f}%" for c, v in top))
            rows = m["shap"].get("by_attribute", [])
            if rows:
                lines.append(f"  SHAP by attribute for {m['shap']['target_class']} (% share of this model's total evidence; + pushes toward, - away):")
                for r in rows:
                    note = " [filled-in default: an artefact, not a reason]" if r.get("filled") else ""
                    lines.append(f"    {r['attribute']} = {r['value']}: {r['share']:+.1f}%{note}")
            contrast = m.get("contrast")
            if contrast:
                lines.append(f"  Why not {contrast['class']} ({contrast['reason']}, {(contrast['probability'] or 0) * 100:.0f}%)? "
                             f"SHAP by attribute toward {contrast['class']}: "
                             + ", ".join(f"{r['attribute']} {r['share']:+.0f}%" for r in contrast["by_attribute"]))
                lines.append("    " + "; ".join(contrast["human_explanations"]))

        guess = current_case.get("llm")
        if guess:
            lines.append(f"- Gemini's own guess from the same 6 attributes: {guess['predicted_class']} ({guess['reason']})")
        return "\n".join(lines) + "\n"

    def _build_prompt(self, user_message: str, history: List[Dict[str, str]], current_case: Optional[Dict]) -> str:
        context_str = self._case_context(current_case)
        # Recent turns so follow-up questions ("why?", "and XGBoost?") make sense.
        recent = [h for h in history if h.get("content")][-HISTORY_TURNS:]
        convo = "".join(f"{h['role'].upper()}: {h['content'][:1500]}\n" for h in recent)
        convo_str = f"\n\nCONVERSATION SO FAR:\n{convo}" if convo else ""
        return f"{SYSTEM_PROMPT}\n{context_str}{convo_str}\n\nUser Question:\n{user_message}"

    def _stream_gemini(self, prompt: str) -> Generator[str, None, None]:
        """Stream from a pinned, fast model; fall back down a short list only if a model fails before answering.

        The old version listed every model on the key before each request and tried them in API order,
        so it burned seconds on retired (404) and image-only models and could land on an overloaded or
        very slow model (one took 200s to start). Pinning avoids all of that.
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            yield "\n⚠️ The `google-genai` package is not installed. Run `pip install -r requirements.txt`."
            return

        client = make_gemini_client(self.api_key)

        # Gemini's fast models answer ~1 in 5 requests with "503 high demand", sometimes mid-answer.
        # Each failure moves on to the next model; if text was already shown, RESTART tells the
        # caller to clear it so the user sees one clean answer.
        last_err = None
        for model_name in gemini_models():
            started = False
            try:
                for chunk in client.models.generate_content_stream(model=model_name, contents=prompt):
                    if getattr(chunk, "text", None):
                        started = True
                        yield chunk.text
                return
            except Exception as e:
                last_err = e
                if started:
                    yield RESTART

        yield f"\n⚠️ Gemini API Error: {last_err}. Please verify your Gemini API key in the sidebar."

    def local_answer(self, user_message: str, current_case: Optional[Dict]) -> Generator[str, None, None]:
        """The built-in summary, without any LLM call (used when Gemini is off, busy or over its limit)."""
        yield from self._stream_local_fallback(user_message, current_case)

    def _stream_local_fallback(self, user_message: str, current_case: Optional[Dict]) -> Generator[str, None, None]:
        msg = user_message.lower()
        if "thai" in msg or any("\u0e00" <= c <= "\u0e7f" for c in user_message):
            # Thai fallback
            resp = "สวัสดีครับ! ผมคือ **Disaster Auditor AI**\n\n"
            if current_case:
                c = current_case["case"]
                resp += f"สำหรับเคสปัจจุบัน **`{c['case_id']}`** (เหตุการณ์จริง: **{c['ground_truth_class'] or 'ยังไม่ระบุ'}** ที่ประเทศ {c['summary']['country']}):\n"
                for m_name, m_data in current_case["models"].items():
                    c_status = "ยังไม่ยืนยันเฉลย" if m_data["is_correct"] is None else "ถูกต้อง ✓" if m_data["is_correct"] else "ทำนายผิด ✗"
                    resp += f"• **{m_name}**: ทำนายเป็น `{m_data['prediction']['predicted_class']}` ({m_data['prediction']['confidence']*100:.1f}%) — {c_status}\n"
                resp += "\n💡 *คำแนะนำ: คุณสามารถใส่ **Gemini API Key** ในแท็บ Settings ที่แถบเมนูด้านซ้าย เพื่อเปิดการสนทนาและวิเคราะห์เชิงลึกด้วย LLM ได้แบบเรียลไทม์ครับ*"
            else:
                resp += "ระบบกำลังทำงานในโหมด **Local Benchmark Engine** (โมเดล Random Forest, LightGBM, และ XGBoost พร้อมใช้งาน)\n\n"
                resp += "คุณสามารถ:\n"
                resp += "1. คลิกปุ่ม **Quick Actions** ด้านบน (เช่น 🎲 Random Case หรือ ⚡ Disagreement Case)\n"
                resp += "2. พิมพ์คำสั่ง เช่น `/random`, `/hard`, `/compare`\n"
                resp += "3. ใส่ **Gemini API Key** ที่ Sidebar ซ้ายมือเพื่อเปิดใช้งานการถาม-ตอบอัจฉริยะด้วย Gemini ครับ"
        else:
            resp = "Hello! I am the **Disaster Auditor AI**.\n\n"
            if current_case:
                c = current_case["case"]
                resp += f"Regarding the active case **`{c['case_id']}`** (Ground Truth: **{c['ground_truth_class'] or 'not confirmed'}**, {c['summary']['country']}):\n"
                for m_name, m_data in current_case["models"].items():
                    c_status = "UNVERIFIED" if m_data["is_correct"] is None else "CORRECT ✓" if m_data["is_correct"] else "MISCLASSIFIED ✗"
                    resp += f"• **{m_name}**: Predicted `{m_data['prediction']['predicted_class']}` ({m_data['prediction']['confidence']*100:.1f}%) — {c_status}\n"
                resp += "\n💡 *Tip: Provide your **Gemini API Key** in the sidebar settings to unlock live generative dialogue and deep XAI explanations.*"
            else:
                resp += "The application is currently running with the **Local Zero-Leakage ML Engines** (Random Forest, LightGBM, XGBoost).\n\n"
                resp += "You can test cases using the quick buttons above or slash commands like `/random` and `/compare`.\n"
                resp += "To enable live natural language reasoning, enter your **Gemini API Key** in the sidebar."

        # Simulate streaming chunks
        for word in resp.split(" "):
            yield word + " "
            time.sleep(0.015)
