import json
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics import f1_score

# Keys in a local .env file become environment variables (already-set variables win).
load_dotenv()

from src.agent import RESTART, DisasterAuditorLLM, agent_turn_text, reply_language, set_usage_hook
from src.auditor_agent import AuditorTools, ClaudeAuditorAgent, GeminiAuditorAgent, SessionCases, agent_sources
from src.event_features import RAW_ATTRIBUTES, attribute_columns, countries_from, event_from_case, make_event_case, compare_events, type_stated_in_text
from src.llm_tasks import llm_guess, route_question, write_scenario
from src.answer_check import check_answer
from src.code_version import build_info, build_label, code_version
from src.agent import SYSTEM_PROMPT as CHAT_PROMPT
from src.auditor_agent import SYSTEM_PROMPT as AGENT_PROMPT
from src.models.registry import ModelRegistry

# One minimal mark for the whole app: browser tab, sidebar title, assistant avatar and greeting.
APP_ICON = ":material/cyclone:"

st.set_page_config(
    page_title="Disaster Auditor",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Chat-style polish that theme config can't express
# (colors and fonts live in .streamlit/config.toml)
# -------------------------------------------------------------
st.html("""
<style>
    /* Assistant turns: no bubble, just text on the page */
    [data-testid="stChatMessage"] {
        background: transparent;
        padding: 0.25rem 0;
    }

    /* User turns: right-aligned soft bubble, no avatar */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        background: #F0EEE6;
        border-radius: 1.1rem;
        padding: 0.55rem 1rem;
        margin-left: auto;
        width: fit-content;
        max-width: 85%;
    }
    [data-testid="stChatMessageAvatarUser"] { display: none; }

    /* Composer: floating rounded box */
    [data-testid="stChatInput"] {
        border-radius: 1.25rem;
        box-shadow: 0 0.25rem 1.25rem rgba(31, 30, 29, 0.07);
    }

    /* Thin scrollbar when a long draft overflows the box */
    [data-testid="stChatInputTextArea"] { scrollbar-width: thin; }

    /* Toolbar sits inside the composer's bottom edge, left of the send button */
    .st-key-composer { gap: 0; container-type: inline-size; }
    .st-key-composer_tools {
        margin-top: -2.9rem;
        padding: 0 3.4rem 0.45rem 0.6rem;
        position: relative;
        z-index: 2;
        pointer-events: none;
        flex-wrap: nowrap;  /* never let the model picker drop out of the box */
    }
    .st-key-composer_tools > * { pointer-events: auto; }
    .st-key-composer_tools [data-testid="stElementContainer"] { margin: 0; }
    .st-key-composer_tools > :nth-child(2) { min-width: 0; }  /* the spacer gives way first */

    /* "+" reads as a plain icon button: drop the menu chevron */
    .st-key-quick_menu [data-testid="stMenuButtonButton"] [data-testid="stIconMaterial"] { display: none; }

    /* Narrow composer: icon-only toggle and a slimmer model picker */
    @container (max-width: 420px) {
        .st-key-composer_tools [data-testid="stCheckbox"]:has(input[aria-label="Compare"]) [data-testid="stMarkdownContainer"] { display: none; }
        .st-key-model_picker { width: 7.5rem !important; }
    }

    /* Model picker reads as plain text, like a chat app's model name */
    .st-key-model_picker [data-baseweb="select"] > div {
        background: transparent;
        border-color: transparent;
        font-size: 0.875rem;
    }
    .st-key-model_picker [data-baseweb="select"] > div:hover { background: rgba(31, 30, 29, 0.05); }

    /* Slash-command menu (driven by the script in the sidebar) */
    #slash-menu {
        display: none;
        position: fixed;
        z-index: 1000;
        background: #FFFFFF;
        border: 1px solid #E8E6DC;
        border-radius: 0.9rem;
        box-shadow: 0 0.5rem 1.75rem rgba(31, 30, 29, 0.12);
        padding: 0.35rem;
        max-height: 18rem;
        overflow-y: auto;
        font-family: Inter, sans-serif;
    }
    #slash-menu .slash-head {
        font-size: 0.72rem;
        color: #8A8780;
        padding: 0.3rem 0.6rem 0.25rem;
    }
    #slash-menu .slash-item {
        display: flex;
        gap: 0.75rem;
        align-items: baseline;
        padding: 0.45rem 0.6rem;
        border-radius: 0.55rem;
        cursor: pointer;
    }
    #slash-menu .slash-item.active { background: #F0EEE6; }
    #slash-menu .slash-cmd {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #8B5E3C;
        white-space: nowrap;
    }
    #slash-menu .slash-desc { font-size: 0.85rem; color: #5E5D59; }

    /* Responsive frame: in Chat view the header, messages and composer share one wide column that
       fills the page; the cap only kicks in on very wide monitors so lines stay readable */
    .st-key-chat_column, .st-key-topbar_chat {
        max-width: 80rem;
        margin-left: auto;
        margin-right: auto;
        width: 100%;
    }
    /* ...but the composer stays a comfortable centered width under the wide messages */
    .st-key-chat_column .st-key-composer_dock,
    .st-key-chat_column .st-key-quick_pills {
        max-width: 48rem;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
    }
    /* Start-screen suggestion chips sit centered under the composer */
    .st-key-quick_pills [role="radiogroup"] { justify-content: center; }
    /* Message list fills the viewport instead of a fixed 600px, leaving room for header and composer */
    /* (Streamlit also pins the layout wrapper with flex: 0 0 600px, so size both) */
    [data-testid="stLayoutWrapper"]:has(> .st-key-chat_scroll) { flex: 0 0 auto !important; }
    [data-testid="stLayoutWrapper"]:has(> .st-key-chat_scroll),
    .st-key-chat_scroll {
        height: calc(100dvh - 19rem) !important;
        min-height: 16rem;
    }
    /* In a conversation the composer dock is pinned to the bottom of the viewport: messages scroll
       behind it and it never gets pushed down. (Sticky goes on the dock's layout wrapper, the
       element that is a direct child of the chat column, or it would have no room to stick in.) */
    [data-testid="stVerticalBlock"]:has(.st-key-chat_scroll) > [data-testid="stLayoutWrapper"]:has(> .st-key-composer_dock) {
        position: sticky;
        bottom: 0;
        z-index: 5;
        background: #FAF9F5;
        padding: 0.5rem 0 1rem;
        box-shadow: 0 -1rem 1rem -0.5rem #FAF9F5;  /* soft fade where messages pass underneath */
    }
    /* First question from the start screen: the message list takes the start screen's slot, and
       Streamlit keeps that screen's old children inside it (greyed out) until the reply finishes.
       Hide those leftovers so the greeting, chips and old composer vanish immediately. */
    .st-key-chat_scroll > [data-stale="true"]:not(:has([data-testid="stChatMessage"])),
    .st-key-chat_scroll [data-testid="stLayoutWrapper"]:has(> .st-key-composer_dock),
    .st-key-chat_scroll .st-key-quick_pills {
        display: none !important;
    }

    /* Top bar (model + Macro-F1 + view switcher) stays pinned just under Streamlit's 3.75rem header
       in every view, with the page color behind it so content scrolls cleanly underneath. */
    [data-testid="stLayoutWrapper"]:has(> .st-key-topbar),
    [data-testid="stLayoutWrapper"]:has(> .st-key-topbar_chat) {
        position: sticky;
        top: 3.75rem;
        z-index: 6;
        background: #FAF9F5;
        padding: 0.25rem 0 0.5rem;
        box-shadow: 0 1rem 1rem -0.5rem #FAF9F5;  /* soft fade where content passes underneath */
    }

    /* Chat view with a conversation: the page itself never scrolls. Every level from the page
       container down to the message list becomes a flex column, so the list takes exactly the
       space left between the top bar and the composer and is the only thing that scrolls. */
    [data-testid="stMain"]:has(.st-key-chat_column .st-key-chat_scroll) { overflow: hidden; }
    [data-testid="stMainBlockContainer"]:has(.st-key-chat_column .st-key-chat_scroll) {
        height: 100%;
        display: flex;
        flex-direction: column;
        box-sizing: border-box;
        /* Start below Streamlit's fixed 3.75rem header, where the sticky top bar sits. Streamlit's own
           top padding changes with the sidebar (16px open, 76px closed), so set it explicitly. */
        padding-top: 3.75rem;
    }
    [data-testid="stMainBlockContainer"]:has(.st-key-chat_column .st-key-chat_scroll) [data-testid="stLayoutWrapper"]:has(> .st-key-topbar_chat) {
        flex-shrink: 0;
    }
    [data-testid="stMainBlockContainer"]:has(.st-key-chat_column .st-key-chat_scroll) > [data-testid="stVerticalBlock"],
    [data-testid="stLayoutWrapper"]:has(> .st-key-chat_column),
    .st-key-chat_column {
        flex: 1 1 0 !important;
        min-height: 0;
    }
    .st-key-chat_column [data-testid="stLayoutWrapper"]:has(> .st-key-chat_scroll),
    .st-key-chat_column .st-key-chat_scroll {
        flex: 1 1 0 !important;
        height: auto !important;
        min-height: 0;
    }

    /* Chat + details view (wider than a phone): same idea as Chat view. The page is locked, the
       chat column's messages scroll above its pinned composer, and the details panel scrolls on its
       own. On phones the two columns stack, so the page keeps scrolling normally there. */
    @media (min-width: 641px) {
        [data-testid="stMain"]:has(.st-key-details_panel) { overflow: hidden; }
        [data-testid="stMainBlockContainer"]:has(.st-key-details_panel) {
            height: 100%;
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            padding-top: 3.75rem;  /* start below Streamlit's fixed 3.75rem header, where the sticky top bar sits */
            padding-bottom: 1rem;
        }
        [data-testid="stMainBlockContainer"]:has(.st-key-details_panel) > [data-testid="stVerticalBlock"],
        [data-testid="stLayoutWrapper"]:has(> [data-testid="stHorizontalBlock"] .st-key-details_panel),
        [data-testid="stHorizontalBlock"]:has(.st-key-details_panel) {
            flex: 1 1 0 !important;
            min-height: 0;
        }
        [data-testid="stHorizontalBlock"]:has(.st-key-details_panel) { align-items: stretch; }
        /* The top bar keeps its full height; only the columns give way to fit the screen */
        [data-testid="stMainBlockContainer"]:has(.st-key-details_panel) [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {
            flex-shrink: 0;
        }
        [data-testid="stHorizontalBlock"]:has(.st-key-details_panel) > [data-testid="stColumn"] {
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        [data-testid="stHorizontalBlock"]:has(.st-key-details_panel) > [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
        [data-testid="stLayoutWrapper"]:has(> .st-key-details_chat),
        .st-key-details_chat,
        [data-testid="stLayoutWrapper"]:has(> .st-key-details_panel) {
            flex: 1 1 0 !important;
            min-height: 0;
        }
        .st-key-details_chat [data-testid="stLayoutWrapper"]:has(> .st-key-chat_scroll),
        .st-key-details_chat .st-key-chat_scroll {
            flex: 1 1 0 !important;
            height: auto !important;
            min-height: 0;
        }
        .st-key-details_panel { height: 100%; overflow-y: auto; }
    }

    /* Drop Streamlit's default 10rem bottom padding while chatting, so the page itself doesn't scroll */
    [data-testid="stMainBlockContainer"]:has(.st-key-chat_scroll) { padding-bottom: 0; }

    @media (max-width: 640px) {
        /* Phones: tighter page gutters, header stacks with the view switcher full width */
        [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; padding-bottom: 2rem; }
        .st-key-topbar, .st-key-topbar_chat { flex-wrap: wrap; row-gap: 0.5rem; }
        [data-testid="stLayoutWrapper"]:has(> .st-key-chat_scroll),
        .st-key-chat_scroll { height: calc(100dvh - 21rem) !important; }  /* header wraps to two rows here */
        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { max-width: 92%; }
    }

    /* Start-screen greeting */
    .greeting {
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: clamp(1.55rem, 4.5vw, 2.3rem);
        line-height: 1.25;
        font-weight: 400;
        letter-spacing: -0.01em;
        color: #1F1E1D;
        text-align: center;
        margin: 0;
    }
    /* The greeting mark is the app icon, drawn with Streamlit's Material Symbols font */
    .greeting-mark {
        font-family: 'Material Symbols Rounded';
        font-weight: 400;
        font-size: 0.95em;
        vertical-align: -0.12em;
        color: #8B5E3C;
        margin-right: 0.4rem;
    }
</style>
""")


CODE_VERSION = code_version()  # cached objects below are rebuilt when src/ changes (see src/code_version.py)


@st.cache_data(ttl=300, show_spinner=False)  # git runs at most every 5 minutes, not on every rerun
def cached_build_info():
    return build_info()


@st.cache_resource(max_entries=1)  # one registry in memory: a new code version replaces the old one
def load_registry(version=None):
    return ModelRegistry()


registry = load_registry(version=CODE_VERSION)


@st.cache_resource(max_entries=1)
def load_agent_tools(_registry, version=None):
    # Precomputes the per-class error breakdown once per server process (and code version).
    return AuditorTools(_registry)


agent_tools = load_agent_tools(registry, version=CODE_VERSION)

FEATURE_NAMES = registry.metadata["feature_names"]
COUNTRIES = countries_from(FEATURE_NAMES)


@st.cache_resource(max_entries=1)
def load_country_regions(_registry, version=None):
    return _registry.country_regions()


COUNTRY_REGIONS = load_country_regions(registry, version=CODE_VERSION)

VIEWS =["Chat", "Chat + details", "Benchmarks"]

SLASH_COMMANDS = (
    [
        {"cmd": "/random", "desc": "Audit a random test case"},
        {"cmd": "/hard", "desc": "A case at least one model got wrong"},
        {"cmd": "/compare", "desc": "All three models on your last event (or a case they disagree on)"},
    ]
    + [
        {"cmd": f"/category {c}", "desc": f"Audit {'an' if c[0] in 'AEIOU' else 'a'} {c} case"}
        for c in registry.class_names
    ]
    + [{"cmd": "/export", "desc": "Open benchmarks and the session report"}]
)

# Shows a command menu above the composer while the text starts with "/".
# st.chat_input has no keystroke events, so this listens on the page (st.html isn't iframed).
SLASH_MENU_JS = """
<script>
(() => {
  if (window.__slashMenu) return;
  window.__slashMenu = true;
  const COMMANDS = __COMMANDS__;
  const SEL = '[data-testid="stChatInputTextArea"]';
  const menu = document.createElement('div');
  menu.id = 'slash-menu';
  menu.setAttribute('role', 'listbox');
  document.body.appendChild(menu);
  let items = [], active = 0, ta = null;

  const open = () => menu.style.display === 'block';
  const hide = () => { menu.style.display = 'none'; items = []; };

  // React tracks the textarea value itself; use the native setter + input event so it notices.
  const setValue = (el, v) => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(el, v);
    el.dispatchEvent(new Event('input', { bubbles: true }));
  };

  // Build rows with DOM calls: the HTML sanitizer drops a script tag whose text looks like markup.
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text) n.textContent = text;
    return n;
  };
  const render = () => {
    menu.replaceChildren(el('div', 'slash-head', 'Commands'));
    items.forEach((c, i) => {
      const row = el('div', 'slash-item' + (i === active ? ' active' : ''));
      row.setAttribute('role', 'option');
      row.dataset.idx = String(i);
      row.append(el('span', 'slash-cmd', c.cmd), el('span', 'slash-desc', c.desc));
      menu.append(row);
    });
    const box = ta.closest('[data-testid="stChatInput"]') || ta;
    const r = box.getBoundingClientRect();
    menu.style.left = r.left + 'px';
    menu.style.width = r.width + 'px';
    menu.style.bottom = (window.innerHeight - r.top + 8) + 'px';
    menu.style.display = 'block';
    const cur = menu.querySelector('.slash-item.active');
    if (cur) cur.scrollIntoView({ block: 'nearest' });
  };

  const update = (el) => {
    const q = el.value.toLowerCase();
    if (!q.startsWith('/') || q.includes('\\n')) { hide(); return; }
    items = COMMANDS.filter(c => c.cmd.toLowerCase().startsWith(q.trimEnd()) || c.cmd.toLowerCase().startsWith(q));
    if (!items.length || (items.length === 1 && items[0].cmd.toLowerCase() === q.trim())) { hide(); return; }
    ta = el;
    active = Math.min(active, items.length - 1);
    render();
  };

  const pick = (i) => {
    const c = items[i];
    if (!c || !ta) return;
    setValue(ta, c.cmd);
    hide();
    ta.focus();
  };

  document.addEventListener('input', (e) => {
    if (e.target.matches && e.target.matches(SEL)) { active = 0; update(e.target); }
  }, true);

  // Capture phase on document runs before React's handlers, so Enter picks instead of sending.
  document.addEventListener('keydown', (e) => {
    if (!open() || e.target !== ta) return;
    if (e.key === 'ArrowDown') { active = (active + 1) % items.length; render(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { active = (active - 1 + items.length) % items.length; render(); e.preventDefault(); }
    else if (e.key === 'Enter' || e.key === 'Tab') { e.preventDefault(); e.stopImmediatePropagation(); pick(active); }
    else if (e.key === 'Escape') { e.preventDefault(); hide(); }
  }, true);

  menu.addEventListener('mousedown', (e) => {
    const row = e.target.closest('[data-idx]');
    if (row) { e.preventDefault(); pick(Number(row.dataset.idx)); }
  });
  document.addEventListener('focusout', (e) => { if (e.target === ta) setTimeout(hide, 120); }, true);
  window.addEventListener('resize', hide);
})();
</script>
""".replace("__COMMANDS__", json.dumps(SLASH_COMMANDS))


# -------------------------------------------------------------
# Session state
# -------------------------------------------------------------
def new_stats():
    # One record per scored case (test-bank case, or a user's event once its true type is known).
    # Model and Gemini scores are both computed from these records, so they always cover the same cases.
    return {"records": []}


def config_value(name: str) -> str:
    """A setting from .streamlit/secrets.toml, else the environment (which .env feeds), else ""."""
    try:
        value = st.secrets.get(name)
    except Exception:  # no secrets file at all
        value = None
    return str(value or os.getenv(name, "")).strip()


def default_provider() -> str:
    """LLM_PROVIDER if set ("gemini" / "claude"); otherwise whichever provider has a key, Gemini first."""
    choice = config_value("LLM_PROVIDER").lower()
    if choice in ("claude", "anthropic"):
        return "Anthropic Claude"
    if choice == "gemini":
        return "Google Gemini"
    if not config_value("GEMINI_API_KEY") and config_value("ANTHROPIC_API_KEY"):
        return "Anthropic Claude"
    return "Google Gemini"


DEFAULTS = {
    "messages": [],
    "selected_model": "Random Forest",
    "comparison_mode": False,
    "comparison_models": ["Random Forest", "LightGBM", "XGBoost"],
    "layout_mode": "Chat",
    "last_audit": None,
    "hard_threshold": 0.55,
    "api_provider": default_provider(),
    "gemini_api_key": config_value("GEMINI_API_KEY"),
    "claude_api_key": config_value("ANTHROPIC_API_KEY"),
    # Only needed for organization-level keys that aren't scoped to a workspace.
    "claude_workspace_id": config_value("ANTHROPIC_WORKSPACE_ID"),
    "pending_quick_action": None,
    "pending_question": None,
    # Events the user described in this chat (EVENT-1, EVENT-2, ...), so tools can find them again.
    "session_cases": None,
    # Claude's Messages API transcript (tool calls and results included); only ever appended to.
    "agent_history": [],
    # Gemini agent mode: Gemini picks the tools itself (function calling); off = fixed route-then-answer flow.
    "gemini_agent": True,
    "gemini_agent_history": [],  # google-genai Content objects, tool calls and results included
    # Optional LLM baseline: Gemini also guesses the type on its own. Off: the ML models are the only predictors.
    "llm_baseline": False,
    # A Typed test in flight: a test-bank case Gemini rewrote as a typed message (true type kept from Gemini).
    "typed_test": None,
    "gemini_calls": 0,  # Gemini requests this session (counted for the shared key's quota guard)
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
if st.session_state.session_cases is None:
    st.session_state.session_cases = SessionCases()


# -------------------------------------------------------------
# Quota guard for the shared Gemini key
# -------------------------------------------------------------
# The public app runs on the owner's free-tier key. Without a cap, one visitor (or one demo) could use up
# the day's quota for everyone. Visitors who paste their own key in Settings are not limited.
def _limit(name, default):
    try:
        return int(config_value(name) or default)
    except ValueError:
        return default


SESSION_LIMIT = _limit("GEMINI_SESSION_LIMIT", 40)  # Gemini requests per visitor session
DAILY_LIMIT = _limit("GEMINI_DAILY_LIMIT", 400)     # Gemini requests per day across all visitors
QUOTA_COOLDOWN_S = 60                               # pause after Gemini says the quota is exhausted (429)


@st.cache_resource
def shared_gemini_usage():
    """One counter for the whole server process, shared by every session."""
    return {"lock": threading.Lock(), "day": None, "calls": 0, "cooldown_until": 0.0}


def using_shared_key():
    shared = config_value("GEMINI_API_KEY")
    return bool(shared) and st.session_state.gemini_api_key.strip() == shared


def _today_usage():
    usage = shared_gemini_usage()
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if usage["day"] != today:
        usage["day"], usage["calls"] = today, 0
    return usage


def record_gemini(kind):
    """Usage hook: src/agent reports every Gemini request ("call") and every 429 ("quota")."""
    if not using_shared_key():
        if kind == "call":
            st.session_state.gemini_calls += 1
        return
    usage = shared_gemini_usage()
    with usage["lock"]:
        if kind == "call":
            st.session_state.gemini_calls += 1
            _today_usage()["calls"] += 1
        elif kind == "quota":
            usage["cooldown_until"] = time.time() + QUOTA_COOLDOWN_S


set_usage_hook(record_gemini)  # thread-local: re-set on every run of this script


def gemini_block_reason():
    """Why Gemini can't be used right now on the shared key, or None if it can."""
    if not st.session_state.gemini_api_key.strip() or not using_shared_key():
        return None
    usage = shared_gemini_usage()
    with usage["lock"]:
        wait = usage["cooldown_until"] - time.time()
        calls_today = _today_usage()["calls"]
    if wait > 0:
        return f"Gemini's free quota is busy right now (try again in about {wait:.0f}s)"
    if st.session_state.gemini_calls >= SESSION_LIMIT:
        return f"This session has used its {SESSION_LIMIT} Gemini requests"
    if calls_today >= DAILY_LIMIT:
        return "The app has used today's shared Gemini budget"
    return None
if "stats" not in st.session_state or "records" not in st.session_state.stats:
    st.session_state.stats = new_stats()
if st.session_state.layout_mode not in VIEWS:
    st.session_state.layout_mode = "Chat"


# -------------------------------------------------------------
# Audit controller
# -------------------------------------------------------------
def active_models():
    return st.session_state.comparison_models if st.session_state.comparison_mode else [st.session_state.selected_model]


def record_audit(case, models_to_test):
    """Run the audit, store it as the active case, and update the session scoreboard."""
    audit_result = registry.audit_case(case, models_to_test)
    register_audit(audit_result, models_to_test)
    return audit_result


def register_audit(audit_result, models_to_test):
    """Make an audit the active case, attach the optional LLM baseline, and score it if the true type is known."""
    st.session_state.last_audit = audit_result
    tag_typed_test(audit_result)
    attach_llm_guess(audit_result)
    score_audit(audit_result)


def tag_typed_test(audit):
    """The first event classified during a Typed test is that test's: give it the hidden true type and
    record how well the LLM read the 6 attributes back from the typed message."""
    test, case = st.session_state.typed_test, audit["case"]
    if not test or case.get("source") != "user" or case.get("typed_test"):
        return
    st.session_state.typed_test = None
    case["typed_test"] = {"from_case": test["case_id"], "read": compare_events(test["event"], case["event"])}
    set_true_type(audit, test["truth"])


def attach_llm_guess(audit):
    """Gemini predicts the type from the same 6 attributes the models see, so the two can be scored side by side."""
    if "llm" in audit or not st.session_state.llm_baseline or gemini_block_reason():
        return
    key = st.session_state.gemini_api_key.strip()
    if not key:
        audit["llm"] = None
        return
    case = audit["case"]
    with st.spinner("Gemini baseline is making its own guess…"):
        guess = llm_guess(key, case.get("event") or event_from_case(case), registry.class_names, COUNTRIES)
    if guess:
        truth = case.get("ground_truth_class")
        guess["is_correct"] = None if truth is None else guess["predicted_class"] == truth
    audit["llm"] = guess


def score_audit(audit):
    """Add the case to the scoreboard once (a case counts once per session, however often it's reopened)."""
    case = audit["case"]
    truth = case.get("ground_truth_class")
    records = st.session_state.stats["records"]
    if truth is None or any(r["case_id"] == case["case_id"] for r in records):
        return
    model_name = next(iter(audit["models"]))  # the active model; in compare mode, the first one
    pred = audit["models"][model_name]["prediction"]
    guess = audit.get("llm")
    typed = case.get("typed_test")
    records.append({
        "case_id": case["case_id"],
        "source": "Typed test" if typed else "Your event" if case.get("source") == "user" else "Test bank",
        "true_type": truth,
        "model": model_name,
        "model_prediction": pred["predicted_class"],
        "model_confidence": round(pred["confidence"], 4),
        "gemini_prediction": guess["predicted_class"] if guess else None,
        "from_case": typed["from_case"] if typed else None,
        "attributes_read_correctly": typed["read"]["correct"] if typed else None,
    })


def confirm_true_type(audit, truth):
    """The user says what their event really was: mark every prediction right or wrong, then score it."""
    set_true_type(audit, truth)
    score_audit(audit)


def set_true_type(audit, truth):
    case = audit["case"]
    case["ground_truth_class"] = truth
    case["ground_truth_id"] = registry.class_names.index(truth)
    for m in audit["models"].values():
        m["is_correct"] = m["prediction"]["is_correct"] = m["prediction"]["predicted_class"] == truth
    if audit.get("llm"):
        audit["llm"]["is_correct"] = audit["llm"]["predicted_class"] == truth
    registry.refresh_contrast(audit)  # "why not X?" now asks about the confirmed type


def _score(pairs):
    if not pairs:
        return {"n": 0, "accuracy": None, "macro_f1": None}
    y_true, y_pred = zip(*pairs)
    return {
        "n": len(pairs),
        "accuracy": sum(t == p for t, p in pairs) / len(pairs),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


SOURCES = ["Test bank", "Typed test", "Your event"]


def session_scores():
    """The ML model's scores this session, overall and per source (plus the optional Gemini baseline)."""
    records = st.session_state.stats["records"]
    threshold = st.session_state.hard_threshold
    typed = [r for r in records if r["source"] == "Typed test"]
    return {
        "model": _score([(r["true_type"], r["model_prediction"]) for r in records]),
        "by_source": {src: _score([(r["true_type"], r["model_prediction"]) for r in records if r["source"] == src])
                      for src in SOURCES},
        "gemini": _score([(r["true_type"], r["gemini_prediction"]) for r in records if r["gemini_prediction"]]),
        # How often the LLM read the typed message back into the right attribute values.
        "attributes_read": (sum(r["attributes_read_correctly"] for r in typed) / (6 * len(typed))) if typed else None,
        "tested": len(records),
        # Hard case: the model was wrong, or right with confidence below the threshold in settings.
        "hard_cases": sum(r["model_prediction"] != r["true_type"] or r["model_confidence"] < threshold for r in records),
    }


def run_audit(case, prompt_label=None, target_models=None):
    if not case:
        st.toast("No matching case found in the benchmark bank.", icon=":material/warning:")
        return

    models_to_test = target_models or active_models()
    audit_result = record_audit(case, models_to_test)

    user_text = prompt_label or f"Audit case `{case['case_id']}` ({case['ground_truth_class']})"
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.session_state.messages.append({"role": "assistant", "audit": audit_result, "models_tested": models_to_test})
    st.rerun()


def compare_user_event(last):
    """/compare after the user described an event: every model on THAT event, not on a test-bank case.

    In the app, "10 days of rain in Thailand in October" then /compare showed a Tajikistan drought from the
    test bank, and the user had to argue "no, it's Thailand!"."""
    models = st.session_state.comparison_models
    case = last["case"]
    audit = registry.audit_case(case, models)  # a scale not given is averaged the same way as before
    if "llm" in last:
        audit["llm"] = last["llm"]  # same event, same Gemini guess: don't ask Gemini again
    register_audit(audit, models)
    st.session_state.messages.append({"role": "user", "content": f"`/compare` {case['case_id']}"})
    st.session_state.messages.append({"role": "assistant", "audit": audit, "models_tested": models})
    st.rerun()


QUICK_ACTIONS = {
    ":material/shuffle: Random case": ({}, "Audit a random historical case from the test bank"),
    ":material/warning: Hard case": ({"only_hard": True}, "Examine a case at least one model misclassified"),
    ":material/compare_arrows: Model disagreement": ({"only_disagreement": True}, "Find a case where the models disagree"),
    ":material/water_drop: Flood": ({"disaster_type": "Flood"}, "Audit a Flood case"),
    ":material/landslide: Earthquake": ({"disaster_type": "Earthquake"}, "Audit an Earthquake case"),
    ":material/storm: Storm": ({"disaster_type": "Storm"}, "Audit a Storm case"),
}


def queue_quick_action():
    # Widget callback: remember the pick and clear the pill so it doesn't re-fire on every rerun.
    st.session_state.pending_quick_action = st.session_state.quick_pills
    st.session_state.quick_pills = None


def run_quick_action(label):
    filters, prompt_label = QUICK_ACTIONS[label]
    if filters.get("only_disagreement"):
        st.session_state.comparison_mode = True
    run_audit(registry.get_random_case(**filters), prompt_label)


def handle_slash_command(command_text: str):
    cmd = command_text.strip().lower()
    if cmd.startswith("/random"):
        run_audit(registry.get_random_case(), "`/random`")
    elif cmd.startswith("/hard"):
        run_audit(registry.get_random_case(only_hard=True), "`/hard`")
    elif cmd.startswith("/compare"):
        st.session_state.comparison_mode = True
        last = st.session_state.last_audit
        if last and last["case"].get("source") == "user":
            compare_user_event(last)
        else:
            run_audit(registry.get_random_case(only_disagreement=True), "`/compare`")
    elif cmd.startswith("/category"):
        parts = cmd.split(maxsplit=1)
        cat_query = parts[1].strip() if len(parts) > 1 else "Flood"
        matched = next((c for c in registry.class_names if c.lower() == cat_query.lower()), None)
        if matched:
            run_audit(registry.get_random_case(disaster_type=matched), f"`/category {matched}`")
        else:
            st.toast(f"Unknown category '{cat_query}'. Available: {', '.join(registry.class_names)}", icon=":material/info:")
    elif cmd.startswith("/export"):
        st.session_state.layout_mode = "Benchmarks"
        st.rerun()
    else:
        st.toast(f"Unrecognized command '{command_text}'. Try /random, /hard, /compare, /category <name>, /export", icon=":material/help:")


def start_typed_test(disaster_type=None):
    """Rewrite a random test case as a typed message (without its type) and send it like a user question."""
    blocked = gemini_block_reason()
    if blocked:
        st.toast(blocked + ". Add your own Gemini key in Settings to continue.", icon=":material/hourglass_empty:")
        return
    case = registry.get_random_case(disaster_type=disaster_type)
    if not case:
        st.toast("No test case matches that category.", icon=":material/warning:")
        return
    event = event_from_case(case)
    with st.spinner("Gemini is writing the test case as a typed message…"):
        message = write_scenario(st.session_state.gemini_api_key.strip(), event,
                                 language="Thai" if len(st.session_state.stats["records"]) % 2 == 0 else "English")
    if not message:
        st.toast("Gemini couldn't write the message (busy or out of quota). Try again.", icon=":material/warning:")
        return
    st.session_state.typed_test = {"case_id": case["case_id"], "truth": case["ground_truth_class"], "event": event,
                                   "message": message}
    st.session_state.messages.append({"role": "user", "content": message,
                                      "typed_test_of": case["case_id"]})
    st.session_state.pending_question = message
    st.rerun()


def submit_question(text: str):
    """Show the user's turn right away; the reply streams on the next run."""
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.pending_question = text
    st.rerun()


def claude_agent_enabled():
    return st.session_state.api_provider == "Anthropic Claude" and bool(st.session_state.claude_api_key.strip())


def gemini_agent_enabled():
    return (st.session_state.api_provider == "Google Gemini" and st.session_state.gemini_agent
            and bool(st.session_state.gemini_api_key.strip()))


def answer_pending_question():
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    test = st.session_state.typed_test
    if test and test["message"] != question and st.session_state.session_cases.asked_details_on != test["message"]:
        st.session_state.typed_test = None  # an earlier typed test that never reached the model
        # (kept while the app waits for the details it asked for: the answer is still part of that test)
    blocked = gemini_block_reason() if st.session_state.api_provider == "Google Gemini" else None
    if blocked:
        answer_with_plain_llm(question, offline_reason=blocked)
    elif claude_agent_enabled() or gemini_agent_enabled():
        answer_with_agent(question)
    else:
        answer_with_plain_llm(question)


def ui_context():
    parts = []
    audit = st.session_state.last_audit
    if audit:
        c = audit["case"]
        s_ = c["summary"]
        parts.append(
            f"case open on screen: {c['case_id']} (ground truth {c['ground_truth_class']}; "
            f"{s_['country']}, {s_['region']}, month {s_['start_month']})"
        )
    parts.append(f"selected models: {', '.join(active_models())}")
    return parts


def steps_label(steps):
    n = len(steps)
    return f"Used {n} tool{'s' if n != 1 else ''}" if n else "Answered directly"


def render_step(step):
    with st.status(step["label"], type="step", state="error" if step["is_error"] else "complete"):
        st.caption(step["detail"])


def answer_with_agent(question):
    """The LLM (Claude or Gemini) decides which registry tools to call; steps, case cards and the answer render as they arrive."""
    if claude_agent_enabled():
        agent = ClaudeAuditorAgent(
            agent_tools,
            st.session_state.claude_api_key.strip(),
            workspace_id=st.session_state.claude_workspace_id,
        )
        history, llm_name = st.session_state.agent_history, "Claude"
    else:
        agent = GeminiAuditorAgent(agent_tools, st.session_state.gemini_api_key.strip())
        history, llm_name = st.session_state.gemini_agent_history, "Gemini"
    text, steps, audits, error = "", [], [], None

    with st.chat_message("assistant", avatar=APP_ICON):
        steps_box = st.status(":shimmer[Working on it]", type="compact")
        cards_box = st.container()
        answer_ph = st.empty()

        st.session_state.session_cases.message = question  # as typed: code-side checks read this, not the UI context
        for kind, payload in agent.run_turn(
            history, agent_turn_text(question, ui_context()), active_models(),
            st.session_state.session_cases
        ):
            if kind == "text":
                text += payload
                answer_ph.markdown(text)
            elif kind == "retract":
                # A request attempt failed mid-stream and is being retried; drop its partial text.
                text = text[:-payload]
                answer_ph.markdown(text)
            elif kind == "tool":
                step = {
                    "label": payload["summary"],
                    "result": payload.get("result"),
                    "is_error": payload["is_error"],
                    "detail": f"`{payload['name']}` {json.dumps(payload['input'], ensure_ascii=False)}",
                }
                steps.append(step)
                with steps_box:
                    render_step(step)
                if payload["audit"]:
                    register_audit(payload["audit"], payload["models"])
                    audits.append({"audit": payload["audit"], "models": payload["models"]})
                    with cards_box:
                        render_audit_card(payload["audit"], payload["models"], f"live_{len(audits)}")
            elif kind == "error":
                error = payload

        if error:
            if not text.strip():
                # Nothing came back (quota, outage): still give the built-in summary of the case on screen.
                fallback = "".join(DisasterAuditorLLM(provider="gemini").local_answer(question, st.session_state.last_audit))
                text = f":orange[:material/warning:] {error}\n\nHere is the built-in summary instead:\n\n{fallback}"
            else:
                text += "\n\n" + f":orange[:material/warning:] {error}"
            answer_ph.markdown(text)
        steps_box.update(label=steps_label(steps), state="error" if error and not steps else "complete")

    sources = agent_sources(steps, text, llm_name)
    # New case cards must have their prediction named; other tool answers are checked on the figures they quote.
    tool_results = [s.get("result") for s in steps if not s["is_error"]]
    check = None
    if tool_results:
        check = check_answer(text, [item["audit"] for item in audits] or [st.session_state.last_audit],
                             tool_results + [registry.leaderboard, registry.mcnemar, AGENT_PROMPT],
                             name_prediction=bool(audits))
    st.session_state.messages.append({"role": "assistant", "content": text, "steps": steps, "audits": audits,
                                      "sources": sources, "check": check})
    st.rerun()


def keyword_route(question):
    """Fallback router when Gemini isn't available: keyword rules for 'give me a case' requests."""
    lower_q = question.lower()
    test_keywords = ["สุ่ม", "audit", "test", "case", "ลอง", "random", "hard", "ผิด", "misclassified", "เคส"]
    if not any(w in lower_q for w in test_keywords):
        return {"intent": "question"}
    return {
        "intent": "test_bank_case",
        "disaster_type": next((c for c in registry.class_names if c.lower() in lower_q), None),
        "only_hard": any(w in lower_q for w in ["hard", "ผิด", "wrong", "misclassified"]),
        "only_disagreement": False,
        "region": None,
    }


def busy_reply(question, reason):
    """The answer when Gemini can't be used and nothing ran: say so plainly, in the message's language.

    Reading a described event takes Gemini, so the models have not run on it. The built-in summary would
    greet the user or describe an older case instead, which reads as if the message had been ignored."""
    if reply_language(question) == "Thai":
        return (f":orange[:material/hourglass_empty:] {reason} การอ่านเหตุการณ์ที่พิมพ์มาต้องใช้ Gemini "
                "จึงยังไม่ได้รันโมเดลกับข้อความนี้ครับ ลองส่งใหม่อีกครั้งในอีกสักครู่ หรือใส่ Gemini key ของคุณเอง "
                "(ฟรี) ใน Settings ระหว่างนี้ปุ่มด้านบน เช่น *Random case* หรือ *Hard case* ใช้ได้โดยไม่ต้องใช้ Gemini")
    return (f":orange[:material/hourglass_empty:] {reason}. Reading a described event needs Gemini, so the models "
            "haven't run on this message yet. Send it again in a moment, or add your own free Gemini key in "
            "Settings. Meanwhile the buttons above (*Random case*, *Hard case*, ...) work without Gemini.")


def audit_for_route(route):
    """Run whatever the question asked for: a test-bank case, or the models on the user's own event."""
    models_to_test = active_models()
    if route["intent"] == "test_bank_case":
        case = registry.get_random_case(
            disaster_type=route.get("disaster_type"),
            only_hard=bool(route.get("only_hard")),
            only_disagreement=bool(route.get("only_disagreement")),
            region=route.get("region"),
        )
        if not case:
            st.toast("No test case matches that request.", icon=":material/warning:")
            return None
    elif route["intent"] == "new_event" and route.get("event"):
        event = dict(route["event"])
        if event.get("country") and not event.get("region"):
            event["region"] = COUNTRY_REGIONS.get(event["country"])
        truth = route.get("stated_true_type")
        truth = truth if type_stated_in_text(truth, st.session_state.session_cases.message) else None
        case = make_event_case(event, FEATURE_NAMES, st.session_state.session_cases.next_id(),
                               truth=truth, class_names=registry.class_names)
        st.session_state.session_cases.add(case)
    else:
        return None
    return record_audit(case, models_to_test), models_to_test


def answer_sources(audit, provider_code):
    """What a chat answer was allowed to draw on, shown under it (the teacher's 'where does it answer from?')."""
    if provider_code != "gemini" or not st.session_state.gemini_api_key.strip():
        return "Answered by the app's built-in summary (no LLM key set)."
    parts = ["the app's model benchmark figures"]
    if audit:
        kind = "your event" if audit["case"].get("source") == "user" else "test case"
        parts.append(f"{kind} {audit['case']['case_id']} on screen")
    parts.append("Gemini's general knowledge where the answer says so")
    return "Based on " + ", ".join(parts[:-1]) + " and " + parts[-1] + "."


def answer_with_plain_llm(question, offline_reason=None):
    """Gemini or the local fallback: work out what the question wants, run it, then stream a reply.

    offline_reason: Gemini is over its limit; route by keywords and answer with the built-in summary."""
    prov_code = "gemini" if st.session_state.api_provider == "Google Gemini" else "anthropic"
    # Earlier turns only: the question itself is the last message and goes in separately.
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m.get("content")
    ]

    st.session_state.session_cases.message = question  # as typed: code-side checks (e.g. a stated true type) read this
    route = None
    gemini_key = "" if offline_reason else st.session_state.gemini_api_key.strip()
    if gemini_key:
        with st.spinner("Reading your question…"):
            route = route_question(gemini_key, question, registry.class_names, COUNTRIES)
    route = route or keyword_route(question)

    result = audit_for_route(route)
    if result:
        audit_result, models_to_test = result
        msg = {"role": "assistant", "audit": audit_result, "models_tested": models_to_test}
        st.session_state.messages.append(msg)
        render_message(msg, len(st.session_state.messages) - 1)

    api_key = st.session_state.gemini_api_key if prov_code == "gemini" else st.session_state.claude_api_key
    llm = DisasterAuditorLLM(provider=prov_code, api_key=api_key)
    with st.chat_message("assistant", avatar=APP_ICON):
        reply, answer_ph = "", st.empty()
        if offline_reason and not result:
            reply = busy_reply(question, offline_reason)
            answer_ph.markdown(reply)
        elif offline_reason:
            reply = (f":orange[:material/hourglass_empty:] {offline_reason}, so this answer uses the built-in summary "
                     "(the ML models still ran). Add your own free Gemini key in Settings to keep chatting.\n\n")
        stream = ([] if offline_reason and not result
                  else llm.local_answer(question, st.session_state.last_audit) if offline_reason
                  else llm.chat_stream(user_message=question, history=history, current_case=st.session_state.last_audit))
        for piece in stream:
            # RESTART: the model failed mid-answer and a fallback model is starting over.
            reply = "" if piece is RESTART else reply + piece
            answer_ph.markdown(reply)
    check = (check_answer(reply, [st.session_state.last_audit], [registry.leaderboard, registry.mcnemar, CHAT_PROMPT])
             if prov_code == "gemini" and gemini_key and result else None)
    st.session_state.messages.append({
        "role": "assistant", "content": reply,
        "sources": ("Gemini limit reached: nothing was run on this message." if offline_reason and not result
                    else "Answered by the app's built-in summary (Gemini limit reached)." if offline_reason
                    else answer_sources(st.session_state.last_audit, prov_code)),
        "check": check,
    })
    st.rerun()


# -------------------------------------------------------------
# Rendering helpers
# -------------------------------------------------------------
def verdict_badge(is_correct):
    if is_correct is None:  # a user's event before they confirm what it really was
        st.badge("Unverified", icon=":material/help:", color="gray")
    elif is_correct:
        st.badge("Correct", icon=":material/check:", color="green")
    else:
        st.badge("Misclassified", icon=":material/close:", color="red")


def render_audit_card(audit, tested_models, key_suffix):
    case = audit["case"]
    summary = case["summary"]
    is_user_event = case.get("source") == "user"
    truth = case.get("ground_truth_class")
    with st.container(border=True):
        if is_user_event:
            truth_text = f"true type **{truth}**" if truth else "true type **not confirmed yet**"
            label = "Typed test" if case.get("typed_test") else "Your event"
            st.markdown(f":material/edit_note: **{label} {case['case_id']}** · {truth_text}")
            event = case["event"]
            details = [f"{summary['country']} ({summary['region']})",
                       f"month {event['start_month']}" if event.get("start_month") else "month not given"]
            if event.get("duration_days") is not None:
                details.append(f"{event['duration_days']:g} days")
            if event.get("magnitude_value") is not None:
                details.append(f"magnitude {event['magnitude_value']:g} {summary['magnitude_scale']}")
            else:
                details.append(f"magnitude scale `{summary['magnitude_scale']}`")
            st.caption(" · ".join(details))
            typed = case.get("typed_test")
            if typed:
                read = typed["read"]
                misread = "; ".join(f"{w['attribute']}: typed {w['typed']}, read {w['read']}" for w in read["wrong"])
                st.caption(f":material/keyboard: Typed test from {typed['from_case']} · the LLM read "
                           f"**{read['correct']}/{read['total']}** attributes correctly" + (f" ({misread})" if misread else ""))
            if case.get("filled"):
                st.caption(f":orange[:material/info:] Not given, filled with training defaults: {', '.join(case['filled'])}. "
                           "Add them for a more reliable prediction.")
            seen = case.get("scale_observed")
            if seen:
                st.caption(f":material/visibility: Magnitude scale **{seen['scale']}** comes from what you observed "
                           f"(\"{seen['words']}\") and you confirmed it; no figure was given.")
            mix = next(iter(audit["models"].values()), {}).get("prediction", {}).get("scale_mix")
            if mix:
                ruled = case.get("scale_ruled_out")
                st.caption(":material/balance: What was measured was not given, so each model ran once per scale usually "
                           "recorded there and the results were averaged: "
                           + ", ".join(f"{m['scale']} {m['weight']:.0%}" for m in mix)
                           + (f" ({', '.join(ruled)} left out: you said it was not that)" if ruled else "") + ".")
        else:
            st.markdown(f":material/description: **Case {case['case_id']}** · ground truth **{truth}**")
            st.caption(
                f"{summary['country']} ({summary['region']}) · month {summary['start_month']} · "
                f"magnitude scale `{summary['magnitude_scale']}`"
            )

        if len(tested_models) == 1:
            m_name = tested_models[0]
            m_data = audit["models"][m_name]
            pred = m_data["prediction"]
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f"{m_name} predicted **{pred['predicted_class']}** · {pred['confidence'] * 100:.1f}% confidence")
                verdict_badge(m_data["is_correct"])
            st.progress(pred["confidence"])
            st.markdown("**Top decision signals (SHAP)**")
            for exp in m_data["shap"]["human_explanations"][:3]:
                st.markdown(f"- {exp}")
            contrast = m_data.get("contrast")
            if m_data["is_correct"] is False and contrast:
                st.markdown(f"**Why not {contrast['class']}?** (the true type, {contrast['probability'] * 100:.0f}% probability)")
                for exp in contrast["human_explanations"]:
                    st.markdown(f"- {exp}")
        else:
            cols = st.columns(len(tested_models))
            for col, m_name in zip(cols, tested_models):
                m_data = audit["models"][m_name]
                with col:
                    st.markdown(f"**{m_name}**")
                    st.markdown(f"{m_data['prediction']['predicted_class']} · {m_data['prediction']['confidence'] * 100:.1f}%")
                    verdict_badge(m_data["is_correct"])

        # The LLM's own prediction from the same 6 attributes, scored like the models.
        guess = audit.get("llm")
        if guess:
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.markdown(f":material/smart_toy: Gemini guessed **{guess['predicted_class']}** · {guess['confidence'] * 100:.0f}%")
                verdict_badge(guess.get("is_correct"))
            st.caption(guess["reason"])

        if is_user_event and truth is None:
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                picked = st.selectbox(
                    "What type was it really?",
                    registry.class_names,
                    index=None,
                    placeholder="What type was it really?",
                    label_visibility="collapsed",
                    key=f"truth_{case['case_id']}_{key_suffix}",
                )
                if st.button("Confirm", icon=":material/check:", key=f"confirm_{case['case_id']}_{key_suffix}",
                             disabled=picked is None, help="Adds this event to the session scoreboard"):
                    confirm_true_type(audit, picked)
                    st.rerun()

        is_latest = audit is st.session_state.last_audit
        if is_latest and st.session_state.layout_mode == "Chat":
            if st.button("Open details", icon=":material/right_panel_open:", type="tertiary", key=f"details_{key_suffix}"):
                st.session_state.layout_mode = "Chat + details"
                st.rerun()


def render_message(msg, idx):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
            if msg.get("typed_test_of"):
                st.caption(f":material/keyboard: Typed test written by Gemini from {msg['typed_test_of']} "
                           "(its true type is hidden until the model has predicted)")
        return
    with st.chat_message("assistant", avatar=APP_ICON):
        if msg.get("steps"):
            with st.expander(steps_label(msg["steps"]), type="compact"):
                for step in msg["steps"]:
                    render_step(step)
        if msg.get("audit"):
            render_audit_card(msg["audit"], msg.get("models_tested", [st.session_state.selected_model]), idx)
        for j, item in enumerate(msg.get("audits", [])):
            render_audit_card(item["audit"], item["models"], f"{idx}_{j}")
        if msg.get("content"):
            st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f":material/info: {msg['sources']}")
        check = msg.get("check")
        if check:
            if check["ok"]:
                st.caption(":green[:material/verified:] Explanation matches the model's output")
            else:
                st.caption(":orange[:material/warning:] Explanation doesn't match the model's output: " + "; ".join(check["issues"]))


def sync_model_picker():
    # Separate widget key + callback: selected_model survives views where the dropdown isn't rendered.
    st.session_state.selected_model = st.session_state.model_picker


def render_composer(placeholder):
    """Claude-style composer: text box with + on the bottom left and the model picker bottom right.

    Everything lives in one dock so that, in a conversation, CSS can pin it to the bottom of the viewport.
    """
    with st.container(key="composer_dock", gap="small"):
        with st.container(key="composer"):
            user_query = st.chat_input(placeholder, submit_mode="disable", height=110)
            with st.container(key="composer_tools", horizontal=True, vertical_alignment="center", gap="small"):
                picked = st.menu_button(
                    ":material/add:",
                    options=list(QUICK_ACTIONS),
                    type="tertiary",
                    key="quick_menu",
                    help="Quick audit",
                )
                st.space("stretch")
                compare = st.toggle("Compare", value=st.session_state.comparison_mode)
                st.selectbox(
                    "Active model",
                    options=registry.available_models,
                    index=registry.available_models.index(st.session_state.selected_model),
                    key="model_picker",
                    on_change=sync_model_picker,
                    label_visibility="collapsed",
                    width=150,
                )

        if st.session_state.comparison_mode:
            selected_comps = st.pills(
                "Models to compare",
                options=registry.available_models,
                default=st.session_state.comparison_models,
                selection_mode="multi",
                label_visibility="collapsed",
            )
            if selected_comps and list(selected_comps) != st.session_state.comparison_models:
                st.session_state.comparison_models = list(selected_comps)

    if compare != st.session_state.comparison_mode:
        st.session_state.comparison_mode = compare
        st.rerun()
    if picked:
        run_quick_action(picked)
    return user_query


def handle_input(user_query):
    if not user_query:
        return
    if user_query.startswith("/"):
        handle_slash_command(user_query)
    else:
        submit_question(user_query)


def greeting_text(now=None):
    """Good morning/afternoon/evening by the viewer's clock: the server runs on UTC, 7 hours behind Thailand."""
    if now is None:
        now = datetime.now()
        try:
            now = datetime.now(ZoneInfo(st.context.timezone))
        except Exception:  # no browser time zone (tests, bare mode): the server's clock is all there is
            pass
    hour = now.hour
    if hour < 12:
        return "Good morning"
    if hour < 18:
        return "Good afternoon"
    return "Good evening"


def render_chat(composer_placeholder):
    # Streamlit matches elements between runs by position and keeps unmatched old ones on screen
    # (greyed out) until the run ends. So the start screen is ONE container in slot 0, and the
    # conversation's message list takes that same slot: the greeting disappears the moment the
    # first question is asked, instead of lingering while the answer streams.
    if not st.session_state.messages:
        # Start screen: greeting, centered composer, suggestion chips
        with st.container(key="start_screen"):
            st.space("large")
            st.space("large")
            st.html(
                f'<p class="greeting"><span class="greeting-mark">{APP_ICON.split("/")[1].strip(":")}</span>{greeting_text()}. '
                "Describe a disaster, or pick a test case below.</p>"
            )
            st.space("medium")
            user_query = render_composer(composer_placeholder)
            st.pills(
                "Suggestions",
                options=list(QUICK_ACTIONS),
                label_visibility="collapsed",
                key="quick_pills",
                on_change=queue_quick_action,
            )
        handle_input(user_query)
        return

    # Conversation: scrolling message area with the composer pinned below it.
    # Height is overridden in CSS (.st-key-chat_scroll) to follow the viewport.
    message_area = st.container(height=600, border=False, key="chat_scroll")
    # Draw the composer before filling the messages so it is already in place while a reply streams.
    user_query = render_composer(composer_placeholder)
    with message_area:
        for i, msg in enumerate(st.session_state.messages):
            render_message(msg, i)
        if st.session_state.pending_question:
            answer_pending_question()
    handle_input(user_query)


def render_case_details():
    if st.session_state.last_audit is None:
        st.info("Run an audit in the chat to inspect its SHAP attributions and probability distribution.", icon=":material/info:")
        return

    audit = st.session_state.last_audit
    case = audit["case"]
    primary_model = st.session_state.selected_model
    m_res = audit["models"].get(primary_model) or list(audit["models"].values())[0]
    pred = m_res["prediction"]

    with st.container(horizontal=True):
        st.metric("Ground truth", case["ground_truth_class"], border=True)
        st.metric(f"{primary_model} prediction", pred["predicted_class"], border=True)
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption(f"Confidence `{pred['confidence'] * 100:.2f}%` · case `{case['case_id']}` · {case['summary']['country']}")
        verdict_badge(m_res["is_correct"])

    st.markdown("**8-class probability distribution**")
    probs_df = pd.DataFrame(
        list(pred["probabilities"].items()),
        columns=["Disaster class", "Probability"],
    ).sort_values("Probability", ascending=True)
    st.bar_chart(
        probs_df.set_index("Disaster class"),
        horizontal=True,
        color="#8B5E3C",
        height=240,
    )

    st.markdown(f"**Why {pred['predicted_class']}? SHAP by attribute**")
    shap_df = pd.DataFrame([
        {
            "Attribute": row["attribute"] + (" *" if row["filled"] else ""),
            "Share": row["share"],
            "Value": row["value"],
        }
        for row in m_res["shap"]["by_attribute"]
    ])
    st.dataframe(
        shap_df,
        column_config={"Share": st.column_config.NumberColumn(format="%+.1f%%", help="Signed % of this model's total evidence: + supports, − opposes")},
        hide_index=True,
    )
    st.caption(
        "The 245 features summed back into the 6 attributes; + supports, − opposes. Shares compare across models."
        + (" \\* Not given, so the model used a training default." if any(r["filled"] for r in m_res["shap"]["by_attribute"]) else "")
    )

    st.markdown("**In plain language**")
    for exp in m_res["shap"]["human_explanations"][:3]:
        st.caption(f"• {exp}")

    contrast = m_res.get("contrast")
    if contrast:
        predicted = pred["predicted_class"]
        other = contrast["class"]
        st.markdown(f"**Why {predicted}, not {other}?**")
        st.caption(f"{other} is the {contrast['reason']} ({contrast['probability'] * 100:.0f}% probability). "
                   "Same attributes, scored toward each class.")
        toward_other = {r["attribute"]: r["share"] for r in contrast["by_attribute"]}
        st.dataframe(
            pd.DataFrame([
                {"Attribute": r["attribute"] + (" *" if r["filled"] else ""),
                 predicted: r["share"], other: toward_other[r["attribute"]]}
                for r in m_res["shap"]["by_attribute"]
            ]),
            column_config={predicted: st.column_config.NumberColumn(format="%+.0f%%"),
                           other: st.column_config.NumberColumn(format="%+.0f%%")},
            hide_index=True,
        )
        for exp in contrast["human_explanations"]:
            st.caption(f"• {exp}")


def render_leaderboard():
    lb_rows = []
    for m_name, metrics in registry.leaderboard.items():
        ci = metrics["macro_f1_ci"]
        lb_rows.append({
            "Model": m_name,
            "Macro-F1": metrics["macro_f1"],
            "95% CI": f"[{ci[0]:.4f}, {ci[1]:.4f}]",
            "Accuracy": metrics["accuracy"] * 100,
            "Weighted-F1": metrics["weighted_f1"],
        })
    st.dataframe(
        pd.DataFrame(lb_rows),
        column_config={
            "Macro-F1": st.column_config.NumberColumn(format="%.4f"),
            "Accuracy": st.column_config.NumberColumn(format="%.2f%%"),
            "Weighted-F1": st.column_config.NumberColumn(format="%.4f"),
        },
        hide_index=True,
    )

    st.markdown("**Per-class F1**")
    class_f1 = {"Disaster class": registry.class_names}
    for m_name, metrics in registry.leaderboard.items():
        class_f1[m_name] = [metrics["per_class_f1"].get(cls, 0.0) for cls in registry.class_names]
    st.bar_chart(pd.DataFrame(class_f1).set_index("Disaster class"), height=260, stack=False)

    st.markdown("**Paired significance (McNemar)**")
    for pair, result in registry.mcnemar.items():
        note = "significant (p < 0.05)" if result["significant"] else "within noise (p ≥ 0.05)"
        st.caption(f"• **{pair}**: p = `{result['p_value']:.4f}`, {note}")

    st.markdown(f"**Model inputs: {len(FEATURE_NAMES)} features from {len(RAW_ATTRIBUTES)} attributes**")
    st.table(pd.DataFrame([
        {"Attribute": name, "Features": len(attribute_columns(columns, FEATURE_NAMES)), "Encoding": encoding}
        for name, columns, encoding in RAW_ATTRIBUTES
    ]), hide_index=True)
    st.caption("All attributes are known before or at the start of an event; impact data (deaths, damage) "
               "is left out to avoid leakage. Gemini's guesses get the same 6 attributes as plain text.")


def render_disagreement():
    meta = registry.metadata
    dis_pct = meta["disagreement_cases"] / meta["total_test_cases"] * 100
    st.markdown(
        f"The test bank has **{meta['disagreement_cases']}** cases (**{dis_pct:.1f}%**) where "
        "Random Forest, LightGBM, and XGBoost predict different classes."
    )
    if st.button("Audit a disagreement case", icon=":material/shuffle:"):
        st.session_state.comparison_mode = True
        if st.session_state.layout_mode == "Benchmarks":
            st.session_state.layout_mode = "Chat + details"
        c = registry.get_random_case(only_disagreement=True)
        run_audit(c, f"Evaluate disagreement case `{c['case_id']}`" if c else None)
    st.markdown("**Why disagreements happen**")
    st.caption("• Boundary confusion: Flood and Storm overlap during monsoon months.")
    st.caption("• Minority-class recall: Random Forest's bagging vs. the boosters' stronger regularization on Drought and Wildfire.")


def render_export():
    scores = session_scores()

    def pct(v):
        return f"{v * 100:.1f}%" if v is not None else "—"

    def f1(v):
        return f"{v:.4f}" if v is not None else "—"

    st.markdown("**Session: the ML model by source**")
    rows = [{"Source": "All", "Cases scored": scores["model"]["n"],
             "Accuracy": pct(scores["model"]["accuracy"]), "Macro-F1": f1(scores["model"]["macro_f1"])}]
    rows += [{"Source": src, "Cases scored": sc["n"], "Accuracy": pct(sc["accuracy"]), "Macro-F1": f1(sc["macro_f1"])}
             for src, sc in scores["by_source"].items()]
    if scores["gemini"]["n"]:
        rows.append({"Source": "Gemini baseline (own guess)", "Cases scored": scores["gemini"]["n"],
                     "Accuracy": pct(scores["gemini"]["accuracy"]), "Macro-F1": f1(scores["gemini"]["macro_f1"])})
    compare_df = pd.DataFrame(rows)
    st.dataframe(compare_df, hide_index=True)
    read = scores["attributes_read"]
    st.caption(
        "Test bank = the model on the recorded features. Typed test = the same kind of case written as a message, "
        "read by the LLM, then predicted by the model: the gap between the two is what typing costs. "
        + (f"The LLM read {read:.0%} of typed-test attributes correctly. " if read is not None else "")
        + f"{scores['hard_cases']} hard cases. A handful of cases is a small sample: compare with the full "
        "test-set figures on the Leaderboard tab."
    )

    records = pd.DataFrame(st.session_state.stats["records"])
    if not records.empty:
        st.markdown("**Every scored case**")
        st.dataframe(records, hide_index=True)
    with st.container(horizontal=True):
        st.download_button(
            "Download summary (CSV)",
            data=compare_df.to_csv(index=False).encode("utf-8"),
            file_name="disaster_auditor_session_summary.csv",
            mime="text/csv",
            icon=":material/download:",
        )
        st.download_button(
            "Download scored cases (CSV)",
            data=records.to_csv(index=False).encode("utf-8"),
            file_name="disaster_auditor_scored_cases.csv",
            mime="text/csv",
            icon=":material/download:",
            disabled=records.empty,
        )


# -------------------------------------------------------------
# Sidebar: new audit, session, settings
# -------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### {APP_ICON} Disaster Auditor")

    if st.button("New audit", icon=":material/edit_square:", type="primary", width="stretch"):
        st.session_state.messages = []
        st.session_state.stats = new_stats()
        st.session_state.last_audit = None
        st.session_state.pending_question = None
        st.session_state.agent_history = []
        st.session_state.gemini_agent_history = []
        st.session_state.session_cases = SessionCases()
        st.rerun()

    scores = session_scores()

    def fmt(value, pattern):
        return pattern.format(value) if value is not None else "—"

    st.markdown("**This session · ML model**")
    with st.container(horizontal=True):
        st.metric("Model Macro-F1", fmt(scores["model"]["macro_f1"], "{:.3f}"),
                  help="Macro-F1 of the active ML model over every case scored this session")
        st.metric("Model accuracy", fmt(scores["model"]["accuracy"], "{:.0%}"))
    st.dataframe(
        pd.DataFrame([{"Source": src, "Cases": sc["n"], "Accuracy": fmt(sc["accuracy"], "{:.0%}"),
                       "Macro-F1": fmt(sc["macro_f1"], "{:.3f}")}
                      for src, sc in scores["by_source"].items()]),
        hide_index=True,
    )
    st.caption(
        "Test bank: the model gets the recorded features. Typed test / Your event: someone types it, the LLM "
        "reads it into the 6 attributes, then the model predicts."
        + (f" The LLM read {scores['attributes_read']:.0%} of typed-test attributes correctly."
           if scores["attributes_read"] is not None else "")
    )
    if st.session_state.llm_baseline:
        with st.container(horizontal=True):
            st.metric("Gemini baseline F1", fmt(scores["gemini"]["macro_f1"], "{:.3f}"),
                      help="Optional: Gemini guessing the type itself from the same 6 attributes, on the same cases")
            st.metric("Baseline accuracy", fmt(scores["gemini"]["accuracy"], "{:.0%}"))
    with st.container(horizontal=True):
        st.metric("Tested", scores["tested"])
        st.metric("Hard cases", scores["hard_cases"],
                  help="Cases the active model misclassified, or got right with confidence below the threshold in settings")

    st.markdown("**Audit a category**")
    with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
        cat_select = st.selectbox(
            "Disaster type",
            ["All categories"] + registry.class_names,
            label_visibility="collapsed",
        )
        target_cat = None if cat_select == "All categories" else cat_select
        if st.button("Run", icon=":material/play_arrow:", help="The model gets a test case's recorded features"):
            run_audit(registry.get_random_case(disaster_type=target_cat), f"Audit a **{cat_select}** case")
    blocked = gemini_block_reason()
    if st.button("Typed test", icon=":material/keyboard:", width="stretch",
                 disabled=not st.session_state.gemini_api_key.strip() or bool(blocked),
                 help="Gemini rewrites a random test case of this category as a typed message (without its type); "
                      "it then goes through the chat like anything you type, and is scored against the real answer."):
        start_typed_test(target_cat)

    is_gemini = st.session_state.api_provider == "Google Gemini"
    active_key = st.session_state.gemini_api_key if is_gemini else st.session_state.claude_api_key
    llm_connected = bool(active_key and active_key.strip())
    llm_label = (("Gemini (agent with tools)" if st.session_state.gemini_agent else "Gemini (chat)")
                 if is_gemini else "Claude Opus 5.5 (agent with tools)")

    with st.expander("Settings", icon=":material/settings:", expanded=not llm_connected):
        prov_choice = st.selectbox(
            "LLM provider",
            ["Google Gemini", "Anthropic Claude"],
            index=0 if is_gemini else 1,
        )
        if prov_choice != st.session_state.api_provider:
            st.session_state.api_provider = prov_choice
            st.rerun()

        if is_gemini:
            gem_key = st.text_input(
                "Gemini API key",
                type="password",
                value=st.session_state.gemini_api_key,
                placeholder="AIzaSy...",
                help="Get a key at https://aistudio.google.com",
            )
            if gem_key.startswith("sk-ant-"):
                st.session_state.claude_api_key = gem_key
                st.session_state.api_provider = "Anthropic Claude"
                st.rerun()
            elif gem_key != st.session_state.gemini_api_key:
                st.session_state.gemini_api_key = gem_key
                st.rerun()
            agent_mode = st.toggle(
                "Agent mode",
                value=st.session_state.gemini_agent,
                help="On: Gemini decides which tools to call (find a case, classify your event, SHAP, metrics) and "
                     "answers from their results. Off: a fixed flow (work out the request, run it, then answer). "
                     "Agent mode can use more of Gemini's free quota per question.",
            )
            if agent_mode != st.session_state.gemini_agent:
                st.session_state.gemini_agent = agent_mode
                st.rerun()
        baseline = st.toggle(
            "LLM baseline",
            value=st.session_state.llm_baseline,
            disabled=not st.session_state.gemini_api_key.strip(),
            help="Optional comparison: Gemini also guesses the disaster type itself from the same 6 attributes, "
                 "scored next to the ML model. The ML model is always the predictor; this costs one Gemini call per case.",
        )
        if baseline != st.session_state.llm_baseline:
            st.session_state.llm_baseline = baseline
            st.rerun()
        else:
            ant_key = st.text_input(
                "Anthropic API key",
                type="password",
                value=st.session_state.claude_api_key,
                placeholder="sk-ant-api03-...",
            )
            if ant_key.startswith("AIzaSy"):
                st.session_state.gemini_api_key = ant_key
                st.session_state.api_provider = "Google Gemini"
                st.rerun()
            elif ant_key != st.session_state.claude_api_key:
                st.session_state.claude_api_key = ant_key
                st.rerun()
            ws_id = st.text_input(
                "Workspace ID (optional)",
                value=st.session_state.claude_workspace_id,
                placeholder="wrkspc_...",
                help="Only needed if your key isn't scoped to a workspace. Copy it from the Claude Console's Workspaces page.",
            )
            if ws_id.strip() != st.session_state.claude_workspace_id:
                st.session_state.claude_workspace_id = ws_id.strip()
                st.rerun()

        st.session_state.hard_threshold = st.slider(
            "Hard-case confidence threshold",
            min_value=0.30,
            max_value=0.90,
            value=st.session_state.hard_threshold,
            step=0.05,
            help="Correct predictions below this confidence also count as hard cases",
        )

    blocked = gemini_block_reason() if is_gemini else None
    status = (":orange[●] paused, built-in answers" if blocked else ":green[●] connected" if llm_connected
              else ":orange[●] offline, add an API key")
    st.caption(f"{llm_label}: {status}")
    if is_gemini and using_shared_key():
        st.caption(f"Gemini requests this session: {st.session_state.gemini_calls}/{SESSION_LIMIT} on the shared key"
                   + (f". {blocked}; add your own free key in Settings to continue." if blocked else ""),
                   help="The public app shares one free-tier key, so each session gets a budget. Your own key isn't limited.")
    st.caption(":green[●] Random Forest · LightGBM · XGBoost ready")
    st.caption(
        f"Test bank: {registry.metadata['total_test_cases']:,} records · "
        f"{len(registry.class_names)} classes · {registry.metadata['disagreement_cases']} disagreements"
    )
    build = cached_build_info()
    st.caption(f":material/commit: {build_label(build)}",
               help=(f"Latest commit: {build['subject']}. " if build.get("subject") else "")
               + "Match this against `git log` to know which code the app is running.")


# -------------------------------------------------------------
# Main area
# -------------------------------------------------------------
# In Chat view the header narrows to the same centered column as the conversation.
topbar_key = "topbar_chat" if st.session_state.layout_mode == "Chat" else "topbar"
with st.container(horizontal=True, vertical_alignment="center", key=topbar_key):
    st.markdown(
        f"**{st.session_state.selected_model}** · test Macro-F1 "
        f"{registry.leaderboard[st.session_state.selected_model]['macro_f1']:.4f}"
    )
    st.space("stretch")
    view_choice = st.segmented_control(
        "View",
        options=VIEWS,
        default=st.session_state.layout_mode,
        label_visibility="collapsed",
    )
    if view_choice and view_choice != st.session_state.layout_mode:
        st.session_state.layout_mode = view_choice
        st.rerun()

# Quick-action chip picked on the previous run (queued by its callback)
if st.session_state.pending_quick_action:
    pick = st.session_state.pending_quick_action
    st.session_state.pending_quick_action = None
    run_quick_action(pick)

COMPOSER = "Ask about a case, or type / for commands"
view = st.session_state.layout_mode

if view == "Chat":
    with st.container(key="chat_column"):
        render_chat(COMPOSER)

elif view == "Chat + details":
    # Keys let the CSS lock the page and give each column its own scroll (see "Chat + details view").
    col_chat, col_details = st.columns([0.55, 0.45], gap="large")
    with col_chat:
        with st.container(key="details_chat"):
            render_chat(COMPOSER)
    with col_details:
        with st.container(border=True, key="details_panel"):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown("**Case details**")
                st.space("stretch")
                if st.button("Close", icon=":material/close:", type="tertiary"):
                    st.session_state.layout_mode = "Chat"
                    st.rerun()
            render_case_details()

else:  # Benchmarks
    tab_leader, tab_disagree, tab_case, tab_export = st.tabs([
        ":material/leaderboard: Leaderboard",
        ":material/compare_arrows: Disagreements",
        ":material/search: Last case",
        ":material/download: Export",
    ])
    with tab_leader:
        render_leaderboard()
    with tab_disagree:
        render_disagreement()
    with tab_case:
        render_case_details()
    with tab_export:
        render_export()

# Last on the page: the slash-command menu script (renders nothing visible).
# Kept out of the sidebar because a collapsed sidebar doesn't mount its contents.
st.html(SLASH_MENU_JS, unsafe_allow_javascript=True)
