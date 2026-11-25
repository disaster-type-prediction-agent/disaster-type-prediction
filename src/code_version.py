"""A fingerprint of the app's own code, used as a cache key.

st.cache_resource keys an object on its loader's code and arguments only. Objects built from src/ classes
(the model registry, the agent tools) would otherwise outlive a push: Streamlit reloads the changed modules,
but the cached objects still belong to the old classes until the server restarts."""
import hashlib
from pathlib import Path

SRC = Path(__file__).resolve().parent


def code_version(folder: Path = SRC) -> str:
    digest = hashlib.sha1()
    for path in sorted(folder.rglob("*.py")):
        digest.update(str(path.relative_to(folder)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def build_info(folder: Path = SRC) -> dict:
    """Which version of the code this app is running, to show in the UI.

    The commit (short hash, time, branch) when the app runs from a git checkout, as on Streamlit Cloud;
    otherwise (e.g. a Docker image without .git) the fingerprint of src/ above, so a deploy can always be
    matched to the code without guessing from the LLM's behaviour."""
    import subprocess
    from datetime import datetime

    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h|%cI|%s"], cwd=folder.parent, capture_output=True,
                             text=True, timeout=3)
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=folder.parent,
                                capture_output=True, text=True, timeout=3).stdout.strip()
        commit, when, subject = out.stdout.strip().split("|", 2)
        return {"commit": commit, "time": datetime.fromisoformat(when).strftime("%Y-%m-%d %H:%M %z"),
                "subject": subject, "branch": branch if branch and branch != "HEAD" else None}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"commit": None, "time": None, "subject": None, "branch": None, "code": code_version(folder)[:7]}


def build_label(info: dict) -> str:
    """One line for the UI, e.g. 'Version 37561c5 · 2026-11-22 21:40 +0700 · agentic-ai-auditor'."""
    if not info.get("commit"):
        return f"Version: code {info.get('code')} (no git info)"
    parts = [f"Version {info['commit']}", info["time"]] + ([info["branch"]] if info.get("branch") else [])
    return " · ".join(parts)
