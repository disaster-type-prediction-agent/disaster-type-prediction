"""Shared fixtures. Tests run offline: no Anthropic or Gemini calls are made."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KEY_VARS = ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_WORKSPACE_ID", "LLM_PROVIDER", "GEMINI_MODEL")


@pytest.fixture(scope="session")
def registry():
    from src.models.registry import ModelRegistry

    cwd = os.getcwd()
    os.chdir(ROOT)  # the registry loads models/ by relative path
    try:
        return ModelRegistry()
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="session")
def tools(registry):
    from src.auditor_agent import AuditorTools

    return AuditorTools(registry)


@pytest.fixture
def app(tmp_path, clean_env):
    """Run app.py headlessly from an empty temp folder (models/ linked in), so the developer's real
    .streamlit/secrets.toml never leaks into a test. Call it with secrets/env to set keys."""
    from streamlit.testing.v1 import AppTest

    (tmp_path / "models").symlink_to(ROOT / "models")
    clean_env.chdir(tmp_path)

    def make(secrets=None, env=None):
        for k, v in (env or {}).items():
            clean_env.setenv(k, v)
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        for k, v in (secrets or {}).items():
            at.secrets[k] = v
        return at.run()

    return make


@pytest.fixture
def clean_env(monkeypatch):
    """No API keys in the environment, and any a test (or Streamlit) sets are removed afterwards."""
    for name in KEY_VARS:
        monkeypatch.setenv(name, "")  # records the original so teardown restores it
        monkeypatch.delenv(name)
    return monkeypatch
