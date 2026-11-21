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
