"""Cached objects follow the code: a push that changes src/ must not keep serving the old classes."""
from unittest import mock

from src.code_version import code_version


def test_fingerprint_changes_with_any_source_file(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "models" / "b.py").write_text("y = 1\n")
    first = code_version(tmp_path)
    assert code_version(tmp_path) == first
    (tmp_path / "models" / "b.py").write_text("y = 2\n")
    assert code_version(tmp_path) != first


def test_app_rebuilds_the_agent_tools_when_the_code_changes(app):
    import src.auditor_agent as aa

    at = app({})
    at.run()  # builds and caches the tools for the current fingerprint

    built = []
    real_init = aa.AuditorTools.__init__

    def counting_init(self, registry):
        built.append(1)
        real_init(self, registry)

    with mock.patch.object(aa.AuditorTools, "__init__", counting_init):
        at.run()                                   # same code: cached tools reused
        assert built == []
        with mock.patch("src.code_version.code_version", return_value="new-version"):
            at.run()                               # code changed: tools rebuilt
    assert built == [1]
