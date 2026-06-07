from __future__ import annotations

import os

from tsm_env import load_project_env


def test_load_project_env_reads_key_values(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        """
        # comments are ignored
        export KOREAEXIM_AUTHKEY="abc123"
        KEEP_EXISTING=from_file
        """,
        encoding="utf-8",
    )
    monkeypatch.delenv("KOREAEXIM_AUTHKEY", raising=False)
    monkeypatch.setenv("KEEP_EXISTING", "original")

    load_project_env(env_path)

    assert os.environ["KOREAEXIM_AUTHKEY"] == "abc123"
    assert os.environ["KEEP_EXISTING"] == "original"


def test_load_project_env_can_override_existing_values(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("KOREAEXIM_AUTHKEY=new-value\n", encoding="utf-8")
    monkeypatch.setenv("KOREAEXIM_AUTHKEY", "old-value")

    load_project_env(env_path, override=True)

    assert os.environ["KOREAEXIM_AUTHKEY"] == "new-value"
