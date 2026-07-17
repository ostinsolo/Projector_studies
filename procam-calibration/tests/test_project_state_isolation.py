"""Regression: AutoWall must never mutate the repository PROJECT_STATE.md from tests."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from procam_calibrate.auto_wall import AutoWall, AutoWallConfig
from procam_calibrate.paths import Roots


def _repo_project_state() -> Path:
    return Roots.resolve().integration.parent / "PROJECT_STATE.md"


def test_temp_autowall_does_not_modify_repo_project_state():
    root_ps = _repo_project_state()
    assert root_ps.exists()
    before = root_ps.read_bytes()
    before_hash = hashlib.sha256(before).hexdigest()

    tmp = Path(tempfile.mkdtemp()) / "isolation_run"
    tmp.mkdir()
    live = tmp / "PROJECT_STATE_LIVE.md"
    aw = AutoWall(
        AutoWallConfig(run_dir=tmp, pipeline="homography", project_state_path=live)
    )
    aw.state["state"] = "PREFLIGHT"
    aw._save_state()
    aw._transition("DISCOVER_CAMERA", "test")
    aw.state["state"] = "BLOCKED_EXTERNAL"
    aw.state["data"]["blocker"] = "test blocker"
    aw._save_state()
    aw.state["state"] = "ACCEPTANCE_FAILED"
    aw._save_state()
    aw.state["state"] = "DONE"
    aw._save_state()

    after = root_ps.read_bytes()
    assert hashlib.sha256(after).hexdigest() == before_hash
    assert b"/var/folders/" not in after
    assert b"staterun" not in after
    assert live.exists()
    body = live.read_text()
    assert "DONE" in body
    assert str(tmp) in body


def test_none_project_state_path_writes_nowhere():
    root_ps = _repo_project_state()
    before_hash = hashlib.sha256(root_ps.read_bytes()).hexdigest()
    tmp = Path(tempfile.mkdtemp()) / "no_write_run"
    tmp.mkdir()
    aw = AutoWall(AutoWallConfig(run_dir=tmp, pipeline="homography", project_state_path=None))
    aw.state["state"] = "RUNNING"
    aw._save_state()
    assert not (tmp / "PROJECT_STATE_LIVE.md").exists()
    assert hashlib.sha256(root_ps.read_bytes()).hexdigest() == before_hash


def test_configured_path_used_for_success_fail_blocked():
    tmp = Path(tempfile.mkdtemp()) / "cfg_path_run"
    tmp.mkdir()
    live = tmp / "nested" / "live_state.md"
    aw = AutoWall(
        AutoWallConfig(run_dir=tmp, pipeline="homography", project_state_path=live)
    )
    for st, token in (
        ("DONE", "DONE"),
        ("ACCEPTANCE_FAILED", "ACCEPTANCE_FAILED"),
        ("BLOCKED_EXTERNAL", "BLOCKED_EXTERNAL"),
    ):
        aw.state["state"] = st
        if st == "BLOCKED_EXTERNAL":
            aw.state["data"]["blocker"] = "blocked for test"
        aw._save_state()
        text = live.read_text()
        assert f"**{token}**" in text
        assert "`" + st + "`" in text


def test_repo_project_state_has_no_temp_paths():
    text = _repo_project_state().read_text()
    assert "/var/folders/" not in text
    assert "/tmp/" not in text
    assert "staterun" not in text
    assert "DONE / VALIDATED" in text
    assert "COMPLETE" in text
    assert "0.34" in text
