#!/usr/bin/env python3
"""Read-only external progress monitor for an active CSPR-Net FIT stage.

Does not attach to, signal, restart, patch, or replace the training process.

Continuous writes (every --interval seconds):
  <run-dir>/logs/cspr_fit_monitor_latest.json   (atomic replace only)

PROJECT_STATE.md is updated ONLY on milestone events (see EVENT_*).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

ITER_RE = re.compile(
    r"Iter\s+(\d+)\s+\[Img\s+(\d+)\]\s+\|\s+Loss:\s+([0-9.]+)",
    re.IGNORECASE,
)
CKPT_ITER_RE = re.compile(r"net_c2p\.iter(\d+)\.pth$", re.IGNORECASE)

# Events that may update PROJECT_STATE.md
EVENT_CKPT_1000 = "checkpoint_1000"
EVENT_CKPT_2000 = "checkpoint_2000"
EVENT_TRAIN_COMPLETE = "training_complete"
EVENT_TRAIN_ERROR = "training_error"
EVENT_TRAINER_GONE = "trainer_disappeared"
EVENT_STALL = "suspected_stall"


@dataclass
class MonitorState:
    prev_cputime_s: Optional[float] = None
    prev_sample_t: Optional[float] = None
    rss_history: list[tuple[float, int]] = field(default_factory=list)
    cpu_delta_history: list[tuple[float, float]] = field(default_factory=list)
    last_cpu_increase_t: float = 0.0
    trainer_start_wall: Optional[float] = None
    t_ckpt_1000: Optional[float] = None
    t_ckpt_2000: Optional[float] = None
    t_complete: Optional[float] = None
    events_fired: set[str] = field(default_factory=set)
    last_project_state_event: Optional[str] = None


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=False)
            f.write("\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")
        f.flush()


def etime_to_seconds(etime: str) -> Optional[float]:
    try:
        days = 0
        rest = etime.strip()
        if "-" in rest:
            d, rest = rest.split("-", 1)
            days = int(d)
        parts = rest.split(":")
        if len(parts) == 2:
            h = 0
            m = int(parts[0])
            s = float(parts[1])
        elif len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            s = float(parts[2])
        else:
            return None
        return float(days * 86400 + h * 3600 + m * 60 + s)
    except Exception:
        return None


def cputime_to_seconds(cputime: str) -> Optional[float]:
    """Parse ps TIME field: macOS often uses mm:ss.xx or hh:mm:ss.xx."""
    return etime_to_seconds(cputime)


def ps_row(pid: int) -> Optional[dict]:
    if pid <= 0:
        return None
    try:
        out = subprocess.check_output(
            [
                "ps",
                "-p",
                str(pid),
                "-o",
                "pid=,ppid=,state=,etime=,time=,%cpu=,rss=,lstart=",
            ],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None
    if not out:
        return None
    parts = out.split(None, 7)
    if len(parts) < 8:
        return None
    cputime_s = cputime_to_seconds(parts[4])
    return {
        "pid": int(parts[0]),
        "ppid": int(parts[1]),
        "state": parts[2],
        "etime": parts[3],
        "cputime": parts[4],
        "cputime_s": cputime_s,
        "cpu_pct": float(parts[5]),
        "rss_kb": int(parts[6]),
        "lstart": parts[7].strip(),
    }


def parse_lstart(lstart: str) -> Optional[float]:
    # e.g. Fri Jul 17 14:53:32 2026
    for fmt in ("%a %b %d %H:%M:%S %Y", "%c"):
        try:
            return datetime.strptime(lstart.strip(), fmt).timestamp()
        except ValueError:
            continue
    return None


def count_matching_procs(pattern: str) -> list[int]:
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True).strip()
    except subprocess.CalledProcessError:
        return []
    return [int(x) for x in out.split() if x.isdigit()] if out else []


def auto_wall_state(run_dir: Path) -> str:
    p = run_dir / "auto_wall_state.json"
    if not p.exists():
        return "missing"
    try:
        return str(json.loads(p.read_text()).get("state", "unknown"))
    except Exception:
        return "unreadable"


def find_checkpoints(run_dir: Path) -> dict[int, Path]:
    root = run_dir / "calibration" / "cspr_work"
    found: dict[int, Path] = {}
    if not root.exists():
        return found
    for p in root.rglob("net_c2p*.pth"):
        m = CKPT_ITER_RE.search(p.name)
        if m:
            found[int(m.group(1))] = p
        elif p.name == "net_c2p.pth":
            found[0] = p  # final unmarked
    return found


def list_output_files(run_dir: Path) -> list[dict]:
    out_dir = run_dir / "calibration" / "cspr_work" / "neural_results_exp"
    if not out_dir.exists():
        return []
    rows = []
    for p in sorted(out_dir.iterdir(), key=lambda x: x.stat().st_mtime):
        if p.is_file():
            st = p.stat()
            rows.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "bytes": st.st_size,
                    "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                }
            )
    return rows


def newest_checkpoint_info(ckpts: dict[int, Path]) -> tuple[Optional[str], Optional[str], Optional[int]]:
    # Prefer highest numbered iter*; ignore sentinel 0 unless it's the only one
    numbered = {k: v for k, v in ckpts.items() if k > 0}
    if numbered:
        it = max(numbered)
        p = numbered[it]
        return str(p), datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"), it
    if 0 in ckpts:
        p = ckpts[0]
        return str(p), datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"), None
    return None, None, None


def format_duration(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def compute_progress(
    state: MonitorState,
    total_iters: int,
    ckpts: dict[int, Path],
    run_dir: Path,
    trainer_elapsed_s: Optional[float],
) -> dict:
    """Checkpoint-based progress only — never invent iter/ETA from CPU%."""
    current_iter: Optional[int] = None
    sec_per_iter: Optional[float] = None
    recent_spi: Optional[float] = None
    remaining: Optional[float] = None
    eta_clock: Optional[str] = None
    confidence = "NONE"
    source = "awaiting_checkpoint"
    complete = False
    latest_loss: Optional[float] = None

    # Final artifacts
    loss_csv = run_dir / "calibration" / "cspr_work" / "neural_results_exp" / "loss_metrics.csv"
    final_c2p = run_dir / "calibration" / "cspr_work" / "neural_results_exp" / "net_c2p.pth"
    stdout_log = run_dir / "logs" / "cspr_train_stdout.txt"

    # Final unmarked checkpoint / CSV appear only after the training loop finishes.
    if loss_csv.exists() or final_c2p.exists():
        current_iter = total_iters
        complete = True
        confidence = "HIGH"
        source = "final_outputs"
        marker = loss_csv if loss_csv.exists() else final_c2p
        if state.trainer_start_wall:
            state.t_complete = marker.stat().st_mtime
            total_dur = state.t_complete - state.trainer_start_wall
        else:
            total_dur = trainer_elapsed_s
        if total_dur and total_iters:
            sec_per_iter = total_dur / total_iters
        remaining = 0.0
        eta_clock = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Parse streamed/captured Iter lines if available (future runs / after exit)
    if stdout_log.exists() and stdout_log.stat().st_size > 0:
        try:
            text = stdout_log.read_text(errors="replace")
            matches = list(ITER_RE.finditer(text))
            if matches:
                m = matches[-1]
                lit = int(m.group(1))
                latest_loss = float(m.group(3))
                if current_iter is None or lit > (current_iter or 0):
                    if not complete:
                        current_iter = lit
                        source = "stdout_iter_lines"
        except OSError:
            pass

    # Checkpoint milestones
    if 1000 in ckpts and state.t_ckpt_1000 is None:
        state.t_ckpt_1000 = ckpts[1000].stat().st_mtime
    if 2000 in ckpts and state.t_ckpt_2000 is None:
        state.t_ckpt_2000 = ckpts[2000].stat().st_mtime

    if not complete and 2000 in ckpts and state.trainer_start_wall:
        current_iter = 2000
        source = "net_c2p.iter2000.pth"
        full_spi = (state.t_ckpt_2000 - state.trainer_start_wall) / 2000.0
        if state.t_ckpt_1000:
            recent_spi = (state.t_ckpt_2000 - state.t_ckpt_1000) / 1000.0
            # Prefer recent when stable (within 40% of full-run rate)
            if recent_spi > 0 and abs(recent_spi - full_spi) / max(full_spi, 1e-9) <= 0.4:
                sec_per_iter = recent_spi
            else:
                sec_per_iter = recent_spi if recent_spi > 0 else full_spi
        else:
            sec_per_iter = full_spi
        remaining = sec_per_iter * (total_iters - 2000)
        confidence = "HIGH"
        eta_clock = (datetime.now() + timedelta(seconds=remaining)).strftime("%Y-%m-%d %H:%M:%S")
    elif not complete and 1000 in ckpts and state.trainer_start_wall:
        current_iter = 1000
        source = "net_c2p.iter1000.pth"
        sec_per_iter = (state.t_ckpt_1000 - state.trainer_start_wall) / 1000.0
        remaining = sec_per_iter * (total_iters - 1000)
        confidence = "MEDIUM"
        eta_clock = (datetime.now() + timedelta(seconds=remaining)).strftime("%Y-%m-%d %H:%M:%S")
    elif not complete:
        current_iter = None
        confidence = "NONE"
        source = (
            "no_mid_run_iter_exposure; stdout pipe-captured until train exits; "
            "next readable milestone net_c2p.iter1000.pth"
        )

    pct = None
    if current_iter is not None:
        pct = round(100.0 * min(current_iter, total_iters) / total_iters, 2)

    return {
        "current_iteration": current_iter,
        "pct_complete": pct,
        "seconds_per_iteration": round(sec_per_iter, 6) if sec_per_iter else None,
        "recent_seconds_per_iteration": round(recent_spi, 6) if recent_spi else None,
        "remaining_seconds": round(remaining, 1) if remaining is not None else None,
        "remaining_human": format_duration(remaining),
        "eta_clock": eta_clock,
        "eta_confidence": confidence,
        "progress_source": source,
        "training_complete": complete,
        "latest_loss": latest_loss,
        "t_ckpt_1000": datetime.fromtimestamp(state.t_ckpt_1000).isoformat(timespec="seconds")
        if state.t_ckpt_1000
        else None,
        "t_ckpt_2000": datetime.fromtimestamp(state.t_ckpt_2000).isoformat(timespec="seconds")
        if state.t_ckpt_2000
        else None,
        "total_duration_s": round(state.t_complete - state.trainer_start_wall, 1)
        if state.t_complete and state.trainer_start_wall
        else None,
    }


def assess_health(
    state: MonitorState,
    parent: Optional[dict],
    trainer: Optional[dict],
    parent_pid: int,
    trainer_pid: int,
    run_dir: Path,
    progress: dict,
    cpu_delta_s: Optional[float],
    now: float,
) -> dict:
    parents = count_matching_procs(f"procam-calibrate auto-wall.*{run_dir.name}")
    trainers = count_matching_procs(f"{run_dir.name}/calibration/cspr_work/run_train.py")
    relationship_ok = bool(
        parent and trainer and trainer["ppid"] == parent["pid"] and parent["pid"] == parent_pid
    )
    state_ok = bool(trainer and trainer["state"][:1] in "RSUT")

    # Memory trend
    mem_note = "insufficient_samples"
    mem_pressure = False
    if len(state.rss_history) >= 8:
        recent = state.rss_history[-18:]
        growth = recent[-1][1] - recent[0][1]
        span = recent[-1][0] - recent[0][0]
        if span >= 60:
            if growth > 400_000 and growth / max(recent[0][1], 1) > 0.25:
                mem_note = f"rising (+{growth} KB / {span:.0f}s)"
                # toward pressure if > 12 GiB RSS
                if recent[-1][1] > 12 * 1024 * 1024:
                    mem_pressure = True
                    mem_note += " toward_system_pressure"
            else:
                mem_note = f"stable (delta {growth} KB / {span:.0f}s)"

    # CPU liveness
    cpu_stalled = False
    negligible_cpu = False
    since_cpu = now - state.last_cpu_increase_t if state.last_cpu_increase_t else None
    if since_cpu is not None and since_cpu >= 300:
        cpu_stalled = True
    if trainer and since_cpu is not None and since_cpu >= 300 and trainer["cpu_pct"] < 1.0:
        negligible_cpu = True

    stderr = run_dir / "logs" / "cspr_train_stderr.txt"
    train_error = False
    if stderr.exists() and stderr.stat().st_size > 0:
        try:
            err = stderr.read_text(errors="replace")
            if "Traceback" in err or "Error" in err:
                train_error = True
        except OSError:
            pass

    trainer_gone = trainer is None
    parent_gone = parent is None

    # Stall policy: before ckpt 1000, do NOT stall merely from missing checkpoints
    before_1000 = progress.get("current_iteration") is None or (
        progress.get("current_iteration") is not None and progress["current_iteration"] < 1000
    )
    suspected_stall = False
    stall_reasons: list[str] = []
    if trainer and not progress.get("training_complete"):
        if cpu_stalled:
            suspected_stall = True
            stall_reasons.append(f"cpu_time_flat_{since_cpu:.0f}s")
        if negligible_cpu:
            suspected_stall = True
            stall_reasons.append(f"negligible_cpu_{since_cpu:.0f}s")
        if mem_pressure:
            suspected_stall = True
            stall_reasons.append("memory_pressure")
    if trainer_gone and not progress.get("training_complete"):
        # unexpected exit handled as separate event; also mark unhealthy
        pass

    healthy = bool(
        relationship_ok
        and state_ok
        and not suspected_stall
        and not trainer_gone
        and not parent_gone
        and (cpu_delta_s is None or cpu_delta_s >= 0)
    )
    if before_1000 and trainer and relationship_ok and state_ok and not cpu_stalled and not negligible_cpu:
        # Explicit: missing checkpoint is OK
        healthy = not mem_pressure and not parent_gone

    return {
        "healthy": healthy,
        "exactly_one_auto_wall_parent": len(parents) == 1,
        "exactly_one_run_train_child": len(trainers) == 1,
        "auto_wall_pids": parents,
        "run_train_pids": trainers,
        "parent_child_ok": relationship_ok,
        "trainer_state_valid": state_ok,
        "memory_trend": mem_note,
        "memory_pressure": mem_pressure,
        "cpu_time_increasing": (cpu_delta_s is not None and cpu_delta_s > 0.01)
        or (since_cpu is not None and since_cpu < 30),
        "seconds_since_cpu_increase": round(since_cpu, 1) if since_cpu is not None else None,
        "train_error_output_detected": train_error,
        "trainer_gone": trainer_gone,
        "parent_gone": parent_gone,
        "suspected_stall": suspected_stall,
        "stall_reasons": stall_reasons,
        "stall_action": "report_only_do_not_kill",
        "before_checkpoint_1000": before_1000,
    }


def write_project_state_event(run_dir: Path, event: str, report: dict) -> None:
    """Event-gated PROJECT_STATE update only — never on the 10s cadence."""
    root = run_dir.parents[2]
    ps = root / "PROJECT_STATE.md"
    prog = report["progress"]
    health = report["health"]
    body = (
        f"# PROJECT_STATE\n\n"
        f"## Status\n\n**RUNNING**\n\n"
        f"## Auto-wall state\n\n`{report['auto_wall_state']}`\n\n"
        f"Run: `{run_dir}`\n\n"
        f"## CSPR fit monitor event\n\n"
        f"- Event: **{event}**\n"
        f"- Time: {report['timestamp']}\n"
        f"- Parent/trainer: {report['parent_pid']} → {report['trainer_pid']} "
        f"(child_ok={report['parent_child_ok']})\n"
        f"- Iteration: {prog.get('current_iteration')} / {report['total_iters']} "
        f"({prog.get('pct_complete')}%)\n"
        f"- s/iter: {prog.get('seconds_per_iteration')} "
        f"(recent={prog.get('recent_seconds_per_iteration')})\n"
        f"- ETA: {prog.get('remaining_human')} → {prog.get('eta_clock')} "
        f"(confidence={prog.get('eta_confidence')})\n"
        f"- Device: CPU (active run; do not switch)\n"
        f"- Health: healthy={health.get('healthy')} stall={health.get('suspected_stall')} "
        f"reasons={health.get('stall_reasons')}\n"
        f"- Continuous detail (atomic): "
        f"`{run_dir / 'logs' / 'cspr_fit_monitor_latest.json'}`\n"
    )
    if event == EVENT_TRAIN_COMPLETE:
        body = body.replace("**RUNNING**", "**RUNNING** (fit complete — auto-wall continuing)")
    try:
        ps.write_text(body)
    except OSError as e:
        print(f"[monitor] PROJECT_STATE write failed: {e}", file=sys.stderr)


def build_sample(
    run_dir: Path,
    total_iters: int,
    parent_pid: int,
    trainer_pid: int,
    state: MonitorState,
) -> tuple[dict, list[str]]:
    now = time.time()
    events: list[str] = []

    parent = ps_row(parent_pid)
    trainer = ps_row(trainer_pid)

    # Rediscover if PIDs went stale but unique match exists
    if parent is None:
        cands = count_matching_procs(f"procam-calibrate auto-wall.*{run_dir.name}")
        if len(cands) == 1:
            parent_pid = cands[0]
            parent = ps_row(parent_pid)
    if trainer is None:
        cands = count_matching_procs(f"{run_dir.name}/calibration/cspr_work/run_train.py")
        if len(cands) == 1:
            trainer_pid = cands[0]
            trainer = ps_row(trainer_pid)

    if trainer and state.trainer_start_wall is None:
        state.trainer_start_wall = parse_lstart(trainer["lstart"]) or (
            now - (etime_to_seconds(trainer["etime"]) or 0)
        )
        state.last_cpu_increase_t = now

    cpu_delta = None
    if trainer and trainer.get("cputime_s") is not None:
        if state.prev_cputime_s is not None:
            cpu_delta = max(0.0, trainer["cputime_s"] - state.prev_cputime_s)
            if cpu_delta > 0.05:
                state.last_cpu_increase_t = now
            state.cpu_delta_history.append((now, cpu_delta))
            state.cpu_delta_history = state.cpu_delta_history[-120:]
        state.prev_cputime_s = trainer["cputime_s"]
    state.prev_sample_t = now

    if trainer:
        state.rss_history.append((now, trainer["rss_kb"]))
        state.rss_history = state.rss_history[-120:]

    ckpts = find_checkpoints(run_dir)
    # Detect checkpoint appearance events
    if 1000 in ckpts and EVENT_CKPT_1000 not in state.events_fired:
        state.t_ckpt_1000 = ckpts[1000].stat().st_mtime
        events.append(EVENT_CKPT_1000)
        state.events_fired.add(EVENT_CKPT_1000)
    if 2000 in ckpts and EVENT_CKPT_2000 not in state.events_fired:
        state.t_ckpt_2000 = ckpts[2000].stat().st_mtime
        events.append(EVENT_CKPT_2000)
        state.events_fired.add(EVENT_CKPT_2000)

    progress = compute_progress(state, total_iters, ckpts, run_dir, etime_to_seconds(trainer["etime"]) if trainer else None)

    if progress["training_complete"] and EVENT_TRAIN_COMPLETE not in state.events_fired:
        events.append(EVENT_TRAIN_COMPLETE)
        state.events_fired.add(EVENT_TRAIN_COMPLETE)

    ckpt_path, ckpt_mtime, ckpt_iter = newest_checkpoint_info(ckpts)
    health = assess_health(
        state, parent, trainer, parent_pid, trainer_pid, run_dir, progress, cpu_delta, now
    )

    if health["suspected_stall"] and EVENT_STALL not in state.events_fired:
        events.append(EVENT_STALL)
        state.events_fired.add(EVENT_STALL)

    if health["trainer_gone"] and not progress["training_complete"]:
        # Check exit via stdout presence + return
        stdout = run_dir / "logs" / "cspr_train_stdout.txt"
        adapt = run_dir / "calibration" / "cspr_adapt.json"
        if adapt.exists():
            try:
                ad = json.loads(adapt.read_text())
                if ad.get("exit_code", 0) != 0 and EVENT_TRAIN_ERROR not in state.events_fired:
                    events.append(EVENT_TRAIN_ERROR)
                    state.events_fired.add(EVENT_TRAIN_ERROR)
            except Exception:
                pass
        if EVENT_TRAINER_GONE not in state.events_fired and EVENT_TRAIN_COMPLETE not in state.events_fired:
            if not stdout.exists() or not progress["training_complete"]:
                events.append(EVENT_TRAINER_GONE)
                state.events_fired.add(EVENT_TRAINER_GONE)

    parent_child_ok = bool(
        parent and trainer and trainer["ppid"] == parent["pid"]
    )

    report = {
        "timestamp": now_iso(),
        "wall_clock_unix": now,
        "parent_pid": parent_pid,
        "trainer_pid": trainer_pid,
        "parent_child_ok": parent_child_ok,
        "trainer_elapsed": trainer["etime"] if trainer else None,
        "trainer_elapsed_s": etime_to_seconds(trainer["etime"]) if trainer else None,
        "trainer_cputime": trainer["cputime"] if trainer else None,
        "trainer_cputime_s": trainer.get("cputime_s") if trainer else None,
        "cputime_delta_s": round(cpu_delta, 3) if cpu_delta is not None else None,
        "cpu_pct": trainer["cpu_pct"] if trainer else None,
        "rss_kb": trainer["rss_kb"] if trainer else None,
        "rss_gb": round(trainer["rss_kb"] / 1024 / 1024, 3) if trainer else None,
        "process_state": trainer["state"] if trainer else None,
        "parent_state": parent["state"] if parent else None,
        "auto_wall_state": auto_wall_state(run_dir),
        "newest_checkpoint": ckpt_path,
        "checkpoint_mtime": ckpt_mtime,
        "checkpoint_iteration": ckpt_iter,
        "output_files": list_output_files(run_dir),
        "total_iters": total_iters,
        "progress": progress,
        "monitor_confidence": progress["eta_confidence"],
        "health": health,
        "device": {
            "active_run": "cpu",
            "note": "Do not switch this run; Config is cuda-if-available else cpu",
        },
        "events_this_sample": events,
    }
    return report, events


def print_sample(report: dict) -> None:
    p = report["progress"]
    h = report["health"]
    print(
        f"[{report['timestamp']}] state={report['auto_wall_state']} "
        f"iter={p.get('current_iteration')}/{report['total_iters']} "
        f"conf={p.get('eta_confidence')} "
        f"ETA={p.get('remaining_human')}~{p.get('eta_clock')} "
        f"cpuΔ={report.get('cputime_delta_s')}s "
        f"cpu%={report.get('cpu_pct')} rss={report.get('rss_gb')}GiB "
        f"ckpt={report.get('checkpoint_iteration')} "
        f"healthy={h.get('healthy')} stall={h.get('suspected_stall')}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only CSPR fit monitor (atomic JSON)")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--total-iters", type=int, default=2500)
    ap.add_argument("--parent-pid", type=int, default=0)
    ap.add_argument("--trainer-pid", type=int, default=0)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--max-hours", type=float, default=24.0)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"run-dir not found: {run_dir}", file=sys.stderr)
        return 2

    latest_path = run_dir / "logs" / "cspr_fit_monitor_latest.json"
    samples_path = run_dir / "logs" / "cspr_fit_monitor_samples.jsonl"
    events_path = run_dir / "logs" / "cspr_fit_monitor_events_fired.json"
    state = MonitorState()
    if events_path.exists():
        try:
            state.events_fired = set(json.loads(events_path.read_text()))
        except Exception:
            pass

    # Seed checkpoint mtimes without re-firing PROJECT_STATE if already recorded
    ckpts0 = find_checkpoints(run_dir)
    if 1000 in ckpts0:
        state.t_ckpt_1000 = ckpts0[1000].stat().st_mtime
    if 2000 in ckpts0:
        state.t_ckpt_2000 = ckpts0[2000].stat().st_mtime

    deadline = time.time() + args.max_hours * 3600
    while time.time() < deadline:
        report, events = build_sample(
            run_dir, args.total_iters, args.parent_pid, args.trainer_pid, state
        )
        atomic_write_json(latest_path, report)
        append_jsonl(samples_path, {
            "timestamp": report["timestamp"],
            "cputime_delta_s": report["cputime_delta_s"],
            "cpu_pct": report["cpu_pct"],
            "rss_kb": report["rss_kb"],
            "current_iteration": report["progress"]["current_iteration"],
            "eta_confidence": report["progress"]["eta_confidence"],
            "remaining_seconds": report["progress"]["remaining_seconds"],
            "healthy": report["health"]["healthy"],
            "suspected_stall": report["health"]["suspected_stall"],
            "events": events,
        })
        print_sample(report)

        for ev in events:
            write_project_state_event(run_dir, ev, report)
            state.last_project_state_event = ev
            print(f"[monitor] PROJECT_STATE.md updated for event: {ev}", flush=True)
        if events:
            atomic_write_json(events_path, sorted(state.events_fired))

        if report["progress"]["training_complete"]:
            print("[monitor] Training complete observed; exiting monitor.", flush=True)
            break
        if report["health"]["trainer_gone"] and EVENT_TRAIN_COMPLETE not in state.events_fired:
            # Keep monitoring briefly already handled via events; exit after reporting
            print("[monitor] Trainer gone unexpectedly; exiting monitor.", flush=True)
            break
        if args.once:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
