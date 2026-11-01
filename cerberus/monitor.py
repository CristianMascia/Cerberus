"""
Cerberus — live monitor of one or more batches (`cerberus batch monitor`).

Runs on the login node and needs only the standard library plus `squeue`.
For every batch dir it shows, per shard: the SLURM state of the array task
(from squeue; a task that left the queue is classified from its result file
and its SLURM log, since sacct may not be available), the node, the elapsed
time, units done / total, errors, throughput and ETA.

Results files are read incrementally (only the bytes appended since the last
refresh), so refreshing every few seconds is cheap even with large results.
Throughput is measured over a sliding window of the refreshes; on the first
refresh it falls back to done / elapsed of the running task.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .batch import BatchError, _read_jsonl, _shard_name, load_manifest

# --------------------------------------------------------------------------- #
# SLURM
# --------------------------------------------------------------------------- #
@dataclass
class TaskInfo:
    state: str          # RUNNING, PENDING, COMPLETING, ...
    elapsed: str        # squeue %M, e.g. "1:02:03"
    node: str           # squeue %N (running) or the pending reason
    elapsed_s: Optional[float] = None


def _parse_elapsed(s: str) -> Optional[float]:
    """squeue time ([D-]HH:MM:SS, MM:SS) -> seconds."""
    s = s.strip()
    if not s or s in ("0:00", "INVALID", "N/A"):
        return 0.0 if s == "0:00" else None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    try:
        parts = [int(p) for p in s.split(":")]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + sec


def squeue_tasks(job_ids: list[str]) -> dict[str, dict[int, TaskInfo]]:
    """{job_id: {array_index: TaskInfo}} for the jobs still in the queue."""
    out: dict[str, dict[int, TaskInfo]] = {j: {} for j in job_ids}
    if not job_ids or not shutil.which("squeue"):
        return out
    # By user, not -j: squeue fails on a job id that already left the queue.
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    cmd = ["squeue", "-h", "-r", "-o", "%F|%K|%T|%M|%N|%r"] + (["-u", user] if user else [])
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        jid, idx, state, elapsed, node, reason = parts[:6]
        if jid not in out or not idx.isdigit():
            continue
        where = node if state in ("RUNNING", "COMPLETING") else f"({reason})"
        out[jid][int(idx)] = TaskInfo(state, elapsed, where, _parse_elapsed(elapsed))
    return out


def has_squeue() -> bool:
    return shutil.which("squeue") is not None


# --------------------------------------------------------------------------- #
# Progress tracking
# --------------------------------------------------------------------------- #
@dataclass
class ShardTracker:
    """Incremental reader of results/shard_XXX.jsonl (last line per unit wins)."""
    path: Path
    total: int
    offset: int = 0
    last: dict = field(default_factory=dict)     # (custom_id, repeat) -> has_error
    tokens: int = 0                              # completion tokens seen so far
    latency_sum: float = 0.0
    latency_n: int = 0
    samples: deque = field(default_factory=deque)  # (t, done, tokens)

    def refresh(self) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size < self.offset:  # rewritten: read again from the start
            self.offset, self.last, self.tokens = 0, {}, 0
            self.latency_sum, self.latency_n = 0.0, 0
        if size == self.offset:
            return
        with self.path.open("rb") as fh:
            fh.seek(self.offset)
            chunk = fh.read(size - self.offset)
        end = chunk.rfind(b"\n")
        if end < 0:  # no complete line yet
            return
        self.offset += end + 1
        for line in chunk[:end].splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (row.get("custom_id"), row.get("repeat"))
            self.last[key] = row.get("error") is not None
            resp = row.get("response") or {}
            ct = (resp.get("usage") or {}).get("completion_tokens")
            if isinstance(ct, int):
                self.tokens += ct
            lat = row.get("latency_s")
            if isinstance(lat, (int, float)):
                self.latency_sum += lat
                self.latency_n += 1

    @property
    def done(self) -> int:
        return len(self.last)

    @property
    def errors(self) -> int:
        return sum(self.last.values())

    def sample(self, now: float, window: float) -> None:
        self.samples.append((now, self.done, self.tokens))
        while len(self.samples) > 2 and now - self.samples[1][0] >= window:
            self.samples.popleft()

    def rate(self) -> Optional[tuple[float, float]]:
        """(units/s, completion tokens/s) over the window, None if not measurable."""
        if len(self.samples) < 2:
            return None
        t0, d0, k0 = self.samples[0]
        t1, d1, k1 = self.samples[-1]
        if t1 - t0 < 1:
            return None
        return (d1 - d0) / (t1 - t0), (k1 - k0) / (t1 - t0)


def _log_verdict(batch_dir: Path, shard: int, job_id: Optional[str]) -> Optional[str]:
    """Why a task that left the queue unfinished ended, from the tail of its log."""
    logs = batch_dir / "logs"
    pattern = f"shard_{shard}_{job_id}.out" if job_id else f"shard_{shard}_*.out"
    files = sorted(logs.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    try:
        with files[-1].open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 8192))
            tail = fh.read().decode(errors="replace")
    except OSError:
        return None
    if "DUE TO TIME LIMIT" in tail:
        return "TIMEOUT"
    if "CANCELLED" in tail:
        return "CANCELLED"
    if "out of memory" in tail.lower() or "OutOfMemory" in tail:
        return "OOM"
    if "Traceback" in tail or "batch error" in tail or "Error" in tail:
        return "FAILED"
    if "finished:" in tail:
        return "ENDED"
    return None


@dataclass
class BatchTracker:
    batch_dir: Path
    manifest: dict
    shards: list[ShardTracker]

    @classmethod
    def open(cls, batch_dir: Path) -> "BatchTracker":
        manifest = load_manifest(batch_dir)
        shards = []
        for i in range(manifest["shards"]):
            total = len(_read_jsonl(batch_dir / "shards" / _shard_name(i)))
            shards.append(ShardTracker(batch_dir / "results" / _shard_name(i), total))
        return cls(batch_dir, manifest, shards)

    @property
    def job_id(self) -> Optional[str]:
        j = self.manifest.get("job_id")
        return str(j) if j else None

    def reload_manifest(self) -> None:
        if self.job_id is None:  # submitted after the monitor started
            try:
                self.manifest = load_manifest(self.batch_dir)
            except (BatchError, json.JSONDecodeError):
                pass


def find_batch_dirs(paths: list[str]) -> list[Path]:
    """Batch dirs among `paths`; a path that is not one is searched one level down."""
    found: list[Path] = []
    for p in paths:
        p = Path(p).expanduser()
        if (p / "manifest.json").is_file():
            found.append(p)
        elif p.is_dir():
            subs = sorted(d for d in p.iterdir() if (d / "manifest.json").is_file())
            if not subs:
                raise BatchError(f"{p}: not a batch dir and no batch dir inside")
            found += subs
        else:
            raise BatchError(f"{p}: not found")
    seen, uniq = set(), []
    for d in found:
        r = d.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(d)
    return uniq


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
class _Style:
    def __init__(self, color: bool):
        self.color = color

    def __call__(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text


_STATE_COLOR = {"RUNNING": "32", "COMPLETED": "36", "PENDING": "33", "COMPLETING": "32",
                "FAILED": "31", "TIMEOUT": "31", "CANCELLED": "31", "OOM": "31",
                "STOPPED": "31", "ENDED": "35", "NOT SUBMITTED": "2"}


def _fmt_dur(s: Optional[float]) -> str:
    if s is None:
        return "-"
    s = int(round(s))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


def _bar(done: int, total: int, width: int) -> str:
    if width <= 0:
        return ""
    frac = done / total if total else 1.0
    full = int(frac * width)
    return "█" * full + "░" * (width - full)


def _shard_state(bt: BatchTracker, i: int, st: ShardTracker,
                 tasks: dict[int, TaskInfo], squeue_ok: bool) -> tuple[str, str, str]:
    """(state, node/reason, elapsed) of one shard."""
    if bt.job_id is None:
        return ("COMPLETED" if st.total and st.done >= st.total else "NOT SUBMITTED"), "", ""
    t = tasks.get(i)
    if t is not None:
        return t.state, t.node, t.elapsed
    if st.done >= st.total:
        return "COMPLETED", "", ""
    if not squeue_ok:
        return "?", "", ""
    return _log_verdict(bt.batch_dir, i, bt.job_id) or "STOPPED", "", ""


def render(trackers: list[BatchTracker], tasks: dict[str, dict[int, TaskInfo]], *,
           squeue_ok: bool, interval: Optional[float], color: bool,
           width: int = 120, show_shards: bool = True) -> tuple[str, bool]:
    """The screen as text, and whether every batch is finished (nothing left to wait)."""
    sty = _Style(color)
    lines = []
    head = f"Cerberus batch monitor — {datetime.now():%Y-%m-%d %H:%M:%S}"
    if interval:
        head += f" — refresh {interval:g}s, Ctrl-C to exit"
    lines.append(sty(head, "1"))
    if not squeue_ok:
        lines.append(sty("squeue not found: SLURM states unavailable", "33"))
    lines.append("")

    all_finished = True
    g_done = g_total = g_err = 0
    g_rate = g_tok = 0.0
    g_rate_ok = False
    bar_w = max(10, min(30, width - 92))

    for bt in trackers:
        m = bt.manifest
        jt = tasks.get(bt.job_id or "", {})
        models = ", ".join(m.get("models") or [])
        title = (f"{bt.batch_dir.name}  job {bt.job_id or '-'}  [{models}]  "
                 f"{m.get('n_requests')} req x {m.get('repeats')} = {m.get('n_units')} units")
        lines.append(sty(title, "1;34"))
        if show_shards:
            lines.append(sty(f"  {'shard':>5}  {'state':<11} {'node / reason':<22} "
                             f"{'elapsed':>9}  {'done/total':>13} {'':<{bar_w}} "
                             f"{'err':>5} {'/min':>7} {'tok/s':>7} {'ETA':>7}", "2"))
        b_done = b_total = b_err = 0
        b_rate = b_tok = 0.0
        b_rate_ok = False
        counts: dict[str, int] = {}
        for i, st in enumerate(bt.shards):
            state, node, elapsed = _shard_state(bt, i, st, jt, squeue_ok)
            counts[state] = counts.get(state, 0) + 1
            r = st.rate()
            t = jt.get(i)
            if r is None and t is not None and t.state == "RUNNING" and t.elapsed_s:
                r = (st.done / t.elapsed_s, st.tokens / t.elapsed_s)  # first refresh
            remaining = st.total - st.done
            eta = None
            if state in ("RUNNING", "COMPLETING") and r and r[0] > 0:
                eta = remaining / r[0]
            if r and state in ("RUNNING", "COMPLETING"):
                b_rate += r[0]; b_tok += r[1]; b_rate_ok = True
            b_done += st.done; b_total += st.total; b_err += st.errors
            if t is not None or state in ("?", "NOT SUBMITTED"):
                all_finished = False  # still queued, unknown, or may be run by hand
            if show_shards:
                pct = f"{100 * st.done / st.total:5.1f}%" if st.total else "  -  "
                rate_s = f"{r[0] * 60:7.1f}" if r and state == "RUNNING" else f"{'-':>7}"
                tok_s = f"{r[1]:7.0f}" if r and state == "RUNNING" else f"{'-':>7}"
                err_s = f"{st.errors:5d}"
                if st.errors:
                    err_s = sty(err_s, "31")
                lines.append(
                    f"  {i:5d}  {sty(f'{state:<11}', _STATE_COLOR.get(state, '0'))} "
                    f"{node[:22]:<22} {elapsed:>9}  {st.done:>6}/{st.total:<6} "
                    f"{_bar(st.done, st.total, bar_w)} {err_s} {rate_s} {tok_s} "
                    f"{_fmt_dur(eta):>7}  {pct}")
        b_eta = (b_total - b_done) / b_rate if b_rate_ok and b_rate > 0 else None
        summary = ", ".join(f"{n} {s.lower()}" for s, n in sorted(counts.items()))
        pct = 100 * b_done / b_total if b_total else 100.0
        lat = sum(s.latency_sum for s in bt.shards)
        lat_n = sum(s.latency_n for s in bt.shards)
        lat_s = f", lat {lat / lat_n:.1f}s" if lat_n else ""
        lines.append(
            f"  {sty('total', '1')}: {b_done}/{b_total} ({pct:.1f}%)  "
            f"{_bar(b_done, b_total, bar_w)}  err {b_err}  "
            + (f"{b_rate * 60:.1f}/min, {b_tok:.0f} tok/s, ETA {_fmt_dur(b_eta)}"
               if b_rate_ok else "rate -")
            + f"{lat_s}  [{summary}]")
        lines.append("")
        g_done += b_done; g_total += b_total; g_err += b_err
        if b_rate_ok:
            g_rate += b_rate; g_tok += b_tok; g_rate_ok = True

    if len(trackers) > 1:
        pct = 100 * g_done / g_total if g_total else 100.0
        lines.append(sty(f"ALL {len(trackers)} batches: {g_done}/{g_total} ({pct:.1f}%), "
                         f"err {g_err}"
                         + (f", {g_rate * 60:.1f}/min, {g_tok:.0f} tok/s" if g_rate_ok else ""),
                         "1"))
    if all_finished:
        lines.append(sty("Nothing running or pending.", "1;36"))
    return "\n".join(lines), all_finished


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def monitor(paths: list[str], *, interval: float = 10.0, once: bool = False,
            window: float = 300.0, show_shards: bool = True, keep: bool = False,
            color: Optional[bool] = None,
            squeue: Callable[[list[str]], dict] = squeue_tasks,
            out=sys.stdout) -> int:
    trackers = [BatchTracker.open(d) for d in find_batch_dirs(paths)]
    if color is None:
        color = out.isatty() and os.environ.get("NO_COLOR") is None
    squeue_ok = has_squeue() if squeue is squeue_tasks else True
    clear = "\033[H\033[2J" if (out.isatty() and not once) else ""
    try:
        while True:
            now = time.monotonic()
            for bt in trackers:
                bt.reload_manifest()
                for st in bt.shards:
                    st.refresh()
                    st.sample(now, window)
            tasks = squeue([bt.job_id for bt in trackers if bt.job_id])
            width = shutil.get_terminal_size((120, 40)).columns
            text, finished = render(trackers, tasks, squeue_ok=squeue_ok,
                                    interval=None if once else interval, color=color,
                                    width=width, show_shards=show_shards)
            out.write(clear + text + "\n")
            out.flush()
            if once or (finished and not keep):
                return 0
            time.sleep(interval)
    except KeyboardInterrupt:
        out.write("\n")
        return 0
