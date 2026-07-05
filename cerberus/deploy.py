"""
Cerberus — deploy orchestrator (`cerberus up`).

Runs on the master (inside a SLURM allocation you created). It:
  1. resolves the allocated nodes,
  2. computes per-model GPU needs and packs them onto nodes+GPUs,
  3. writes a plan the per-node launchers read,
  4. launches one `srun` step per node (node_launch.sh) that starts the servers,
  5. waits for every server to report a port and become healthy,
  6. writes <project_dir>/endpoints.json,
  7. blocks (keeping the servers up) until Ctrl-C / signal, then tears down.

Only stdlib is used here (subprocess, urllib); no singularity is needed on the
master — that runs inside the srun steps on the compute nodes.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from .config import Config
from . import placement
from .download import resolve_local_gguf

_HERE = Path(__file__).resolve().parent
NODE_LAUNCH = _HERE / "node_launch.sh"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _settings() -> dict:
    home = os.path.expanduser("~")
    return {
        "sandbox": _env("CERBERUS_SANDBOX", f"{home}/tools/cuda-build"),
        "bin": _env("CERBERUS_BIN", f"{home}/tools/llama.cpp-master/build/bin"),
        "bind": _env("CERBERUS_BIND", f"{home}/tools"),
        "hf_cache": _env("HF_HOME", f"{home}/.cache/huggingface"),
        "threads": _env("CERBERUS_THREADS", "8"),
        "threads_http": _env("CERBERUS_THREADS_HTTP", "4"),
        "ngl": _env("CERBERUS_NGL", "999"),
        "port_min": _env("CERBERUS_PORT_MIN", "8081"),
        "port_max": _env("CERBERUS_PORT_MAX", "8999"),
        "srun_gres": _env("CERBERUS_SRUN_GRES", ""),  # e.g. "--gpus-per-node=3"
        "health_timeout": int(_env("CERBERUS_HEALTH_TIMEOUT", "600")),
    }


def slurm_nodes() -> list[str]:
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        raise RuntimeError(
            "not inside a SLURM allocation (SLURM_JOB_NODELIST unset); "
            "run 'cerberus up' from an salloc/sbatch job"
        )
    out = subprocess.run(["scontrol", "show", "hostnames", nodelist],
                         capture_output=True, text=True, check=True)
    return [h for h in out.stdout.split() if h]


def _container_path(host_path: Path, hf_cache: str) -> str:
    rel = os.path.relpath(host_path, hf_cache)
    return f"/hf/{rel}"


def _health_ok(host: str, port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as r:
            return b'"ok"' in r.read()
    except Exception:
        return False


def up(config: Config, project_dir: Path, token: str | None = None) -> Path:
    st = _settings()
    nodes = slurm_nodes()
    print(f"[cerberus] allocation: {len(nodes)} node(s): {' '.join(nodes)}")

    # 1-2. needs + placement
    print("[cerberus] computing VRAM needs and placement...")
    needs = placement.compute_needs(config, token=token)
    placements, n_used = placement.plan(needs, config.gpus_per_node, max_nodes=len(nodes))
    print(f"[cerberus] plan uses {n_used}/{len(nodes)} node(s):")
    for p in placements:
        print(f"  {p.spec.label}: node{p.node_idx} ({nodes[p.node_idx]}) "
              f"dev={p.device_arg} ctx={p.spec.ctx_size} (~{p.footprint_gib:.1f} GiB)")

    # 3. runtime dir + plan file (shared Lustre so all nodes read it)
    job = os.environ.get("SLURM_JOB_ID", "local")
    rundir = project_dir / ".cerberus" / job
    logdir = rundir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    plan_tsv = rundir / "plan.tsv"
    parts = rundir / "endpoints.parts"
    parts.write_text("")  # truncate
    with plan_tsv.open("w") as fh:
        for p in placements:
            host_gguf = resolve_local_gguf(p.spec, st["hf_cache"])
            if host_gguf is None:
                raise RuntimeError(
                    f"GGUF for '{p.spec.label}' not in cache — run 'cerberus download' first"
                )
            container = _container_path(host_gguf, st["hf_cache"])
            rb = p.spec.reasoning_budget if p.spec.reasoning_budget is not None else ""
            # columns: node, label, gguf_container, device, ctx, parallel, kv, reasoning, reasoning_budget
            fh.write("\t".join(str(x) for x in [
                p.node_idx, p.spec.label, container, p.device_arg, p.spec.ctx_size,
                p.spec.parallel, p.spec.kv_cache_type, p.spec.reasoning, rb,
            ]) + "\n")

    # export settings for node_launch (srun propagates the environment)
    child_env = dict(os.environ)
    child_env.update({
        "CERBERUS_SANDBOX": st["sandbox"], "CERBERUS_BIN": st["bin"],
        "CERBERUS_BIND": st["bind"], "HF_CACHE": st["hf_cache"],
        "CERBERUS_NGL": st["ngl"], "CERBERUS_THREADS": st["threads"],
        "CERBERUS_THREADS_HTTP": st["threads_http"],
        "CERBERUS_PORT_MIN": st["port_min"], "CERBERUS_PORT_MAX": st["port_max"],
        "CERBERUS_LOGDIR": str(logdir),
    })

    # 4. one srun step per used node
    used_nodes = sorted({p.node_idx for p in placements})
    procs = []
    for idx in used_nodes:
        host = nodes[idx]
        cmd = ["srun", "--nodes=1", "--ntasks=1", f"--nodelist={host}", "--overlap"]
        if st["srun_gres"]:
            cmd += st["srun_gres"].split()
        else:
            cmd.append(f"--gpus-per-node={config.gpus_per_node}")
        cmd += ["bash", str(NODE_LAUNCH), str(idx), str(plan_tsv), str(parts)]
        print(f"[cerberus] launching node {idx} ({host})")
        srun_log = open(logdir / f"node_{idx}_srun.log", "w")
        procs.append(subprocess.Popen(cmd, env=child_env, stdout=srun_log, stderr=srun_log))

    def cleanup(*_):
        print("\n[cerberus] tearing down (stopping srun steps)...")
        for pr in procs:
            pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=15)
            except Exception:
                pr.kill()
        print("[cerberus] done.")

    signal.signal(signal.SIGINT, lambda *a: (cleanup(), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *a: (cleanup(), sys.exit(0)))

    # 5. wait for every model to report host:port and become healthy
    try:
        endpoints = _await_ready(placements, nodes, parts, st["health_timeout"], logdir)
    except Exception:
        cleanup()
        raise

    # 6. write endpoints.json into the project dir
    out = {
        "job_id": job,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": endpoints,
    }
    ep_path = project_dir / "endpoints.json"
    ep_path.write_text(json.dumps(out, indent=2))
    print(f"[cerberus] ready — {len(endpoints)} model(s). endpoints: {ep_path}")
    for label, m in endpoints.items():
        print(f"  {label}: {m['base_url']}  (reasoning={m['reasoning']})")

    # 7. block, keeping servers up, until interrupted
    print("[cerberus] serving. Press Ctrl-C to stop.")
    try:
        while all(pr.poll() is None for pr in procs):
            time.sleep(5)
        print("[cerberus] a node step exited; tearing down.")
    finally:
        cleanup()
    return ep_path


def _await_ready(placements, nodes, parts: Path, timeout: int, logdir: Path) -> dict:
    """Poll endpoints.parts until every model reports a port and passes /health."""
    want = {p.spec.label for p in placements}
    reasoning_of = {p.spec.label: p.spec.reasoning for p in placements}
    reported: dict[str, tuple[str, int]] = {}
    waited = 0
    while True:
        for line in parts.read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 3 and f[0] not in reported:
                reported[f[0]] = (f[1], int(f[2]))
        healthy = {lbl: hp for lbl, hp in reported.items() if _health_ok(*hp)}
        if want <= set(healthy):
            return {
                lbl: {
                    "base_url": f"http://{healthy[lbl][0]}:{healthy[lbl][1]}/v1",
                    "host": healthy[lbl][0], "port": healthy[lbl][1],
                    "reasoning": reasoning_of[lbl] != "off",
                }
                for lbl in want
            }
        time.sleep(3)
        waited += 3
        if waited >= timeout:
            missing = want - set(healthy)
            for lbl in missing:
                log = logdir / f"{lbl}.log"
                if log.exists():
                    print(f"--- last log for {lbl} ---", file=sys.stderr)
                    print("\n".join(log.read_text().splitlines()[-15:]), file=sys.stderr)
            raise RuntimeError(f"timeout: models not healthy: {sorted(missing)}")
