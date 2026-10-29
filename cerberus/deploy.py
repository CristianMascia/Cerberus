"""
Cerberus — deploy orchestrator (`cerberus up`, and the servers of `cerberus batch run`).

Runs on the master (inside a SLURM allocation you created). It:
  1. resolves the allocated nodes,
  2. computes per-model GPU needs and packs them onto nodes+GPUs,
  3. writes a plan the per-node launchers read,
  4. launches one `srun` step per node (node_launch.sh) that starts the servers,
  5. waits for every server to report a port and become healthy,
  6. writes <project_dir>/endpoints.json,
  7. blocks (keeping the servers up) until Ctrl-C / signal, then tears down.

Steps 1-5 are `start()` and the teardown is `stop()`, so `cerberus batch run`
reuses them without the blocking part.

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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import Config
from . import placement
from .download import resolve_local_gguf
from .vllm_backend import resolve_local_snapshot

_HERE = Path(__file__).resolve().parent
NODE_LAUNCH = _HERE / "node_launch.sh"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _settings() -> dict:
    home = os.path.expanduser("~")
    return {
        "image": _env("CERBERUS_IMAGE", ""),  # .sif with llama-server inside (wins)
        "vllm_image": _env("CERBERUS_VLLM_IMAGE", ""),  # .sif with `vllm serve` inside
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
        # vLLM compiles (torch.compile) and captures CUDA graphs at first start
        "health_timeout_vllm": int(_env("CERBERUS_HEALTH_TIMEOUT_VLLM", "1500")),
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


def plan_row(p: placement.Placed, hf_cache: str) -> str:
    """One plan.tsv line for node_launch.sh.

    columns: node, label, model_container, device, ctx, parallel, kv, reasoning,
             reasoning_format, reasoning_budget, backend, num_gpus,
             gpu_memory_utilization, reasoning_parser
    Empty values are written as "-": `read` with a tab IFS merges empty fields.
    """
    s = p.spec
    if s.is_vllm:
        host_path = resolve_local_snapshot(s, hf_cache)
        what = "snapshot"
    else:
        host_path = resolve_local_gguf(s, hf_cache)
        what = "GGUF"
    if host_path is None:
        raise RuntimeError(f"{what} for '{s.label}' not in cache — run 'cerberus download' first")
    cols = [
        p.node_idx, s.label, _container_path(host_path, hf_cache), p.device_arg,
        s.ctx_size, s.parallel, s.kv_cache_type, s.reasoning, s.reasoning_format,
        s.reasoning_budget, s.backend, p.num_gpus, s.gpu_memory_utilization,
        s.reasoning_parser,
    ]
    return "\t".join("-" if c is None or c == "" else str(c) for c in cols)


def _health_ok(host: str, port: int) -> bool:
    # llama.cpp: 200 {"status":"ok"} (503 while loading); vLLM: 200 with an empty body.
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3) as r:
            body = r.read().strip()
            return r.status == 200 and (not body or b'"ok"' in body)
    except Exception:
        return False


@dataclass
class Deployment:
    """Servers started by `start()`; pass it to `stop()` to tear them down."""
    job: str
    rundir: Path
    endpoints: dict
    procs: list = field(default_factory=list)

    def alive(self) -> bool:
        return all(pr.poll() is None for pr in self.procs)


def start(config: Config, project_dir: Path, token: str | None = None,
          needs: list[placement.ModelNeed] | None = None) -> Deployment:
    """Launch every model of `config` on the current SLURM allocation.

    Returns once all servers are healthy. `needs` lets the caller pass a placement
    input computed elsewhere (e.g. on the login node by `cerberus batch submit`),
    so the compute node does not re-read the GGUF headers over HTTP.
    """
    st = _settings()
    nodes = slurm_nodes()
    print(f"[cerberus] allocation: {len(nodes)} node(s): {' '.join(nodes)}")

    # 1-2. needs + placement
    if needs is None:
        print("[cerberus] computing VRAM needs and placement...")
        needs = placement.compute_needs(config, token=token)
    placements, n_used = placement.plan(needs, config.gpus_per_node, config.vram_gib,
                                         max_nodes=len(nodes))
    print(f"[cerberus] plan uses {n_used}/{len(nodes)} node(s):")
    for p in placements:
        print(f"  {p.spec.label}: node{p.node_idx} ({nodes[p.node_idx]}) "
              f"dev={p.device_arg} ctx={p.spec.ctx_size} (~{p.footprint_gib:.1f} GiB)")

    # 3. runtime dir + plan file (shared Lustre so all nodes read it).
    # In a job array SLURM_JOB_ID differs per task, so the dirs do not collide.
    job = os.environ.get("SLURM_JOB_ID", "local")
    rundir = project_dir / ".cerberus" / job
    logdir = rundir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    plan_tsv = rundir / "plan.tsv"
    parts = rundir / "endpoints.parts"
    parts.write_text("")  # truncate
    with plan_tsv.open("w") as fh:
        for p in placements:
            fh.write(plan_row(p, st["hf_cache"]) + "\n")
    if any(p.spec.is_vllm for p in placements) and not st["vllm_image"]:
        raise RuntimeError("vllm models need CERBERUS_VLLM_IMAGE (the vLLM .sif); "
                           "see docs/hpc_unina_h100.md")

    # export settings for node_launch (srun propagates the environment)
    child_env = dict(os.environ)
    child_env.update({
        "CERBERUS_IMAGE": st["image"], "CERBERUS_VLLM_IMAGE": st["vllm_image"],
        "CERBERUS_SANDBOX": st["sandbox"], "CERBERUS_BIN": st["bin"],
        "CERBERUS_BIND": st["bind"], "HF_CACHE": st["hf_cache"],
        "CERBERUS_NGL": st["ngl"], "CERBERUS_THREADS": st["threads"],
        "CERBERUS_THREADS_HTTP": st["threads_http"],
        "CERBERUS_PORT_MIN": st["port_min"], "CERBERUS_PORT_MAX": st["port_max"],
        "CERBERUS_LOGDIR": str(logdir),
    })

    # 4. one srun step per used node
    used_nodes = sorted({p.node_idx for p in placements})
    dep = Deployment(job=job, rundir=rundir, endpoints={})
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
        dep.procs.append(subprocess.Popen(cmd, env=child_env, stdout=srun_log, stderr=srun_log))

    # 5. wait for every model to report host:port and become healthy
    timeout = st["health_timeout"]
    if any(p.spec.is_vllm for p in placements):
        timeout = max(timeout, st["health_timeout_vllm"])
    try:
        dep.endpoints = _await_ready(placements, nodes, parts, timeout, logdir)
    except BaseException:
        stop(dep)
        raise
    return dep


def stop(dep: Deployment) -> None:
    """Stop the srun steps (and thus the servers) of a deployment."""
    print("\n[cerberus] tearing down (stopping srun steps)...")
    for pr in dep.procs:
        if pr.poll() is None:
            pr.terminate()
    for pr in dep.procs:
        try:
            pr.wait(timeout=15)
        except Exception:
            pr.kill()
    print("[cerberus] done.")


def write_endpoints(dep: Deployment, path: Path) -> Path:
    out = {
        "job_id": dep.job,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": dep.endpoints,
    }
    path.write_text(json.dumps(out, indent=2))
    return path


def up(config: Config, project_dir: Path, token: str | None = None) -> Path:
    dep = start(config, project_dir, token=token)

    signal.signal(signal.SIGINT, lambda *a: (stop(dep), sys.exit(0)))
    signal.signal(signal.SIGTERM, lambda *a: (stop(dep), sys.exit(0)))

    # 6. write endpoints.json into the project dir
    ep_path = write_endpoints(dep, project_dir / "endpoints.json")
    print(f"[cerberus] ready — {len(dep.endpoints)} model(s). endpoints: {ep_path}")
    for label, m in dep.endpoints.items():
        print(f"  {label}: {m['base_url']}  (reasoning={m['reasoning']})")

    # 7. block, keeping servers up, until interrupted
    print("[cerberus] serving. Press Ctrl-C to stop.")
    try:
        while dep.alive():
            time.sleep(5)
        print("[cerberus] a node step exited; tearing down.")
    finally:
        stop(dep)
    return ep_path


def _await_ready(placements, nodes, parts: Path, timeout: int, logdir: Path) -> dict:
    """Poll endpoints.parts until every model reports a port and passes /health."""
    want = {p.spec.label for p in placements}
    reasoning_of = {p.spec.label: p.spec.reasoning for p in placements}
    backend_of = {p.spec.label: p.spec.backend for p in placements}
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
                    "backend": backend_of[lbl],
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
