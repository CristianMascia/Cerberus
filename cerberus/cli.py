"""
Cerberus — command-line interface.

    cerberus download [-c models.conf]        # fetch the GGUFs into the HF cache
    cerberus validate [-c models.conf]        # check schema + report GPU/node needs
    cerberus up       [-c models.conf] [--project-dir DIR]   # deploy + write endpoints.json
    cerberus status   [--project-dir DIR]     # health of the running models
    cerberus down     [--project-dir DIR]     # remove the endpoint map (stop with Ctrl-C on `up`)

Run as `python -m cerberus <cmd>` (or via the `cerberus` console wrapper).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from .config import ConfigError, load_config


def _load(path):
    try:
        return load_config(path)
    except ConfigError as exc:
        sys.exit(f"[cerberus] config error: {exc}")


def cmd_download(args):
    from .download import download_models
    cfg = _load(args.config)
    download_models(cfg)


def cmd_validate(args):
    from . import placement
    cfg = _load(args.config)
    print(f"[cerberus] {cfg.path}: OK — gpus_per_node={cfg.gpus_per_node}, "
          f"{len(cfg.models)} model(s)")
    if args.schema_only:
        return
    print("[cerberus] computing GPU/VRAM needs (reads GGUF headers over HTTP)...")
    needs = placement.compute_needs(cfg)
    placements, n_nodes = placement.plan(needs, cfg.gpus_per_node)
    for p in placements:
        print(f"  {p.spec.label}: {p.spec.alloc_mode} num_gpus={p.num_gpus} "
              f"~{p.footprint_gib:.1f} GiB  ctx={p.spec.ctx_size}")
    total_gpus = sum(n.num_gpus for n in needs)
    print(f"[cerberus] needs {total_gpus} GPU(s) -> {n_nodes} node(s) "
          f"at {cfg.gpus_per_node} GPU/node.")
    print(f"[cerberus] allocate e.g.: salloc --nodes={n_nodes} --ntasks-per-node=1 "
          f"--gpus-per-node={cfg.gpus_per_node} ...")


def cmd_up(args):
    from .deploy import up
    cfg = _load(args.config)
    project_dir = Path(args.project_dir) if args.project_dir else cfg.path.parent
    up(cfg, project_dir)


def _endpoints_path(project_dir) -> Path:
    import os
    env = os.environ.get("CERBERUS_ENDPOINTS")
    if env:
        return Path(env)
    base = Path(project_dir) if project_dir else Path.cwd()
    return base / "endpoints.json"


def cmd_status(args):
    ep = _endpoints_path(args.project_dir)
    if not ep.is_file():
        sys.exit(f"[cerberus] no endpoint map at {ep}")
    data = json.loads(ep.read_text())
    print(f"[cerberus] job {data.get('job_id')} — {ep}")
    for label, m in data["models"].items():
        ok = False
        try:
            with urllib.request.urlopen(f"http://{m['host']}:{m['port']}/health", timeout=3) as r:
                ok = b'"ok"' in r.read()
        except Exception:
            ok = False
        print(f"  {label:20s} {m['base_url']:40s} {'UP' if ok else 'DOWN'}")


def cmd_down(args):
    ep = _endpoints_path(args.project_dir)
    if ep.is_file():
        ep.unlink()
        print(f"[cerberus] removed {ep}")
    print("[cerberus] to stop the servers, Ctrl-C the running 'cerberus up' "
          "or 'scancel' the SLURM job.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cerberus", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_conf(sp):
        sp.add_argument("-c", "--config", default="models.conf", help="path to models.conf")

    sp = sub.add_parser("download", help="download the GGUFs into the HF cache")
    add_conf(sp); sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("validate", help="validate config and report GPU/node needs")
    add_conf(sp)
    sp.add_argument("--schema-only", action="store_true", help="skip VRAM estimation")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("up", help="deploy the models and write endpoints.json")
    add_conf(sp)
    sp.add_argument("--project-dir", help="where endpoints.json is written (default: config dir)")
    sp.set_defaults(func=cmd_up)

    sp = sub.add_parser("status", help="health of the running models")
    sp.add_argument("--project-dir", help="dir containing endpoints.json")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("down", help="remove the endpoint map")
    sp.add_argument("--project-dir", help="dir containing endpoints.json")
    sp.set_defaults(func=cmd_down)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
