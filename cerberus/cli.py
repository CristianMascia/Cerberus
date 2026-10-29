"""
Cerberus — command-line interface.

    cerberus download [-c models.conf]        # fetch the GGUFs into the HF cache
    cerberus validate [-c models.conf]        # check schema + report GPU/node needs
    cerberus up       [-c models.conf] [--project-dir DIR]   # deploy + write endpoints.json
    cerberus status   [--project-dir DIR]     # health of the running models
    cerberus down     [--project-dir DIR]     # remove the endpoint map (stop with Ctrl-C on `up`)

    cerberus batch validate -i in.jsonl [--model L]            # check a batch input
    cerberus batch submit   -i in.jsonl [--shards N] [--repeats R] [--dry-run]
    cerberus batch run      --batch-dir D [--shard I] [--endpoints endpoints.json]
    cerberus batch status   --batch-dir D
    cerberus batch merge    --batch-dir D

Run as `python -m cerberus <cmd>` (or via the `cerberus` console wrapper).
"""

from __future__ import annotations

import argparse
import json
import sys
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
          f"vram_gib={cfg.vram_gib:g}, "
          f"{len(cfg.models)} model(s)")
    if args.schema_only:
        return
    print("[cerberus] computing GPU/VRAM needs (reads model metadata over HTTP)...")
    needs = placement.compute_needs(cfg)
    placements, n_nodes = placement.plan(needs, cfg.gpus_per_node, cfg.vram_gib)
    for p in placements:
        print(f"  {p.instance}: {p.spec.alloc_mode} num_gpus={p.num_gpus} "
              f"~{p.footprint_gib:.1f} GiB  ctx={p.spec.ctx_size}")
    total_gpus = sum(n.num_gpus * n.spec.replicas for n in needs)
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
    from .deploy import _health_ok
    for label, m in data["models"].items():
        for i, r in enumerate(m.get("replicas") or [m]):
            ok = _health_ok(r["host"], r["port"])
            name = label if i == 0 else ""
            print(f"  {name:20s} {r['base_url']:40s} {'UP' if ok else 'DOWN'}")


def cmd_down(args):
    ep = _endpoints_path(args.project_dir)
    if ep.is_file():
        ep.unlink()
        print(f"[cerberus] removed {ep}")
    print("[cerberus] to stop the servers, Ctrl-C the running 'cerberus up' "
          "or 'scancel' the SLURM job.")


# --------------------------------------------------------------------------- #
# cerberus batch ...
# --------------------------------------------------------------------------- #
def _batch_fail(exc):
    sys.exit(f"[cerberus] batch error: {exc}")


def cmd_batch_validate(args):
    from . import batch
    cfg = _load(args.config)
    try:
        reqs = batch.load_jsonl(args.input, default_model=args.model)
        warnings = batch.check_against_config(reqs, cfg)
    except batch.BatchError as exc:
        _batch_fail(exc)
    for w in warnings:
        print(f"[cerberus] warning: {w}")
    per_model = {}
    for r in reqs:
        per_model[r.model] = per_model.get(r.model, 0) + 1
    print(f"[cerberus] {args.input}: OK — {len(reqs)} request(s)")
    for label, n in per_model.items():
        print(f"  {label}: {n}")
    if args.schema_only:
        return
    from . import placement
    sub = batch.subset_config(cfg, per_model)
    print("[cerberus] computing GPU/VRAM needs (reads model metadata over HTTP)...")
    needs = placement.compute_needs(sub)
    _, n_nodes = placement.plan(needs, sub.gpus_per_node, sub.vram_gib)
    print(f"[cerberus] each shard needs {n_nodes} node(s) x {sub.gpus_per_node} GPU(s)")


def cmd_batch_submit(args):
    from . import batch
    cfg = _load(args.config)
    batch_dir = Path(args.batch_dir) if args.batch_dir else batch.default_batch_dir(cfg)
    try:
        manifest = batch.prepare(cfg, args.input, batch_dir.resolve(), model=args.model,
                                 shards=args.shards, repeats=args.repeats)
        sbatch = batch.write_sbatch(batch_dir.resolve(), manifest)
        print(f"[cerberus] wrote {sbatch}")
        if args.dry_run:
            print("[cerberus] --dry-run: not submitted. Submit with: "
                  f"sbatch {sbatch}  (or run shards with 'cerberus batch run')")
            return
        job = batch.submit(batch_dir.resolve(), manifest)
    except batch.BatchError as exc:
        _batch_fail(exc)
    print(f"[cerberus] submitted job array {job} ({manifest['shards']} task(s)).")
    print(f"[cerberus] follow it with: cerberus batch status --batch-dir {batch_dir}")


def cmd_batch_run(args):
    import os
    from . import batch
    shard = args.shard
    if shard is None:
        env = os.environ.get("SLURM_ARRAY_TASK_ID")
        shard = int(env) if env is not None else 0
    try:
        n_err = batch.run_shard(args.batch_dir, shard, endpoints=args.endpoints,
                                retry_failed=args.retry_failed)
    except batch.BatchError as exc:
        _batch_fail(exc)
    if n_err:
        print(f"[cerberus] {n_err} request(s) failed; rerun with --retry-failed to retry them")


def cmd_batch_status(args):
    from . import batch
    bd = Path(args.batch_dir)
    try:
        manifest = batch.load_manifest(bd)
    except batch.BatchError as exc:
        _batch_fail(exc)
    job = manifest.get("job_id")
    states = batch.slurm_state(job) if job else {}
    print(f"[cerberus] batch {bd} — job {job or '-'}, {manifest['n_units']} unit(s)")
    tot = done = err = 0
    for row in batch.shard_progress(bd, manifest):
        tot += row["total"]; done += row["done"]; err += row["errors"]
        st = states.get(row["shard"], "")
        print(f"  shard {row['shard']:3d}: {row['done']:6d}/{row['total']:<6d} "
              f"errors {row['errors']:<5d} {st}")
    print(f"[cerberus] total: {done}/{tot} done, {err} error(s)")


def cmd_batch_merge(args):
    from . import batch
    try:
        out, missing = batch.merge(args.batch_dir, args.output)
    except batch.BatchError as exc:
        _batch_fail(exc)
    print(f"[cerberus] wrote {out}")
    if missing:
        print(f"[cerberus] warning: {len(missing)} unit(s) missing, e.g. {missing[:5]}")
        sys.exit(1)


def _add_batch_parser(sub, add_conf):
    bp = sub.add_parser("batch", help="run a JSONL batch of requests (see docs/batch.md)")
    bsub = bp.add_subparsers(dest="batch_cmd", required=True)

    sp = bsub.add_parser("validate", help="check a batch input against models.conf")
    add_conf(sp)
    sp.add_argument("-i", "--input", required=True, help="requests JSONL")
    sp.add_argument("--model", help="model label for lines without 'model'")
    sp.add_argument("--schema-only", action="store_true", help="skip VRAM estimation")
    sp.set_defaults(func=cmd_batch_validate)

    sp = bsub.add_parser("submit", help="split the batch and submit a SLURM job array")
    add_conf(sp)
    sp.add_argument("-i", "--input", required=True, help="requests JSONL")
    sp.add_argument("--model", help="model label for lines without 'model'")
    sp.add_argument("--shards", type=int, help="parallel jobs (default: [batch] shards)")
    sp.add_argument("--repeats", type=int, help="repetitions (default: [batch] repeats)")
    sp.add_argument("--batch-dir", help="output dir (default: batch_<timestamp> next to the config)")
    sp.add_argument("--dry-run", action="store_true", help="prepare everything, do not sbatch")
    sp.set_defaults(func=cmd_batch_submit)

    sp = bsub.add_parser("run", help="process one shard (inside the allocation)")
    sp.add_argument("--batch-dir", required=True)
    sp.add_argument("--shard", type=int, help="shard index (default: SLURM_ARRAY_TASK_ID)")
    sp.add_argument("--endpoints", help="use already running servers from this endpoints.json")
    sp.add_argument("--retry-failed", action="store_true", help="redo units that ended with an error")
    sp.set_defaults(func=cmd_batch_run)

    sp = bsub.add_parser("status", help="progress per shard")
    sp.add_argument("--batch-dir", required=True)
    sp.set_defaults(func=cmd_batch_status)

    sp = bsub.add_parser("merge", help="join the shard results into results.jsonl")
    sp.add_argument("--batch-dir", required=True)
    sp.add_argument("-o", "--output", help="output file (default: <batch-dir>/results.jsonl)")
    sp.set_defaults(func=cmd_batch_merge)


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

    _add_batch_parser(sub, add_conf)
    return p


def main(argv=None):
    # line-buffered even when redirected (cerberus up > up.log &), so logs show progress
    sys.stdout.reconfigure(line_buffering=True)
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
