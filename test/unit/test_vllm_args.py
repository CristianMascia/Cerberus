"""Test di vllm_args: opzioni extra di `vllm serve` da models.conf, passate
intatte da config -> plan.tsv (deploy.plan_row) -> node_launch.sh."""

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from cerberus.config import ConfigError, load_config
from cerberus.deploy import plan_row
from cerberus.placement import Placed

HEAD = """
[allocation]
gpus_per_node = 2
vram_gib = 92
[defaults]
{defaults}
"""

VLLM = """
[[model]]
label = "q"
backend = "vllm"
hf_repo = "Qwen/Qwen3-8B"
alloc_mode = "MANUAL"
num_gpus = 1
max_input_tokens = 100
max_output_tokens = 100
{extra}
"""

LLAMA = """
[[model]]
label = "g"
hf_repo = "o/g-GGUF"
gguf_file = "g.gguf"
alloc_mode = "AUTO"
max_input_tokens = 100
max_output_tokens = 100
{extra}
"""

# spazi, virgolette, $, backslash, ; e un argomento vuoto o "-" devono arrivare intatti
TRICKY = ["--safetensors-load-strategy", "eager", "--x", "a b  c", "--y", "$HOME;'\"\\*",
          "", "-"]
TRICKY_TOML = "vllm_args = " + json.dumps(TRICKY)   # una stringa JSON è anche TOML

NODE_LAUNCH = Path(__file__).resolve().parents[2] / "cerberus" / "node_launch.sh"

# un argomento per riga, per vedere i confini tra gli argomenti
FAKE_VLLM = """#!/usr/bin/env bash
printf '%s\\n' "$@" > "$(dirname "$0")/../args.txt"
"""


def load(text):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "models.conf"
        p.write_text(text)
        return load_config(p)


def make_snapshot(hf: Path) -> str:
    """Snapshot completo di Qwen/Qwen3-8B nella cache finta; ritorna il path relativo."""
    repo = hf / "hub" / "models--Qwen--Qwen3-8B"
    snap = repo / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    for f in ("config.json", "tokenizer.json", "model.safetensors"):
        (snap / f).write_text("{}")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)
    return str(snap.relative_to(hf))


class TestConfig(unittest.TestCase):
    def test_default_empty(self):
        self.assertEqual(load(HEAD.format(defaults="") + VLLM.format(extra=""))
                         .models[0].vllm_args, [])

    def test_model_and_defaults(self):
        m = load(HEAD.format(defaults="") + VLLM.format(extra=TRICKY_TOML)).models[0]
        self.assertEqual(m.vllm_args, TRICKY)
        cfg = load(HEAD.format(defaults='vllm_args = ["--enforce-eager"]')
                   + VLLM.format(extra="")
                   + VLLM.format(extra='vllm_args = []').replace('"q"', '"q2"')
                   + LLAMA.format(extra=""))
        q, q2, g = cfg.models
        self.assertEqual(q.vllm_args, ["--enforce-eager"])   # da [defaults]
        self.assertEqual(q2.vllm_args, [])                   # il modello vince
        self.assertEqual(g.vllm_args, [])                    # [defaults] non vale per llamacpp

    def test_llamacpp_rejects(self):
        with self.assertRaises(ConfigError):
            load(HEAD.format(defaults="") + LLAMA.format(extra='vllm_args = ["--x"]'))

    def test_bad_values(self):
        for bad in ('vllm_args = "--enforce-eager"', 'vllm_args = ["--x", 1]',
                    'vllm_args = ["a\\tb"]', 'vllm_args = ["a\\nb"]'):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                load(HEAD.format(defaults="") + VLLM.format(extra=bad))


class TestPlanRow(unittest.TestCase):
    def test_column(self):
        with tempfile.TemporaryDirectory() as d:
            make_snapshot(Path(d))
            cfg = load(HEAD.format(defaults="") + VLLM.format(extra=TRICKY_TOML)
                       + VLLM.format(extra="").replace('"q"', '"q0"'))
            q, q0 = cfg.models
            row = plan_row(Placed(q, 0, [0], 1, 92.0), d)
            cols = row.split("\t")
            self.assertEqual(len(cols), 17)
            self.assertEqual(cols[16], "".join(a + "\x1f" for a in TRICKY))
            self.assertNotIn("\n", row)
            self.assertEqual(plan_row(Placed(q0, 0, [1], 1, 92.0), d).split("\t")[16], "-")


class TestNodeLaunch(unittest.TestCase):
    def run_launch(self, root: Path, row: str, env_col: bool):
        env = root / "vllm-env"
        (env / "bin").mkdir(parents=True)
        vllm = env / "bin" / "vllm"
        vllm.write_text(FAKE_VLLM)
        vllm.chmod(vllm.stat().st_mode | stat.S_IXUSR)
        plan = root / "plan.tsv"
        plan.write_text(row.replace("ENV", str(env)) + "\n")
        parts = root / "endpoints.parts"
        parts.write_text("")
        path = os.pathsep.join(p for p in ("/usr/bin", "/bin") if os.path.isdir(p))
        run_env = {"PATH": path, "HOME": str(root),
                   "CERBERUS_VLLM_ENV": "" if env_col else str(env),
                   "CERBERUS_VLLM_IMAGE": "", "HF_CACHE": str(root / "hf"),
                   "CERBERUS_LOGDIR": str(root / "logs"), "CUDA_VISIBLE_DEVICES": "0,1"}
        out = subprocess.run(["bash", str(NODE_LAUNCH), "0", str(plan), str(parts)],
                             env=run_env, capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        return (root / "vllm-env" / "args.txt").read_text().split("\n")[:-1]

    def test_args_verbatim_at_the_end(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            hf = root / "hf"
            make_snapshot(hf)
            cfg = load(HEAD.format(defaults="") + VLLM.format(extra=TRICKY_TOML
                                                              + '\nvllm_env = "ENV"'))
            row = plan_row(Placed(cfg.models[0], 0, [1], 1, 92.0), str(hf))
            args = self.run_launch(root, row, env_col=True)
            self.assertEqual(args[0], "serve")
            self.assertEqual(args[-len(TRICKY):], TRICKY)
            self.assertIn("--max-num-seqs", args[:-len(TRICKY)])

    def test_old_plan_without_column(self):
        # plan.tsv a 15 colonne (senza vllm_env né vllm_args): nessun argomento extra
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            snap = make_snapshot(root / "hf")
            row = "\t".join(["0", "q", f"/hf/{snap}", "CUDA0", "200", "64", "f16", "off",
                             "-", "-", "vllm", "1", "0.9", "-", "q"])
            args = self.run_launch(root, row, env_col=False)
            self.assertEqual(args[-2:], ["--max-num-seqs", "64"])


if __name__ == "__main__":
    unittest.main()
