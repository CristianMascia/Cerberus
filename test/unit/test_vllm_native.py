"""Test del vLLM nativo (CERBERUS_VLLM_ENV): scelta del runtime e lancio da
node_launch.sh con un `vllm` finto, senza Apptainer nel PATH."""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from types import SimpleNamespace

from cerberus.config import load_config
from cerberus.deploy import vllm_runtime

CONF = """
[allocation]
gpus_per_node = 2
vram_gib = 92
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

NODE_LAUNCH = Path(__file__).resolve().parents[2] / "cerberus" / "node_launch.sh"

FAKE_VLLM = """#!/usr/bin/env bash
{ echo "ARGS $*"; echo "CVD=$CUDA_VISIBLE_DEVICES"; echo "HF_HOME=$HF_HOME";
  echo "OFFLINE=$HF_HUB_OFFLINE"; echo "PP=${PYTHONPATH-unset}";
  echo "LDP=${LD_LIBRARY_PATH-unset}"; } > "$(dirname "$0")/../calls.txt"
"""


def make_env(root: Path) -> Path:
    env = root / "vllm-env"
    (env / "bin").mkdir(parents=True)
    vllm = env / "bin" / "vllm"
    vllm.write_text(FAKE_VLLM)
    vllm.chmod(vllm.stat().st_mode | stat.S_IXUSR)
    return env


class TestRuntime(unittest.TestCase):
    def test_env_wins_over_image(self):
        with tempfile.TemporaryDirectory() as d:
            env = make_env(Path(d))
            self.assertEqual(vllm_runtime({"vllm_env": str(env), "vllm_image": "x.sif"}),
                             f"env {env}")

    def test_image(self):
        self.assertEqual(vllm_runtime({"vllm_env": "", "vllm_image": "x.sif"}), "image x.sif")

    def test_env_without_vllm(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                vllm_runtime({"vllm_env": d, "vllm_image": "x.sif"})

    def test_model_env_wins(self):
        with tempfile.TemporaryDirectory() as d:
            env = make_env(Path(d))
            spec = SimpleNamespace(vllm_env=str(env))
            self.assertEqual(vllm_runtime({"vllm_env": "", "vllm_image": "x.sif"}, spec),
                             f"env {env}")
            self.assertEqual(vllm_runtime({"vllm_env": "", "vllm_image": "x.sif"},
                                          SimpleNamespace(vllm_env=None)), "image x.sif")

    def test_config_field(self):
        with tempfile.TemporaryDirectory() as d:
            conf = Path(d) / "models.conf"
            conf.write_text(CONF.format(extra='vllm_env = "~/miniforge3/envs/vllm-base"'))
            m = load_config(conf).models[0]
            self.assertEqual(m.vllm_env, os.path.expanduser("~/miniforge3/envs/vllm-base"))
            conf.write_text(CONF.format(extra=""))
            self.assertIsNone(load_config(conf).models[0].vllm_env)

    def test_neither(self):
        with self.assertRaises(RuntimeError):
            vllm_runtime({"vllm_env": "", "vllm_image": ""})


class TestNodeLaunch(unittest.TestCase):
    def test_native_launch_without_apptainer(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = make_env(root)
            hf = root / "hf"
            plan = root / "plan.tsv"
            snap = "hub/models--Qwen--Qwen3-8B/snapshots/" + "a" * 40
            plan.write_text("\t".join([
                "0", "q", f"/hf/{snap}", "CUDA1", "10000", "64", "fp8", "off", "-", "-",
                "vllm", "1", "0.9", "qwen3", "q#1"]) + "\n")
            parts = root / "endpoints.parts"
            parts.write_text("")
            # PATH without apptainer/singularity: only the basic tools
            path = os.pathsep.join(p for p in ("/usr/bin", "/bin") if os.path.isdir(p))
            run_env = {"PATH": path, "HOME": str(root), "CERBERUS_VLLM_ENV": str(env),
                       "CERBERUS_VLLM_IMAGE": "", "HF_CACHE": str(hf),
                       "CERBERUS_LOGDIR": str(root / "logs"),
                       "CUDA_VISIBLE_DEVICES": "2,3", "PYTHONPATH": "/bad",
                       "LD_LIBRARY_PATH": "/bad"}
            out = subprocess.run(["bash", str(NODE_LAUNCH), "0", str(plan), str(parts)],
                                 env=run_env, capture_output=True, text=True, timeout=30)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            calls = (root / "vllm-env" / "calls.txt").read_text().splitlines()
            args = calls[0].split()
            self.assertEqual(args[1:3], ["serve", f"{hf}/{snap}"])
            self.assertIn("--served-model-name", args)
            self.assertEqual(args[args.index("--kv-cache-dtype") + 1], "fp8")
            self.assertEqual(args[args.index("--reasoning-parser") + 1], "qwen3")
            self.assertIn("CVD=3", calls)                  # CUDA1 -> seconda GPU dello step
            self.assertIn(f"HF_HOME={hf}", calls)
            self.assertIn("OFFLINE=1", calls)
            self.assertIn("PP=unset", calls)
            self.assertIn("LDP=unset", calls)
            self.assertTrue(parts.read_text().startswith("q#1\t"))


    def test_native_launch_env_from_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = make_env(root)
            hf = root / "hf"
            plan = root / "plan.tsv"
            snap = "hub/models--Qwen--Qwen3-8B/snapshots/" + "a" * 40
            plan.write_text("\t".join([
                "0", "q", f"/hf/{snap}", "CUDA1", "10000", "64", "fp8", "off", "-", "-",
                "vllm", "1", "0.9", "qwen3", "q#1", str(env)]) + "\n")
            parts = root / "endpoints.parts"
            parts.write_text("")
            # PATH without apptainer/singularity: only the basic tools
            path = os.pathsep.join(p for p in ("/usr/bin", "/bin") if os.path.isdir(p))
            run_env = {"PATH": path, "HOME": str(root), "CERBERUS_VLLM_ENV": "",
                       "CERBERUS_VLLM_IMAGE": "", "HF_CACHE": str(hf),
                       "CERBERUS_LOGDIR": str(root / "logs"),
                       "CUDA_VISIBLE_DEVICES": "2,3", "PYTHONPATH": "/bad",
                       "LD_LIBRARY_PATH": "/bad"}
            out = subprocess.run(["bash", str(NODE_LAUNCH), "0", str(plan), str(parts)],
                                 env=run_env, capture_output=True, text=True, timeout=30)
            self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
            calls = (root / "vllm-env" / "calls.txt").read_text().splitlines()
            args = calls[0].split()
            self.assertEqual(args[1:3], ["serve", f"{hf}/{snap}"])
            self.assertIn("--served-model-name", args)
            self.assertEqual(args[args.index("--kv-cache-dtype") + 1], "fp8")
            self.assertEqual(args[args.index("--reasoning-parser") + 1], "qwen3")
            self.assertIn("CVD=3", calls)                  # CUDA1 -> seconda GPU dello step
            self.assertIn(f"HF_HOME={hf}", calls)
            self.assertIn("OFFLINE=1", calls)
            self.assertIn("PP=unset", calls)
            self.assertIn("LDP=unset", calls)
            self.assertTrue(parts.read_text().startswith("q#1\t"))


if __name__ == "__main__":
    unittest.main()
