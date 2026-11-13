"""Uscita immediata quando un server muore durante l'avvio: node_launch.sh scrive
endpoints.dead, deploy._await_ready lo legge e fallisce subito con nodo, GPU e
riga di errore, senza aspettare l'health timeout."""

import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from cerberus import deploy

NODE_LAUNCH = Path(__file__).resolve().parents[2] / "cerberus" / "node_launch.sh"

VLLM_LOG = """\
INFO 11-13 10:00:00 [api_server.py:1000] vLLM API server version 0.32.0
(EngineCore_DP0 pid=4242) ERROR 11-13 10:00:31 [core.py:708] EngineCore failed to start.
(EngineCore_DP0 pid=4242) ERROR 11-13 10:00:31 [core.py:708] Traceback (most recent call last):
(EngineCore_DP0 pid=4242) ERROR 11-13 10:00:31 [core.py:708]   File "worker.py", line 12, in init_device
(EngineCore_DP0 pid=4242) ERROR 11-13 10:00:31 [core.py:708] RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu.
Traceback (most recent call last):
  File "/env/bin/vllm", line 8, in <module>
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
"""

LLAMA_LOG = """\
ggml_cuda_init: found 1 CUDA devices:
llama_model_load: error loading model: tensor 'blk.0.attn_q.weight' data is not within the file bounds
llama_model_load_from_file_impl: failed to load model
main: exiting due to model loading error
"""

FAKE_VLLM = """#!/usr/bin/env bash
echo "RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu."
exit 1
"""


def placed(inst, node_idx=0, label="q"):
    return SimpleNamespace(instance=inst, node_idx=node_idx,
                           spec=SimpleNamespace(label=label, reasoning="off", backend="vllm"))


class TestErrorLine(unittest.TestCase):
    def test_vllm_root_cause(self):
        self.assertEqual(deploy.error_line(VLLM_LOG.splitlines()),
                         "RuntimeError: CUDA driver initialization failed, "
                         "you might not have a CUDA gpu.")

    def test_vllm_only_generic(self):
        lines = ["RuntimeError: Engine core initialization failed. See root cause above."]
        self.assertIn("Engine core", deploy.error_line(lines))

    def test_llamacpp(self):
        # nessuna eccezione Python: l'ultima riga che sembra un errore
        self.assertEqual(deploy.error_line(LLAMA_LOG.splitlines()),
                         "main: exiting due to model loading error")

    def test_fallback(self):
        self.assertEqual(deploy.error_line(["", "Killed"]), "Killed")
        self.assertEqual(deploy.error_line(["hello", ""]), "hello")
        self.assertEqual(deploy.error_line([]), "(empty log)")


class TestAwaitReady(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = Path(self.d.name)
        self.logdir = self.root / "logs"
        self.logdir.mkdir()
        self.parts = self.root / "endpoints.parts"
        self.parts.write_text("")

    def tearDown(self):
        self.d.cleanup()

    def test_dead_replica_fails_fast(self):
        (self.logdir / "q#1.log").write_text(VLLM_LOG)
        deploy.dead_file(self.parts).write_text("q#1\tgpu03\t1\t1\tvllm\n")
        t0 = time.monotonic()
        with self.assertRaises(RuntimeError) as cm:
            deploy._await_ready([placed("q#0"), placed("q#1")], ["gpu03"], self.parts,
                                3600, self.logdir)
        self.assertLess(time.monotonic() - t0, 2)
        msg = str(cm.exception)
        self.assertIn("q#1 on gpu03 GPU 1 (exit 1)", msg)
        self.assertIn("CUDA driver initialization failed", msg)

    def test_dead_other_job_ignored(self):
        # un'istanza che non è nel piano non conta: si arriva al timeout
        deploy.dead_file(self.parts).write_text("other\tgpu03\t0\t1\tvllm\n")
        with self.assertRaises(RuntimeError) as cm:
            deploy._await_ready([placed("q")], ["gpu03"], self.parts, 1, self.logdir)
        self.assertIn("timeout", str(cm.exception))

    def test_step_exited(self):
        pr = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"])
        pr.wait()
        (self.logdir / "node_0_srun.log").write_text("srun: error: Unable to create step\n")
        with self.assertRaises(RuntimeError) as cm:
            deploy._await_ready([placed("q")], ["gpu03"], self.parts, 3600, self.logdir,
                                {0: pr})
        self.assertIn("srun step exited with code 1", str(cm.exception))


class TestNodeLaunchReportsDeath(unittest.TestCase):
    def test_crash_written_to_dead(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = root / "vllm-env"
            (env / "bin").mkdir(parents=True)
            vllm = env / "bin" / "vllm"
            vllm.write_text(FAKE_VLLM)
            vllm.chmod(vllm.stat().st_mode | stat.S_IXUSR)
            plan = root / "plan.tsv"
            snap = "hub/models--Qwen--Qwen3-8B/snapshots/" + "a" * 40
            plan.write_text("\t".join([
                "0", "q", f"/hf/{snap}", "CUDA1", "10000", "64", "auto", "off", "-", "-",
                "vllm", "1", "0.9", "-", "q#1", str(env)]) + "\n")
            parts = root / "endpoints.parts"
            parts.write_text("")
            path = os.pathsep.join(p for p in ("/usr/bin", "/bin") if os.path.isdir(p))
            run_env = {"PATH": path, "HOME": str(root), "CERBERUS_VLLM_ENV": "",
                       "CERBERUS_VLLM_IMAGE": "", "HF_CACHE": str(root / "hf"),
                       "CERBERUS_LOGDIR": str(root / "logs"), "CUDA_VISIBLE_DEVICES": "0,1"}
            out = subprocess.run(["bash", str(NODE_LAUNCH), "0", str(plan), str(parts)],
                                 env=run_env, capture_output=True, text=True, timeout=30)
            self.assertEqual(out.returncode, 1, out.stdout + out.stderr)
            host = (root / "endpoints.dead").read_text().split("\t")[1]
            self.assertEqual((root / "endpoints.dead").read_text(),
                             f"q#1\t{host}\t1\t1\tvllm\n")
            self.assertIn("exited with code 1", out.stdout)


if __name__ == "__main__":
    unittest.main()
