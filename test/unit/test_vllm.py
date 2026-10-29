"""Test unitari del backend vLLM: config, stima GPU, piano, health e client."""

import http.server
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from cerberus.client import CerberusClient, _reasoning_field
from cerberus.config import ConfigError, ModelSpec, load_config
from cerberus.deploy import _health_ok, plan_row
from cerberus.placement import ModelNeed, Placed, plan
from cerberus.vllm_backend import (kv_dtype_bytes, kv_geometry, required_gpus_for,
                                   resolve_local_snapshot)

HEAD = """
[allocation]
gpus_per_node = 2
vram_gib = 92
[defaults]
parallel = 2
kv_cache_type = "q8_0"
"""

LLAMA = """
[[model]]
label = "g"
hf_repo = "o/g-GGUF"
gguf_file = "g.gguf"
alloc_mode = "AUTO"
max_input_tokens = 1000
max_output_tokens = 500
"""

VLLM = """
[[model]]
label = "q"
backend = "vllm"
hf_repo = "Qwen/Qwen3-8B"
alloc_mode = "AUTO"
max_input_tokens = 8000
max_output_tokens = 2000
kv_cache_type = "fp8"
reasoning_parser = "qwen3"
"""


def load(text):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "models.conf"
        p.write_text(text)
        return load_config(p)


def vspec(**kw):
    base = dict(label="q", hf_repo="Qwen/Qwen3-8B", gguf_file="", alloc_mode="AUTO",
                max_input_tokens=8000, max_output_tokens=2000, parallel=128,
                backend="vllm")
    base.update(kw)
    return ModelSpec(**base)


class TestConfig(unittest.TestCase):
    def test_mixed_backends(self):
        cfg = load(HEAD + LLAMA + VLLM)
        g, q = cfg.models
        self.assertEqual(g.backend, "llamacpp")
        self.assertEqual((g.parallel, g.ctx_size), (2, 3000))  # [defaults] valgono
        self.assertEqual(q.backend, "vllm")
        # vLLM: [defaults] parallel non si applica; ctx = una sola sequenza
        self.assertEqual((q.parallel, q.ctx_size), (64, 10000))
        self.assertEqual((q.kv_cache_type, q.reasoning_parser), ("fp8", "qwen3"))
        self.assertAlmostEqual(q.gpu_memory_utilization, 0.90)

    def test_vllm_rejects_gguf_file(self):
        with self.assertRaises(ConfigError):
            load(HEAD + VLLM.replace('alloc_mode', 'gguf_file = "x.gguf"\nalloc_mode'))

    def test_vllm_rejects_llamacpp_kv(self):
        with self.assertRaises(ConfigError):
            load(HEAD + VLLM.replace('kv_cache_type = "fp8"', 'kv_cache_type = "q8_0"'))

    def test_vllm_rejects_reasoning_budget(self):
        with self.assertRaises(ConfigError):
            load(HEAD + VLLM + "reasoning_budget = 100\n")

    def test_llamacpp_rejects_vllm_fields(self):
        with self.assertRaises(ConfigError):
            load(HEAD + LLAMA + 'gpu_memory_utilization = 0.8\n')

    def test_bad_backend_and_utilization(self):
        with self.assertRaises(ConfigError):
            load(HEAD + LLAMA.replace('alloc_mode', 'backend = "tgi"\nalloc_mode'))
        with self.assertRaises(ConfigError):
            load(HEAD + VLLM + "gpu_memory_utilization = 1.5\n")


class TestEstimate(unittest.TestCase):
    CFG = {"num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096,
           "num_hidden_layers": 36, "head_dim": 128, "torch_dtype": "bfloat16"}

    def test_kv_geometry(self):
        elems, heads = kv_geometry(self.CFG)
        self.assertEqual((elems, heads), (2 * 36 * 8 * 128, 32))
        # i config multimodali hanno il modello di testo in text_config
        self.assertEqual(kv_geometry({"text_config": self.CFG}), (elems, heads))

    def test_kv_dtype(self):
        self.assertEqual(kv_dtype_bytes(vspec(kv_cache_type="fp8"), self.CFG), 1)
        self.assertEqual(kv_dtype_bytes(vspec(kv_cache_type="f16"), self.CFG), 2)

    def test_one_gpu_when_it_fits(self):
        self.assertEqual(required_gpus_for(16, 1, 32, 2, 92, 0.9), 1)

    def test_two_gpus_for_big_model(self):
        # 140 GiB di pesi (70B bf16): 70 + 6 per GPU su due H100
        self.assertEqual(required_gpus_for(140, 2, 64, 2, 92, 0.9), 2)

    def test_tp_must_divide_heads(self):
        # 3 non divide 40 teste: si salta a 4
        self.assertEqual(required_gpus_for(200, 0, 40, 4, 92, 0.9), 4)

    def test_does_not_fit(self):
        with self.assertRaises(RuntimeError):
            required_gpus_for(400, 0, 64, 2, 92, 0.9)


class TestPlan(unittest.TestCase):
    def test_vllm_takes_whole_gpus(self):
        q = vspec()
        small = ModelSpec(label="s", hf_repo="o/r", gguf_file="s.gguf", alloc_mode="AUTO",
                          max_input_tokens=10, max_output_tokens=10)
        pl, n = plan([ModelNeed(q, 1, 92.0), ModelNeed(small, 1, 5.0)], 2, 92)
        by = {p.spec.label: p.gpu_indices for p in pl}
        self.assertEqual(n, 1)
        self.assertNotEqual(by["q"], by["s"])          # il piccolo non entra sulla GPU vLLM

    def test_plan_row_and_snapshot(self):
        with tempfile.TemporaryDirectory() as hf:
            repo = Path(hf) / "hub" / "models--Qwen--Qwen3-8B"
            snap = repo / "snapshots" / ("a" * 40)
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}")
            (repo / "refs").mkdir()
            (repo / "refs" / "main").write_text("a" * 40)
            q = vspec(reasoning_parser=None)
            self.assertEqual(resolve_local_snapshot(q, hf), snap)
            self.assertIsNone(resolve_local_snapshot(vspec(revision="v2"), hf))
            row = plan_row(Placed(q, 0, [0, 1], 2, 184.0), hf).split("\t")
            self.assertEqual(len(row), 14)                 # nessun campo vuoto
            self.assertEqual(row[2], f"/hf/hub/models--Qwen--Qwen3-8B/snapshots/{'a' * 40}")
            self.assertEqual(row[3:6], ["CUDA0,CUDA1", "10000", "128"])
            self.assertEqual(row[9:], ["-", "vllm", "2", "0.9", "-"])


class _Handler(http.server.BaseHTTPRequestHandler):
    status, body = 200, b""

    def do_GET(self):
        self.send_response(self.server.reply[0])
        self.end_headers()
        self.wfile.write(self.server.reply[1])

    def log_message(self, *a):
        pass


class TestHealth(unittest.TestCase):
    def check(self, status, body):
        srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        srv.reply = (status, body)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            return _health_ok("127.0.0.1", srv.server_address[1])
        finally:
            srv.shutdown()
            srv.server_close()

    def test_vllm_empty_200(self):
        self.assertTrue(self.check(200, b""))

    def test_llamacpp_ok(self):
        self.assertTrue(self.check(200, b'{"status":"ok"}'))

    def test_loading_503(self):
        self.assertFalse(self.check(503, b'{"error":{"message":"Loading model"}}'))

    def test_other_body(self):
        self.assertFalse(self.check(200, b'{"status":"loading"}'))


class TestClient(unittest.TestCase):
    def client(self, backend):
        d = tempfile.mkdtemp()
        ep = {"job_id": "1", "models": {"m": {"base_url": "http://h:1/v1", "host": "h",
                                              "port": 1, "reasoning": True,
                                              "backend": backend}}}
        (Path(d) / "endpoints.json").write_text(json.dumps(ep))
        return CerberusClient(project_dir=d)

    def test_repeat_penalty_renamed_for_vllm(self):
        body = {"repeat_penalty": 1.1, "top_k": 20}
        out = self.client("vllm")._adapt_body("m", body)
        self.assertEqual(out, {"repetition_penalty": 1.1, "top_k": 20})
        self.assertIn("repeat_penalty", body)               # l'originale non cambia

    def test_llamacpp_body_unchanged(self):
        body = {"repeat_penalty": 1.1}
        self.assertEqual(self.client("llamacpp")._adapt_body("m", body), body)

    def test_reasoning_field_names(self):
        self.assertEqual(_reasoning_field(SimpleNamespace(reasoning_content="a")), "a")
        self.assertEqual(_reasoning_field(SimpleNamespace(reasoning="b")), "b")
        msg = SimpleNamespace(model_extra={"reasoning": "c"})
        self.assertEqual(_reasoning_field(msg), "c")
        self.assertIsNone(_reasoning_field(SimpleNamespace(model_extra=None)))


if __name__ == "__main__":
    unittest.main()
