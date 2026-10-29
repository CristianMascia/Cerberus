"""Test unitari delle repliche: config, placement, endpoints e round-robin del client."""

import json
import tempfile
import unittest
from pathlib import Path

from cerberus.client import CerberusClient
from cerberus.config import ConfigError, ModelSpec, load_config
from cerberus.deploy import plan_row
from cerberus.placement import ModelNeed, PlacementError, plan

CONF = """
[allocation]
gpus_per_node = 2
vram_gib = 92
[[model]]
label = "q"
backend = "vllm"
hf_repo = "Qwen/Qwen3-8B"
alloc_mode = "AUTO"
max_input_tokens = 1000
max_output_tokens = 500
"""


def load(text):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "models.conf"
        p.write_text(text)
        return load_config(p)


def spec(label="s", replicas=1, **kw):
    base = dict(label=label, hf_repo="o/r", gguf_file="s.gguf", alloc_mode="AUTO",
                max_input_tokens=10, max_output_tokens=10, replicas=replicas)
    base.update(kw)
    return ModelSpec(**base)


class TestConfig(unittest.TestCase):
    def test_default_and_value(self):
        self.assertEqual(load(CONF).models[0].replicas, 1)
        self.assertEqual(load(CONF + "replicas = 2\n").models[0].replicas, 2)

    def test_invalid(self):
        for bad in ("0", "1.5", "true", '"2"'):
            with self.assertRaises(ConfigError):
                load(CONF + f"replicas = {bad}\n")


class TestPlacement(unittest.TestCase):
    def test_copies_never_share_a_gpu(self):
        # 2 GiB l'una: ci starebbero tutte su una GPU, ma le copie vanno separate
        pl, n = plan([ModelNeed(spec(replicas=2), 1, 2.0)], 2, 92)
        self.assertEqual(n, 1)
        self.assertEqual([p.gpu_indices for p in pl], [[0], [1]])
        self.assertEqual([p.instance for p in pl], ["s#0", "s#1"])

    def test_more_copies_than_gpus_add_nodes(self):
        pl, n = plan([ModelNeed(spec(replicas=3), 1, 2.0)], 2, 92)
        self.assertEqual(n, 2)
        self.assertEqual(len({(p.node_idx, p.gpu_indices[0]) for p in pl}), 3)

    def test_other_models_can_share(self):
        # un modello diverso può stare sulla GPU di una replica
        needs = [ModelNeed(spec("a", replicas=2), 1, 10.0), ModelNeed(spec("b"), 1, 10.0)]
        pl, n = plan(needs, 2, 92)
        self.assertEqual(n, 1)
        self.assertEqual([p.instance for p in pl], ["a#0", "a#1", "b"])

    def test_vllm_replicas_take_whole_gpus(self):
        q = spec("q", replicas=2, backend="vllm", gguf_file="")
        pl, n = plan([ModelNeed(q, 1, 92.0)], 2, 92)
        self.assertEqual((n, [p.gpu_indices for p in pl]), (1, [[0], [1]]))
        q3 = spec("q", replicas=3, backend="vllm", gguf_file="")
        with self.assertRaises(PlacementError):          # 3 GPU intere su un nodo da 2
            plan([ModelNeed(q3, 1, 92.0)], 2, 92, max_nodes=1)

    def test_instance_in_plan_row(self):
        with tempfile.TemporaryDirectory() as hf:
            snap = Path(hf) / "hub" / "models--o--r" / "snapshots" / ("a" * 40)
            snap.mkdir(parents=True)
            (snap / "s.gguf").write_text("")
            refs = Path(hf) / "hub" / "models--o--r" / "refs"
            refs.mkdir()
            (refs / "main").write_text("a" * 40)
            pl, _ = plan([ModelNeed(spec(replicas=2), 1, 2.0)], 2, 92)
            rows = [plan_row(p, hf).split("\t") for p in pl]
            self.assertEqual([r[1] for r in rows], ["s", "s"])     # served name = label
            self.assertEqual([r[14] for r in rows], ["s#0", "s#1"])


class TestClient(unittest.TestCase):
    def client(self, entry):
        d = tempfile.mkdtemp()
        (Path(d) / "endpoints.json").write_text(json.dumps({"models": {"m": entry}}))
        return CerberusClient(project_dir=d)

    def test_round_robin(self):
        urls = ["http://a:1/v1", "http://b:2/v1"]
        c = self.client({"base_url": urls[0], "host": "a", "port": 1, "reasoning": False,
                         "replicas": [{"base_url": u} for u in urls]})
        self.assertEqual(c.base_urls("m"), urls)
        self.assertEqual([c._next_replica("m") for _ in range(4)], [0, 1, 0, 1])

    def test_old_endpoints_without_replicas(self):
        c = self.client({"base_url": "http://a:1/v1", "host": "a", "port": 1})
        self.assertEqual(c.base_urls("m"), ["http://a:1/v1"])
        self.assertEqual([c._next_replica("m") for _ in range(2)], [0, 0])


if __name__ == "__main__":
    unittest.main()
