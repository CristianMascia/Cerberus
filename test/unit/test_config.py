"""Test unitari del parser/validatore di models.conf (cerberus.config)."""

import tempfile
import unittest
from pathlib import Path

from cerberus.config import ConfigError, load_config

VALID = """
[allocation]
gpus_per_node = 3
[defaults]
parallel = 2
kv_cache_type = "q8_0"
[[model]]
label = "a"
hf_repo = "org/repo"
gguf_file = "a.gguf"
alloc_mode = "AUTO"
max_input_tokens = 100
max_output_tokens = 50
[[model]]
label = "b"
hf_repo = "org/repo2"
gguf_file = "b.gguf"
alloc_mode = "MANUAL"
num_gpus = 2
max_input_tokens = 10
max_output_tokens = 10
"""


def _write(d, text):
    p = Path(d) / "models.conf"
    p.write_text(text)
    return p


class TestConfig(unittest.TestCase):
    def _load(self, text):
        with tempfile.TemporaryDirectory() as d:
            return load_config(_write(d, text))

    def test_valid_parse_and_defaults(self):
        cfg = self._load(VALID)
        self.assertEqual(cfg.gpus_per_node, 3)
        self.assertEqual(len(cfg.models), 2)
        a = cfg.by_label("a")
        self.assertEqual(a.parallel, 2)              # ereditato da [defaults]
        self.assertEqual(a.kv_cache_type, "q8_0")    # ereditato da [defaults]
        self.assertEqual(a.ctx_size, (100 + 50) * 2)  # (in+out)*parallel
        self.assertEqual(cfg.by_label("b").num_gpus, 2)

    def test_manual_requires_num_gpus(self):
        with self.assertRaises(ConfigError):
            self._load(VALID.replace("num_gpus = 2\n", ""))

    def test_auto_rejects_num_gpus(self):
        bad = VALID.replace('alloc_mode = "AUTO"', 'alloc_mode = "AUTO"\nnum_gpus = 1')
        with self.assertRaises(ConfigError):
            self._load(bad)

    def test_duplicate_labels(self):
        with self.assertRaises(ConfigError):
            self._load(VALID.replace('label = "b"', 'label = "a"'))

    def test_num_gpus_exceeds_node(self):
        with self.assertRaises(ConfigError):
            self._load(VALID.replace("num_gpus = 2", "num_gpus = 9"))

    def test_bad_kv_cache_type(self):
        with self.assertRaises(ConfigError):
            self._load(VALID.replace('kv_cache_type = "q8_0"', 'kv_cache_type = "int4"'))

    def test_missing_required_field(self):
        with self.assertRaises(ConfigError):
            self._load(VALID.replace('gguf_file = "a.gguf"\n', ""))


if __name__ == "__main__":
    unittest.main()
