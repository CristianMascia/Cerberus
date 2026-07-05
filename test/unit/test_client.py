"""Test unitari del client: separazione del thinking e risoluzione della mappa."""

import json
import tempfile
import unittest
from pathlib import Path

from cerberus.client import CerberusClient, CerberusUnavailable, _split_thinking
from cerberus.download import _files_to_fetch
from cerberus.config import ModelSpec


class TestSplitThinking(unittest.TestCase):
    def test_reasoning_content_field(self):
        self.assertEqual(_split_thinking("Risposta.", "ragiono"), ("ragiono", "Risposta."))

    def test_inline_think_closed(self):
        self.assertEqual(_split_thinking("<think>penso</think>\nEcco.", None),
                         ("penso", "Ecco."))

    def test_inline_think_truncated(self):
        self.assertEqual(_split_thinking("<think>penso e non finisco", None),
                         ("penso e non finisco", ""))

    def test_no_thinking(self):
        self.assertEqual(_split_thinking("Solo risposta.", None), (None, "Solo risposta."))


class TestEndpointMap(unittest.TestCase):
    def test_unavailable_off_cluster(self):
        with tempfile.TemporaryDirectory() as d:
            c = CerberusClient(project_dir=d)     # nessun endpoints.json
            self.assertFalse(c.is_available())
            with self.assertRaises(CerberusUnavailable):
                c.list_models()

    def test_reads_map(self):
        with tempfile.TemporaryDirectory() as d:
            ep = {"job_id": "1", "models": {
                "m": {"base_url": "http://h:8081/v1", "host": "h", "port": 8081,
                      "reasoning": True}}}
            (Path(d) / "endpoints.json").write_text(json.dumps(ep))
            c = CerberusClient(project_dir=d)
            self.assertTrue(c.is_available())
            self.assertEqual(c.list_models(), ["m"])
            self.assertEqual(c.endpoint("m")["port"], 8081)
            with self.assertRaises(KeyError):
                c.endpoint("assente")


class TestDownloadSplit(unittest.TestCase):
    def _spec(self, fname):
        return ModelSpec(label="x", hf_repo="o/r", gguf_file=fname, alloc_mode="AUTO",
                         max_input_tokens=1, max_output_tokens=1)

    def test_single_file(self):
        self.assertEqual(_files_to_fetch(self._spec("model-Q4_K_M.gguf")),
                         ["model-Q4_K_M.gguf"])

    def test_split_expands_all_parts(self):
        self.assertEqual(
            _files_to_fetch(self._spec("model-Q8_0-00001-of-00003.gguf")),
            ["model-Q8_0-00001-of-00003.gguf", "model-Q8_0-00002-of-00003.gguf",
             "model-Q8_0-00003-of-00003.gguf"])


if __name__ == "__main__":
    unittest.main()
