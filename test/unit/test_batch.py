"""Test unitari della batch API (cerberus.batch): parsing, split, ripresa, merge."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from cerberus import batch
from cerberus.client import Response
from cerberus.config import ConfigError, load_config

CONF = """
[allocation]
gpus_per_node = 2
[batch]
shards = 2
repeats = 3
time = "00:30:00"
[[model]]
label = "a"
hf_repo = "org/a"
gguf_file = "a.gguf"
alloc_mode = "MANUAL"
num_gpus = 1
max_input_tokens = 100
max_output_tokens = 50
parallel = 2
[[model]]
label = "b"
hf_repo = "org/b"
gguf_file = "b.gguf"
alloc_mode = "MANUAL"
num_gpus = 1
max_input_tokens = 100
max_output_tokens = 50
"""


def _msgs(text="ciao"):
    return [{"role": "user", "content": text}]


class FakeClient:
    """Stand-in for CerberusClient: answers with the prompt, fails on 'boom'."""

    calls = []

    def __init__(self, *a, **kw):
        pass

    def list_models(self):
        return ["a", "b"]

    async def achat(self, label, messages, **kw):
        FakeClient.calls.append((label, messages[-1]["content"], kw))
        if messages[-1]["content"] == "boom":
            raise ValueError("bad request")
        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=5)
        return Response(content=f"{label}:{messages[-1]['content']}", reasoning=None,
                        finish_reason="stop", raw=SimpleNamespace(usage=usage))


class BatchTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "models.conf").write_text(CONF)
        self.cfg = load_config(self.dir / "models.conf")
        FakeClient.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def write_input(self, rows, name="in.jsonl"):
        p = self.dir / name
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return p


class TestConfigBatch(BatchTestCase):
    def test_batch_section(self):
        self.assertEqual(self.cfg.batch.shards, 2)
        self.assertEqual(self.cfg.batch.repeats, 3)
        self.assertEqual(self.cfg.batch.time, "00:30:00")
        self.assertEqual(self.cfg.batch.max_retries, 3)  # default

    def test_batch_unknown_field(self):
        (self.dir / "bad.conf").write_text(CONF.replace("shards = 2", "shard = 2"))
        with self.assertRaises(ConfigError):
            load_config(self.dir / "bad.conf")


class TestParsing(BatchTestCase):
    def test_native_and_default_model(self):
        p = self.write_input([
            {"custom_id": "1", "messages": _msgs(), "params": {"temperature": 0.1}},
            {"custom_id": "2", "model": "b", "messages": _msgs()},
        ])
        reqs = batch.load_jsonl(p, default_model="a")
        self.assertEqual([r.model for r in reqs], ["a", "b"])
        self.assertEqual(reqs[0].params, {"temperature": 0.1})

    def test_openai_batch_format(self):
        p = self.write_input([{
            "custom_id": "x", "method": "POST", "url": "/v1/chat/completions",
            "body": {"model": "a", "messages": _msgs(), "max_completion_tokens": 20,
                     "temperature": 0},
        }])
        (r,) = batch.load_jsonl(p)
        self.assertEqual(r.model, "a")
        self.assertEqual(r.params, {"max_tokens": 20, "temperature": 0})

    def test_errors(self):
        cases = [
            [{"custom_id": "1", "model": "a", "messages": _msgs()},
             {"custom_id": "1", "model": "a", "messages": _msgs()}],       # duplicate id
            [{"custom_id": "1", "messages": _msgs()}],                     # no model
            [{"custom_id": "1", "model": "a", "messages": []}],            # no messages
            [{"custom_id": "1", "model": "a", "messages": _msgs(),
              "params": {"temprature": 1}}],                               # unknown param
        ]
        for rows in cases:
            with self.assertRaises(batch.BatchError):
                batch.load_jsonl(self.write_input(rows))

    def test_check_against_config(self):
        bad_label = batch.load_jsonl(self.write_input(
            [{"custom_id": "1", "model": "zzz", "messages": _msgs()}]))
        with self.assertRaises(batch.BatchError):
            batch.check_against_config(bad_label, self.cfg)
        too_long = batch.load_jsonl(self.write_input(
            [{"custom_id": "1", "model": "a", "messages": _msgs(),
              "params": {"max_tokens": 51}}]))
        with self.assertRaises(batch.BatchError):
            batch.check_against_config(too_long, self.cfg)
        big_prompt = batch.load_jsonl(self.write_input(
            [{"custom_id": "1", "model": "a", "messages": _msgs("x" * 1000)}]))
        self.assertEqual(len(batch.check_against_config(big_prompt, self.cfg)), 1)


class TestSplit(unittest.TestCase):
    def test_round_robin_deterministic(self):
        units = [(str(i), 0) for i in range(7)]
        shards = batch.make_shards(units, 3)
        self.assertEqual([len(s) for s in shards], [3, 2, 2])
        self.assertEqual(shards, batch.make_shards(units, 3))
        self.assertEqual(sorted(u for s in shards for u in s), sorted(units))

    def test_never_more_shards_than_units(self):
        self.assertEqual(len(batch.make_shards([("a", 0), ("b", 0)], 5)), 2)

    def test_repeats_land_in_different_shards(self):
        reqs = [batch.BatchRequest("q", "a", _msgs())]
        shards = batch.make_shards(batch.expand_units(reqs, 2), 2)
        self.assertEqual(shards, [[("q", 0)], [("q", 1)]])

    def test_request_kwargs(self):
        kw = batch._request_kwargs({"seed": 7, "top_k": 20, "reasoning": False,
                                    "reasoning_effort": "low", "top_p": 0.9}, 2, 50)
        self.assertEqual(kw["seed"], 9)          # seed + repeat
        self.assertEqual(kw["max_tokens"], 50)   # default = max_output_tokens
        self.assertFalse(kw["reasoning"])
        self.assertEqual(kw["top_p"], 0.9)
        self.assertEqual(kw["extra_body"], {"top_k": 20, "chat_template_kwargs":
                                            {"reasoning_effort": "low"}})


class TestRunAndMerge(BatchTestCase):
    def prepare(self, rows, **kw):
        p = self.write_input(rows)
        bd = self.dir / "bd"
        manifest = batch.prepare(self.cfg, p, bd, estimate=False, **kw)
        return bd, manifest

    def run_all(self, bd, manifest, **kw):
        ep = self.dir / "endpoints.json"
        with mock.patch("cerberus.client.CerberusClient", FakeClient):
            return [batch.run_shard(bd, i, endpoints=str(ep), **kw)
                    for i in range(manifest["shards"])]

    def test_prepare_writes_batch_dir(self):
        rows = [{"custom_id": str(i), "model": "a", "messages": _msgs()} for i in range(4)]
        bd, manifest = self.prepare(rows)
        self.assertEqual(manifest["n_units"], 12)          # 4 x [batch] repeats=3
        self.assertEqual(manifest["shards"], 2)
        self.assertEqual(manifest["models"], ["a"])        # only the used models
        sb = batch.write_sbatch(bd, manifest, python="python3")
        text = sb.read_text()
        self.assertIn("#SBATCH --array=0-1", text)
        self.assertIn("#SBATCH --time=00:30:00", text)
        with self.assertRaises(batch.BatchError):          # never overwrite a batch
            batch.prepare(self.cfg, self.dir / "in.jsonl", bd, estimate=False)

    def test_run_merge_in_input_order(self):
        rows = [{"custom_id": c, "model": m, "messages": _msgs(c)}
                for c, m in [("z", "a"), ("y", "b"), ("x", "a")]]
        bd, manifest = self.prepare(rows, repeats=2)
        self.assertEqual(self.run_all(bd, manifest), [0, 0])
        out, missing = batch.merge(bd)
        self.assertEqual(missing, [])
        res = [json.loads(l) for l in out.read_text().splitlines()]
        self.assertEqual([(r["custom_id"], r["repeat"]) for r in res],
                         [("z", 0), ("z", 1), ("y", 0), ("y", 1), ("x", 0), ("x", 1)])
        self.assertEqual(res[2]["response"]["content"], "b:y")
        self.assertEqual({r["shard"] for r in res}, {0, 1})

    def test_resume_and_retry_failed(self):
        rows = [{"custom_id": "ok", "model": "a", "messages": _msgs("ok")},
                {"custom_id": "bad", "model": "a", "messages": _msgs("boom")}]
        bd, manifest = self.prepare(rows, shards=1, repeats=1)
        self.assertEqual(self.run_all(bd, manifest), [1])   # error recorded, not raised
        n_calls = len(FakeClient.calls)
        self.run_all(bd, manifest)                          # resume: nothing to redo
        self.assertEqual(len(FakeClient.calls), n_calls)
        self.run_all(bd, manifest, retry_failed=True)       # only the failed one
        self.assertEqual([c[1] for c in FakeClient.calls[n_calls:]], ["boom"])
        out, missing = batch.merge(bd)
        res = [json.loads(l) for l in out.read_text().splitlines()]
        self.assertEqual(len(res), 2)                        # last line per unit wins
        self.assertEqual(res[1]["error"]["type"], "ValueError")

    def test_status_and_missing(self):
        rows = [{"custom_id": str(i), "model": "a", "messages": _msgs()} for i in range(3)]
        bd, manifest = self.prepare(rows, shards=3, repeats=1)
        with mock.patch("cerberus.client.CerberusClient", FakeClient):
            batch.run_shard(bd, 0, endpoints=str(self.dir / "e.json"))
        prog = batch.shard_progress(bd, manifest)
        self.assertEqual([p["done"] for p in prog], [1, 0, 0])
        _, missing = batch.merge(bd)
        self.assertEqual(missing, [("1", 0), ("2", 0)])


if __name__ == "__main__":
    unittest.main()
