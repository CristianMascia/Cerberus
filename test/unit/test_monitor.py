"""Test unitari del monitor dei batch (cerberus.monitor): lettura incrementale, stati, render."""

import io
import json
import tempfile
import unittest
from pathlib import Path

from cerberus import monitor


def make_batch(root: Path, name: str, shards: int, per_shard: int, job_id="100"):
    bd = root / name
    for d in ("shards", "results", "logs"):
        (bd / d).mkdir(parents=True)
    for i in range(shards):
        rows = [{"custom_id": f"{i}-{k}", "repeat": 0} for k in range(per_shard)]
        (bd / "shards" / f"shard_{i:03d}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows))
    (bd / "manifest.json").write_text(json.dumps({
        "models": ["m"], "n_requests": shards * per_shard, "repeats": 1,
        "shards": shards, "n_units": shards * per_shard, "job_id": job_id}))
    return bd


def add_results(bd: Path, shard: int, ids, error=False, partial=""):
    with (bd / "results" / f"shard_{shard:03d}.jsonl").open("a") as fh:
        for cid in ids:
            fh.write(json.dumps({
                "custom_id": cid, "repeat": 0,
                "response": None if error else {"usage": {"completion_tokens": 10}},
                "error": {"type": "X", "message": "x"} if error else None,
                "latency_s": 2.0}) + "\n")
        fh.write(partial)


class TestMonitor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_incremental_read_and_last_wins(self):
        bd = make_batch(self.root, "b", 1, 4)
        st = monitor.BatchTracker.open(bd).shards[0]
        add_results(bd, 0, ["0-0", "0-1"], partial='{"custom_id": "0-2"')  # cut line
        st.refresh()
        self.assertEqual((st.done, st.errors, st.tokens), (2, 0, 20))
        with (bd / "results" / "shard_000.jsonl").open("a") as fh:
            fh.write(', "repeat": 0, "response": null, "error": {"type": "E"}}\n')
        add_results(bd, 0, ["0-0"], error=True)   # resumed unit: the last line wins
        st.refresh()
        self.assertEqual((st.done, st.errors), (3, 2))

    def test_rate_from_samples(self):
        bd = make_batch(self.root, "b", 1, 10)
        st = monitor.BatchTracker.open(bd).shards[0]
        st.refresh(); st.sample(0.0, 300)
        add_results(bd, 0, ["0-0", "0-1", "0-2"])
        st.refresh(); st.sample(60.0, 300)
        units_s, tok_s = st.rate()
        self.assertAlmostEqual(units_s * 60, 3.0)
        self.assertAlmostEqual(tok_s, 0.5)

    def test_find_dirs_in_parent(self):
        make_batch(self.root, "b1", 1, 1)
        make_batch(self.root, "b2", 1, 1)
        (self.root / "other").mkdir()
        dirs = monitor.find_batch_dirs([str(self.root), str(self.root / "b1")])
        self.assertEqual([d.name for d in dirs], ["b1", "b2"])
        with self.assertRaises(monitor.BatchError):
            monitor.find_batch_dirs([str(self.root / "other")])

    def test_states_and_exit(self):
        b1 = make_batch(self.root, "b1", 3, 2, job_id="100")
        b2 = make_batch(self.root, "b2", 1, 2, job_id="200")
        add_results(b1, 0, ["0-0", "0-1"])            # shard 0 complete, left the queue
        add_results(b1, 1, ["1-0"])                   # shard 1 running
        (b1 / "logs" / "shard_2_100.out").write_text("CANCELLED AT ... DUE TO TIME LIMIT\n")
        add_results(b2, 0, ["0-0", "0-1"])
        queue = {"100": {1: monitor.TaskInfo("RUNNING", "1:00", "gpu01", 60.0)}, "200": {}}
        out = io.StringIO()
        monitor.monitor([str(b1), str(b2)], once=True, color=False,
                        squeue=lambda ids: queue, out=out)
        text = out.getvalue()
        self.assertIn("COMPLETED", text)
        self.assertIn("RUNNING", text)
        self.assertIn("TIMEOUT", text)
        self.assertIn("ALL 2 batches: 5/8", text)
        self.assertNotIn("Nothing running", text)

        # nothing in the queue any more: the loop ends by itself
        queue["100"] = {}
        out = io.StringIO()
        rc = monitor.monitor([str(self.root)], interval=0.01, color=False,
                             squeue=lambda ids: queue, out=out)
        self.assertEqual(rc, 0)
        self.assertIn("Nothing running or pending.", out.getvalue())

    def test_parse_elapsed(self):
        self.assertEqual(monitor._parse_elapsed("1-02:03:04"), 93784)
        self.assertEqual(monitor._parse_elapsed("5:07"), 307)
        self.assertEqual(monitor._parse_elapsed("0:00"), 0)


if __name__ == "__main__":
    unittest.main()
