"""Test unitari del bin-packing di posizionamento (cerberus.placement)."""

import unittest

from cerberus.config import ModelSpec
from cerberus.placement import ModelNeed, PlacementError, plan


def need(label, num_gpus, footprint):
    spec = ModelSpec(label=label, hf_repo="o/r", gguf_file="x.gguf",
                     alloc_mode="AUTO", max_input_tokens=10, max_output_tokens=10)
    return ModelNeed(spec=spec, num_gpus=num_gpus, footprint_gib=footprint)


class TestPlacement(unittest.TestCase):
    def test_single_node_sharing(self):
        # due modelli piccoli condividono una GPU, tutto su un nodo
        pl, n = plan([need("a", 1, 10), need("b", 1, 8)], gpus_per_node=3)
        self.assertEqual(n, 1)
        self.assertEqual({p.spec.label: p.gpu_indices for p in pl}, {"a": [0], "b": [0]})

    def test_multi_gpu_whole_and_spill_to_second_node(self):
        # un modello a 2 GPU riempie il nodo 0 (con un piccolo), il secondo va sul nodo 1
        pl, n = plan([need("big", 2, 40), need("big2", 2, 40), need("s", 1, 5)],
                     gpus_per_node=3)
        self.assertEqual(n, 2)
        by = {p.spec.label: (p.node_idx, p.gpu_indices) for p in pl}
        self.assertEqual(by["big"], (0, [0, 1]))
        self.assertEqual(by["big2"], (1, [0, 1]))
        self.assertEqual(by["s"][0], 0)                 # il piccolo impacchettato sul nodo 0

    def test_device_arg_format(self):
        pl, _ = plan([need("m", 2, 40)], gpus_per_node=3)
        self.assertEqual(pl[0].device_arg, "CUDA0,CUDA1")

    def test_order_preserved(self):
        needs = [need("z", 1, 1), need("a", 1, 1), need("m", 1, 1)]
        pl, _ = plan(needs, gpus_per_node=3)
        self.assertEqual([p.spec.label for p in pl], ["z", "a", "m"])

    def test_overflow_raises(self):
        with self.assertRaises(PlacementError):
            plan([need("a", 2, 40), need("b", 2, 40)], gpus_per_node=3, max_nodes=1)

    def test_single_gpu_too_big_raises(self):
        with self.assertRaises(PlacementError):
            plan([need("huge", 1, 50)], gpus_per_node=3, max_nodes=1)


if __name__ == "__main__":
    unittest.main()
