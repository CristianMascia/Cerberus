"""
Cerberus — automatic node + GPU placement.

Given each model's GPU count and VRAM footprint, pack the models onto an
allocation of `nodes x gpus_per_node` V100s (32 GiB each):

  * multi-GPU models (tensor-split) take that many WHOLE GPUs on a SINGLE node
    (tensor-split cannot cross nodes);
  * single-GPU models pack onto GPUs and may SHARE one when their footprints sum
    to <= 32 GiB (first-fit-decreasing).

`compute_needs()` (network: reads GGUF headers) turns a Config into per-model
needs; `plan()` is pure and does the packing so it can be tested offline.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config, ModelSpec
from . import vram

VRAM_GIB = 32.0  # per V100-SXM2-32GB


@dataclass
class ModelNeed:
    spec: ModelSpec
    num_gpus: int
    footprint_gib: float          # total VRAM footprint across its GPUs


@dataclass
class Placed:
    spec: ModelSpec
    node_idx: int                 # index into the allocation's node list
    gpu_indices: list[int]        # logical GPU indices on that node (-> CUDA<i>)
    num_gpus: int
    footprint_gib: float

    @property
    def device_arg(self) -> str:
        return ",".join(f"CUDA{i}" for i in self.gpu_indices)


class PlacementError(Exception):
    pass


def compute_needs(config: Config, vram_gib: float = VRAM_GIB,
                  token: str | None = None) -> list[ModelNeed]:
    """Resolve each model's GPU count + footprint (reads GGUF headers over HTTP)."""
    needs = []
    for m in config.models:
        n_gpus, est = vram.required_gpus(m, config.gpus_per_node, vram_gib, token)
        needs.append(ModelNeed(spec=m, num_gpus=n_gpus,
                               footprint_gib=vram.footprint_gib(est)))
    return needs


def plan(needs: list[ModelNeed], gpus_per_node: int, vram_gib: float = VRAM_GIB,
         max_nodes: int | None = None) -> tuple[list[Placed], int]:
    """Pack needs onto nodes. Returns (placements, n_nodes_used).

    Nodes grow on demand; if `max_nodes` is set and exceeded, raise PlacementError.
    """
    # Per-node GPU free capacity (GiB). nodes[k] is a list of length gpus_per_node.
    nodes: list[list[float]] = []

    def add_node() -> int:
        if max_nodes is not None and len(nodes) >= max_nodes:
            raise PlacementError(
                f"models do not fit in {max_nodes} node(s) x {gpus_per_node} GPU(s)"
            )
        nodes.append([vram_gib] * gpus_per_node)
        return len(nodes) - 1

    placements: list[Placed] = []

    # Order: multi-GPU first (hardest to place), then single-GPU by footprint desc.
    multi = sorted([n for n in needs if n.num_gpus > 1],
                   key=lambda n: n.num_gpus, reverse=True)
    single = sorted([n for n in needs if n.num_gpus == 1],
                    key=lambda n: n.footprint_gib, reverse=True)

    def place_multi(need: ModelNeed):
        # need num_gpus GPUs at full capacity on one node
        for k, gpus in enumerate(nodes):
            free = [i for i, cap in enumerate(gpus) if cap >= vram_gib]
            if len(free) >= need.num_gpus:
                chosen = free[:need.num_gpus]
                for i in chosen:
                    gpus[i] = 0.0
                return k, chosen
        k = add_node()
        chosen = list(range(need.num_gpus))
        for i in chosen:
            nodes[k][i] = 0.0
        return k, chosen

    def place_single(need: ModelNeed):
        # first GPU (any node) with enough remaining capacity
        for k, gpus in enumerate(nodes):
            for i, cap in enumerate(gpus):
                if cap >= need.footprint_gib:
                    gpus[i] -= need.footprint_gib
                    return k, [i]
        k = add_node()
        if need.footprint_gib > vram_gib:
            raise PlacementError(
                f"model '{need.spec.label}' needs {need.footprint_gib:.1f} GiB on a "
                f"single GPU but a GPU has only {vram_gib:.0f} GiB"
            )
        nodes[k][0] -= need.footprint_gib
        return k, [0]

    for need in multi:
        k, gpus = place_multi(need)
        placements.append(Placed(need.spec, k, gpus, need.num_gpus, need.footprint_gib))
    for need in single:
        k, gpus = place_single(need)
        placements.append(Placed(need.spec, k, gpus, need.num_gpus, need.footprint_gib))

    # Keep output in the original models.conf order for readability.
    order = {n.spec.label: idx for idx, n in enumerate(needs)}
    placements.sort(key=lambda p: order[p.spec.label])
    return placements, len(nodes)
