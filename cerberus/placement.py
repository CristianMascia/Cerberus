"""
Cerberus — automatic node + GPU placement.

Given each model's GPU count and VRAM footprint, pack the models onto an
allocation of `nodes x gpus_per_node` GPUs of `vram_gib` GiB each
([allocation] vram_gib; 32 GiB = IBiSCo's V100s by default):

  * multi-GPU models (tensor-split) take that many WHOLE GPUs on a SINGLE node
    (tensor-split cannot cross nodes);
  * single-GPU models pack onto GPUs and may SHARE one when their footprints sum
    to <= vram_gib (first-fit-decreasing);
  * a model with `replicas = N` is placed N times, never two copies on the same
    GPU (a second copy on a busy GPU adds no throughput).

`compute_needs()` (network: reads GGUF headers, or safetensors sizes and
config.json for vllm models) turns a Config into per-model needs; a vllm model's
footprint is its whole GPUs, since vLLM preallocates them; `plan()` is pure and does the packing so it can be tested offline.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_VRAM_GIB, Config, ModelSpec
from . import vram, vllm_backend

VRAM_GIB = DEFAULT_VRAM_GIB  # kept for callers that import it


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
    replica: int = 0              # 0..spec.replicas-1

    @property
    def instance(self) -> str:
        """Name of this server (logs, endpoints.parts): the label, `label#i` for replicas."""
        return self.spec.label if self.spec.replicas == 1 else f"{self.spec.label}#{self.replica}"

    @property
    def device_arg(self) -> str:
        return ",".join(f"CUDA{i}" for i in self.gpu_indices)


class PlacementError(Exception):
    pass


def compute_needs(config: Config, vram_gib: float | None = None,
                  token: str | None = None) -> list[ModelNeed]:
    """Resolve each model's GPU count + footprint (reads model metadata over HTTP).

    `vram_gib` defaults to the config's [allocation] vram_gib.
    """
    if vram_gib is None:
        vram_gib = config.vram_gib
    needs = []
    for m in config.models:
        if m.is_vllm:
            n_gpus, fp = vllm_backend.required_gpus(m, config.gpus_per_node, vram_gib, token)
        else:
            n_gpus, est = vram.required_gpus(m, config.gpus_per_node, vram_gib, token)
            fp = vram.footprint_gib(est)
        needs.append(ModelNeed(spec=m, num_gpus=n_gpus, footprint_gib=fp))
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

    # One entry per server: (need, replica index).
    copies = [(n, r) for n in needs for r in range(n.spec.replicas)]
    # Order: multi-GPU first (hardest to place), then single-GPU by footprint desc.
    multi = sorted([c for c in copies if c[0].num_gpus > 1],
                   key=lambda c: c[0].num_gpus, reverse=True)
    single = sorted([c for c in copies if c[0].num_gpus == 1],
                    key=lambda c: c[0].footprint_gib, reverse=True)
    used_by: dict[str, set[tuple[int, int]]] = {}   # label -> (node, gpu) of its copies

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
        # first GPU (any node) with enough remaining capacity, not holding another copy
        taken = used_by.get(need.spec.label, set())
        for k, gpus in enumerate(nodes):
            for i, cap in enumerate(gpus):
                if cap >= need.footprint_gib and (k, i) not in taken:
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

    for need, r in multi + single:
        k, gpus = (place_multi if need.num_gpus > 1 else place_single)(need)
        used_by.setdefault(need.spec.label, set()).update((k, i) for i in gpus)
        placements.append(Placed(need.spec, k, gpus, need.num_gpus, need.footprint_gib, r))

    # Keep output in the original models.conf order for readability.
    order = {n.spec.label: idx for idx, n in enumerate(needs)}
    placements.sort(key=lambda p: (order[p.spec.label], p.replica))
    return placements, len(nodes)
