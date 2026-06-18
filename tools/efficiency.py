#!/usr/bin/env python3
"""Report backend parameter counts (and optionally MACs) for SpAArSIST configs.

Backend = the AASIST/SpAArSIST pooling module only; the XLS-R front-end is held
constant and excluded, matching Table 2 of the paper.

MAC counting requires `ptflops` (pip install ptflops) and is approximate for the
custom graph ops; parameter counts are exact.

    python tools/efficiency.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from spaarsist.backend import SpAArSIST


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


# Expected backend param counts (paper Table 2): Base 611,784; rank-1 586,434.
CONFIGS = [
    ("AASIST baseline (Base)",   dict(node_scoring="learned", stack_aggregation="attention",
                                      pool_ratio_train=0.5, pool_ratio_infer=0.5)),
    ("AST-03-01-Mag (rank 1)",   dict(node_scoring="magnitude", stack_aggregation="mean",
                                      pool_ratio_train=0.3, pool_ratio_infer=0.1)),
    ("Mag-only (attention kept)", dict(node_scoring="magnitude", stack_aggregation="attention",
                                       pool_ratio_train=0.3, pool_ratio_infer=0.1)),
]


def main():
    print(f"{'Configuration':<26}{'BE params':>12}")
    print("-" * 38)
    for name, kw in CONFIGS:
        backend = SpAArSIST(inputs_dim=1024, outputs_dim=1024, **kw)
        backend.eval()
        print(f"{name:<26}{count_params(backend):>12,}")

    # Optional MAC estimate for one forward pass through the backend.
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        print("\n(install ptflops for an approximate MAC count)")
        return

    print("\nApproximate backend MACs (one ~4 s utterance):")
    for name, kw in CONFIGS:
        backend = SpAArSIST(inputs_dim=1024, outputs_dim=1024, **kw).eval()
        # backend input: (T, D) frame-level features; T ~ 200 for ~4 s at XLS-R stride
        macs, _ = get_model_complexity_info(
            backend, (200, 1024), as_strings=False, print_per_layer_stat=False, verbose=False,
        )
        print(f"  {name:<26}{macs/1e6:>10.2f} M-MACs")


if __name__ == "__main__":
    main()
