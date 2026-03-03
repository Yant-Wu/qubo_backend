"""dimod 背包問題求解協調器（單一職責：串接 builder → sampler → parser 的流程）。"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from .cqm_builder import build_knapsack_cqm
from .cqm_sampler import SimulatedAnnealingCQMSampler
from .cqm_parser import parse_solution




def solve_knapsack_with_dimod(
    items: List[Dict[str, Any]],
    capacity: float,
    penalty: float,
    num_reads: int = 1000,
    seed: int | None = None,
) -> Dict[str, Any]:
    """完整求解流程：build_knapsack_cqm -> sample_cqm -> sampleset.first -> parse_solution。"""
    start = time.perf_counter()

    cqm, item_vars = build_knapsack_cqm(items, capacity, penalty)

    sampler = SimulatedAnnealingCQMSampler()
    sampleset = sampler.sample_cqm(cqm, num_reads=num_reads, seed=seed)
    best = sampleset.first

    parsed = parse_solution(items, best.sample, item_vars)

    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "status": "success",
        "solution": parsed["solution"],
        "energy": float(best.energy),
        "interpretation": parsed["interpretation"],
        "computation_time_ms": round(elapsed_ms, 3),
    }
