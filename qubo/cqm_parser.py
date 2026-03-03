"""CQM 結果解析器（單一職責：將 SampleSet 解析為業務層可用的結果結構）。"""
from __future__ import annotations

from typing import Any, Dict, List


def parse_solution(
    items: List[Dict[str, Any]],
    sample: Dict[str, int],
    item_vars: List[str],
) -> Dict[str, Any]:
    """
    從 sampleset.first.sample 解析出選取物品、總價值、總重量。

    Args:
        items:     原始物品列表
        sample:    sampleset.first.sample（變數名 → 0/1 值）
        item_vars: 物品對應的 QUBO 變數名稱列表

    Returns:
        {
            "solution": [0, 1, ...],
            "interpretation": {
                "selected_items": [{"name": ..., "weight": ..., "value": ...}],
                "total_value": float,
                "total_weight": float
            }
        }
    """
    solution = [int(sample[var]) for var in item_vars]

    selected_items: List[Dict[str, Any]] = []
    total_value = 0.0
    total_weight = 0.0

    for bit, item in zip(solution, items):
        if bit == 1:
            selected_items.append({
                "name": str(item["name"]),
                "weight": float(item["weight"]),
                "value": float(item["value"]),
            })
            total_weight += float(item["weight"])
            total_value += float(item["value"])

    return {
        "solution": solution,
        "interpretation": {
            "selected_items": selected_items,
            "total_value": total_value,
            "total_weight": total_weight,
        },
    }
