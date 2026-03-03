# problem_generator.py — 隨機問題資料生成（單一職責：測試與開發用資料產生）
import random
from typing import Any, Dict, Optional


def generate_random_problem_data(
    problem_type: str,
    n_variables: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    依 problem_type 與 n_variables 生成隨機問題資料。

    Returns:
        dict 格式，與各 ProblemData schema 相容（可直接傳給 builder/interpreter）。
    """
    rng = random.Random(seed)

    if problem_type == "knapsack":
        items = [
            {
                "weight": round(rng.uniform(1.0, 10.0), 1),
                "value": round(rng.uniform(1.0, 20.0), 1),
            }
            for _ in range(n_variables)
        ]
        total_weight = sum(it["weight"] for it in items)
        max_weight = int(total_weight * rng.uniform(0.5, 0.7))
        max_weight = max(max_weight, 1)
        return {"items": items, "max_weight": float(max_weight), "penalty": 10.0}

    if problem_type == "max_cut":
        edges = []
        for i in range(n_variables):
            for j in range(i + 1, n_variables):
                if rng.random() < 0.5:
                    edges.append({"from": i, "to": j, "weight": round(rng.uniform(0.5, 2.0), 2)})
        if not edges:
            edges.append({"from": 0, "to": 1, "weight": 1.0})
        return {"nodes": n_variables, "edges": edges}

    raise ValueError(f"problem_type '{problem_type}' 不支援隨機生成")
