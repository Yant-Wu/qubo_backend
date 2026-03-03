"""QUBO CQM 建構器（單一職責：將背包問題轉換為 ConstrainedQuadraticModel）。"""
from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, List, Tuple

from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, QuadraticModel


def _build_slack_coefficients(capacity: int) -> List[int]:
    """將不等式約束轉成等式時使用的 slack 位元權重（二進位編碼）。"""
    num_bits = max(1, math.ceil(math.log2(capacity + 1)))
    return [2 ** bit for bit in range(num_bits)]


def build_knapsack_cqm(
    items: List[Dict[str, Any]],
    capacity: float,
    penalty: float,
) -> Tuple[ConstrainedQuadraticModel, List[str]]:
    """
    建立背包問題 CQM。

    - 目標函數：BinaryQuadraticModel(vartype='BINARY')，最大化 value
    - 約束條件：QuadraticModel()，weight 總和 <= capacity
    - 使用 slack variables 將不等式轉為等式
    - 懲罰項：P × (constraint)²

    Returns:
        (cqm, item_vars) — CQM 模型與物品變數名稱列表
    """
    if capacity <= 0:
        raise ValueError("capacity 必須大於 0")
    if penalty <= 0:
        raise ValueError("penalty 必須大於 0")
    if not float(capacity).is_integer():
        raise ValueError("capacity 目前需為整數，才能使用 slack variables 編碼")

    cap_int = int(capacity)
    item_vars = [f"x_{idx}" for idx in range(len(items))]
    slack_coeffs = _build_slack_coefficients(cap_int)
    slack_vars = [f"s_{idx}" for idx in range(len(slack_coeffs))]

    # 目標函數（最小化負 value）
    objective = BinaryQuadraticModel(vartype="BINARY")
    for var_name, item in zip(item_vars, items):
        objective.add_linear(var_name, -float(item["value"]))

    # 懲罰項展開加入目標函數
    coeff_by_var: Dict[str, float] = {
        **{v: float(item["weight"]) for v, item in zip(item_vars, items)},
        **{v: float(c) for v, c in zip(slack_vars, slack_coeffs)},
    }
    variables = list(coeff_by_var.items())
    for var_name, coeff in variables:
        objective.add_linear(var_name, penalty * (coeff ** 2 - 2 * cap_int * coeff))
    for (u, cu), (v, cv) in combinations(variables, 2):
        objective.add_quadratic(u, v, penalty * 2 * cu * cv)

    # 約束條件（等式形式）
    constraint = QuadraticModel()
    for var_name, item in zip(item_vars, items):
        constraint.add_variable("BINARY", var_name)
        constraint.add_linear(var_name, float(item["weight"]))
    for var_name, coeff in zip(slack_vars, slack_coeffs):
        constraint.add_variable("BINARY", var_name)
        constraint.add_linear(var_name, float(coeff))

    cqm = ConstrainedQuadraticModel()
    cqm.set_objective(objective)
    cqm.add_constraint(constraint == cap_int, label="capacity_with_slack")

    return cqm, item_vars
