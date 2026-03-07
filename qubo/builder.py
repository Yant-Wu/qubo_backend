"""QUBO 矩陣構建器 - 將問題轉換為 Q 矩陣"""
import math
import numpy as np
from typing import Dict, List, Any


def build_qubo_matrix(problem_type: str, problem_data: Dict[str, Any]) -> np.ndarray:
    """
    根據問題類型和數據構建 QUBO 矩陣。
    
    Args:
        problem_type: 問題類型 (knapsack, max_cut, custom)
        problem_data: 問題具體數據
    
    Returns:
        Q: QUBO 矩陣 (上三角矩陣)
    """
    if problem_type == "knapsack":
        return build_knapsack_qubo(problem_data)
    elif problem_type == "max_cut":
        return build_max_cut_qubo(problem_data)
    elif problem_type == "custom":
        return build_custom_qubo(problem_data)
    else:
        raise ValueError(f"不支援的問題類型: {problem_type}")


def build_knapsack_qubo(data: Dict[str, Any]) -> np.ndarray:
    """
    背包問題 QUBO 構建。
    
    目標：最大化價值，受重量限制約束
    
    QUBO 形式：
    H = -Σ value_i * x_i + penalty * (Σ weight_i * x_i - max_weight)^2
    
    Args:
        data: {
            "items": [{"weight": float, "value": float}, ...],
            "max_weight": float,
            "penalty": float (懲罰係數，預設 10.0)
        }
    
    Returns:
        Q: QUBO 矩陣
    """
    items = data.get("items")
    if items is None:
        raise ValueError("knapsack 問題缺少必要欄位: items")
    max_weight = data.get("max_weight")
    if max_weight is None:
        raise ValueError("knapsack 問題缺少必要欄位: max_weight")
    penalty = data.get("penalty", 10.0)
    
    n = len(items)

    # Slack variables：將 Σw_i x_i ≤ C 轉成 Σw_i x_i + Σ2^k s_k = C
    # K 由使用者設定，未設定則自動推算 ⌈log2(C+1)⌉
    auto_K = max(1, math.ceil(math.log2(max_weight + 1))) if max_weight > 0 else 1
    K = int(data.get("slack_bits") or auto_K)
    if K < 1:
        raise ValueError("slack_bits 必須 ≥ 1")
    total = n + K
    Q = np.zeros((total, total))

    # 目標：最大化價值 → 最小化負價值（只作用在前 n 個物品變數）
    for i in range(n):
        Q[i, i] -= items[i]["value"]

    # 約束係數：物品用 weight，slack 用 2^k
    coeffs = [float(items[i]["weight"]) for i in range(n)] + [float(2 ** k) for k in range(K)]

    # 展開 penalty * (Σ coeffs[i]*y_i - C)^2
    for i in range(total):
        Q[i, i] += penalty * (coeffs[i] ** 2 - 2 * max_weight * coeffs[i])
        for j in range(i + 1, total):
            Q[i, j] += penalty * 2 * coeffs[i] * coeffs[j]

    # 常數項 (penalty * C^2) 不影響最優解，可忽略
    return Q


def build_max_cut_qubo(data: Dict[str, Any]) -> np.ndarray:
    """
    最大割問題 QUBO 構建。
    
    目標：最大化割邊權重總和
    
    QUBO 形式：
    H = -Σ (i,j)∈E  weight_ij * (x_i + x_j - 2*x_i*x_j)
      = -Σ (i,j)∈E  weight_ij * x_i - Σ (i,j)∈E  weight_ij * x_j + 2 * Σ (i,j)∈E  weight_ij * x_i * x_j
    
    Args:
        data: {
            "nodes": int (節點數量),
            "edges": [{"from": int, "to": int, "weight": float}, ...]
        }
    
    Returns:
        Q: QUBO 矩陣
    """
    n_nodes = data.get("nodes")
    if n_nodes is None:
        raise ValueError("max_cut 問題缺少必要欄位: nodes")
    if not isinstance(n_nodes, int) or n_nodes < 1:
        raise ValueError("nodes 必須為正整數")
    if n_nodes > 500:
        raise ValueError("nodes 不可超過 500")
    edges = data.get("edges")
    if edges is None:
        raise ValueError("max_cut 問題缺少必要欄位: edges")
    
    Q = np.zeros((n_nodes, n_nodes))
    
    for edge in edges:
        i, j = int(edge["from"]), int(edge["to"])
        if not (0 <= i < n_nodes and 0 <= j < n_nodes):
            raise ValueError(f"edge 索引 ({i}, {j}) 超出節點範圍 [0, {n_nodes})")
        weight = edge["weight"]
        
        # 確保 i < j（上三角矩陣）
        if i > j:
            i, j = j, i
        
        # H = -weight * (x_i + x_j - 2*x_i*x_j)
        # 對角線項：-weight * x_i, -weight * x_j
        Q[i, i] -= weight
        Q[j, j] -= weight
        
        # 非對角線項：2 * weight * x_i * x_j
        Q[i, j] += 2 * weight
    
    return Q


def build_custom_qubo(data: Dict[str, Any]) -> np.ndarray:
    """
    自定義 QUBO 矩陣。
    
    Args:
        data: {
            "Q_matrix": List[List[float]] (直接傳入 Q 矩陣)
        }
    
    Returns:
        Q: QUBO 矩陣
    """
    Q_matrix = data.get("Q_matrix")
    if Q_matrix is None:
        raise ValueError("custom 類型需要提供 Q_matrix")

    if not Q_matrix:
        raise ValueError("Q_matrix 不可為空")

    n = len(Q_matrix)
    for i, row in enumerate(Q_matrix):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"Q_matrix 第 {i} 行不是陣列")
        if len(row) != n:
            raise ValueError(f"Q_matrix 不是方陣：第 {i} 行長度 {len(row)} ≠ {n}")

    Q = np.array(Q_matrix, dtype=float)
    
    # 確保是方陣
    if Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q_matrix 必須是方陣，當前形狀: {Q.shape}")
    
    # 對稱化：若輸入只有上三角或只有下三角，補齊另一側後直接回傳完整對稱矩陣。
    # 對稱矩陣的 x^T Q x = Σ Q_ii x_i + 2*Σ_{i<j} Q_ij x_i x_j，
    # solver 的 x @ Q @ x 計算方式與此完全一致，不需縮減為上三角。
    Q = (Q + Q.T) / 2
    
    return Q

