"""QUBO 矩陣構建器 - 將問題轉換為 Q 矩陣"""
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
    Q = np.zeros((n, n))
    
    # 目標：最大化價值 → 最小化負價值
    for i in range(n):
        Q[i, i] -= items[i]["value"]
    
    # 約束：(Σ weight_i * x_i - max_weight)^2
    # 展開：Σ w_i^2 * x_i + 2 * Σ Σ (i<j) w_i * w_j * x_i * x_j - 2 * max_weight * Σ w_i * x_i + max_weight^2
    for i in range(n):
        w_i = items[i]["weight"]
        # 對角線項：w_i^2 * x_i - 2 * max_weight * w_i * x_i
        Q[i, i] += penalty * (w_i ** 2 - 2 * max_weight * w_i)
        
        # 非對角線項：2 * w_i * w_j * x_i * x_j
        for j in range(i + 1, n):
            w_j = items[j]["weight"]
            Q[i, j] += penalty * 2 * w_i * w_j
    
    # 常數項 (penalty * max_weight^2) 不影響最優解，可忽略
    
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
    edges = data.get("edges")
    if edges is None:
        raise ValueError("max_cut 問題缺少必要欄位: edges")
    
    Q = np.zeros((n_nodes, n_nodes))
    
    for edge in edges:
        i, j = edge["from"], edge["to"]
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
    
    Q = np.array(Q_matrix, dtype=float)
    
    # 確保是方陣
    if Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q_matrix 必須是方陣，當前形狀: {Q.shape}")
    
    # 轉換為上三角矩陣（QUBO 標準形式）
    Q = np.triu(Q) + np.triu(Q, k=1).T
    Q = np.triu(Q)
    
    return Q


def calculate_energy(x: np.ndarray, Q: np.ndarray) -> float:
    """
    計算給定解的能量值。
    
    E(x) = x^T * Q * x
    
    Args:
        x: 解向量 (0/1)
        Q: QUBO 矩陣
    
    Returns:
        energy: 能量值
    """
    return float(x.T @ Q @ x)
