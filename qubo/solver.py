"""AEQTS 量子啟發演化求解器"""
import numpy as np
import time
from math import sqrt
from typing import Callable, Dict, Any, Optional


# ─────────────────────────────────────────────
#  內部輔助
# ─────────────────────────────────────────────

def _measure(qindividuals: np.ndarray) -> np.ndarray:
    """量子測量：將 Q-bit 族群崩縮為 0/1 古典解。"""
    probs = qindividuals[:, 1] ** 2          # |β|² 為 |1⟩ 的機率
    return (np.random.rand(len(qindividuals)) < probs).astype(float)


def _gen_nbrs(qindividuals: np.ndarray, N: int):
    """生成 N 個鄰居解（量子測量 N 次）。"""
    return [_measure(qindividuals) for _ in range(N)]


def _evaluate(neighbours, Q: np.ndarray):
    """對所有鄰居計算 QUBO 能量並由低到高排序。"""
    energies = [float(s @ Q @ s) for s in neighbours]
    return [neighbours[i] for i in np.argsort(energies)]


def _update_qbits(neighbours, qindividuals: np.ndarray, N: int) -> np.ndarray:
    """
    量子旋轉更新：根據最佳/最差鄰居調整 Q-bit 轉動角。

    使用排名加權旋轉角（rank-based theta），sign 依 α·β 決定方向。
    """
    num_pairs = N // 2
    best_group  = np.array(neighbours[:num_pairs])
    worst_group = np.array(neighbours[N - 1: N - num_pairs - 1: -1])

    diff     = best_group - worst_group                        # (num_pairs, n)
    ranks    = np.arange(1, num_pairs + 1).reshape(-1, 1)
    thetas   = np.sum(diff * (0.01 * np.pi / ranks), axis=0)  # (n,)

    alpha, beta = qindividuals[:, 0], qindividuals[:, 1]
    sign   = np.where((alpha * beta) < 0, -1.0, 1.0)
    thetas = thetas * sign

    c, s = np.cos(thetas), np.sin(thetas)
    qindividuals[:, 0] = alpha * c - beta * s
    qindividuals[:, 1] = alpha * s + beta * c
    return qindividuals


def _entropy(qindividuals: np.ndarray) -> float:
    """計算 Q-bit 族群平均 von Neumann entropy。"""
    p1 = np.clip(qindividuals[:, 1] ** 2, 1e-9, 1 - 1e-9)
    p0 = 1.0 - p1
    return float(-np.mean(p0 * np.log2(p0) + p1 * np.log2(p1)))


# ─────────────────────────────────────────────
#  主求解器
# ─────────────────────────────────────────────

def aeqts_solver(
    Q: np.ndarray,
    num_iterations: int = 1000,
    N: int = 50,
    seed: Optional[int] = None,
    feasibility_checker: Optional[Callable[[np.ndarray], bool]] = None,
    objective_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> Dict[str, Any]:
    """
    AEQTS（Adaptive Evolutionary Quantum-inspired Tabu Search）求解器。

    Args:
        Q            : QUBO 矩陣 (n×n，上三角或對稱皆可)
        num_iterations: 演化迭代次數
        N            : 每代鄰居數（對應原始碼中的 N_list）
        seed         : 隨機種子
        feasibility_checker: f(x) → bool，判斷解是否滿足約束
        objective_fn : f(x) → float，計算真實目標値（如背包總價値）。
                       若為 None 則以 QUBO 能量代替。

    Returns:
        {
          "solution"          : 最優 0/1 解,
          "energy"            : 最低 QUBO 能量,
          "history"           : 每個 checkpoint 的能量紀錄,
          "computation_time_ms": 計算時間 (ms)
        }
    """
    if seed is not None:
        np.random.seed(seed)

    start_time = time.time()
    n = Q.shape[0]

    # ── 初始化 Q-bits：α = β = 1/√2 ──────────────────────────
    qindividuals = np.full((n, 2), 1.0 / sqrt(2))

    # ── 初始鄰居評估 ─────────────────────────────────────────
    init_nbrs   = _gen_nbrs(qindividuals, N)
    init_sorted = _evaluate(init_nbrs, Q)
    global_best_sol    = init_sorted[0].copy()
    global_best_energy = float(global_best_sol @ Q @ global_best_sol)

    # ── 歷史記錄間隔 ─────────────────────────────────────────
    record_interval = max(1, num_iterations // 100)
    history = []

    # ── 主迭代 ───────────────────────────────────────────────
    for it in range(num_iterations):
        nbrs         = _gen_nbrs(qindividuals, N)
        nbrs_sorted  = _evaluate(nbrs, Q)
        current_best = nbrs_sorted[0]
        current_energy = float(current_best @ Q @ current_best)

        if current_energy < global_best_energy:
            global_best_energy = current_energy
            global_best_sol    = current_best.copy()

        qindividuals = _update_qbits(nbrs_sorted, qindividuals, N)

        if it % record_interval == 0:
            is_feasible = bool(feasibility_checker(global_best_sol)) if feasibility_checker else None
            obj_val = float(objective_fn(global_best_sol)) if objective_fn else global_best_energy
            history.append({
                "iteration"  : it,
                "energy"     : global_best_energy,
                "objective"  : obj_val,
                "entropy"    : _entropy(qindividuals),
                "is_feasible": is_feasible,
            })

    # ── 最後一次記錄 ─────────────────────────────────────────
    is_feasible_final = bool(feasibility_checker(global_best_sol)) if feasibility_checker else None
    obj_val_final = float(objective_fn(global_best_sol)) if objective_fn else global_best_energy
    history.append({
        "iteration"  : num_iterations,
        "energy"     : global_best_energy,
        "objective"  : obj_val_final,
        "entropy"    : _entropy(qindividuals),
        "is_feasible": is_feasible_final,
    })

    computation_time_ms = (time.time() - start_time) * 1000

    return {
        "solution"           : global_best_sol.astype(int).tolist(),
        "energy"             : global_best_energy,
        "history"            : history,
        "computation_time_ms": round(computation_time_ms, 2),
    }


# ─────────────────────────────────────────────
#  保留 SA（供 multi_run_solver 使用）
# ─────────────────────────────────────────────

def simulated_annealing_solver(
    Q: np.ndarray,
    num_iterations: int = 10000,
    T_start: float = 10.0,
    T_end: float = 0.01,
    seed: int = None,
    feasibility_checker: Optional[Callable[[np.ndarray], bool]] = None,
) -> Dict[str, Any]:
    """模擬退火求解器（備用）。"""
    from .builder import calculate_energy

    if seed is not None:
        np.random.seed(seed)

    start_time = time.time()
    n = Q.shape[0]

    x_current = np.random.randint(0, 2, size=n).astype(float)
    E_current = calculate_energy(x_current, Q)
    x_best, E_best = x_current.copy(), E_current

    record_interval = max(1, num_iterations // 100)
    alpha = (T_end / T_start) ** (1.0 / num_iterations)
    T = T_start
    history = []

    for iteration in range(num_iterations):
        flip_idx = np.random.randint(0, n)
        x_new = x_current.copy()
        x_new[flip_idx] = 1 - x_new[flip_idx]

        delta_E = calculate_energy_flip(x_current, Q, flip_idx)
        E_new = E_current + delta_E

        if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
            x_current, E_current = x_new, E_new

        if E_current < E_best:
            x_best, E_best = x_current.copy(), E_current

        if iteration % record_interval == 0:
            is_feasible = bool(feasibility_checker(x_best)) if feasibility_checker else None
            history.append({"iteration": iteration, "energy": float(E_best),
                            "temperature": float(T), "is_feasible": is_feasible})
        T *= alpha

    is_feasible_final = bool(feasibility_checker(x_best)) if feasibility_checker else None
    history.append({"iteration": num_iterations, "energy": float(E_best),
                    "temperature": float(T), "is_feasible": is_feasible_final})

    return {
        "solution": x_best.astype(int).tolist(),
        "energy": float(E_best),
        "history": history,
        "computation_time_ms": round((time.time() - start_time) * 1000, 2),
    }


def calculate_energy_flip(x: np.ndarray, Q: np.ndarray, flip_idx: int) -> float:
    """計算翻轉單個比特後的能量變化（優化版）。"""
    i = flip_idx
    n = len(x)
    delta = Q[i, i]
    for j in range(n):
        if j != i:
            delta += (Q[i, j] if j > i else Q[j, i]) * x[j]
    delta *= 2
    if x[i] == 1:
        delta = -delta
    return delta
