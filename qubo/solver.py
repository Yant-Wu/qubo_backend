"""AEQTS 求解器：優先呼叫 CUDA 二進位檔（aeqts.cu 編譯產生），
無 CUDA binary 時自動 fallback 至純 Python 實作。"""
import json
import os
import shutil
import subprocess
import time
from math import sqrt
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# ══════════════════════════════════════════════════════════════════
#  CUDA binary 封裝層
# ══════════════════════════════════════════════════════════════════

_BINARY_NAME = "solve_cuda"

def _find_binary() -> Optional[str]:
    """搜尋編譯完成的 solve_cuda 二進位檔（開發機 / Docker 兩種路徑）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", _BINARY_NAME),   # backend/solve_cuda（開發）
        os.path.join("/app", _BINARY_NAME),        # Docker /app/solve_cuda
        shutil.which(_BINARY_NAME) or "",          # PATH
    ]
    for p in candidates:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return os.path.abspath(p)
    return None


def is_cuda_available() -> bool:
    """回傳 True 代表 solve_cuda binary 存在且可執行。"""
    return _find_binary() is not None


def cuda_knapsack_solver(
    weights: List[float],
    values: List[float],
    capacity: float,
    penalty: float,
    N: int = 50,
    num_iterations: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    呼叫 CUDA binary（solve_cuda）執行 knapsack AEQTS 求解。

    Returns:
        同 aeqts_solver 相同結構的 dict：
        {solution, energy, history, computation_time_ms, device}
    """
    binary = _find_binary()
    if binary is None:
        raise RuntimeError(
            "solve_cuda binary not found. "
            "Build aeqts.cu first: nvcc -O3 -std=c++17 aeqts.cu -o solve_cuda"
        )

    cmd = [
        binary,
        "--weights",    ",".join(f"{float(w)}" for w in weights),
        "--values",     ",".join(f"{float(v)}" for v in values),
        "--capacity",   str(float(capacity)),
        "--penalty",    str(float(penalty)),
        "--population", str(int(N)),
        "--iterations", str(int(num_iterations)),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(
            f"solve_cuda exited with code {proc.returncode}:\n{proc.stderr}"
        )

    raw = json.loads(proc.stdout)

    # 統一輸出格式（solver.py history 欄位使用 "objective" key，router/worker 讀取相同）
    return {
        "solution"            : raw["solution"],
        "energy"              : raw["energy"],
        "history"             : raw["history"],   # [{iteration, energy, current_energy, objective, entropy, is_feasible}]
        "computation_time_ms" : raw["computation_time_ms"],
        "device"              : "cuda",
    }


# ══════════════════════════════════════════════════════════════════
#  Python fallback（GPU 自動偵測 CuPy / 退回 numpy）
# ══════════════════════════════════════════════════════════════════

# ── CuPy 自動偵測 ────────────────────────────────────────────────
try:
    import cupy as cp
    _GPU_AVAILABLE: bool = cp.cuda.is_available()
except Exception:
    cp = None
    _GPU_AVAILABLE = False


def _xp(use_gpu: bool):
    """回傳 cupy 或 numpy 模組。"""
    return cp if (use_gpu and _GPU_AVAILABLE) else np


def _to_np(arr, xp) -> np.ndarray:
    """GPU array → CPU numpy（CPU 時直接 asarray）。"""
    return arr.get() if xp is not np else np.asarray(arr)


def _entropy(alpha, beta, xp) -> float:
    """計算 Q-bit 族群平均 von Neumann entropy。"""
    p0 = alpha ** 2
    p1 = beta  ** 2
    p0_s = xp.where(p0 == 0, 1e-9, p0)
    p1_s = xp.where(p1 == 0, 1e-9, p1)
    ent  = -(p0 * xp.log2(p0_s) + p1 * xp.log2(p1_s))
    ent  = xp.where((p0 == 0) | (p1 == 0), 0.0, ent)
    return float(xp.mean(ent))


def _gen_nbrs(beta, N: int, xp):
    """
    向量化生成 N 個鄰居解，形狀 (N, n)。
    對應 test.cu gen_neighbours_kernel：機率 = |β|²。
    """
    probs = beta ** 2                              # (n,)
    rnd   = xp.random.rand(N, beta.shape[0])       # (N, n)
    return (rnd < probs[xp.newaxis, :]).astype(xp.float64)


def _evaluate(neighbours, Q_dev, xp):
    """
    向量化計算 N 個鄰居的 QUBO 能量，返回排序索引與能量陣列。
    energy[i] = neigh[i] @ Q @ neigh[i]
    對應 test.cu energy_kernel + Thrust sort_by_key。
    """
    energies = (neighbours @ Q_dev * neighbours).sum(axis=1)  # (N,)
    return xp.argsort(energies), energies


def _update_qbits(neighbours, sorted_idx, N: int,
                  alpha, beta, theta_scale: float, xp):
    """
    量子旋轉更新（rank-based theta）。
    對應 test.cu updateQ_pairs_kernel。
    """
    num_pairs  = N // 2
    ranks      = xp.arange(1, num_pairs + 1, dtype=xp.float64).reshape(-1, 1)
    base_theta = theta_scale * xp.pi / ranks             # (num_pairs, 1)

    best_idx  = sorted_idx[:num_pairs]
    worst_idx = sorted_idx[N - 1: N - num_pairs - 1: -1]

    diff = neighbours[best_idx] - neighbours[worst_idx]  # (num_pairs, n)
    raw  = (diff * base_theta).sum(axis=0)               # (n,)

    sign   = xp.where(alpha * beta < 0, -1.0, 1.0)
    thetas = raw * sign
    c, s   = xp.cos(thetas), xp.sin(thetas)
    return alpha * c - beta * s, alpha * s + beta * c


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
    use_gpu: bool = True,
) -> Dict[str, Any]:
    """
    AEQTS（Adaptive Evolutionary Quantum-inspired Tabu Search）求解器。
    自動偵測 CUDA GPU，無 GPU 時 fallback 至 CPU numpy。

    Args:
        Q             : QUBO 矩陣 (n×n，對稱)
        num_iterations: 最大迭代次數
        N             : 每代鄰居數（對應原始碼的 N_list）
        seed          : 隨機種子
        feasibility_checker: f(x) → bool，判斷解是否滿足約束
        objective_fn  : f(x) → float，計算真實目標値；None 則用 QUBO 能量
        use_gpu       : True = 嘗試 GPU（無 GPU 自動 fallback CPU）

    Returns:
        {
          "solution"           : 最優 0/1 解,
          "energy"             : 最低 QUBO 能量,
          "history"            : 每個 checkpoint 的能量紀錄,
          "computation_time_ms": 計算時間 (ms),
          "device"             : "gpu" 或 "cpu"
        }
    """
    xp     = _xp(use_gpu)
    device = "gpu" if (xp is not np) else "cpu"

    if seed is not None:
        np.random.seed(seed)
        if xp is not np:
            xp.random.seed(seed)

    start_time = time.time()

    # theta 同原始腳本：0.01 ~ 0.10，每次 run 隨機抽一個
    _theta_list   = np.round(np.arange(0.01, 0.11, 0.01), 2).tolist()
    current_theta = float(np.random.choice(_theta_list))

    Q_dev = xp.asarray(Q)
    n     = Q_dev.shape[0]

    # ── 初始化 Q-bits：α = β = 1/√2 ──────────────────────────
    val   = 1.0 / sqrt(2)
    alpha = xp.full(n, val, dtype=xp.float64)
    beta  = xp.full(n, val, dtype=xp.float64)

    # ── 初始鄰居評估 ─────────────────────────────────────────
    nbrs_init    = _gen_nbrs(beta, N, xp)
    sidx_init, _ = _evaluate(nbrs_init, Q_dev, xp)
    best_sol     = nbrs_init[sidx_init[0]].copy()
    best_energy  = float((best_sol @ Q_dev) @ best_sol)

    # ── 歷史記錄間隔 ─────────────────────────────────────────
    record_interval   = max(1, num_iterations // 100)
    history: list     = []
    ENTROPY_THRESHOLD = 0.02

    # ── 主迭代 ───────────────────────────────────────────────
    for it in range(num_iterations):
        current_entropy = _entropy(alpha, beta, xp)

        nbrs      = _gen_nbrs(beta, N, xp)
        sidx, eng = _evaluate(nbrs, Q_dev, xp)
        this_energy = float(eng[sidx[0]])

        if this_energy < best_energy:
            best_energy = this_energy
            best_sol    = nbrs[sidx[0]].copy()

        alpha, beta = _update_qbits(nbrs, sidx, N, alpha, beta, current_theta, xp)

        if it % record_interval == 0:
            sol_np = _to_np(best_sol, xp)
            is_feasible = bool(feasibility_checker(sol_np)) if feasibility_checker else None
            obj_val     = float(objective_fn(sol_np)) if objective_fn else best_energy
            history.append({
                "iteration"     : it,
                "energy"        : best_energy,    # 目前最佳 QUBO 能量（running best）
                "current_energy": this_energy,    # 當前迭代的最佳候選能量（每次都變）
                "objective"     : obj_val,
                "entropy"       : current_entropy,
                "is_feasible"   : is_feasible,
            })

        # entropy 提前停止（同原始腳本）
        if current_entropy <= ENTROPY_THRESHOLD:
            break

    # ── 最後一次記錄 ─────────────────────────────────────────
    sol_np_final      = _to_np(best_sol, xp)
    is_feasible_final = bool(feasibility_checker(sol_np_final)) if feasibility_checker else None
    obj_val_final     = float(objective_fn(sol_np_final)) if objective_fn else best_energy
    history.append({
        "iteration"     : num_iterations,
        "energy"        : best_energy,
        "current_energy": best_energy,   # 結束時 current = best
        "objective"     : obj_val_final,
        "entropy"       : _entropy(alpha, beta, xp),
        "is_feasible"   : is_feasible_final,
    })

    computation_time_ms = (time.time() - start_time) * 1000

    return {
        "solution"           : np.asarray(sol_np_final).astype(int).tolist(),
        "energy"             : best_energy,
        "history"            : history,
        "computation_time_ms": round(computation_time_ms, 2),
        "device"             : device,
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
