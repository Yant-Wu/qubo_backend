"""AEQTS 求解器：優先呼叫 CUDA 二進位檔（aeqts.cu 編譯產生），
無 CUDA binary 時自動 fallback 至純 Python 實作。"""
import json
import os
import shutil
import subprocess
import time
from math import sqrt
from typing import Any, Callable, Dict, List, Optional, Generator

import numpy as np

# ══════════════════════════════════════════════════════════════════
#  CUDA binary 封裝層
# ══════════════════════════════════════════════════════════════════

_BINARY_NAME = "solve_cuda"

def _find_binary() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", _BINARY_NAME),
        os.path.join("/app", _BINARY_NAME),
        shutil.which(_BINARY_NAME) or "",
    ]
    for p in candidates:
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return os.path.abspath(p)
    return None

def is_cuda_available() -> bool:
    return _find_binary() is not None

def cuda_knapsack_solver(
    weights: List[float],
    values: List[float],
    capacity: float,
    penalty: float,
    N: int = 50,
    num_iterations: int = 1000,
    seed: Optional[int] = None,
    timeout: float = 30.0,
) -> Generator[Dict[str, Any], None, None]:
    binary = _find_binary()
    if binary is None:
        raise RuntimeError("solve_cuda binary not found. Build aeqts.cu first.")

    cmd = [
        binary,
        "--weights",    ",".join(f"{float(w)}" for w in weights),
        "--values",     ",".join(f"{float(v)}" for v in values),
        "--capacity",   str(float(capacity)),
        "--penalty",    str(float(penalty)),
        "--population", str(int(N)),
        "--iterations", str(int(num_iterations)),
        "--timeout",    str(float(timeout)),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])

    # 💡 使用 Popen 並且逐行讀取 stdout 達成即時回傳
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            yield data
        except json.JSONDecodeError:
            pass
            
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"solve_cuda exited with code {proc.returncode}")


# ══════════════════════════════════════════════════════════════════
#  Python fallback（GPU 自動偵測 CuPy / 退回 numpy）
# ══════════════════════════════════════════════════════════════════

try:
    import cupy as cp
    _GPU_AVAILABLE: bool = cp.cuda.is_available()
except Exception:
    cp = None
    _GPU_AVAILABLE = False

def _xp(use_gpu: bool):
    return cp if (use_gpu and _GPU_AVAILABLE) else np

def _to_np(arr, xp) -> np.ndarray:
    return arr.get() if xp is not np else np.asarray(arr)

def _entropy(alpha, beta, xp) -> float:
    p0, p1 = alpha ** 2, beta ** 2
    p0_s, p1_s = xp.where(p0 == 0, 1e-9, p0), xp.where(p1 == 0, 1e-9, p1)
    ent  = -(p0 * xp.log2(p0_s) + p1 * xp.log2(p1_s))
    ent  = xp.where((p0 == 0) | (p1 == 0), 0.0, ent)
    return float(xp.mean(ent))

def _gen_nbrs(beta, N: int, xp):
    probs = beta ** 2
    rnd   = xp.random.rand(N, beta.shape[0])
    return (rnd < probs[xp.newaxis, :]).astype(xp.float64)

def _evaluate(neighbours, Q_dev, xp):
    energies = (neighbours @ Q_dev * neighbours).sum(axis=1)
    return xp.argsort(energies), energies

def _update_qbits(neighbours, sorted_idx, N: int, alpha, beta, theta_scale: float, xp):
    num_pairs  = N // 2
    ranks      = xp.arange(1, num_pairs + 1, dtype=xp.float64).reshape(-1, 1)
    base_theta = theta_scale * xp.pi / ranks
    best_idx  = sorted_idx[:num_pairs]
    worst_idx = sorted_idx[N - 1: N - num_pairs - 1: -1]
    diff = neighbours[best_idx] - neighbours[worst_idx]
    raw  = (diff * base_theta).sum(axis=0)
    sign   = xp.where(alpha * beta < 0, -1.0, 1.0)
    thetas = raw * sign
    c, s   = xp.cos(thetas), xp.sin(thetas)
    return alpha * c - beta * s, alpha * s + beta * c

def aeqts_solver(
    Q: np.ndarray,
    num_iterations: int = 1000,
    N: int = 50,
    seed: Optional[int] = None,
    feasibility_checker: Optional[Callable[[np.ndarray], bool]] = None,
    objective_fn: Optional[Callable[[np.ndarray], float]] = None,
    use_gpu: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    
    xp     = _xp(use_gpu)
    device = "gpu" if (xp is not np) else "cpu"

    if seed is not None:
        np.random.seed(seed)
        if xp is not np: xp.random.seed(seed)

    start_time = time.time()
    _theta_list   = np.round(np.arange(0.01, 0.11, 0.01), 2).tolist()
    current_theta = float(np.random.choice(_theta_list))

    Q_dev = xp.asarray(Q)
    n     = Q_dev.shape[0]

    val   = 1.0 / sqrt(2)
    alpha, beta = xp.full(n, val, dtype=xp.float64), xp.full(n, val, dtype=xp.float64)

    nbrs_init    = _gen_nbrs(beta, N, xp)
    sidx_init, _ = _evaluate(nbrs_init, Q_dev, xp)
    best_sol     = nbrs_init[sidx_init[0]].copy()
    best_energy  = float((best_sol @ Q_dev) @ best_sol)

    record_interval   = max(1, num_iterations // 100)
    ENTROPY_THRESHOLD = 0.02

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
            qubit_probs = _to_np(beta ** 2, xp).tolist()
            
            # 💡 Python 版也改為即時 yield
            yield {
                "type"          : "progress",
                "iteration"     : it,
                "energy"        : best_energy,
                "current_energy": this_energy,
                "objective"     : obj_val,
                "entropy"       : current_entropy,
                "is_feasible"   : is_feasible,
                "qubit_probs"   : qubit_probs,
            }

        if current_entropy <= ENTROPY_THRESHOLD:
            break

    computation_time_ms = (time.time() - start_time) * 1000
    
    yield {
        "type"               : "final",
        "solution"           : np.asarray(_to_np(best_sol, xp)).astype(int).tolist(),
        "energy"             : best_energy,
        "computation_time_ms": round(computation_time_ms, 2),
        "device"             : device,
    }