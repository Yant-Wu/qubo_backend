# worker.py — 後台任務執行器（單一職責：任務狀態流轉與真實運算）
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from sqlalchemy.orm import Session

from database import Job, JobHistory, SessionLocal
from schemas import HistoryPoint
from qubo import build_qubo_matrix, aeqts_solver
from qubo.solver import cuda_knapsack_solver, is_cuda_available

# Job 的 problem_type 欄位（首字大寫）→ QUBO builder 格式（小寫/底線）
_PROBLEM_TYPE_MAP: Dict[str, str] = {
    "knapsack": "knapsack",
    "Knapsack": "knapsack",
    "maxcut": "max_cut",
    "MaxCut": "max_cut",
    "max_cut": "max_cut",
    "custom": "custom",
}


def process_pending_jobs():
    """
    背景任務：掃描所有 pending jobs，更新狀態流轉。
    
    流程：pending → running → 模擬運算（寫入 history）→ completed/failed
    """
    db = SessionLocal()
    try:
        # 查找所有 pending 任務
        pending_jobs = db.query(Job).filter(Job.status == "pending").all()
        
        for job in pending_jobs:
            # 1. 標記為 running
            job.status = "running"
            db.commit()
            print(f"[worker] Job {job.id} → running")
            
            # 2. 模擬運算：生成隨機迭代數據
            try:
                _simulate_job(db, job)
                
                # 3. 標記為 completed
                job.status = "completed"
                db.commit()
                print(f"[worker] Job {job.id} → completed")
                
            except Exception as e:
                # 4. 失敗：標記為 failed
                job.status = "failed"
                job.error_message = str(e)
                db.commit()
                print(f"[worker] Job {job.id} → failed: {e}")
    
    finally:
        db.close()


def _make_feasibility_checker(qubo_type: str, raw: Dict[str, Any]):
    """建立各問題類型的可行解判斷函式。"""
    if qubo_type == "knapsack":
        items = raw.get("items", [])
        max_weight = float(raw.get("max_weight", float("inf")))
        weights = [float(it["weight"]) for it in items]

        def check_knapsack(x) -> bool:
            return float(sum(w * int(xi) for w, xi in zip(weights, x))) <= max_weight

        return check_knapsack

    if qubo_type == "max_cut":
        # Max-cut 無不等式約束，所有解皆可行
        return lambda x: True

    return None


def _make_objective_fn(qubo_type: str, raw: Dict[str, Any]):
    """建立各問題類型的真實目標値函式（用於圖表顯示）。"""
    if qubo_type == "knapsack":
        items = raw.get("items", [])
        values = [float(it["value"]) for it in items]
        # 對應 test.py： np.sum(global_best_sol * values)
        def knapsack_objective(x) -> float:
            return float(sum(v * int(xi) for v, xi in zip(values, x)))
        return knapsack_objective

    if qubo_type == "max_cut":
        # Max-cut 使用 QUBO energy 即為切割目標（negated），直接回傳 None 讓 solver 以 energy 替代
        return None

    return None


def _simulate_job(db: Session, job: Job):
    """
    真實任務運算：根據 problem_type、problem_data 建立 QUBO 矩陣並執行 AEQTS 求解。

    流程：
      1. 將 problem_data 展開成實際問題資料（支援 generation_method: random）
      2. build_qubo_matrix() 建立 Q 矩陣
      3. aeqts_solver() 真實求解（附可行性追蹤）
      4. 將 energy + is_feasible 歷史寫入 JobHistory
      5. 更新 Job：computation_time_ms, t_start（=N）, t_end（=num_iterations）
    """
    # --- 1. 解析 problem_type ---
    qubo_type = _PROBLEM_TYPE_MAP.get(job.problem_type)
    if qubo_type is None:
        raise ValueError(
            f"problem_type '{job.problem_type}' 目前不支援真實計算"
        )

    # --- 2. 展開 problem_data，先提取使用者讀入的參數 ---
    raw: Dict[str, Any] = dict(job.problem_data) if job.problem_data else {}
    # 在 raw 被 generate_random_problem_data 替換前，先存下 AEQTS 參數
    user_num_iterations = raw.get("num_iterations")   # 前端傳入的 Iterations
    user_timeout        = raw.get("timeout_seconds")   # 前端傳入的執行時限（秒）
    # --- 3. 建立 QUBO 矩陣（非 knapsack 才需要）──────────────────────────────────
    # knapsack + CUDA binary 路徑：binary 自己建 QUBO，不需要 Python 建矩陣
    # 相容處理：前端可能存 capacity，builder 需要 max_weight
    if qubo_type == "knapsack" and "max_weight" not in raw and "capacity" in raw:
        raw["max_weight"] = raw["capacity"]

    Q = feasibility_checker = objective_fn = None
    if not (qubo_type == "knapsack" and is_cuda_available()):
        Q = build_qubo_matrix(problem_type=qubo_type, problem_data=raw)
        feasibility_checker = _make_feasibility_checker(qubo_type, raw)
        objective_fn = _make_objective_fn(qubo_type, raw)

    # --- 4. 決定求解參數 ---
    n_vars = len(raw.get("items", [])) if qubo_type == "knapsack" else (Q.shape[0] if Q is not None else 0)
    N = int(job.core_limit or 50)          # 鄰域大小（前端 Neighbors N）
    num_iterations = int(user_num_iterations) if user_num_iterations else max(1000, n_vars * 100)
    timeout_secs = float(user_timeout) if user_timeout else 30.0

    import time as _time
    run_start = _time.time()
    best_result = None
    best_energy = float("inf")
    run = 0

    # --- CUDA 路徑（knapsack + binary 可用）---
    use_cuda = qubo_type == "knapsack" and is_cuda_available()
    if use_cuda:
        items    = raw.get("items", [])
        _weights  = [float(it["weight"]) for it in items]
        _values   = [float(it["value"])  for it in items]
        _capacity = float(raw.get("max_weight") or raw.get("capacity", 0))
        _penalty  = float(raw.get("penalty", 10.0))
        while True:
            result = cuda_knapsack_solver(
                weights=_weights, values=_values,
                capacity=_capacity, penalty=_penalty,
                N=N, num_iterations=num_iterations, seed=run,
            )
            run += 1
            if result["energy"] < best_energy:
                best_energy = result["energy"]
                best_result = result
            if (_time.time() - run_start) >= timeout_secs:
                break
    else:
        # --- Python fallback（非 knapsack 或無 binary）---
        while True:
            result = aeqts_solver(
                Q=Q,
                num_iterations=num_iterations,
                N=N,
                seed=run,
                feasibility_checker=feasibility_checker,
                objective_fn=objective_fn,
            )
            run += 1
            if result["energy"] < best_energy:
                best_energy = result["energy"]
                best_result = result
            if (_time.time() - run_start) >= timeout_secs:
                break

    # --- 5. 將 objective + entropy + is_feasible 歷史寫入 JobHistory ---
    for point in best_result["history"]:
        db.add(JobHistory(
            job_id=job.id,
            iteration=point["iteration"],
            value=round(point["objective"], 6),        # 真實目標值（背包：總價值，max_cut：切割值）
            qubo_energy=round(point["current_energy"], 6) if point.get("current_energy") is not None else None,
            entropy=round(point["entropy"], 6) if point.get("entropy") is not None else None,
            is_feasible=point.get("is_feasible"),
        ))

    # --- 6. 更新 Job 的求解統計欄位 ---
    job.computation_time_ms = round(best_result["computation_time_ms"], 2)
    job.t_start = float(N)              # 儲存鄰域大小 N
    job.t_end = float(num_iterations)   # 儲存迭代次數
    job.compute_device = "gpu" if best_result.get("device") in ("gpu", "cuda") else "cpu"
    db.commit()
    return best_result

