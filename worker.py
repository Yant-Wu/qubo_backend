# worker.py — 後台任務執行器（單一職責：任務狀態流轉與真實運算）
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from database import Job, JobHistory, SessionLocal
from schemas import HistoryPoint
from qubo import build_qubo_matrix, aeqts_solver
from qubo.solver import cuda_knapsack_solver, is_cuda_available

_PROBLEM_TYPE_MAP: Dict[str, str] = {
    "knapsack": "knapsack", "Knapsack": "knapsack",
    "maxcut": "max_cut", "MaxCut": "max_cut", "max_cut": "max_cut",
    "custom": "custom",
}

def process_pending_jobs():
    db = SessionLocal()
    try:
        pending_jobs = db.query(Job).filter(Job.status == "pending").all()
        for job in pending_jobs:
            job.status = "running"
            db.commit()
            print(f"[worker] Job {job.id} → running")
            try:
                _simulate_job(db, job)
                job.status = "completed"
                db.commit()
                print(f"[worker] Job {job.id} → completed")
            except Exception as e:
                job.status = "failed"
                job.error_message = type(e).__name__
                db.commit()
                print(f"[worker] Job {job.id} → failed: {type(e).__name__}")
    finally:
        db.close()

def _make_feasibility_checker(qubo_type: str, raw: Dict[str, Any]):
    if qubo_type == "knapsack":
        items = raw.get("items", [])
        max_weight = float(raw.get("max_weight", float("inf")))
        weights = [float(it["weight"]) for it in items]
        def check_knapsack(x) -> bool:
            return float(sum(w * int(xi) for w, xi in zip(weights, x))) <= max_weight
        return check_knapsack
    if qubo_type == "max_cut": return lambda x: True
    return None

def _make_objective_fn(qubo_type: str, raw: Dict[str, Any]):
    if qubo_type == "knapsack":
        items = raw.get("items", [])
        values = [float(it["value"]) for it in items]
        def knapsack_objective(x) -> float:
            return float(sum(v * int(xi) for v, xi in zip(values, x)))
        return knapsack_objective
    return None

def _simulate_job(db: Session, job: Job):
    qubo_type = _PROBLEM_TYPE_MAP.get(job.problem_type)
    if qubo_type is None:
        raise ValueError(f"problem_type '{job.problem_type}' 目前不支援真實計算")

    raw: Dict[str, Any] = dict(job.problem_data) if job.problem_data else {}
    user_num_iterations = raw.get("num_iterations")
    user_timeout        = raw.get("timeout_seconds")

    if qubo_type == "knapsack" and "max_weight" not in raw and "capacity" in raw:
        raw["max_weight"] = raw["capacity"]

    Q = feasibility_checker = objective_fn = None
    if not (qubo_type == "knapsack" and is_cuda_available()):
        Q = build_qubo_matrix(problem_type=qubo_type, problem_data=raw)
        feasibility_checker = _make_feasibility_checker(qubo_type, raw)
        objective_fn = _make_objective_fn(qubo_type, raw)

    n_vars = Q.shape[0] if Q is not None else len(raw.get("items", []))
    N = int(job.core_limit or 50)
    num_iterations = int(user_num_iterations) if user_num_iterations else max(1000, n_vars * 100)
    timeout_secs = float(user_timeout) if user_timeout else 30.0

    import time as _time
    run_start = _time.time()
    best_result = None

    use_cuda = qubo_type == "knapsack" and is_cuda_available()
    
    # 決定要用哪個 solver generator
    solver_gen = None
    if use_cuda:
        items    = raw.get("items", [])
        _weights  = [float(it["weight"]) for it in items]
        _values   = [float(it["value"])  for it in items]
        _capacity = float(raw.get("max_weight") or raw.get("capacity", 0))
        _penalty  = float(raw.get("penalty", 10.0))
        solver_gen = cuda_knapsack_solver(
            weights=_weights, values=_values, capacity=_capacity, penalty=_penalty,
            N=N, num_iterations=num_iterations, seed=None, timeout=timeout_secs
        )
    else:
        solver_gen = aeqts_solver(
            Q=Q, num_iterations=num_iterations, N=N, seed=None,
            feasibility_checker=feasibility_checker, objective_fn=objective_fn,
        )

    # 💡 逐筆接收並即時存入 DB (Streaming)
    for data in solver_gen:
        if data.get("type") == "progress":
            db.add(JobHistory(
                job_id=job.id,
                iteration=data["iteration"],
                value=round(data["objective"], 6),
                qubo_energy=round(data.get("current_energy"), 6) if data.get("current_energy") is not None else None,
                entropy=round(data.get("entropy"), 6) if data.get("entropy") is not None else None,
                is_feasible=data.get("is_feasible"),
                qubit_probs=data.get("qubit_probs"),
            ))
            db.commit() # 每次都 commit 讓前端拉得到
            
        elif data.get("type") == "final":
            best_result = data

    if not best_result: return

    job.computation_time_ms = round(best_result["computation_time_ms"], 2)
    job.t_start = float(N)
    job.t_end = float(num_iterations)
    job.compute_device = "gpu" if best_result.get("device") in ("gpu", "cuda") else "cpu"

    if Q is not None:
        total_vars = int(Q.shape[0])
        n_items = len(raw.get("items", [])) if qubo_type == "knapsack" else 0
        n_slack = total_vars - n_items
    elif use_cuda and qubo_type == "knapsack":
        n_items = len(raw.get("items", []))
        _cap = float(raw.get("max_weight") or raw.get("capacity", 0))
        _auto_K = max(1, math.ceil(math.log2(_cap + 1))) if _cap > 0 else 1
        n_slack = int(raw.get("slack_bits") or _auto_K)
        total_vars = n_items + n_slack
    else:
        total_vars = n_slack = 0

    if total_vars > 0:
        job.n_variables = total_vars
        if n_slack > 0:
            _pd = dict(job.problem_data or {})
            _pd["n_slack"] = n_slack
            job.problem_data = _pd
            flag_modified(job, "problem_data")

    if qubo_type == "knapsack":
        items_list = raw.get("items", [])
        solution   = best_result.get("solution", [])
        selected   = [
            {"name": it["name"], "weight": it["weight"], "value": it["value"]}
            for it, xi in zip(items_list, solution) if xi
        ]
        _pd3 = dict(job.problem_data or {})
        _pd3["selected_items"] = selected
        _pd3["total_value"]    = round(sum(float(s["value"])  for s in selected), 6)
        _pd3["total_weight"]   = round(sum(float(s["weight"]) for s in selected), 6)
        job.problem_data = _pd3
        flag_modified(job, "problem_data")

    db.commit()
    return best_result