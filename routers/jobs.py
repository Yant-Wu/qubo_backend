# routers/jobs.py — 完整的 REST API 路由（單一職責：HTTP 層）
from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

# 最多允許 10 個求解任務並行（CPU/GPU bound，各佔一個 thread）
_solve_executor = ThreadPoolExecutor(max_workers=10)

import store
from database import get_db
from schemas import (
    ApiErrorResponse,
    ApiResponse,
    HistoryPointCreate,
    JobCreateRequest,
    JobDetail,
    JobListItem,
    SolveAndCreateResponse,
    StatusUpdate,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


# ===== 1. GET /api/jobs =====
@router.get("", response_model=ApiResponse[List[JobListItem]])
async def list_jobs(
    algorithm: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """列出所有任務（新到舊）。可選過濾 solver_backend。"""
    jobs = store.list_jobs(db, solver_backend=algorithm)
    items = [
        JobListItem(
            id=j.id,
            task_name=j.task_name,
            status=j.status,
            created_at=j.created_at,
        )
        for j in jobs
    ]
    return ApiResponse(data=items, total=len(items))


# ===== 2. POST /api/jobs =====
@router.post("", response_model=ApiResponse[JobDetail], status_code=201)
async def create_job(req: JobCreateRequest, db: Session = Depends(get_db)):
    """建立新任務。初始狀態為 pending，後續由 worker 更新。"""
    job_id = str(uuid.uuid4())
    job = store.create_job(db, req, job_id)
    return ApiResponse(data=job, message="Job created successfully")


# ===== 3. GET /api/jobs/{id} =====
@router.get("/{job_id}", response_model=ApiResponse[JobDetail])
async def get_job(job_id: str, db: Session = Depends(get_db)):
    """獲取單一任務的完整詳情（包括歷史資料）。"""
    job = store.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return ApiResponse(data=job)


# ===== 4. DELETE /api/jobs/{id} =====
@router.delete("/{job_id}", status_code=200)
async def delete_job(job_id: str, db: Session = Depends(get_db)):
    """刪除任務及其歷史記錄。"""
    if not store.delete_job(db, job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return ApiResponse(data=None, message=f"Job '{job_id}' deleted successfully")


# ===== 5. PATCH /api/jobs/{id}/status =====
@router.patch("/{job_id}/status", response_model=ApiResponse[JobDetail])
async def update_job_status(
    job_id: str,
    req: StatusUpdate,
    db: Session = Depends(get_db),
):
    """更新任務狀態（pending|running|completed|failed）。"""
    job = store.update_job_status(db, job_id, req.status, req.error_message)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return ApiResponse(data=job, message=f"Job status updated to '{req.status}'")


# ===== 6. POST /api/jobs/{id}/history =====
@router.post("/{job_id}/history", response_model=ApiResponse[dict], status_code=201)
async def add_history(
    job_id: str,
    req: HistoryPointCreate,
    db: Session = Depends(get_db),
):
    """為任務新增歷史資料點。"""
    # 先檢查任務是否存在
    job = store.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    # 新增歷史點
    if not store.add_history_points(db, job_id, req.points):
        raise HTTPException(status_code=500, detail="Failed to add history points")
    
    return ApiResponse(
        data={"added": len(req.points)},
        message=f"Added {len(req.points)} history point(s)",
    )


def _blocking_solve(job_id: str, job_data: dict) -> dict:
    """
    在獨立 thread 執行（thread-safe）：
    - 建立自己的 DB session（避免跨 thread 共用 SQLAlchemy session）
    - 執行 AEQTS 求解
    - 回傳 best_result dict
    """
    from database import Job as JobModel, SessionLocal
    from worker import _simulate_job

    db = SessionLocal()
    try:
        job_orm = db.query(JobModel).filter(JobModel.id == job_id).first()
        best_result = _simulate_job(db, job_orm)
        job_orm.status = "completed"
        db.commit()
        return best_result
    except Exception as exc:
        job_orm = db.query(JobModel).filter(JobModel.id == job_id).first()
        if job_orm:
            job_orm.status = "failed"
            job_orm.error_message = str(exc)
            db.commit()
        raise
    finally:
        db.close()


# ===== 7. POST /api/jobs/solve =====
@router.post("/solve", response_model=ApiResponse[SolveAndCreateResponse], status_code=201)
async def solve_and_create(req: JobCreateRequest, db: Session = Depends(get_db)):
    """
    統一求解端點（非同步執行）：建立 Job → 丟入 thread pool → AEQTS 求解 → 回傳結果。
    使用 run_in_executor 讓 CPU/GPU 密集運算在獨立 thread 執行，
    event loop 不阻塞，最多支援 10 個任務同時執行。
    """
    from database import Job as JobModel

    job_id = str(uuid.uuid4())
    job_orm = JobModel(
        id=job_id,
        task_name=req.task_name,
        problem_type=req.problem_type,
        n_variables=req.n_variables,
        solver_backend=req.solver_backend,
        core_limit=req.core_limit,
        problem_data=req.problem_data.model_dump(),
        status="running",
    )
    db.add(job_orm)
    db.commit()
    db.close()   # 主 session 用完即關，thread 會自建 session

    try:
        loop = asyncio.get_event_loop()
        best_result = await loop.run_in_executor(
            _solve_executor,
            _blocking_solve,
            job_id,
            req.problem_data.model_dump(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"求解失敗: {exc}") from exc

    # 建立 interpretation（selected_items / total_value / total_weight）
    raw_items = req.problem_data.items or []
    solution = best_result.get("solution", [])
    selected = [
        {"name": it.name, "weight": it.weight, "value": it.value}
        for it, xi in zip(raw_items, solution)
        if xi
    ]
    total_value  = sum(float(it["value"])  for it in selected)
    total_weight = sum(float(it["weight"]) for it in selected)

    # 將求解結果存回資料庫（重新取 session，因原 session 已關閉）
    from database import Job as _Job, SessionLocal
    _db2 = SessionLocal()
    try:
        _job = _db2.query(_Job).filter(_Job.id == job_id).first()
        if _job:
            _pd = dict(_job.problem_data or {})
            _pd["selected_items"] = selected
            _pd["total_value"]    = total_value
            _pd["total_weight"]   = total_weight
            _job.problem_data = _pd
            flag_modified(_job, "problem_data")
            _db2.commit()
    finally:
        _db2.close()

    return ApiResponse(
        data=SolveAndCreateResponse(
            job_id=job_id,
            energy=best_result["energy"],
            selected_items=selected,
            total_value=total_value,
            total_weight=total_weight,
            computation_time_ms=best_result["computation_time_ms"],
            device=best_result.get("device", "cpu"),
        ),
        message="Job solved and created successfully",
    )
