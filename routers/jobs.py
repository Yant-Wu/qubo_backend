# routers/jobs.py — 完整的 REST API 路由（單一職責：HTTP 層）
from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

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


# ===== 7. POST /api/jobs/solve =====
@router.post("/solve", response_model=ApiResponse[SolveAndCreateResponse], status_code=201)
async def solve_and_create(req: JobCreateRequest, db: Session = Depends(get_db)):
    """
    統一求解端點（同步執行）：建立 Job → AEQTS 求解 → 儲存收斂歷史 → 回傳 job_id 與求解結果。
    前端只需呼叫一次，SolveResultPanel 與 QuboMonitorPanel 使用同一份資料。
    """
    from database import Job as JobModel
    from worker import _simulate_job

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
    db.refresh(job_orm)

    try:
        best_result = _simulate_job(db, job_orm)
        job_orm.status = "completed"
        db.commit()
    except Exception as exc:
        job_orm.status = "failed"
        job_orm.error_message = str(exc)
        db.commit()
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

    # 將求解結果存回資料庫（不需要新層資料表）
    pd = dict(job_orm.problem_data or {})
    pd["selected_items"] = selected
    pd["total_value"]    = total_value
    pd["total_weight"]   = total_weight
    job_orm.problem_data = pd
    flag_modified(job_orm, "problem_data")
    db.commit()

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
