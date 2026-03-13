# routers/jobs.py — 完整的 REST API 路由（單一職責：HTTP 層）
from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

_log = logging.getLogger(__name__)

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
async def list_jobs(algorithm: Optional[str] = Query(None), db: Session = Depends(get_db)):
    jobs = store.list_jobs(db, solver_backend=algorithm)
    items = [JobListItem(id=j.id, task_name=j.task_name, status=j.status, created_at=j.created_at) for j in jobs]
    return ApiResponse(data=items, total=len(items))

# ===== 2. POST /api/jobs =====
@router.post("", response_model=ApiResponse[JobDetail], status_code=201)
async def create_job(req: JobCreateRequest, db: Session = Depends(get_db)):
    job_id = str(uuid.uuid4())
    job = store.create_job(db, req, job_id)
    return ApiResponse(data=job, message="Job created successfully")

# ===== 3. GET /api/jobs/{id} =====
@router.get("/{job_id}", response_model=ApiResponse[JobDetail])
async def get_job(job_id: str, db: Session = Depends(get_db)):
    job = store.get_job(db, job_id)
    if not job: raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return ApiResponse(data=job)

# ===== 4. DELETE /api/jobs/{id} =====
@router.delete("/{job_id}", status_code=200)
async def delete_job(job_id: str, db: Session = Depends(get_db)):
    if not store.delete_job(db, job_id): raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return ApiResponse(data=None, message=f"Job '{job_id}' deleted successfully")

# ===== 5. PATCH /api/jobs/{id}/status =====
@router.patch("/{job_id}/status", response_model=ApiResponse[JobDetail])
async def update_job_status(job_id: str, req: StatusUpdate, db: Session = Depends(get_db)):
    job = store.update_job_status(db, job_id, req.status, req.error_message)
    if not job: raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return ApiResponse(data=job, message=f"Job status updated to '{req.status}'")

# ===== 6. POST /api/jobs/{id}/history =====
@router.post("/{job_id}/history", response_model=ApiResponse[dict], status_code=201)
async def add_history(job_id: str, req: HistoryPointCreate, db: Session = Depends(get_db)):
    job = store.get_job(db, job_id)
    if not job: raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if not store.add_history_points(db, job_id, req.points): raise HTTPException(status_code=500, detail="Failed to add history points")
    return ApiResponse(data={"added": len(req.points)}, message=f"Added {len(req.points)} history point(s)")

def _blocking_solve(job_id: str):
    from database import Job as JobModel, SessionLocal
    from worker import _simulate_job
    db = SessionLocal()
    try:
        job_orm = db.query(JobModel).filter(JobModel.id == job_id).first()
        _simulate_job(db, job_orm)
        job_orm.status = "completed"
        db.commit()
    except Exception as exc:
        _log.exception("[_blocking_solve] job %s error: %s", job_id, exc)
        job_orm = db.query(JobModel).filter(JobModel.id == job_id).first()
        if job_orm:
            job_orm.status = "failed"
            job_orm.error_message = type(exc).__name__
            db.commit()
    finally:
        db.close()


# ===== 7. POST /api/jobs/solve =====
@router.post("/solve", response_model=ApiResponse[SolveAndCreateResponse], status_code=201)
async def solve_and_create(req: JobCreateRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    統一求解端點：建立 Job → 將運算丟給背景任務 → 立刻回傳 job_id 讓前端開始輪詢拉取。
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
    
    # 💡 放進背景任務執行，API 立刻回傳
    background_tasks.add_task(_blocking_solve, job_id)

    # 回傳假資料（因為運算還沒完成），前端會利用 job_id 去輪詢正確資料
    return ApiResponse(
        data=SolveAndCreateResponse(
            job_id=job_id,
            energy=0.0,
            selected_items=[],
            total_value=0.0,
            total_weight=0.0,
            computation_time_ms=0.0,
            device="gpu",
        ),
        message="Job started in background"
    )