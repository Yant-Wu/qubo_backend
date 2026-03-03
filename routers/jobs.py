# routers/jobs.py — 完整的 REST API 路由（單一職責：HTTP 層）
from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

import store
from database import get_db
from schemas import (
    ApiErrorResponse,
    ApiResponse,
    HistoryPointCreate,
    JobCreateRequest,
    JobDetail,
    JobListItem,
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
