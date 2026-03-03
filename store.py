# store.py — ORM CRUD 操作層（單一職責：資料訪問）
from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from database import Job, JobHistory
from schemas import HistoryPoint, JobCreateRequest, JobDetail, ProblemData


def create_job(db: Session, req: JobCreateRequest, job_id: str) -> JobDetail:
    """建立新任務。"""
    job = Job(
        id=job_id,
        task_name=req.task_name,
        problem_type=req.problem_type,
        n_variables=req.n_variables,
        solver_backend=req.solver_backend,
        core_limit=req.core_limit,
        problem_data=req.problem_data.model_dump(),
        status="pending",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return _job_to_detail(db, job)


def list_jobs(db: Session, solver_backend: Optional[str] = None) -> List[JobDetail]:
    """列出所有任務（新到舊）。"""
    query = db.query(Job).order_by(Job.created_at.desc())
    if solver_backend:
        query = query.filter(Job.solver_backend == solver_backend)
    jobs = query.all()
    return [_job_to_detail(db, job) for job in jobs]


def get_job(db: Session, job_id: str) -> Optional[JobDetail]:
    """獲取單一任務詳情。"""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return None
    return _job_to_detail(db, job)


def delete_job(db: Session, job_id: str) -> bool:
    """刪除任務及其歷史記錄（ON DELETE CASCADE 自動處理）。"""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return False
    db.delete(job)
    db.commit()
    return True


def update_job_status(db: Session, job_id: str, status: str, error_message: Optional[str] = None) -> Optional[JobDetail]:
    """更新任務狀態。"""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return None
    job.status = status
    if error_message:
        job.error_message = error_message
    db.commit()
    db.refresh(job)
    return _job_to_detail(db, job)


def add_history_points(db: Session, job_id: str, points: List[HistoryPoint]) -> bool:
    """為任務新增歷史資料點。"""
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        return False
    
    for point in points:
        history = JobHistory(
            job_id=job_id,
            iteration=point.iteration,
            value=point.value,
        )
        db.add(history)
    db.commit()
    return True


def get_history_points(db: Session, job_id: str) -> List[HistoryPoint]:
    """獲取任務的所有歷史點。"""
    points = db.query(JobHistory).filter(JobHistory.job_id == job_id).order_by(JobHistory.iteration).all()
    return [HistoryPoint(iteration=p.iteration, value=p.value, entropy=p.entropy, is_feasible=p.is_feasible) for p in points]


# ============ 內部輔助 ============
def _job_to_detail(db: Session, job: Job) -> JobDetail:
    """將 DB Job 物件轉換為 JobDetail。"""
    history = get_history_points(db, job.id)
    return JobDetail(
        id=job.id,
        task_name=job.task_name,
        problem_type=job.problem_type,
        n_variables=job.n_variables,
        solver_backend=job.solver_backend,
        core_limit=job.core_limit,
        problem_data=ProblemData(**job.problem_data) if isinstance(job.problem_data, dict) else job.problem_data,
        status=job.status,
        history_data=history,
        error_message=job.error_message,
        computation_time_ms=job.computation_time_ms,
        t_start=job.t_start,
        t_end=job.t_end,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )
