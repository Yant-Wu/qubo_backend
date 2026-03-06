# database.py — SQLAlchemy models & DB 連接（單一職責：資料模型與 ORM 設定）
from __future__ import annotations

import json
from datetime import datetime
from typing import Generator, Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import DATABASE_URL


# 建立 Base class
class Base(DeclarativeBase):
    pass


# ============ Models ============
class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True)  # UUID
    task_name = Column(String(255), nullable=False)
    problem_type = Column(String(50), nullable=False)  # TSP, MaxCut, Knapsack
    n_variables = Column(Integer, nullable=False)
    solver_backend = Column(String(50), nullable=False)  # exact, simulated_annealing, quantum_annealing
    core_limit = Column(Integer, nullable=True)  # Optional
    problem_data = Column(JSON, nullable=False)  # e.g. {"generation_method": "random", "seed": 42}
    status = Column(String(20), default="pending", nullable=False)  # pending, running, completed, failed
    error_message = Column(String(1000), nullable=True)  # Optional: error details
    computation_time_ms = Column(Float, nullable=True)   # 實際計算時間（ms）
    t_start = Column(Float, nullable=True)               # 鄰域大小 N
    t_end = Column(Float, nullable=True)                 # 迭代次數
    compute_device = Column(String(10), nullable=True)   # 'gpu' 或 'cpu'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class JobHistory(Base):
    __tablename__ = "job_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    iteration = Column(Integer, nullable=False)
    value = Column(Float, nullable=False)
    qubo_energy = Column(Float, nullable=True)    # 當前迭代最佳候選的 QUBO 能量
    entropy = Column(Float, nullable=True)        # AEQTS Q-bit 族群 entropy
    is_feasible = Column(Boolean, nullable=True)  # 該迭代的最佳解是否滿足約束
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# ============ Engine & Session ============
def get_engine():
    """Create database engine."""
    # SQLite 需要確保目錄存在
    # 格式：sqlite:///./relative/path.db 或 sqlite:////absolute/path.db
    if DATABASE_URL.startswith("sqlite:///"):
        from pathlib import Path
        db_path = Path(DATABASE_URL[len("sqlite:///"):])  # 去掉 sqlite:/// 前綴
        db_path.parent.mkdir(parents=True, exist_ok=True)

    return create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})


engine = get_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db():
    """初始化資料庫表（首次運行時執行）。"""
    Base.metadata.create_all(bind=engine)
    # 安全地為舊資料庫補上 compute_device 欄位（已存在時忽略）
    with engine.connect() as conn:
        for ddl in [
            "ALTER TABLE jobs ADD COLUMN compute_device VARCHAR(10)",
            "ALTER TABLE job_history ADD COLUMN qubo_energy FLOAT",
        ]:
            try:
                conn.execute(text(ddl))
                conn.commit()
            except Exception:
                pass  # 欄位已存在，忽略
    print("✓ Database tables initialized")


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
