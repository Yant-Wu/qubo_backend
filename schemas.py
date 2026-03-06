# schemas.py — Pydantic 資料模型（單一職責：API 請求/回應的型別定義）
from __future__ import annotations

from datetime import datetime
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# ============ 通用回應格式 ============
class ApiResponse(BaseModel, Generic[T]):
    """統一的 API 回應格式。"""
    data: T
    message: str = "success"
    total: Optional[int] = None  # 用於列表回應


class ApiErrorResponse(BaseModel):
    """錯誤回應格式。"""
    message: str


# ============ 歷史資料 ============
class HistoryPoint(BaseModel):
    """單一歷史點 (iteration + value)。"""
    iteration: int
    value: float
    entropy: Optional[float] = None   # AEQTS Q-bit entropy
    is_feasible: Optional[bool] = None  # 該迭代最佳解是否滿足約束


class HistoryPointCreate(BaseModel):
    """新建歷史點的請求。"""
    points: List[HistoryPoint] = Field(..., min_items=1)


# ============ 問題資料 ============
class KnapsackItemData(BaseModel):
    """背包問題的單一物品。"""
    name: str
    weight: float
    value: float


class ProblemData(BaseModel):
    """問題參數。"""
    generation_method: str = "random"  # random, upload
    seed: Optional[int] = None
    filename: Optional[str] = None
    n_variables: Optional[int] = Field(default=None, ge=1, description="隨機生成時的變數數量（物品數 / 節點數）")
    num_iterations: Optional[int] = Field(default=None, ge=1, description="AEQTS 迭代次數")
    timeout_seconds: Optional[float] = Field(default=None, gt=0, description="執行時限（秒）")
    Q_matrix: Optional[List[List[float]]] = Field(default=None, description="自訂 QUBO 矩陣（custom 類型）")
    # Knapsack 問題前端表單紀錄（供「套用此設定」還原用）
    items: Optional[List[KnapsackItemData]] = Field(default=None, description="Knapsack 物品清單")
    capacity: Optional[float] = Field(default=None, description="Knapsack 容量")
    penalty: Optional[float] = Field(default=None, description="Knapsack 懲罰係數")


# ============ Job CRUD ============
class JobCreateRequest(BaseModel):
    """POST /api/jobs 的請求格式。"""
    task_name: str = Field(..., min_length=1)
    problem_type: str  # TSP, MaxCut, Knapsack
    n_variables: int = Field(..., ge=1)
    solver_backend: str  # exact, simulated_annealing, quantum_annealing
    core_limit: Optional[int] = Field(default=None, ge=1)   # AEQTS 鄰域大小 N（最小 1）
    problem_data: ProblemData = Field(default_factory=ProblemData)


class JobListItem(BaseModel):
    """GET /api/jobs 回傳的最小單位（列表用）。"""
    id: str
    task_name: str
    status: str
    created_at: datetime


class JobDetail(BaseModel):
    """GET /api/jobs/{id} 的完整詳情。"""
    id: str
    task_name: str
    problem_type: str
    n_variables: int
    solver_backend: str
    core_limit: Optional[int]
    problem_data: ProblemData
    status: str
    history_data: List[HistoryPoint] = []
    error_message: Optional[str] = None
    computation_time_ms: Optional[float] = None  # 實際計算時間
    t_start: Optional[float] = None              # AEQTS 鄰域大小 N
    t_end: Optional[float] = None                # AEQTS 迭代次數

    created_at: datetime
    updated_at: datetime


# ============ 狀態更新 ============
class StatusUpdate(BaseModel):
    """PATCH /api/jobs/{id}/status 的請求格式。"""
    status: str = Field(..., pattern="^(pending|running|completed|failed)$")
    error_message: Optional[str] = None
