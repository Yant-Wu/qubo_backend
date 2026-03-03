# qubo_models.py — QUBO 直接求解 API 的 Pydantic 模型（單一職責：QUBO 求解請求/回應型別定義）
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


# ============ 請求模型 ============

class SolverParams(BaseModel):
    """求解器參數"""
    num_iterations: int = Field(default=10000, ge=100, le=1000000, description="迭代次數")
    T_start: float = Field(default=10.0, gt=0, description="初始溫度")
    T_end: float = Field(default=0.01, gt=0, description="最終溫度")
    seed: Optional[int] = Field(default=None, description="隨機種子")


class KnapsackItem(BaseModel):
    """背包問題中的物品"""
    weight: float = Field(..., ge=0, description="重量")
    value: float = Field(..., ge=0, description="價值")


class KnapsackProblemData(BaseModel):
    """背包問題數據"""
    items: List[KnapsackItem] = Field(..., min_length=1, description="物品列表")
    max_weight: float = Field(..., gt=0, description="最大重量限制")
    penalty: float = Field(default=10.0, gt=0, description="約束懲罰係數")


class Edge(BaseModel):
    """圖的邊"""
    from_node: int = Field(..., alias="from", ge=0, description="起始節點")
    to_node: int = Field(..., alias="to", ge=0, description="目標節點")
    weight: float = Field(default=1.0, description="邊權重")
    
    class Config:
        populate_by_name = True


class MaxCutProblemData(BaseModel):
    """最大割問題數據"""
    nodes: int = Field(..., ge=2, description="節點數量")
    edges: List[Edge] = Field(..., min_length=1, description="邊列表")


class CustomProblemData(BaseModel):
    """自定義問題數據"""
    Q_matrix: List[List[float]] = Field(..., description="QUBO 矩陣")


class SolveRequest(BaseModel):
    """求解請求"""
    problem_type: Literal["knapsack", "max_cut", "custom"] = Field(..., description="問題類型")
    problem_data: Dict[str, Any] = Field(..., description="問題數據")
    solver_params: SolverParams = Field(default_factory=SolverParams, description="求解器參數")


# ============ 回應模型 ============

class HistoryPoint(BaseModel):
    """能量歷史點"""
    iteration: int
    energy: float
    temperature: float


class Interpretation(BaseModel):
    """結果解讀（動態欄位）"""
    pass


class SolveResponse(BaseModel):
    """求解回應"""
    status: Literal["success", "error"] = Field(..., description="執行狀態")
    solution: Optional[List[int]] = Field(None, description="0/1 解")
    energy: Optional[float] = Field(None, description="最優能量值")
    interpretation: Optional[Dict[str, Any]] = Field(None, description="人類可讀的結果")
    history: Optional[List[HistoryPoint]] = Field(None, description="能量歷史")
    computation_time_ms: Optional[float] = Field(None, description="計算時間 (毫秒)")
    error_message: Optional[str] = Field(None, description="錯誤訊息")


# ============ 範例數據模型 ============

class ExampleProblem(BaseModel):
    """範例問題"""
    name: str
    problem_type: str
    description: str
    problem_data: Dict[str, Any]
    expected_solution_description: str


# ============ dimod knapsack 直解 API 模型 ============

class KnapsackSolveItem(BaseModel):
    """/solve 請求中的物品。"""
    name: str = Field(..., min_length=1)
    weight: float = Field(..., gt=0)
    value: float = Field(..., ge=0)


class KnapsackSolveRequest(BaseModel):
    """POST /solve 請求格式。"""
    items: List[KnapsackSolveItem] = Field(..., min_length=1)
    capacity: float = Field(..., gt=0)
    penalty: float = Field(..., gt=0)


class KnapsackInterpretation(BaseModel):
    """背包結果解讀。"""
    selected_items: List[KnapsackSolveItem]
    total_value: float
    total_weight: float


class KnapsackSolveResponse(BaseModel):
    """POST /solve 回應格式。"""
    status: Literal["success"]
    solution: List[int]
    energy: float
    interpretation: KnapsackInterpretation
    computation_time_ms: float
