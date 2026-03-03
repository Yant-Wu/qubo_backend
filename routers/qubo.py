"""QUBO 優化 API 路由"""
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from qubo_models import (
    SolveRequest, SolveResponse, HistoryPoint,
    KnapsackProblemData, MaxCutProblemData, CustomProblemData,
)
from schemas import ApiResponse
from qubo import build_qubo_matrix, simulated_annealing_solver, interpret_solution
from problem_generator import generate_random_problem_data
import numpy as np
from typing import List

# 依問題類型對應 Pydantic 驗證模型
_PROBLEM_DATA_SCHEMAS = {
    "knapsack": KnapsackProblemData,
    "max_cut": MaxCutProblemData,
    "custom": CustomProblemData,
}

router = APIRouter(prefix="/api", tags=["QUBO Solver"])


@router.post("/solve", response_model=ApiResponse[SolveResponse])
async def solve_qubo(request: SolveRequest):
    """
    QUBO 優化求解端點。
    
    支援問題類型：
    - knapsack: 背包問題
    - max_cut: 最大割問題
    - custom: 自定義 Q 矩陣
    
    範例請求（背包問題）：
    ```json
    {
      "problem_type": "knapsack",
      "problem_data": {
        "items": [
          {"weight": 2, "value": 10},
          {"weight": 3, "value": 15},
          {"weight": 5, "value": 30}
        ],
        "max_weight": 7,
        "penalty": 10.0
      },
      "solver_params": {
        "num_iterations": 10000,
        "T_start": 10.0,
        "T_end": 0.01
      }
    }
    ```
    """
    try:
        # 0a. 假如前端傳入 generation_method: random，先展開成實際問題資料
        raw = request.problem_data
        if raw.get("generation_method") == "random":
            n_vars = int(raw.get("n_variables") or 5)  # 未指定預設 5 個變數
            seed = raw.get("seed")  # None 則真隨機
            try:
                raw = generate_random_problem_data(
                    problem_type=request.problem_type,
                    n_variables=n_vars,
                    seed=seed,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))

        # 0b. 依問題類型驗證 problem_data 結構（避免 KeyError 往後傳）
        schema_cls = _PROBLEM_DATA_SCHEMAS.get(request.problem_type)
        if schema_cls is not None:
            try:
                validated = schema_cls.model_validate(raw)
                # by_alias=True 保留 "from"/"to" 等 alias key，讓 builder/interpreter 相容
                problem_data = validated.model_dump(by_alias=True)
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=exc.errors())
        else:
            problem_data = raw

        # 1. 構建 QUBO 矩陣
        Q_matrix = build_qubo_matrix(
            problem_type=request.problem_type,
            problem_data=problem_data
        )
        
        # 2. 使用模擬退火求解
        solver_result = simulated_annealing_solver(
            Q=Q_matrix,
            num_iterations=request.solver_params.num_iterations,
            T_start=request.solver_params.T_start,
            T_end=request.solver_params.T_end,
            seed=request.solver_params.seed
        )
        
        # 3. 解讀結果
        interpretation = interpret_solution(
            problem_type=request.problem_type,
            problem_data=problem_data,
            solution=solver_result.get("solution", [])
        )
        
        # 4. 構建回應
        response = SolveResponse(
            status="success",
            solution=solver_result.get("solution"),
            energy=solver_result.get("energy"),
            interpretation=interpretation,
            history=[
                HistoryPoint(**point) for point in solver_result.get("history", [])
            ],
            computation_time_ms=solver_result.get("computation_time_ms")
        )
        
        return ApiResponse(
            data=response,
            message=f"求解完成，問題類型: {request.problem_type}"
        )
        
    except ValueError as e:
        # 輸入驗證錯誤
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # 其他錯誤
        return ApiResponse(
            data=SolveResponse(
                status="error",
                error_message=str(e)
            ),
            message="求解過程發生錯誤"
        )


@router.get("/examples", response_model=ApiResponse[List[dict]])
async def get_example_problems():
    """
    取得範例問題列表。
    """
    examples = [
        {
            "name": "簡單背包問題",
            "problem_type": "knapsack",
            "description": "3 個物品，容量限制 7",
            "problem_data": {
                "items": [
                    {"weight": 2, "value": 10},
                    {"weight": 3, "value": 15},
                    {"weight": 5, "value": 30}
                ],
                "max_weight": 7,
                "penalty": 10.0
            },
            "expected_solution": "選擇物品 1 和 2（總價值 25）"
        },
        {
            "name": "四節點最大割",
            "problem_type": "max_cut",
            "description": "4 個節點的無向圖",
            "problem_data": {
                "nodes": 4,
                "edges": [
                    {"from": 0, "to": 1, "weight": 1.0},
                    {"from": 1, "to": 2, "weight": 1.0},
                    {"from": 2, "to": 3, "weight": 1.0},
                    {"from": 3, "to": 0, "weight": 1.0}
                ]
            },
            "expected_solution": "將節點分為兩組，最大化割邊數"
        },
        {
            "name": "自定義 2x2 矩陣",
            "problem_type": "custom",
            "description": "簡單的 2 變數 QUBO",
            "problem_data": {
                "Q_matrix": [
                    [-1, 2],
                    [0, -1]
                ]
            },
            "expected_solution": "最小化能量函數"
        }
    ]
    
    return ApiResponse(
        data=examples,
        message="範例問題列表",
        total=len(examples)
    )
