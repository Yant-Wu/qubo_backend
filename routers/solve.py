"""knapsack 直解 API（dimod + CQM + 本機 SA）。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from qubo_models import KnapsackSolveRequest, KnapsackSolveResponse
from qubo.dimod_knapsack import solve_knapsack_with_dimod

router = APIRouter(prefix="/api", tags=["Knapsack Solve"])


@router.post("/solve/knapsack", response_model=KnapsackSolveResponse)
async def solve_knapsack(payload: dict):
    """
    依需求提供 knapsack 直解端點。

    請求格式：
    {
      "items": [{"name": "...", "weight": 1, "value": 2}],
      "capacity": 5,
      "penalty": 10
    }
    """
    try:
        request = KnapsackSolveRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    try:
        result = solve_knapsack_with_dimod(
            items=[item.model_dump() for item in request.items],
            capacity=request.capacity,
            penalty=request.penalty,
        )
        return KnapsackSolveResponse.model_validate(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"運算錯誤: {exc}") from exc
