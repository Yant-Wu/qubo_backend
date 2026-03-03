"""結果解讀器 - 將 0/1 解轉換為人類可讀的答案"""
from typing import Dict, List, Any


def interpret_solution(
    problem_type: str,
    problem_data: Dict[str, Any],
    solution: List[int]
) -> Dict[str, Any]:
    """
    解讀求解結果。
    
    Args:
        problem_type: 問題類型
        problem_data: 問題數據
        solution: 0/1 解
    
    Returns:
        interpretation: 人類可讀的結果
    """
    if problem_type == "knapsack":
        return interpret_knapsack(problem_data, solution)
    elif problem_type == "max_cut":
        return interpret_max_cut(problem_data, solution)
    elif problem_type == "custom":
        return interpret_custom(problem_data, solution)
    else:
        return {"raw_solution": solution}


def interpret_knapsack(data: Dict[str, Any], solution: List[int]) -> Dict[str, Any]:
    """
    解讀背包問題結果。
    
    Returns:
        {
            "selected_items": [索引列表],
            "total_weight": 總重量,
            "total_value": 總價值,
            "weight_limit": 重量限制,
            "is_feasible": 是否滿足約束
        }
    """
    items = data.get("items")
    if items is None:
        raise ValueError("knapsack 問題缺少必要欄位: items")
    max_weight = data.get("max_weight")
    if max_weight is None:
        raise ValueError("knapsack 問題缺少必要欄位: max_weight")
    
    selected_items = [i for i, x in enumerate(solution) if x == 1]
    total_weight = sum(items[i]["weight"] for i in selected_items)
    total_value = sum(items[i]["value"] for i in selected_items)
    is_feasible = total_weight <= max_weight
    
    return {
        "selected_items": selected_items,
        "total_weight": round(total_weight, 2),
        "total_value": round(total_value, 2),
        "weight_limit": max_weight,
        "is_feasible": is_feasible,
        "feasibility_note": "滿足重量限制" if is_feasible else f"超重 {round(total_weight - max_weight, 2)}"
    }


def interpret_max_cut(data: Dict[str, Any], solution: List[int]) -> Dict[str, Any]:
    """
    解讀最大割問題結果。
    
    Returns:
        {
            "partition_0": [節點列表 (組別 0)],
            "partition_1": [節點列表 (組別 1)],
            "cut_edges": [被割的邊],
            "cut_weight": 割邊權重總和
        }
    """
    edges = data.get("edges")
    if edges is None:
        raise ValueError("max_cut 問題缺少必要欄位: edges")
    
    partition_0 = [i for i, x in enumerate(solution) if x == 0]
    partition_1 = [i for i, x in enumerate(solution) if x == 1]
    
    cut_edges = []
    cut_weight = 0.0
    
    for edge in edges:
        i, j = edge["from"], edge["to"]
        weight = edge["weight"]
        
        # 如果兩端點在不同組，則為割邊
        if solution[i] != solution[j]:
            cut_edges.append({
                "from": i,
                "to": j,
                "weight": weight
            })
            cut_weight += weight
    
    return {
        "partition_0": partition_0,
        "partition_1": partition_1,
        "partition_0_size": len(partition_0),
        "partition_1_size": len(partition_1),
        "cut_edges": cut_edges,
        "cut_weight": round(cut_weight, 2),
        "num_cut_edges": len(cut_edges)
    }


def interpret_custom(data: Dict[str, Any], solution: List[int]) -> Dict[str, Any]:
    """
    解讀自定義問題結果。
    
    Returns:
        {
            "solution": 0/1 解,
            "num_ones": 1 的數量,
            "num_zeros": 0 的數量
        }
    """
    num_ones = sum(solution)
    num_zeros = len(solution) - num_ones
    
    return {
        "solution": solution,
        "num_ones": num_ones,
        "num_zeros": num_zeros,
        "solution_density": round(num_ones / len(solution), 3) if len(solution) > 0 else 0
    }
