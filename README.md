# QUBO Optimization Platform Backend

QUBO 優化平台後端服務，提供組合最佳化問題（Knapsack、MaxCut、Custom QUBO）的任務建立、背景執行、進度追蹤與結果查詢。

This backend provides job-based optimization APIs for Knapsack, MaxCut, and Custom QUBO problems, with background execution and progress tracking.

## 1. 專案亮點 Highlights

- FastAPI REST API，具備 Swagger 文件（/docs）
- Job-based 非同步流程（Create Job -> Poll Result）
- 背景排程 Worker（APScheduler）處理 pending 任務
- 支援 CUDA binary 優先執行，找不到時自動 fallback 到 Python solver
- SQLite 開發即用，並可透過 DATABASE_URL 切換到 PostgreSQL

## 2. 技術棧 Tech Stack

- Language: Python 3.13+
- Web Framework: FastAPI, Uvicorn
- ORM: SQLAlchemy
- Scheduler: APScheduler
- Numerics: NumPy
- QUBO Ecosystem: dimod, dwave-samplers
- Solver Runtime: CUDA binary (optional) + Python fallback

## 3. 快速開始 Quick Start

### 3.1 環境需求 Prerequisites

- Python >= 3.13
- 建議使用 uv（專案已提供 uv.lock）
- 若要啟用 GPU：NVIDIA driver + CUDA runtime（容器建議走 Dockerfile.cuda）

### 3.2 安裝依賴 Install Dependencies

建議流程（uv）：

```bash
cp .env.example .env
uv sync
```

替代流程（pip）：

```bash
cp .env.example .env
pip install -e .
```

### 3.3 啟動服務 Run Service

```bash
python3 main.py

or 

uv run main.py
```

啟動後預設位址：

- API Base: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 4. 環境變數 Environment Variables

以 .env.example 與 config.py 為準。

| Variable | Default | 說明 Description |
|---|---|---|
| DATABASE_URL | sqlite:///./database/qubo.db | 資料庫連線字串（本機預設 SQLite） |
| CORS_ORIGINS | http://localhost:5173,http://localhost:3000 | 允許的前端來源，可用逗號分隔或設為 * |
| WORKER_ENABLED | true | 是否啟用排程 worker 掃描 pending 任務 |
| WORKER_CHECK_INTERVAL | 2 | worker 掃描間隔（秒） |
| HOST | 0.0.0.0 | 服務監聽位址 |
| PORT | 8000 | 服務監聽埠 |
| RELOAD | false | 是否啟用熱重載（開發可設 true） |

## 5. API 概覽 API Overview

### 5.1 端點清單 Endpoints

| Method | Path | 說明 |
|---|---|---|
| GET | /health | 健康檢查 |
| GET | /api/jobs | 查詢任務列表（可依 algorithm 篩選） |
| POST | /api/jobs | 建立任務（狀態初始為 pending） |
| GET | /api/jobs/{job_id} | 取得單一任務詳情與歷史資料 |
| DELETE | /api/jobs/{job_id} | 刪除任務 |
| PATCH | /api/jobs/{job_id}/status | 手動更新任務狀態 |
| POST | /api/jobs/{job_id}/history | 補寫歷史點 |
| POST | /api/jobs/solve | 建立任務並立即背景求解 |

### 5.2 任務狀態流轉 Job State Flow

```text
pending -> running -> completed
										-> failed
```

### 5.3 兩種執行模式 Execution Modes

1) 排程模式（Queued + Scheduler）
- POST /api/jobs 建立任務，狀態為 pending
- APScheduler 週期性呼叫 process_pending_jobs 將任務轉 running 並求解

2) 立即背景模式（Immediate Background Task）
- POST /api/jobs/solve 建立任務後立刻以 FastAPI BackgroundTasks 執行
- API 即時回傳 job_id，前端可輪詢 GET /api/jobs/{job_id}

## 6. API 使用範例 Examples

### 6.1 Health Check

```bash
curl http://localhost:8000/health
```

### 6.2 建立並背景求解（Knapsack）

```bash
curl -X POST "http://localhost:8000/api/jobs/solve" \
	-H "Content-Type: application/json" \
	-d '{
		"task_name": "demo-knapsack",
		"problem_type": "knapsack",
		"n_variables": 6,
		"solver_backend": "simulated_annealing",
		"core_limit": 50,
		"problem_data": {
			"items": [
				{"name": "A", "weight": 2, "value": 8},
				{"name": "B", "weight": 3, "value": 11},
				{"name": "C", "weight": 4, "value": 13}
			],
			"capacity": 5,
			"penalty": 10,
			"num_iterations": 500,
			"timeout_seconds": 30
		}
	}'
```

回傳後請使用 job_id 輪詢：

```bash
curl "http://localhost:8000/api/jobs/<job_id>"
```

## 7. 支援問題類型 Supported Problem Types

### 7.1 Knapsack

- 輸入：items, capacity, penalty
- 流程：先轉換為 QUBO，再交由 AEQTS/CUDA 求解
- 輸出：selected_items, total_value, total_weight

### 7.2 MaxCut

- 輸入：nodes, edges
- builder 目前限制 nodes 不可超過 500

### 7.3 Custom QUBO

- 輸入：Q_matrix（方陣）
- 會做基本驗證與對稱化

## 8. 架構與資料流 Architecture & Data Flow

```text
Client
	-> FastAPI Router (/api/jobs)
			-> Store (CRUD)
					-> SQLAlchemy (jobs, job_history)

Worker Path A: APScheduler -> process_pending_jobs -> _simulate_job
Worker Path B: POST /api/jobs/solve -> BackgroundTasks -> _blocking_solve -> _simulate_job

_simulate_job
	-> build_qubo_matrix (except CUDA knapsack fast path)
	-> cuda_knapsack_solver or aeqts_solver
	-> stream progress into job_history
```

## 9. 專案結構 Project Structure

```text
backend/
├─ main.py                # FastAPI 入口、lifespan、scheduler、CORS
├─ config.py              # 環境變數讀取與預設值
├─ database.py            # SQLAlchemy model、engine、init_db
├─ store.py               # 資料存取層 CRUD
├─ schemas.py             # Pydantic request/response models
├─ worker.py              # 任務執行流程與進度寫回
├─ routers/
│  └─ jobs.py             # /api/jobs 路由
├─ qubo/
│  ├─ builder.py          # 問題到 QUBO 矩陣轉換
│  └─ solver.py           # CUDA/Python solver 封裝
├─ Dockerfile             # CPU 映像
├─ Dockerfile.cuda        # GPU 映像（含 aeqts.cu 編譯）
├─ .env.example           # 環境變數範本
└─ pyproject.toml         # 專案依賴與 Python 版本
```

## 10. Docker 部署 Docker Deployment

### 10.1 CPU 版本

```bash
docker build -f Dockerfile -t qubo-backend:cpu .
docker run --rm -p 8000:8000 qubo-backend:cpu
```

### 10.2 GPU 版本

```bash
docker build -f Dockerfile.cuda -t qubo-backend:gpu .
docker run --rm --gpus all -p 8000:8000 qubo-backend:gpu
```

## 11. 已知限制與風險 Known Limitations

- 啟發式求解（AEQTS）不保證全域最優解
- 若缺少 CUDA binary，會 fallback 到 Python solver，耗時可能增加
- SQLite 在高併發場景有限制，正式環境建議改 PostgreSQL
- 長任務建議設置合理 timeout_seconds 與輪詢頻率，避免不必要負載

## 12. 開發者維運建議 For Developers

- 開發期可將 RELOAD 設為 true
- 生產環境請固定 CORS_ORIGINS，不建議設為 *
- 監控資料建議重點：jobs status 分布、失敗率、平均 computation_time_ms
- 若要提高吞吐，優先考慮資料庫切換與 worker 併發策略

