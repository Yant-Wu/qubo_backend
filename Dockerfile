# backend/Dockerfile — FastAPI 後端容器

FROM python:3.13-slim

# 安裝 uv（輕量 Python 套件管理器）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# 確保 Python 輸出即時顯示（不緩衝），Docker logs 才看得到 print
ENV PYTHONUNBUFFERED=1

# 先複製 lock 檔案，讓依賴安裝步驟可被 Docker layer 快取
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 複製後端原始碼
COPY . .

# 建立 SQLite 資料庫目錄（執行期由 SQLAlchemy 自動建立 .db 檔）
RUN mkdir -p /app/database

EXPOSE 8000

# 使用 venv 內的 uvicorn 直接啟動，避免 uv run 封裝帶來的 PID 問題
CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
