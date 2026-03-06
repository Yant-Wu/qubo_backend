# config.py — 環境變數與設定管理（單一職責：應用配置）
import os
from pathlib import Path

from dotenv import load_dotenv

# 載入 .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./database/qubo.db")

# CORS（預設只允許本機開發端口，正式部署請透過環境變數設定）
_cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000")
CORS_ORIGINS = ["*"] if _cors_raw == "*" else [o.strip() for o in _cors_raw.split(",")]

# Worker 設定
WORKER_ENABLED = os.getenv("WORKER_ENABLED", "true").lower() == "true"
WORKER_CHECK_INTERVAL = int(os.getenv("WORKER_CHECK_INTERVAL", "2"))  # 秒

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
RELOAD = os.getenv("RELOAD", "false").lower() == "true"  # 生產環境應關閉
