# main.py — FastAPI 應用初始化（單一職責：應用程式進入點）
import os
import uvicorn
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from config import CORS_ORIGINS, HOST, PORT, RELOAD, WORKER_CHECK_INTERVAL, WORKER_ENABLED
from utils import free_port
from database import init_db
from routers import jobs
from routers import qubo
from routers import solve
from worker import process_pending_jobs

# ============ 全局變量 ============
scheduler = None


# ============ Lifespan 事件處理 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """應用啟動/關閉生命週期管理（替代 on_event）。"""
    # Startup
    global scheduler
    init_db()
    print("✓ Database initialized")
    
    if WORKER_ENABLED:
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            process_pending_jobs,
            "interval",
            seconds=WORKER_CHECK_INTERVAL,
            id="process_jobs",
            max_instances=2,  # 允許最多 2 個並行實例，避免長任務期間 skip warning
        )
        scheduler.start()
        print(f"✓ Background scheduler started (check interval: {WORKER_CHECK_INTERVAL}s)")
    
    yield  # 應用運行期間
    
    # Shutdown
    if scheduler:
        scheduler.shutdown(wait=True)
        print("✓ Background scheduler stopped")


# ============ FastAPI 應用初始化 ============
app = FastAPI(
    title="QUBO Optimization Platform",
    description="Real-time job processing with persistent storage",
    version="2.0.0",
    lifespan=lifespan,  # 注入生命週期
)

# ============ CORS 設定 ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials="*" not in CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ 路由註冊 ============
app.include_router(jobs.router)
app.include_router(qubo.router)
app.include_router(solve.router)


# ============ 健康檢查端點 ============
@app.get("/health")
async def health_check():
    """健康檢查端點。"""
    return {"status": "ok"}


# ============ 前端靜態檔托管（SPA） ============
_DIST_DIR = os.path.join(os.path.dirname(__file__), "..", "qubo-dashboard", "dist")
_DIST_DIR = os.path.abspath(_DIST_DIR)

if os.path.isdir(_DIST_DIR):
    _assets_dir = os.path.join(_DIST_DIR, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """SPA catch-all：將所有非 API 路徑導向 index.html。"""
        file_path = os.path.join(_DIST_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(_DIST_DIR, "index.html"))


if __name__ == "__main__":
    free_port(PORT)
    uvicorn.run("main:app", host=HOST, port=PORT, reload=RELOAD)
