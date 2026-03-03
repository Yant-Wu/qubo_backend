# utils.py — 通用系統工具（單一職責：底層系統工具函式）
import os
import signal
import socket


def free_port(port: int) -> None:
    """若 port 已被佔用，自動終止該程序。"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return  # port 空閒，不需處理
    except OSError:
        return

    try:
        import subprocess
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        for pid_str in pids:
            try:
                os.kill(int(pid_str), signal.SIGKILL)
                print(f"✓ 已終止佔用 port {port} 的程序 (PID {pid_str})")
            except (ProcessLookupError, PermissionError):
                pass
    except Exception:
        pass
