"""
ControlPlane.ai - One-Click Startup Script
Launches the FastAPI governance gateway and Streamlit dashboard.
"""

import subprocess
import sys
import time
import threading
import os
import signal
from typing import Optional


GATEWAY_HOST = "0.0.0.0"
GATEWAY_PORT = 8000
STREAMLIT_PORT = 8501

def start_gateway():
    """Start FastAPI reverse proxy gateway."""
    print("[ControlPlane] Starting FastAPI Gateway on port", GATEWAY_PORT)
    cmd = [
        sys.executable, "-m", "uvicorn",
        "controlplane.proxy_gateway:app",
        "--host", GATEWAY_HOST,
        "--port", str(GATEWAY_PORT),
        "--reload",
        "--log-level", "info",
    ]
    return subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


def start_dashboard():
    """Start Streamlit enterprise dashboard."""
    print("[ControlPlane] Starting Streamlit Dashboard on port", STREAMLIT_PORT)
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", str(STREAMLIT_PORT),
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false",
    ]
    return subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))


def main():
    print("""
===============================================================
  ControlPlane.ai — Enterprise In-Flight AI Governance Layer
  Accenture Innovation Challenge 2026
===============================================================
  Gateway:    http://localhost:8000
  Dashboard:  http://localhost:8501
  API Docs:   http://localhost:8000/docs
  Health:     http://localhost:8000/health
===============================================================
    """)

    gateway_proc: Optional[subprocess.Popen] = None
    dashboard_proc: Optional[subprocess.Popen] = None

    try:
        gateway_proc = start_gateway()
        time.sleep(2)  # Let gateway initialize
        dashboard_proc = start_dashboard()

        print("[ControlPlane] All services started. Press Ctrl+C to stop.")

        # Wait for both processes
        gateway_proc.wait()

    except KeyboardInterrupt:
        print("\n[ControlPlane] Shutting down services...")
        if gateway_proc:
            gateway_proc.terminate()
        if dashboard_proc:
            dashboard_proc.terminate()
        print("[ControlPlane] Shutdown complete.")
    except Exception as e:
        print(f"[ControlPlane] Error: {e}")
        if gateway_proc:
            gateway_proc.terminate()
        if dashboard_proc:
            dashboard_proc.terminate()
        sys.exit(1)


if __name__ == "__main__":
    main()
