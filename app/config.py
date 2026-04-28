import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
ONLINE_UPLOAD_DIR = BASE_DIR / "online_uploads"
CACHE_DIR = BASE_DIR / "benchmark_cache"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
ONLINE_UPLOAD_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Doubao VLM
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
ARK_API_BASE = os.getenv("ARK_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
VLM_MODEL = os.getenv("VLM_MODEL", "ep-20260420011113-6jvmx")

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8501"))
