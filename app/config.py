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
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
ONLINE_UPLOAD_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_DB_PATH = DATA_DIR / "history.sqlite3"

# Doubao VLM
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
ARK_API_BASE = os.getenv("ARK_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
VLM_MODEL = os.getenv("VLM_MODEL", "ep-20260420011113-6jvmx")

# VLM Pod Verify — 纠错置信度阈值
# P 点剔除、F 点补回都必须达到该阈值；越高越保守，纠错越少。
VLM_FP_CONFIDENCE_THRESHOLD = float(os.getenv("VLM_FP_CONFIDENCE_THRESHOLD", "0.8"))

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8501"))
