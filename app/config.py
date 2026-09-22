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

# VLM 请求选项
# 思考模式：实测同一张航拍图的标注调用，思考开启时 281 秒以上、且经常长时间挂住
# 不返回；关闭后 3~16 秒。默认关闭。
#   设为 "default" 则不加任何参数，沿用模型自带配置（思考开启）。
VLM_THINKING = os.getenv("VLM_THINKING", "disabled").strip().lower()

# 单次请求超时（秒）与失败重试次数。
# 不显式设置时 openai SDK 默认 read=600s、max_retries=2，叠加起来最坏要
# 600×3=1800 秒才失败，表现为页面长时间卡住。
VLM_TIMEOUT_S = float(os.getenv("VLM_TIMEOUT_S", "120"))
VLM_MAX_RETRIES = int(os.getenv("VLM_MAX_RETRIES", "1"))

# schema 校验不通过时的重发次数上限。
# 模型偶尔返回漏字段 / 格式不对的结果（如 bbox 少逗号、action 非法），
# 与其静默降级，不如重发请求让它重给一次。上限是必须的——否则模型对某张图
# 若始终给不出合规结果，会无限重发烧额度。
VLM_SCHEMA_MAX_ATTEMPTS = int(os.getenv("VLM_SCHEMA_MAX_ATTEMPTS", "3"))

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8501"))
