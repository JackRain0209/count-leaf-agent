"""
CV Agent — Plant Phenotyping API Server
FastAPI backend for rapeseed plant analysis.
"""

import shutil
import threading
import time
import uuid
from pathlib import Path

import json as _json

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import UPLOAD_DIR, RESULT_DIR, ONLINE_UPLOAD_DIR, HOST, PORT
from app.pipeline.analyzer import analyze_plant_image_stream

# ── Batch test state ──
_batch_cancel = threading.Event()
_batch_running = threading.Event()

app = FastAPI(title="CV Agent - Plant Phenotyping", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve result images
app.mount("/results", StaticFiles(directory=str(RESULT_DIR)), name="results")
# Serve online_uploads (for thumbnail preview in picker)
app.mount("/online_uploads", StaticFiles(directory=str(ONLINE_UPLOAD_DIR)), name="online_uploads")

_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


@app.get("/api/online_images")
def list_online_images():
    """Recursively list all image files under online_uploads."""
    items = []
    for p in sorted(ONLINE_UPLOAD_DIR.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMG_EXT:
            rel = p.relative_to(ONLINE_UPLOAD_DIR).as_posix()
            items.append({
                "path": rel,
                "name": p.name,
                "folder": p.parent.relative_to(ONLINE_UPLOAD_DIR).as_posix() or "/",
                "size": p.stat().st_size,
                "url": f"/online_uploads/{rel}",
            })
    return {"count": len(items), "items": items}


@app.post("/api/analyze_online")
async def analyze_online(payload: dict):
    """Analyze an image that already exists under online_uploads. payload: {"path": "relative/path.jpg"}"""
    rel = (payload or {}).get("path", "")
    method = (payload or {}).get("method", "skeleton")
    if method not in ("skeleton", "graph", "plantcv"):
        method = "skeleton"
    if not rel:
        raise HTTPException(400, "Missing 'path'")
    target = (ONLINE_UPLOAD_DIR / rel).resolve()
    # Security: ensure within ONLINE_UPLOAD_DIR
    try:
        target.relative_to(ONLINE_UPLOAD_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid path")
    if not target.exists() or target.suffix.lower() not in _IMG_EXT:
        raise HTTPException(404, "Image not found")

    def _stream():
        try:
            for event in analyze_plant_image_stream(str(target), method=method):
                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "cv-agent"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), method: str = "skeleton"):
    """
    Upload a plant image → SSE stream of pipeline steps.
    Each line: data: {"type":"step"|"result"|"error", ...}
    """
    ext = Path(file.filename or "unknown.jpg").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    file_id = uuid.uuid4().hex[:8]
    save_name = f"{file_id}{ext}"
    save_path = UPLOAD_DIR / save_name
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    def _stream():
        try:
            for event in analyze_plant_image_stream(str(save_path), method=method):
                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {_json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/api/results/{image_id}")
def get_result(image_id: str):
    """Get previously computed result by image ID."""
    result_dir = RESULT_DIR / image_id
    result_file = result_dir / "result.json"
    if not result_file.exists():
        raise HTTPException(404, "Result not found")
    import json
    with open(result_file) as f:
        return json.load(f)


@app.post("/api/batch_test")
async def batch_test():
    """Run full pipeline (graph method) on all online images. SSE stream of logs."""
    if _batch_running.is_set():
        raise HTTPException(409, "Batch test already running")

    _batch_cancel.clear()
    _batch_running.set()
    method = "graph"

    def _log(msg):
        return f"data: {_json.dumps({'type': 'log', 'message': msg}, ensure_ascii=False)}\n\n"

    def _stream():
        try:
            images = sorted([
                p for p in ONLINE_UPLOAD_DIR.rglob("*")
                if p.is_file() and p.suffix.lower() in _IMG_EXT
            ])
            total = len(images)
            yield _log(f"=== 批量测试开始 (method={method}) ===")
            yield _log(f"共 {total} 张图片")
            yield _log("")

            success_count = 0
            error_count = 0
            all_results = []
            batch_t0 = time.time()

            for idx, img_path in enumerate(images):
                if _batch_cancel.is_set():
                    yield _log(f"\n⛔ 已在第 {idx}/{total} 张处手动停止")
                    break

                rel = img_path.relative_to(ONLINE_UPLOAD_DIR).as_posix()
                yield _log(f"[{idx+1}/{total}] {rel}")

                t0 = time.time()
                result_data = None
                step_count = 0

                try:
                    for event in analyze_plant_image_stream(str(img_path), method=method):
                        if _batch_cancel.is_set():
                            break
                        etype = event.get("type")
                        if etype == "step":
                            step_count += 1
                            desc = event.get("description", "")
                            if any(kw in desc for kw in ["VLM", "复核", "计数", "角果", "Pods"]):
                                yield _log(f"  [{step_count}] {desc}")
                        elif etype == "result":
                            result_data = event
                        elif etype == "error":
                            yield _log(f"  ❌ {event.get('message', 'unknown error')}")
                            error_count += 1
                except Exception as e:
                    yield _log(f"  ❌ Exception: {e}")
                    error_count += 1
                    continue

                elapsed = round(time.time() - t0, 1)
                if result_data:
                    pods = result_data.get("total_pods", 0)
                    parts = result_data.get("plant_count", 0)
                    yield _log(f"  ✅ {parts}部件, {pods}角果, {elapsed}s")
                    all_results.append({
                        "image": rel, "pods": pods,
                        "parts": parts, "time": elapsed,
                    })
                    success_count += 1

                yield _log("")

            batch_elapsed = round(time.time() - batch_t0, 1)
            yield _log(f"{'='*50}")
            yield _log(f"批量测试完成: 成功={success_count}, 失败={error_count}, 总计={total}")
            if all_results:
                total_pods = sum(r["pods"] for r in all_results)
                avg_time = round(sum(r["time"] for r in all_results) / len(all_results), 1)
                yield _log(f"总角果数: {total_pods}, 平均耗时: {avg_time}s/张, 总耗时: {batch_elapsed}s")

            yield f"data: {_json.dumps({'type': 'batch_done', 'success': success_count, 'error': error_count, 'total': total}, ensure_ascii=False)}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield _log(f"❌ 批量测试异常: {e}")
        finally:
            _batch_running.clear()

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/api/batch_stop")
async def batch_stop():
    """Emergency stop for batch test."""
    if not _batch_running.is_set():
        return {"status": "not_running"}
    _batch_cancel.set()
    return {"status": "stopping"}


# Serve frontend (must be after API routes)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
