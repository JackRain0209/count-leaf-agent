"""
CV Agent — Plant Phenotyping API Server
FastAPI backend for rapeseed plant analysis.
"""

import shutil
import uuid
import re
import time
from pathlib import Path

import json as _json

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from urllib.parse import quote
from app.excel_export import build_workbook, workbook_to_bytes, make_filename
from app.config import (
    UPLOAD_DIR,
    RESULT_DIR,
    ONLINE_UPLOAD_DIR,
    HISTORY_DB_PATH,
    HOST,
    PORT,
)
from app.history import HistoryStore
from app.pipeline.analyzer import analyze_plant_image_stream

app = FastAPI(title="CV Agent - Plant Phenotyping", version="0.1.0")
history_store = HistoryStore(HISTORY_DB_PATH)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


def _safe_run_part(value: str, fallback: str = "image") -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe[:80] or fallback


def _make_run_id(source_name: str) -> str:
    stem = _safe_run_part(Path(source_name).stem)
    return f"{stem}__{int(time.time() * 1000)}__{uuid.uuid4().hex[:8]}"


def _write_result_snapshot(run_id: str, result: dict) -> None:
    result_dir = RESULT_DIR / run_id
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / "result.json"
    temp_file = result_dir / "result.json.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        _json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    temp_file.replace(result_file)


def _analysis_stream(
    image_path: str,
    run_id: str,
    source_label: str,
    source_kind: str,
):
    steps = []
    terminal_event_seen = False
    history_store.start(run_id, source_label, image_path, source_kind)
    try:
        for event in analyze_plant_image_stream(
            image_path,
            method="stalk",
            run_id=run_id,
            cache_key=run_id,
            source_label=source_label,
        ):
            event_type = event.get("type")
            if event_type == "step":
                steps.append(event)
            elif event_type == "result":
                event = history_store.complete(run_id, event, steps)
                _write_result_snapshot(run_id, event)
                terminal_event_seen = True
            elif event_type == "error":
                history_store.fail(run_id, event.get("message", "分析失败"), steps)
                terminal_event_seen = True
            yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"

        if not terminal_event_seen:
            message = "分析流程结束但没有返回结果"
            history_store.fail(run_id, message, steps)
            yield f"data: {_json.dumps({'type': 'error', 'run_id': run_id, 'message': message}, ensure_ascii=False)}\n\n"
    except GeneratorExit:
        history_store.fail(run_id, "客户端中断分析", steps)
        raise
    except Exception as exc:
        import traceback
        traceback.print_exc()
        history_store.fail(run_id, str(exc), steps)
        event = {"type": "error", "run_id": run_id, "message": str(exc)}
        yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"


# Serve result images
app.mount("/results", NoCacheStaticFiles(directory=str(RESULT_DIR)), name="results")
# Serve online_uploads (for thumbnail preview in picker)
app.mount("/online_uploads", NoCacheStaticFiles(directory=str(ONLINE_UPLOAD_DIR)), name="online_uploads")

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
    method = (payload or {}).get("method", "stalk")
    if method != "stalk":
        raise HTTPException(400, "Only the stalk method is supported")
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

    run_id = _make_run_id(Path(rel).name)

    return StreamingResponse(
        _analysis_stream(str(target), run_id, rel, "online"),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "cv-agent"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), method: str = "stalk"):
    """
    Upload a plant image → SSE stream of pipeline steps.
    Each line: data: {"type":"step"|"result"|"error", ...}
    """
    ext = Path(file.filename or "unknown.jpg").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    if method != "stalk":
        raise HTTPException(400, "Only the stalk method is supported")

    file_id = uuid.uuid4().hex[:8]
    save_name = f"{file_id}{ext}"
    save_path = UPLOAD_DIR / save_name
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    run_id = _make_run_id(file.filename or save_name)

    return StreamingResponse(
        _analysis_stream(
            str(save_path),
            run_id,
            file.filename or save_name,
            "upload",
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )



@app.get("/api/history")
def list_history(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    q: str = Query("", max_length=120),
):
    return history_store.list(limit=limit, offset=offset, query=q)

@app.get("/api/history/export")
def export_history(
    q: str = Query("", max_length=120),
    run_ids: str = Query("", max_length=4000),
):
    """导出。
    - 传 run_ids（逗号分隔）时：只导出这些 run_id，忽略 q
    - 否则：按 q 筛选导出
    """
    rid_list = [x.strip() for x in run_ids.split(",") if x.strip()] if run_ids else []
    if rid_list:
        records = history_store.query_by_run_ids(rid_list)
        filename = make_filename(tag="选中")
    else:
        records = history_store.query_all_for_export(query=q)
        filename = make_filename(tag=(q or "").strip()[:20])
    wb = build_workbook(records)
    data = workbook_to_bytes(wb)
    return StreamingResponse(
        BytesIO(data),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
            "Content-Length": str(len(data)),
        },
    )
@app.post("/api/history/{run_id}/restore")
def restore_history_result(run_id: str):
    try:
        result = history_store.restore(run_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    _write_result_snapshot(run_id, result)
    return history_store.get(run_id)


@app.put("/api/history/{run_id}")
def update_history_result(run_id: str, payload: dict):
    try:
        result = history_store.update(run_id, payload or {})
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    _write_result_snapshot(run_id, result)
    return history_store.get(run_id)


@app.get("/api/history/{run_id}")
def get_history_result(run_id: str):
    record = history_store.get(run_id)
    if not record:
        raise HTTPException(404, "历史记录不存在")
    return record


@app.get("/api/results/{run_id}")
def get_result(run_id: str):
    """Get the current persisted result, including any manual revisions."""
    record = history_store.get(run_id)
    if record and record.get("result"):
        return record["result"]

    result_dir = RESULT_DIR / run_id
    result_file = result_dir / "result.json"
    if not result_file.exists():
        raise HTTPException(404, "Result not found")
    with open(result_file, encoding="utf-8") as f:
        return _json.load(f)


# Serve frontend (must be after API routes)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", NoCacheStaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
