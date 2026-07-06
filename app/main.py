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

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import UPLOAD_DIR, RESULT_DIR, ONLINE_UPLOAD_DIR, HOST, PORT
from app.pipeline.analyzer import analyze_plant_image_stream

app = FastAPI(title="CV Agent - Plant Phenotyping", version="0.1.0")

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

    def _stream():
        try:
            for event in analyze_plant_image_stream(
                str(target),
                method=method,
                run_id=run_id,
                cache_key=run_id,
                source_label=rel,
            ):
                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {_json.dumps({'type': 'error', 'run_id': run_id, 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _stream(),
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

    def _stream():
        try:
            for event in analyze_plant_image_stream(
                str(save_path),
                method=method,
                run_id=run_id,
                cache_key=run_id,
                source_label=file.filename or save_name,
            ):
                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {_json.dumps({'type': 'error', 'run_id': run_id, 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


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


# Serve frontend (must be after API routes)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", NoCacheStaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
