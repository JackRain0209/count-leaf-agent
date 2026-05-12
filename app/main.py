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
    if method not in ("skeleton", "graph", "plantcv", "stalk"):
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


def _load_ground_truth():
    """Load ground truth from xlsx in online_uploads."""
    import openpyxl
    # Find the xlsx file
    xlsx_files = list(ONLINE_UPLOAD_DIR.rglob("*终版*.xlsx"))
    if not xlsx_files:
        # Fallback: any xlsx
        xlsx_files = list(ONLINE_UPLOAD_DIR.rglob("*.xlsx"))
    xlsx_files = [f for f in xlsx_files if not f.name.startswith("~$")]
    if not xlsx_files:
        print("[GT] No xlsx found in online_uploads")
        return {}
    xlsx_path = xlsx_files[0]
    print(f"[GT] Loading ground truth from {xlsx_path.name}")
    gt = {}
    try:
        wb = openpyxl.load_workbook(str(xlsx_path), read_only=True)
        ws = wb.active
        for row in ws.iter_rows(min_row=2, values_only=True):
            if len(row) < 8:
                continue
            n, rep, variety, photo, branches, main_pods, branch_pods, total_pods = row[:8]
            if not n or photo is None:
                continue
            # Column I (index 8) = 备注, non-empty means dirty data, skip
            remark = row[8] if len(row) > 8 else None
            if remark:
                continue
            # Build photo name: {A}.{B}.{C}.{D} e.g. N0.1.1.1
            # A is string (N0/N1/...), B/C/D are numeric, convert to int
            photo_name = f"{str(n).strip()}.{int(rep)}.{int(variety)}.{int(photo)}"
            # Handle int fields
            try:
                gt[photo_name] = {
                    "gt_main": int(main_pods or 0),
                    "gt_branch": int(branch_pods or 0),
                    "gt_total": int(total_pods or 0),
                    "gt_branches": int(branches or 0),
                }
            except (ValueError, TypeError):
                continue
        wb.close()
    except Exception as e:
        print(f"[GT] Error loading xlsx: {e}")
    print(f"[GT] Loaded {len(gt)} ground truth entries")
    return gt


def _calc_stats(results):
    """Calculate summary stats from a list of result dicts with gt."""
    n = len(results)
    if n == 0:
        return {}
    sum_err = sum(r["err"] for r in results)
    sum_abs_err = sum(abs(r["err"]) for r in results)
    sum_sq_err = sum(r["err"] ** 2 for r in results)
    sum_acc = sum(r["acc"] for r in results)
    sum_gt = sum(r["gt_total"] for r in results)
    sum_algo = sum(r["algo_total"] for r in results)
    mae = round(sum_abs_err / n, 2)
    mse = round(sum_sq_err / n, 2)
    rmse = round(mse ** 0.5, 2)
    avg_acc = round(sum_acc / n, 1)
    overall_acc = round((1 - abs(sum_algo - sum_gt) / sum_gt) * 100, 1) if sum_gt else 0
    avg_time = round(sum(r["time"] for r in results) / n, 1)
    # Main / Branch
    avg_acc_m = round(sum(r.get("acc_m", 0) for r in results) / n, 1)
    avg_acc_b = round(sum(r.get("acc_b", 0) for r in results) / n, 1)
    sum_gt_m = sum(r.get("gt_main", 0) for r in results)
    sum_algo_m = sum(r.get("algo_main", 0) for r in results)
    sum_gt_b = sum(r.get("gt_branch", 0) for r in results)
    sum_algo_b = sum(r.get("algo_branch", 0) for r in results)
    overall_acc_m = round((1 - abs(sum_algo_m - sum_gt_m) / sum_gt_m) * 100, 1) if sum_gt_m else 0
    overall_acc_b = round((1 - abs(sum_algo_b - sum_gt_b) / sum_gt_b) * 100, 1) if sum_gt_b else 0
    return {
        "n": n, "mae": mae, "mse": mse, "rmse": rmse,
        "avg_acc": avg_acc, "overall_acc": overall_acc,
        "sum_gt": sum_gt, "sum_algo": sum_algo,
        "mean_err": round(sum_err / n, 2), "avg_time": avg_time,
        "avg_acc_m": avg_acc_m, "avg_acc_b": avg_acc_b,
        "overall_acc_m": overall_acc_m, "overall_acc_b": overall_acc_b,
        "sum_gt_m": sum_gt_m, "sum_algo_m": sum_algo_m,
        "sum_gt_b": sum_gt_b, "sum_algo_b": sum_algo_b,
    }


def _fmt_stats(s, label=""):
    lines = []
    if label:
        lines.append(f"── {label} ({s['n']}张) ──")
    lines.append(f"  总计: GT={s['sum_gt']} 算={s['sum_algo']}  "
                 f"误差={s['mean_err']}  MAE={s['mae']}  RMSE={s['rmse']}")
    lines.append(f"  总体Acc={s['overall_acc']}%  逐图平均Acc={s['avg_acc']}%  "
                 f"平均耗时={s['avg_time']}s")
    lines.append(f"  主枝: GT={s['sum_gt_m']} 算={s['sum_algo_m']}  "
                 f"总体Acc={s['overall_acc_m']}%  逐图平均Acc={s['avg_acc_m']}%")
    lines.append(f"  分枝: GT={s['sum_gt_b']} 算={s['sum_algo_b']}  "
                 f"总体Acc={s['overall_acc_b']}%  逐图平均Acc={s['avg_acc_b']}%")
    return lines


@app.post("/api/batch_test")
async def batch_test():
    """Run full pipeline on all online images with accuracy metrics. SSE stream."""
    if _batch_running.is_set():
        raise HTTPException(409, "Batch test already running")

    _batch_cancel.clear()
    _batch_running.set()
    method = "stalk"
    gt_map = _load_ground_truth()

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
            yield _log(f"共 {total} 张图片, GT数据 {len(gt_map)} 条")
            yield _log(f"{'Photo':<14} {'GT':>4} {'Algo':>5} {'Err':>5} {'Acc%':>6} "
                       f"{'M_Acc':>6} {'B_Acc':>6} {'Time':>5}")
            yield _log("-" * 70)

            success_count = 0
            error_count = 0
            all_results = []
            batch_t0 = time.time()

            for idx, img_path in enumerate(images):
                if _batch_cancel.is_set():
                    yield _log(f"\n⛔ 已在第 {idx}/{total} 张处手动停止")
                    break

                rel = img_path.relative_to(ONLINE_UPLOAD_DIR).as_posix()
                photo_name = img_path.stem
                yield _log(f"▶ [{idx+1}/{total}] {photo_name} 开始分析...")

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
                            # Show key steps as progress
                            if any(kw in desc for kw in ["VLM", "复核", "计数", "角果", "Pods", "裁剪", "骨架", "主干"]):
                                yield _log(f"  [{step_count}] {desc}")
                        elif etype == "result":
                            result_data = event
                        elif etype == "error":
                            yield _log(f"  ❌ {photo_name}: {event.get('message', 'unknown error')}")
                            error_count += 1
                except Exception as e:
                    yield _log(f"  ❌ {photo_name}: {e}")
                    error_count += 1
                    continue

                elapsed = round(time.time() - t0, 1)
                if result_data:
                    algo_total = result_data.get("total_pods", 0)
                    plants = result_data.get("plants", [])
                    algo_main = sum(p.get("pod_count", 0) for p in plants if p.get("label") == "主枝")
                    algo_branch = sum(p.get("pod_count", 0) for p in plants if p.get("label") == "分枝")
                    parts = len([p for p in plants if p.get("label") != "主干"])

                    g = gt_map.get(photo_name)
                    if g:
                        gt_total = g["gt_total"]
                        gt_main = g["gt_main"]
                        gt_branch = g["gt_branch"]
                        err = algo_total - gt_total
                        err_m = algo_main - gt_main
                        err_b = algo_branch - gt_branch
                        acc = max(0, round((1 - abs(err) / gt_total) * 100, 1)) if gt_total else 0
                        acc_m = max(0, round((1 - abs(err_m) / gt_main) * 100, 1)) if gt_main else 0
                        acc_b = max(0, round((1 - abs(err_b) / gt_branch) * 100, 1)) if gt_branch else 0

                        yield _log(f"{photo_name:<14} {gt_total:>4} {algo_total:>5} "
                                   f"{err:>+5} {acc:>5.1f}% {acc_m:>5.1f}% {acc_b:>5.1f}% {elapsed:>5.1f}s")
                        yield _log(f"  主枝: GT={gt_main} 算={algo_main} err={err_m:+d}  "
                                   f"分枝: GT={gt_branch} 算={algo_branch} err={err_b:+d}  "
                                   f"部件={parts}")
                        # Per-plant detail
                        for p in plants:
                            if p.get("label") == "主干":
                                continue
                            pid = p.get("id", "?")
                            plabel = p.get("label", "?")
                            pc = p.get("pod_count", 0)
                            vlm_v = p.get("vlm_verify")
                            vlm_note = ""
                            if vlm_v:
                                fp = vlm_v.get("false_positives", [])
                                if fp:
                                    vlm_note = f" VLM:-{len(fp)}"
                            yield _log(f"    #{pid} {plabel}: {pc}个{vlm_note}")
                    else:
                        yield _log(f"{photo_name:<14}  --- {algo_total:>5}   ---    ---    ---    ---  {elapsed:>5.1f}s (无GT)")
                        acc = 0; acc_m = 0; acc_b = 0
                        gt_main = 0; gt_branch = 0; gt_total = 0
                        err = 0; err_m = 0; err_b = 0

                    all_results.append({
                        "photo": photo_name,
                        "algo_total": algo_total, "algo_main": algo_main,
                        "algo_branch": algo_branch, "parts": parts,
                        "gt_total": gt_total, "gt_main": gt_main, "gt_branch": gt_branch,
                        "err": err, "err_m": err_m, "err_b": err_b,
                        "acc": acc, "acc_m": acc_m, "acc_b": acc_b,
                        "has_gt": bool(g),
                        "time": elapsed,
                    })
                    success_count += 1

                    # Summary every 10 images
                    gt_results = [r for r in all_results if r["has_gt"]]
                    if len(gt_results) > 0 and len(gt_results) % 10 == 0:
                        s = _calc_stats(gt_results)
                        yield _log("")
                        for line in _fmt_stats(s, f"前 {len(gt_results)} 张小结"):
                            yield _log(line)
                        yield _log("")

            # ── Final summary ──
            batch_elapsed = round(time.time() - batch_t0, 1)
            yield _log("")
            yield _log("=" * 55)

            gt_results = [r for r in all_results if r["has_gt"]]
            if gt_results:
                s = _calc_stats(gt_results)
                for line in _fmt_stats(s, "最终汇总"):
                    yield _log(line)
            yield _log(f"  成功={success_count}  失败={error_count}  总计={len(images)}  总耗时={batch_elapsed}s")
            yield _log("=" * 55)

            yield f"data: {_json.dumps({'type': 'batch_done', 'success': success_count, 'error': error_count, 'total': len(images)}, ensure_ascii=False)}\n\n"

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
