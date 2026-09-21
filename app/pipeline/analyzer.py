"""
Plant phenotyping analyzer — Direct Doubao Vision labeling.

Flow: upload image → send to Doubao Vision → model labels plants.
No CV preprocessing. Just VLM.
"""

import cv2
import numpy as np
from pathlib import Path
import time
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .vlm_counter import (
    vlm_label_plants, vlm_detect_adhesion, mask_adhesion_regions,
    vlm_verify_crop, apply_recrop, measure_ruler_yellow,
)
from .pod_verify_vlm import verify_pod_markers
from .pod_counter_stalk import count_pods_on_branch as _stalk_counter
from app import cancel, trace
from app.config import RESULT_DIR, CACHE_DIR


def _safe_run_part(value: str, fallback: str = "image") -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe[:80] or fallback


def _new_run_id(image_path: str) -> str:
    stem = _safe_run_part(Path(image_path).stem)
    return f"{stem}__{int(time.time() * 1000)}__{uuid.uuid4().hex[:8]}"


def _upscale_annotation_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(3.0, max(1.0, 1800 / max(1, max(h, w))))
    if scale <= 1.01:
        return img
    return cv2.resize(
        img,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


def analyze_plant_image_stream(
    image_path: str,
    method: str = "stalk",
    run_id: Optional[str] = None,
    cache_key: Optional[str] = None,
    source_label: Optional[str] = None,
):
    """
    Generator that yields each pipeline step as it completes.
    Yields: dict with "type" = "step" | "result"
      - step:   {"type":"step", "run_id":..., "step":N, "url":..., "description":...}
      - result: {"type":"result", "run_id":..., "plant_count":..., "total_pods":..., "plants":..., ...}
    """
    t0 = time.time()
    run_id = _safe_run_part(run_id) if run_id else _new_run_id(image_path)
    trace.set_run(run_id)
    source_path = str(Path(image_path).resolve())
    source_label = source_label or Path(image_path).name
    if method != "stalk":
        trace.event("image_end", status="failed", fail_reason="unsupported_method",
                    label=source_label, method=method)
        yield {
            "type": "error",
            "run_id": run_id,
            "message": f"Unsupported method: {method}. Only 'stalk' is available.",
        }
        return

    image = cv2.imread(image_path)
    if image is None:
        trace.event("image_end", status="failed", fail_reason="cannot_read",
                    label=source_label, duration_s=round(time.time() - t0, 1))
        yield {"type": "error", "run_id": run_id, "message": f"Cannot read image: {image_path}"}
        return
    trace.event("image_start", label=source_label, width=int(image.shape[1]),
                height=int(image.shape[0]))

    stem_name = Path(image_path).stem
    debug_dir = RESULT_DIR / run_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    step_counter = [0]

    def _save_step(img, description, *, base_img=None, editable_points=None):
        step_counter[0] += 1
        fname = f"step_{step_counter[0]:02d}.png"
        fpath = debug_dir / fname
        cv2.imwrite(str(fpath), img)
        rel = fpath.relative_to(RESULT_DIR)
        event = {
            "type": "step",
            "run_id": run_id,
            "step": step_counter[0],
            "url": f"/results/{rel}",
            "description": description,
        }
        if base_img is not None:
            base_fname = f"step_{step_counter[0]:02d}_base.png"
            base_path = debug_dir / base_fname
            cv2.imwrite(str(base_path), base_img)
            base_rel = base_path.relative_to(RESULT_DIR)
            event["editable_base_url"] = f"/results/{base_rel}"
        if editable_points:
            event["editable_points"] = editable_points
        return event

    def _write_debug_json(fname, payload):
        try:
            fpath = debug_dir / fname
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"[Debug] failed to write {fname}: {e}")

    def _write_debug_text(fname, text):
        try:
            fpath = debug_dir / fname
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(text or "")
        except Exception as e:
            print(f"[Debug] failed to write {fname}: {e}")

    def _editable_points_from_markers(markers, fp_ids, restored_ids):
        fp_set = set(fp_ids or [])
        restored_set = set(restored_ids or [])
        points = []
        for m in markers:
            mid = m.get("id")
            mtype = m.get("type")
            if mid is None or "x_norm" not in m or "y_norm" not in m:
                continue
            if mtype == "pod":
                color = "purple" if mid in fp_set else "green"
                state = "false_positive" if mid in fp_set else "confirmed"
            else:
                color = "green" if mid in restored_set else "red"
                state = "restored" if mid in restored_set else "filtered"
            points.append({
                "id": mid,
                "source": mtype,
                "state": state,
                "color": color,
                "x": float(m["x_norm"]),
                "y": float(m["y_norm"]),
            })
        return points

    # Step: Original
    yield _save_step(image, f"原始图片：{source_label}，{image.shape[1]}×{image.shape[0]} 像素")

    if cancel.is_cancelled(run_id):
        trace.event("image_end", status="failed", fail_reason="cancelled",
                    label=source_label, duration_s=round(time.time() - t0, 1))
        yield {"type": "error", "run_id": run_id, "message": "分析已被用户停止。"}
        return

    # Step: VLM
    print(f"[Pipeline] Sending {image.shape[1]}x{image.shape[0]} to Doubao Vision...")
    with trace.phase("vlm_annotate"):
        vlm_result = vlm_label_plants(image)
    plants = vlm_result.get("plants", []) or []
    crops = vlm_result.get("crops", {}) or {}
    raw_response = vlm_result.get("raw_response", "")

    _write_debug_text("vlm_label_raw.txt", raw_response)
    _write_debug_json("vlm_label_debug.json", {
        "source_image": source_path,
        "source_label": source_label,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "parsed_plant_count": len(plants),
        "parsed_crop_count": len(crops),
        "parse_error": bool(vlm_result.get("parse_error")),
        "partial_json": bool(vlm_result.get("partial_json")),
        "plants": plants,
        "ruler": vlm_result.get("ruler"),
        "label_tag": vlm_result.get("label_tag"),
        "sample_id": vlm_result.get("sample_id", ""),
        "raw_response_preview": raw_response[:1000],
    })
    print(f"[Pipeline] VLM parsed plants={len(plants)}, crops={len(crops)}")

    labeled_img = vlm_result.get("labeled_image")
    if labeled_img is not None:
        if plants:
            desc = "VLM 部件识别：豆包视觉模型标注主干、主枝、分枝（彩色框）"
        else:
            desc = "VLM 部件识别：未返回任何主干/主枝/分枝，已保存 VLM 调试文件"
        yield _save_step(labeled_img, desc)

    if not plants:
        api_error = vlm_result.get("api_error")
        trace.event("image_end", status="failed",
                    fail_reason="api_error" if api_error else "no_plants",
                    label=source_label, duration_s=round(time.time() - t0, 1),
                    api_error=api_error,
                    api_status=vlm_result.get("api_status"),
                    parse_error=bool(vlm_result.get("parse_error")),
                    partial_json=bool(vlm_result.get("partial_json")),
                    raw_chars=len(raw_response or ""),
                    crops=len(crops))
        if api_error:
            status = vlm_result.get("api_status")
            detail = str(vlm_result.get("api_error_message") or "")[:200]
            message = f"VLM 接口调用失败：{api_error}"
            if status:
                message += f"（HTTP {status}）"
            message += f"。{detail}"
        else:
            message = (
                "VLM 结构识别没有返回任何主干/主枝/分枝，本次已停止。"
                f"调试文件已保存到 {debug_dir / 'vlm_label_debug.json'} 和 "
                f"{debug_dir / 'vlm_label_raw.txt'}。"
            )
        yield {
            "type": "error",
            "run_id": run_id,
            "message": message,
            "result_dir": str(debug_dir),
        }
        return

    if not crops:
        trace.event("image_end", status="failed", fail_reason="no_crops",
                    label=source_label, duration_s=round(time.time() - t0, 1),
                    plants=len(plants), raw_chars=len(raw_response or ""))
        yield {
            "type": "error",
            "run_id": run_id,
            "message": (
                "VLM 返回了结构文本，但没有任何有效 bbox 可用于裁剪，本次已停止。"
                f"调试文件已保存到 {debug_dir / 'vlm_label_debug.json'}。"
            ),
            "result_dir": str(debug_dir),
        }
        return

    # ── Ruler detection + CV measurement ──
    _t_ruler = time.perf_counter()
    px_per_cm = None
    ruler_found = False
    ruler_pixel_length = None

    ruler_data = vlm_result.get("ruler", {})
    if ruler_data and ruler_data.get("found"):
        ruler_bbox_norm = ruler_data["bbox"]
        h_img, w_img = image.shape[:2]
        rx1 = max(0, int(ruler_bbox_norm[0] / 1000.0 * w_img))
        ry1 = max(0, int(ruler_bbox_norm[1] / 1000.0 * h_img))
        rx2 = min(w_img, int(ruler_bbox_norm[2] / 1000.0 * w_img))
        ry2 = min(h_img, int(ruler_bbox_norm[3] / 1000.0 * h_img))
        if rx2 > rx1 and ry2 > ry1:
            ruler_crop = image[ry1:ry2, rx1:rx2].copy()
            ruler_meas = measure_ruler_yellow(ruler_crop)
            if ruler_meas.get("detected"):
                ruler_found = True
                px_per_cm = ruler_meas["px_per_cm"]
                ruler_pixel_length = ruler_meas["yellow_length_px"]
                debug_img = ruler_meas.get("debug_image")
                if debug_img is not None:
                    yield _save_step(debug_img,
                        f"直尺检测：黄色段 {ruler_pixel_length:.0f} px, "
                        f"比例 {px_per_cm:.2f} px/cm（1m = 100cm）")
            else:
                print("[Pipeline] Ruler bbox found but CV yellow detection failed")
        else:
            print(f"[Pipeline] Invalid ruler bbox: {ruler_bbox_norm}")

    trace.event("phase_end", phase="cv_ruler",
                duration_ms=round((time.perf_counter() - _t_ruler) * 1000, 1))

    # ── Label tag processing ──
    sample_id = vlm_result.get("sample_id", "")
    label_found = False
    label_is_flipped = False

    label_data = vlm_result.get("label_tag", {})
    if label_data and label_data.get("found"):
        label_found = True
        label_is_flipped = label_data.get("is_flipped", False)
        label_rows = label_data.get("rows", ["", "", ""])
        if label_is_flipped:
            sample_id += " (标签翻转)"
        yield _save_step(
            cv2.resize(image, (800, int(800 * image.shape[0] / image.shape[1]))),
            f"标签识别：sample_id = {sample_id}"
            f"{'（检测到标签倒置）' if label_is_flipped else ''}"
        )

    # Write ruler/label debug info
    _write_debug_json("ruler_label_debug.json", {
        "ruler_found": ruler_found,
        "ruler_pixel_length": ruler_pixel_length,
        "px_per_cm": px_per_cm,
        "label_found": label_found,
        "sample_id": sample_id,
        "label_is_flipped": label_is_flipped,
    })

    # Step: Crops
    for p in plants:
        pid = p.get("id")
        if pid in crops:
            label = p.get("label", "unknown")
            crop_h, crop_w = crops[pid].shape[:2]
            yield _save_step(crops[pid], f"裁剪 #{pid} {label}：{crop_w}×{crop_h} 像素")

    # ── Step: VLM crop verification (parallel) ──
    # Submit all verify_crop calls concurrently, then process results.
    verify_results = {}
    plants_to_remove = []
    verify_plants = [p for p in plants
                     if p.get("id") in crops and p.get("label", "") != "主干"]

    if cancel.is_cancelled(run_id):
        trace.event("image_end", status="failed", fail_reason="cancelled",
                    label=source_label, duration_s=round(time.time() - t0, 1),
                    cancelled_at="before_verify_crops")
        yield {"type": "error", "run_id": run_id, "message": "分析已被用户停止。"}
        return

    _t_verify = time.perf_counter()
    if verify_plants:
        print(f"[Pipeline] Verifying {len(verify_plants)} crops in parallel...")
        with ThreadPoolExecutor(max_workers=len(verify_plants)) as executor:
            future_map = {}
            for p in verify_plants:
                fut = executor.submit(
                    vlm_verify_crop,
                    original_image=image,
                    crop_image=crops[p["id"]],
                    plant=p,
                    all_plants=plants,
                )
                future_map[fut] = p

            for fut in as_completed(future_map):
                p = future_map[fut]
                pid = p.get("id")
                plabel = p.get("label", "")
                try:
                    verdict = fut.result()
                except Exception as e:
                    print(f"[Pipeline] Verify #{pid} failed: {e}")
                    verdict = {"action": "keep", "reason": f"verify error: {e}"}
                verify_results[pid] = verdict

        # Process results in original order (for consistent step numbering)
        for p in verify_plants:
            pid = p.get("id")
            plabel = p.get("label", "")
            verdict = verify_results.get(pid, {"action": "keep", "reason": "no result"})
            action = verdict.get("action", "keep")
            reason = verdict.get("reason", "")
            p["verify"] = verdict

            if action == "keep":
                yield _save_step(crops[pid],
                                 f"[#{pid} {plabel}] ✅ VLM 复核：通过 — {reason}")
            elif action == "delete":
                yield _save_step(crops[pid],
                                 f"[#{pid} {plabel}] ❌ VLM 复核：剔除 — {reason}")
                plants_to_remove.append(pid)
            elif action == "recrop":
                new_bbox = verdict.get("new_bbox")
                if new_bbox:
                    new_crop, new_pixel_bbox = apply_recrop(image, p, new_bbox)
                    ox1, oy1, ox2, oy2 = new_pixel_bbox
                    def _to_local(pt):
                        if not pt or len(pt) != 2:
                            return None
                        lx = max(ox1, min(ox2 - 1, int(pt[0]))) - ox1
                        ly = max(oy1, min(oy2 - 1, int(pt[1]))) - oy1
                        return [lx, ly]
                    if p.get("stem_start_px"):
                        p["stem_start_local"] = _to_local(p.get("stem_start_px"))
                    if p.get("stem_end_px"):
                        p["stem_end_local"] = _to_local(p.get("stem_end_px"))
                    crops[pid] = new_crop
                    ch, cw = new_crop.shape[:2]
                    yield _save_step(new_crop,
                                     f"[#{pid} {plabel}] 🔄 VLM 复核：重截 ({cw}×{ch}) — {reason}")
                else:
                    yield _save_step(crops[pid],
                                     f"[#{pid} {plabel}] ⚠️ VLM 复核：建议重截但未给新 bbox，保留原图 — {reason}")

    trace.event("phase_end", phase="vlm_verify_crops", n_calls=len(verify_plants),
                duration_ms=round((time.perf_counter() - _t_verify) * 1000, 1))

    # Apply deletions
    if plants_to_remove:
        plants = [p for p in plants if p.get("id") not in plants_to_remove]
        for pid in plants_to_remove:
            crops.pop(pid, None)

    # ── Cache verified crops (post-VLM-verification) ──
    # Saves after keep/delete/recrop so downstream --cv-only runs use clean data.
    try:
        _cache_dir = CACHE_DIR / (cache_key or stem_name)
        _cache_dir.mkdir(parents=True, exist_ok=True)
        _crop_files = {}
        for p in plants:
            pid = p.get("id")
            if pid in crops:
                label = p.get("label", "unknown")
                crop_fname = f"crop_{pid}_{label}.jpg"
                cv2.imwrite(str(_cache_dir / crop_fname), crops[pid])
                _crop_files[str(pid)] = crop_fname
        _meta = {
            "plants": plants,
            "crop_files": _crop_files,
            "raw_response": vlm_result.get("raw_response", ""),
        }
        with open(_cache_dir / "vlm_result.json", "w", encoding="utf-8") as _f:
            json.dump(_meta, _f, ensure_ascii=False, indent=2)
        print(f"[Cache] Saved {len(_crop_files)} verified crops to {_cache_dir}")
    except Exception as _e:
        print(f"[Cache] Warning: failed to save cache: {_e}")

    # # Steps: Adhesion detection per crop → mask foreign plant regions
    # (temporarily disabled — testing tight crop without padding first)
    # for p in plants:
    #     pid = p.get("id")
    #     plabel = p.get("label", "")
    #     if plabel == "主干" or pid not in crops:
    #         continue
    #     print(f"[Pipeline] Checking foreign adhesion on #{pid} {plabel}...")
    #     adhesion_result = vlm_detect_adhesion(
    #         crops[pid], plant_id=pid, label=plabel,
    #         all_plants=plants, current_plant=p)
    #     foreign_regions = adhesion_result.get("regions", [])
    #     p["adhesion"] = {
    #         "has_foreign": adhesion_result.get("has_foreign", False),
    #         "region_count": len(foreign_regions),
    #     }
    #     if foreign_regions:
    #         descs = "; ".join(r.get("description", "") for r in foreign_regions)
    #         masked_crop, debug_crop = mask_adhesion_regions(crops[pid], foreign_regions)
    #         yield _save_step(debug_crop,
    #                          f"[#{pid} {plabel}] 异株检测：发现 {len(foreign_regions)} 个混入区域（红色）— {descs}")
    #         yield _save_step(masked_crop,
    #                          f"[#{pid} {plabel}] 异株遮涂：已涂黑 {len(foreign_regions)} 个区域")
    #         crops[pid] = masked_crop  # use cleaned version for pod counting
    #     else:
    #         yield _save_step(crops[pid],
    #                          f"[#{pid} {plabel}] 异株检测：未发现其他植株混入")

    # Steps: pod counting per branch (CPU, fast)
    if cancel.is_cancelled(run_id):
        trace.event("image_end", status="failed", fail_reason="cancelled",
                    label=source_label, duration_s=round(time.time() - t0, 1),
                    cancelled_at="before_cv_pod_count")
        yield {"type": "error", "run_id": run_id, "message": "分析已被用户停止。"}
        return
    _t_cv = time.perf_counter()
    _counter = _stalk_counter
    pod_plants = [p for p in plants
                  if p.get("label", "") != "主干" and p.get("id") in crops]
    pod_results_map = {}  # pid -> pod_result

    # Also run the stalk counter on trunk — same logic as branches, just take
    # main_stem_length (pod count on trunk is ~0, we don't care).
    trunk_plant = next((p for p in plants
                        if p.get("label", "") == "主干" and p.get("id") in crops), None)
    trunk_pixel_len = 0
    if trunk_plant:
        tid = trunk_plant.get("id")
        print(f"[Pipeline] Measuring trunk #{tid} for stem length...")
        trunk_result = _counter(
            crops[tid],
            stem_start_local=trunk_plant.get("stem_start_local"),
            stem_end_local=trunk_plant.get("stem_end_local"),
            skip_ruler_rejection=True,
        )
        trunk_pixel_len = trunk_result.get("main_stem_length", 0)
        trunk_plant["pod_count"] = trunk_result.get("pod_count", 0)
        trunk_plant["main_stem_length"] = trunk_pixel_len
        pod_results_map[tid] = trunk_result
        for step_img, step_desc in trunk_result.get("step_images", []):
            yield _save_step(step_img, f"[#{tid} 主干] {step_desc}")
        print(f"  [Trunk] main_stem_length = {trunk_pixel_len}px")

    for p in pod_plants:
        pid = p.get("id")
        plabel = p.get("label", "")
        print(f"[Pipeline] Counting pods on #{pid} {plabel} (method={method})...")
        pod_result = _counter(
            crops[pid],
            stem_start_local=p.get("stem_start_local"),
            stem_end_local=p.get("stem_end_local"),
        )
        p["pod_count"] = pod_result.get("pod_count", 0)
        p["main_stem_length"] = pod_result.get("main_stem_length", 0)
        pod_results_map[pid] = pod_result

        for step_img, step_desc in pod_result.get("step_images", []):
            yield _save_step(step_img, f"[#{pid} {plabel}] {step_desc}")

    trace.event("phase_end", phase="cv_pod_count", parts=len(pod_plants),
                duration_ms=round((time.perf_counter() - _t_cv) * 1000, 1))

    # ── VLM pod verification (parallel) ──
    vlm_verify_inputs = []
    for p in pod_plants:
        pid = p.get("id")
        pod_result = pod_results_map[pid]
        markers = pod_result.get("markers", [])
        if markers:
            vlm_verify_inputs.append((p, pod_result, markers))

    # Also verify trunk pods
    if trunk_plant and trunk_pixel_len > 0:
        trunk_markers = trunk_result.get("markers", [])
        if trunk_markers:
            vlm_verify_inputs.append((trunk_plant, trunk_result, trunk_markers))

    if cancel.is_cancelled(run_id):
        trace.event("image_end", status="failed", fail_reason="cancelled",
                    label=source_label, duration_s=round(time.time() - t0, 1),
                    cancelled_at="before_vlm_pod_review")
        yield {"type": "error", "run_id": run_id, "message": "分析已被用户停止。"}
        return

    vlm_verify_results = {}  # pid -> verify_result
    _t_podreview = time.perf_counter()
    if vlm_verify_inputs:
        print(f"[Pipeline] VLM verifying pods on {len(vlm_verify_inputs)} branches in parallel...")
        with ThreadPoolExecutor(max_workers=len(vlm_verify_inputs)) as executor:
            future_map = {}
            for p, pod_result, markers in vlm_verify_inputs:
                pid = p.get("id")
                fut = executor.submit(
                    verify_pod_markers,
                    crop_bgr=crops[pid],
                    debug_image=pod_result.get("debug_image", crops[pid]),
                    markers=markers,
                    pod_count=p["pod_count"],
                )
                future_map[fut] = p

            for fut in as_completed(future_map):
                p = future_map[fut]
                pid = p.get("id")
                try:
                    vlm_verify_results[pid] = fut.result()
                except Exception as e:
                    print(f"[Pipeline] VLM verify #{pid} failed: {e}")
                    vlm_verify_results[pid] = {
                        "missed_count": 0,
                        "restored_filtered_ids": [],
                        "restored_filtered_confidences": {},
                        "false_positive_ids": [],
                        "false_positive_confidences": {},
                        "adjusted_pod_count": p["pod_count"],
                        "reason": f"VLM error: {e}",
                        "verified_image": crops[pid].copy(),
                    }

    trace.event("phase_end", phase="vlm_pod_review", n_calls=len(vlm_verify_inputs),
                duration_ms=round((time.perf_counter() - _t_podreview) * 1000, 1))

    # Yield VLM verify results in original order
    for p, pod_result, markers in vlm_verify_inputs:
        pid = p.get("id")
        plabel = p.get("label", "")
        verify_result = vlm_verify_results.get(pid)
        if not verify_result:
            continue

        missed = verify_result.get("missed_count", 0)
        restored_ids = verify_result.get("restored_filtered_ids", [])
        restored_conf = verify_result.get("restored_filtered_confidences", {})
        fp_ids = verify_result.get("false_positive_ids", [])
        fp_conf = verify_result.get("false_positive_confidences", {})
        adjusted = verify_result.get("adjusted_pod_count", p["pod_count"])
        v_reason = verify_result.get("reason", "")

        if missed > 0 or restored_ids or fp_ids:
            p["pod_count_before_vlm"] = p["pod_count"]
            p["pod_count"] = adjusted
            p["vlm_verify"] = {
                "missed": missed,
                "restored_filtered": restored_ids,
                "restored_filtered_confidences": restored_conf,
                "false_positives": fp_ids,
                "false_positive_confidences": fp_conf,
                "reason": v_reason,
            }
            change_desc = []
            if missed > 0:
                restore_items = []
                for rid in restored_ids:
                    conf = restored_conf.get(str(rid), restored_conf.get(rid))
                    if conf is None:
                        restore_items.append(f"F{rid}")
                    else:
                        restore_items.append(f"F{rid}({float(conf):.2f})")
                if restore_items:
                    change_desc.append(f"补回 +{len(restored_ids)} ({','.join(restore_items)})")
                else:
                    change_desc.append(f"漏检 +{missed}")
            if fp_ids:
                fp_items = []
                for fp_id in fp_ids:
                    conf = fp_conf.get(str(fp_id), fp_conf.get(fp_id))
                    if conf is None:
                        fp_items.append(f"P{fp_id}")
                    else:
                        fp_items.append(f"P{fp_id}({float(conf):.2f})")
                change_desc.append(f"误判 -{len(fp_ids)} ({','.join(fp_items)})")
            yield _save_step(
                verify_result.get("verified_image", crops[pid]),
                f"[#{pid} {plabel}] 🔍 VLM 角果复核：{' / '.join(change_desc)}，"
                f"{p['pod_count_before_vlm']} → {adjusted} — {v_reason}",
                base_img=_upscale_annotation_image(crops[pid]),
                editable_points=_editable_points_from_markers(markers, fp_ids, restored_ids),
            )
        else:
            p["vlm_verify"] = {
                "missed": 0,
                "restored_filtered": [],
                "restored_filtered_confidences": {},
                "false_positives": [],
                "false_positive_confidences": {},
                "reason": v_reason,
            }
            yield _save_step(
                verify_result.get("verified_image", crops[pid]),
                f"[#{pid} {plabel}] ✅ VLM 角果复核：计数无调整 — {v_reason}",
                base_img=_upscale_annotation_image(crops[pid]),
                editable_points=_editable_points_from_markers(markers, [], []),
            )

    # Final result
    trunk_pod_count = int(trunk_plant.get("pod_count", 0)) if trunk_plant else 0
    main_pod_count = sum(int(p.get("pod_count", 0)) for p in plants if p.get("label") == "主枝")
    main_stem_plus_main_pods = trunk_pod_count + main_pod_count
    branch_pod_count = sum(int(p.get("pod_count", 0)) for p in plants if p.get("label") == "分枝")
    branch_pod_breakdown = [
        {
            "id": p.get("id"),
            "label": p.get("label"),
            "pod_count": int(p.get("pod_count", 0)),
        }
        for p in plants
        if p.get("label") == "分枝"
    ]
    branch_count = len(branch_pod_breakdown)
    total_pods = sum(p.get("pod_count", 0) for p in plants)
    elapsed = round(time.time() - t0, 2)

    # Compute real-world stem length
    main_stem_length_cm = None
    if trunk_pixel_len > 0 and px_per_cm and px_per_cm > 0:
        main_stem_length_cm = round(trunk_pixel_len / px_per_cm, 1)

    trace.event("image_end", status="completed", label=source_label,
                duration_s=elapsed, plants=len(plants), crops=len(crops),
                branch_count=branch_count, total_pods=total_pods,
                ruler_found=ruler_found, main_stem_length_cm=main_stem_length_cm)

    yield {
        "type": "result",
        "run_id": run_id,
        "source_image": source_path,
        "source_label": source_label,
        "result_dir": str(debug_dir),
        "plant_count": len(plants),
        "branch_count": branch_count,
        "main_inflorescence_pod_count": main_pod_count,
        "main_stem_plus_main_pod_count": main_stem_plus_main_pods,
        "branch_pod_count": branch_pod_count,
        "branch_total_pod_count": branch_pod_count,
        "main_stem_length": trunk_pixel_len or None,
        "main_stem_length_cm": main_stem_length_cm,
        "total_pods": total_pods,
        "sample_id": sample_id or "",
        "ruler_found": ruler_found,
        "ruler_pixel_length": ruler_pixel_length,
        "px_per_cm": px_per_cm,
        "label_found": label_found,
        "label_is_flipped": label_is_flipped,
        "summary": {
            "branch_count": branch_count,
            "predicted_branch_count": branch_count,
            "main_inflorescence_pod_count": main_pod_count,
            "main_stem_plus_main_pod_count": main_stem_plus_main_pods,
            "branch_pod_count": branch_pod_count,
            "branch_total_pod_count": branch_pod_count,
            "total_pod_count": total_pods,
            "main_stem_length": trunk_pixel_len or None,
            "main_stem_length_cm": main_stem_length_cm,
            "sample_id": sample_id or "",
            "ruler_found": ruler_found,
            "label_found": label_found,
            "branch_pod_breakdown": branch_pod_breakdown,
        },
        "plants": plants,
        "processing_time_s": elapsed,
        "total_steps": step_counter[0],
    }
    print(f"\n[Done] time: {elapsed}s, steps: {step_counter[0]}")
