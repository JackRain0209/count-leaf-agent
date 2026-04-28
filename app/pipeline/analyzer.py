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

from .vlm_counter import (
    vlm_label_plants, vlm_detect_adhesion, mask_adhesion_regions,
    vlm_verify_crop, apply_recrop,
)
from .pod_verify_vlm import verify_pod_markers
from .pod_counter import count_pods_on_branch as _skeleton_counter
from .pod_counter_graph import count_pods_on_branch as _graph_counter
from .pod_counter_plantcv import count_pods_on_branch as _plantcv_counter
from app.config import RESULT_DIR, CACHE_DIR


def analyze_plant_image_stream(image_path: str, method: str = "skeleton"):
    """
    Generator that yields each pipeline step as it completes.
    Yields: dict with "type" = "step" | "result"
      - step:   {"type":"step", "step":N, "url":..., "description":...}
      - result: {"type":"result", "plant_count":..., "total_pods":..., "plants":..., ...}
    """
    t0 = time.time()
    image = cv2.imread(image_path)
    if image is None:
        yield {"type": "error", "message": f"Cannot read image: {image_path}"}
        return

    stem_name = Path(image_path).stem
    debug_dir = RESULT_DIR / stem_name
    debug_dir.mkdir(exist_ok=True)
    step_counter = [0]

    def _save_step(img, description):
        step_counter[0] += 1
        fname = f"step_{step_counter[0]:02d}.png"
        fpath = debug_dir / fname
        cv2.imwrite(str(fpath), img)
        rel = fpath.relative_to(RESULT_DIR)
        url = f"/results/{rel}"
        return {"type": "step", "step": step_counter[0], "url": url, "description": description}

    # Step: Original
    yield _save_step(image, f"原始图片：{image.shape[1]}×{image.shape[0]} 像素")

    # Step: VLM
    print(f"[Pipeline] Sending {image.shape[1]}x{image.shape[0]} to Doubao Vision...")
    vlm_result = vlm_label_plants(image)

    labeled_img = vlm_result.get("labeled_image")
    if labeled_img is not None:
        yield _save_step(labeled_img, "VLM 部件识别：豆包视觉模型标注主干、主枝、分枝（彩色框）")

    # Step: Crops
    crops = vlm_result.get("crops", {})
    plants = vlm_result.get("plants", [])
    for p in plants:
        pid = p.get("id")
        if pid in crops:
            label = p.get("label", "unknown")
            crop_h, crop_w = crops[pid].shape[:2]
            yield _save_step(crops[pid], f"裁剪 #{pid} {label}：{crop_w}×{crop_h} 像素")

    # ── Step: VLM crop verification ──
    # Send original + crop + metadata back to VLM, ask it to verify
    # whether each crop is a complete, single, independent plant.
    # Possible verdicts: keep / delete / recrop
    verify_results = {}
    plants_to_remove = []
    for p in list(plants):
        pid = p.get("id")
        plabel = p.get("label", "")
        if pid not in crops:
            continue
        # Skip the main stem since it's not used for pod counting anyway
        if plabel == "主干":
            continue
        print(f"[Pipeline] Verifying crop #{pid} {plabel}...")
        verdict = vlm_verify_crop(
            original_image=image,
            crop_image=crops[pid],
            plant=p,
            all_plants=plants,
        )
        verify_results[pid] = verdict
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
                # Recompute stem_*_local relative to new bbox if stem_*_px exists
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

    # Apply deletions
    if plants_to_remove:
        plants = [p for p in plants if p.get("id") not in plants_to_remove]
        for pid in plants_to_remove:
            crops.pop(pid, None)

    # ── Cache verified crops (post-VLM-verification) ──
    # Saves after keep/delete/recrop so downstream --cv-only runs use clean data.
    try:
        _cache_dir = CACHE_DIR / stem_name
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

    # Steps: Pod counting per branch
    for p in plants:
        pid = p.get("id")
        plabel = p.get("label", "")
        if plabel == "主干" or pid not in crops:
            continue
        if method == "graph":
            _counter = _graph_counter
        elif method == "plantcv":
            _counter = _plantcv_counter
        else:
            _counter = _skeleton_counter
        print(f"[Pipeline] Counting pods on #{pid} {plabel} (method={method})...")
        pod_result = _counter(
            crops[pid],
            stem_start_local=p.get("stem_start_local"),
            stem_end_local=p.get("stem_end_local"),
        )
        p["pod_count"] = pod_result.get("pod_count", 0)
        p["main_stem_length"] = pod_result.get("main_stem_length", 0)

        for step_img, step_desc in pod_result.get("step_images", []):
            yield _save_step(step_img, f"[#{pid} {plabel}] {step_desc}")

        # ── VLM pod verification (decoupled post-processor) ──
        markers = pod_result.get("markers", [])
        if markers:
            print(f"[Pipeline] VLM verifying pods on #{pid} {plabel}...")
            verify_result = verify_pod_markers(
                crop_bgr=crops[pid],
                debug_image=pod_result.get("debug_image", crops[pid]),
                markers=markers,
                pod_count=p["pod_count"],
            )
            missed = verify_result.get("missed_count", 0)
            fp_ids = verify_result.get("false_positive_ids", [])
            adjusted = verify_result.get("adjusted_pod_count", p["pod_count"])
            v_reason = verify_result.get("reason", "")

            if missed > 0 or fp_ids:
                p["pod_count_before_vlm"] = p["pod_count"]
                p["pod_count"] = adjusted
                p["vlm_verify"] = {
                    "missed": missed,
                    "false_positives": fp_ids,
                    "reason": v_reason,
                }
                change_desc = []
                if missed > 0:
                    change_desc.append(f"漏检 +{missed}")
                if fp_ids:
                    change_desc.append(f"误判 -{len(fp_ids)} (P{',P'.join(str(x) for x in fp_ids)})")
                yield _save_step(
                    verify_result.get("verified_image", crops[pid]),
                    f"[#{pid} {plabel}] 🔍 VLM 角果复核：{' / '.join(change_desc)}，"
                    f"{p['pod_count_before_vlm']} → {adjusted} — {v_reason}")
            else:
                yield _save_step(
                    verify_result.get("verified_image", crops[pid]),
                    f"[#{pid} {plabel}] ✅ VLM 角果复核：计数无调整 — {v_reason}")

    # Final result
    total_pods = sum(p.get("pod_count", 0) for p in plants)
    elapsed = round(time.time() - t0, 2)
    yield {
        "type": "result",
        "plant_count": len(plants),
        "total_pods": total_pods,
        "plants": plants,
        "processing_time_s": elapsed,
        "total_steps": step_counter[0],
    }
    print(f"\n[Done] time: {elapsed}s, steps: {step_counter[0]}")
