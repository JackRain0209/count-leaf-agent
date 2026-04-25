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

from .vlm_counter import vlm_label_plants
from .pod_counter import count_pods_on_branch
from app.config import RESULT_DIR


def analyze_plant_image_stream(image_path: str):
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

    # Steps: Pod counting per branch
    for p in plants:
        pid = p.get("id")
        label = p.get("label", "")
        if label == "主干" or pid not in crops:
            continue
        print(f"[Pipeline] Counting pods on #{pid} {label}...")
        pod_result = count_pods_on_branch(
            crops[pid],
            stem_start_local=p.get("stem_start_local"),
            stem_end_local=p.get("stem_end_local"),
        )
        p["pod_count"] = pod_result.get("pod_count", 0)
        p["main_stem_length"] = pod_result.get("main_stem_length", 0)

        for step_img, step_desc in pod_result.get("step_images", []):
            yield _save_step(step_img, f"[#{pid} {label}] {step_desc}")

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
