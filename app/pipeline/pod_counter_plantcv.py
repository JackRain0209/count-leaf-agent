"""
Pod (角果) counting via PlantCV morphology analysis.

Algorithm:
  1. Binarize (shared with other counters)
  2. PlantCV skeletonize → prune barbs
  3. Find branch points and tip points
  4. Segment skeleton into pieces
  5. Sort segments into leaf (pod) vs stem
  6. Count leaf segments = pod count
"""

import cv2
import numpy as np
from plantcv import plantcv as pcv

# Reuse common utilities from the skeleton-based counter
from .pod_counter import (
    _binarize,
    _remove_ruler_region,
    _select_center_component,
)


def count_pods_on_branch(
    crop_bgr: np.ndarray,
    stem_start_local=None,
    stem_end_local=None,
    min_fork_gap: int = 15,
    prune_size: int = 10,
) -> dict:
    """Count pods on a single branch crop via PlantCV morphology.

    Drop-in replacement for pod_counter.count_pods_on_branch.
    """
    h, w = crop_bgr.shape[:2]

    # 0. Auto-upscale low-res images
    MIN_DIM = 800
    max_side = max(h, w)
    if max_side < MIN_DIM:
        scale = MIN_DIM / max_side
        crop_bgr = cv2.resize(crop_bgr, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LANCZOS4)
        h, w = crop_bgr.shape[:2]

    step_images = []

    # Suppress PlantCV debug output
    pcv.params.debug = None

    # 1. Binarize
    mask = _binarize(crop_bgr)
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    step_images.append((mask_vis, "二值化分割：Otsu + 暗色恢复，白色=植物区域"))
    if cv2.countNonZero(mask) < 50:
        return {"pod_count": 0, "debug_image": crop_bgr.copy(),
                "step_images": step_images, "note": "mask too small"}

    # 1b. Ruler removal
    mask = _remove_ruler_region(mask)
    # 1c. Center component
    mask = _select_center_component(mask)

    # 2. PlantCV Skeletonize
    skeleton = pcv.morphology.skeletonize(mask=mask)
    skel_pixels = cv2.countNonZero(skeleton)
    skel_vis = crop_bgr.copy()
    skel_vis[skeleton > 0] = (0, 255, 255)
    step_images.append((skel_vis, f"PlantCV 骨架提取：共 {skel_pixels} 像素"))
    if skel_pixels < 20:
        return {"pod_count": 0, "debug_image": crop_bgr.copy(),
                "step_images": step_images, "note": "skeleton too short"}

    # 3. Prune small barbs
    pruned_skeleton, seg_img, segment_objects = pcv.morphology.prune(
        skel_img=skeleton, size=prune_size, mask=mask)
    pruned_pixels = cv2.countNonZero(pruned_skeleton)
    prune_vis = crop_bgr.copy()
    prune_vis[pruned_skeleton > 0] = (0, 255, 255)
    # Show pruned portions in red
    pruned_away = cv2.bitwise_and(skeleton, cv2.bitwise_not(pruned_skeleton))
    prune_vis[pruned_away > 0] = (0, 0, 255)
    step_images.append((prune_vis,
                        f"PlantCV 修剪：去除 <{prune_size}px 毛刺（红色），保留 {pruned_pixels} 像素"))

    if pruned_pixels < 20:
        return {"pod_count": 0, "debug_image": crop_bgr.copy(),
                "step_images": step_images, "note": "pruned skeleton too short"}

    # 4. Find branch points and tips
    branch_pts_mask = pcv.morphology.find_branch_pts(skel_img=pruned_skeleton, mask=mask)
    tip_pts_mask = pcv.morphology.find_tips(skel_img=pruned_skeleton, mask=mask)

    branch_pts = set(map(tuple, np.argwhere(branch_pts_mask > 0)))
    tip_pts = set(map(tuple, np.argwhere(tip_pts_mask > 0)))

    pts_vis = crop_bgr.copy()
    pts_vis[pruned_skeleton > 0] = (0, 255, 255)
    for r, c in branch_pts:
        cv2.circle(pts_vis, (c, r), 4, (0, 165, 255), -1)  # orange = branch pts
    for r, c in tip_pts:
        cv2.circle(pts_vis, (c, r), 4, (255, 0, 255), -1)  # magenta = tips
    step_images.append((pts_vis,
                        f"PlantCV 关键点：{len(branch_pts)} 个分支点（橙），{len(tip_pts)} 个末端点（紫）"))

    # 5. Segment skeleton
    seg_img, edge_objects = pcv.morphology.segment_skeleton(
        skel_img=pruned_skeleton, mask=mask)

    if not edge_objects or len(edge_objects) == 0:
        return {"pod_count": 0, "debug_image": crop_bgr.copy(),
                "step_images": step_images, "note": "no segments"}

    # 6. Sort segments into leaf vs stem
    leaf_obj, stem_obj = pcv.morphology.segment_sort(
        skel_img=pruned_skeleton, objects=edge_objects, mask=mask)

    # Count leaves = pods
    leaf_count = len(leaf_obj) if leaf_obj else 0
    stem_count = len(stem_obj) if stem_obj else 0

    # 7. Adaptive filtering by segment length
    # Filter out very short leaf segments (likely noise)
    real_pods = []
    rejected = []
    if leaf_obj and len(leaf_obj) > 0:
        lengths = []
        for obj in leaf_obj:
            length = cv2.arcLength(obj, closed=False)
            lengths.append(length)

        if len(lengths) >= 4:
            median_len = float(np.median(lengths))
            min_len = median_len * 0.25
            for obj, l in zip(leaf_obj, lengths):
                if l >= min_len:
                    real_pods.append((obj, l))
                else:
                    rejected.append((obj, l))
            print(f"  [PlantCV] segments={len(leaf_obj)}, med_len={median_len:.0f}, "
                  f"min_len={min_len:.0f}, keep={len(real_pods)}, reject={len(rejected)}")
        else:
            real_pods = [(obj, cv2.arcLength(obj, closed=False)) for obj in leaf_obj]

    pod_count = len(real_pods)

    # 8. Debug visualization
    debug = crop_bgr.copy()

    # Draw stem segments in orange
    if stem_obj:
        for obj in stem_obj:
            cv2.drawContours(debug, [obj], -1, (255, 128, 0), 2)

    # Draw rejected leaf segments in purple
    for obj, l in rejected:
        cv2.drawContours(debug, [obj], -1, (200, 80, 200), 1)

    # Draw counted leaf segments in green with numbered tips
    for i, (obj, l) in enumerate(real_pods, 1):
        cv2.drawContours(debug, [obj], -1, (0, 255, 0), 2)
        # Mark tip of segment
        if len(obj) > 0:
            tip_pt = tuple(obj[-1][0])
            cv2.circle(debug, tip_pt, 5, (255, 255, 0), -1)
            cv2.putText(debug, str(i), (tip_pt[0] + 6, tip_pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Branch points
    for r, c in branch_pts:
        cv2.circle(debug, (c, r), 3, (0, 165, 255), -1)

    # Count text
    fs = max(0.5, min(h, w) / 500)
    cv2.putText(debug, f"Pods: {pod_count} (plantcv)", (5, int(25 * fs) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), max(1, int(fs * 2)))
    cv2.putText(debug,
                f"leaf_segs={leaf_count} stem_segs={stem_count} "
                f"reject={len(rejected)} tips={len(tip_pts)}",
                (5, int(50 * fs) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (180, 180, 180), 1)

    # Step image: sort visualization
    sort_vis = crop_bgr.copy()
    if stem_obj:
        for obj in stem_obj:
            cv2.drawContours(sort_vis, [obj], -1, (255, 128, 0), 2)
    if leaf_obj:
        for obj in leaf_obj:
            cv2.drawContours(sort_vis, [obj], -1, (0, 255, 0), 2)
    step_images.append((sort_vis,
                        f"PlantCV 分类：橙=茎段({stem_count})，绿=叶/角果段({leaf_count})"))

    # Step image: filter visualization
    filter_vis = crop_bgr.copy()
    if stem_obj:
        for obj in stem_obj:
            cv2.drawContours(filter_vis, [obj], -1, (255, 128, 0), 1)
    for obj, l in real_pods:
        cv2.drawContours(filter_vis, [obj], -1, (0, 255, 0), 2)
        if len(obj) > 0:
            tip_pt = tuple(obj[-1][0])
            txt = f"l={l:.0f}"
            cv2.putText(filter_vis, txt, (tip_pt[0] + 6, tip_pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    for obj, l in rejected:
        cv2.drawContours(filter_vis, [obj], -1, (0, 0, 255), 1)
        if len(obj) > 0:
            tip_pt = tuple(obj[-1][0])
            txt = f"l={l:.0f}"
            cv2.putText(filter_vis, txt, (tip_pt[0] + 6, tip_pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    step_images.append((filter_vis,
                        f"PlantCV 过滤：绿=角果({pod_count})，红=过短({len(rejected)})"))

    # Final result
    step_images.append((debug.copy(),
                        f"PlantCV 计数结果：{pod_count} 个角果"))

    print(f"  [PlantCV] leaf_segs={leaf_count}, stem_segs={stem_count}, "
          f"reject={len(rejected)}, pods={pod_count}")

    # ── Standardized markers for downstream VLM verification ──
    # Coordinates normalized to [0, 1] relative to working image dimensions
    # to avoid mismatch when auto-upscale changes resolution.
    markers = []
    for i, (obj, l) in enumerate(real_pods, 1):
        if len(obj) > 0:
            tip_xy = tuple(obj[-1][0])
            markers.append({"id": i, "type": "pod",
                            "x_norm": tip_xy[0] / w, "y_norm": tip_xy[1] / h})
    for j, (obj, l) in enumerate(rejected, len(real_pods) + 1):
        if len(obj) > 0:
            tip_xy = tuple(obj[-1][0])
            markers.append({"id": j, "type": "filtered",
                            "x_norm": tip_xy[0] / w, "y_norm": tip_xy[1] / h})

    return {
        "pod_count": pod_count,
        "main_stem_length": 0,
        "fork_count": len(tip_pts),
        "skeleton_pixels": skel_pixels,
        "stem_source": "plantcv",
        "debug_image": debug,
        "step_images": step_images,
        "markers": markers,
    }
