"""
Pod (角果) counting via stalk-based method.

Two-pass algorithm:
  Pass 1 — Reuse graph topology to identify and filter dead twigs (枯枝)
  Pass 2 — Remove dead twig pixels from skeleton, rebuild graph,
            count non-main-stem edges emanating from main-stem junctions.
            Each edge with sufficient length = 1 pod stalk (果柄).

Key advantage over tip-based counting:
  When two pods cross at their tips, the skeleton merges tips → undercounts.
  But their stalks (果柄) attach to the main stem at separate points,
  so counting stalks avoids the tip-merging problem.
"""

import cv2
import numpy as np
from collections import deque

from .pod_counter import (
    _binarize,
    _get_skeleton,
    _remove_ruler_region,
    _select_center_component,
    _find_main_stem_numba,
    _nb_find_endpoints,
    _nb_bfs_farthest,
    _nb_bfs_path,
    _neighbors,
)
from .pod_counter_graph import _build_topo_graph, _graph_adjacency


def count_pods_on_branch(
    crop_bgr: np.ndarray,
    stem_start_local=None,
    stem_end_local=None,
    min_fork_gap: int = 15,
) -> dict:
    """Count pods by counting fruit-stalk branches off the main stem.

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

    # 2. Skeletonize
    skeleton = _get_skeleton(mask)
    skel_arr = (skeleton > 0).astype(np.bool_)
    skel_set = set(map(tuple, np.argwhere(skel_arr)))
    skel_vis = crop_bgr.copy()
    skel_vis[skel_arr] = (0, 255, 255)
    step_images.append((skel_vis, f"骨架提取：1px 宽骨架线（黄色），共 {len(skel_set)} 像素"))
    if len(skel_set) < 20:
        return {"pod_count": 0, "debug_image": crop_bgr.copy(),
                "step_images": step_images, "note": "skeleton too short"}

    # 3. Find main stem (with ruler rejection)
    MAX_RULER_RETRIES = 2
    main_path = []
    stem_len = 0
    stem_source = "scored"
    for _ruler_attempt in range(MAX_RULER_RETRIES + 1):
        main_path, stem_len, stem_score = _find_main_stem_numba(skel_arr)
        stem_source = "scored"

        if not main_path:
            endpoints = _nb_find_endpoints(skel_arr)
            if endpoints.shape[0] >= 1:
                fr, fc, _ = _nb_bfs_farthest(skel_arr, endpoints[0, 0], endpoints[0, 1])
                fr2, fc2, _ = _nb_bfs_farthest(skel_arr, fr, fc)
                path_arr = _nb_bfs_path(skel_arr, fr, fc, fr2, fc2)
                main_path = [(int(path_arr[k, 0]), int(path_arr[k, 1]))
                             for k in range(path_arr.shape[0])]
                stem_len = len(main_path)
            stem_source = "bfs_fallback"

        if not main_path:
            break

        if _ruler_attempt < MAX_RULER_RETRIES and stem_len > 30:
            _rdist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            _widths = np.array([_rdist[p[0], p[1]] for p in main_path])
            _mean_w = _widths.mean()
            if _mean_w > 0:
                _cv = _widths.std() / _mean_w
                if _cv < 0.15:
                    mid = main_path[len(main_path) // 2]
                    _flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                    cv2.floodFill(mask, _flood_mask, (mid[1], mid[0]), 0)
                    print(f"  [PodStalk] Ruler rejected: CV={_cv:.3f}")
                    skeleton = _get_skeleton(mask)
                    skel_arr = (skeleton > 0).astype(np.bool_)
                    skel_set = set(map(tuple, np.argwhere(skel_arr)))
                    if len(skel_set) < 20:
                        main_path = []
                        break
                    continue
        break

    if not main_path:
        return {"pod_count": 0, "debug_image": crop_bgr.copy(),
                "step_images": step_images, "note": "no main path"}

    stem_vis = crop_bgr.copy()
    for p in main_path:
        cv2.circle(stem_vis, (p[1], p[0]), 2, (255, 128, 0), -1)
    cv2.circle(stem_vis, (main_path[0][1], main_path[0][0]), 6, (0, 255, 0), 2)
    cv2.circle(stem_vis, (main_path[-1][1], main_path[-1][0]), 6, (0, 255, 0), 2)
    step_images.append((stem_vis, f"主干识别：橙色=主干路径（{stem_len}px），绿色=端点"))

    start_pt, end_pt = main_path[0], main_path[-1]
    main_set = set(main_path)

    # ══════════════════════════════════════════════════════════════════
    # PASS 1 — Graph-based dead twig identification (same as graph method)
    # ══════════════════════════════════════════════════════════════════
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    nodes_p1, edges_p1 = _build_topo_graph(skel_set, main_set)
    if not nodes_p1:
        return {"pod_count": 0, "debug_image": crop_bgr.copy(),
                "step_images": step_images, "note": "no graph nodes"}

    adj_p1 = _graph_adjacency(nodes_p1, edges_p1)
    attach_nodes_p1 = set(n for n, t in nodes_p1.items() if t == 'attach')
    tip_nodes_p1 = [n for n, t in nodes_p1.items() if t == 'tip']

    # BFS from each tip → nearest attach (same as graph method)
    branch_data = []
    for tip in tip_nodes_p1:
        parent = {tip: None}
        queue = deque([tip])
        found_attach = None
        while queue:
            cur = queue.popleft()
            if cur in attach_nodes_p1:
                found_attach = cur
                break
            for nb, ei in adj_p1[cur]:
                if nb not in parent:
                    parent[nb] = (cur, ei)
                    queue.append(nb)
        if found_attach is None:
            continue

        node_path = []
        edge_indices = []
        cur = found_attach
        while cur is not None:
            node_path.append(cur)
            info = parent[cur]
            if info is None:
                break
            edge_indices.append(info[1])
            cur = info[0]
        node_path.reverse()
        edge_indices.reverse()

        full_path = []
        for k in range(len(node_path) - 1):
            n1, n2 = node_path[k], node_path[k + 1]
            ei = edge_indices[k]
            ep = edges_p1[ei][2]
            if ep[0] == n1:
                seg = ep[1:] if full_path else ep
            elif ep[-1] == n1:
                rev = ep[::-1]
                seg = rev[1:] if full_path else rev
            else:
                seg = ep[1:] if full_path else ep
            full_path.extend(seg)

        if len(full_path) < 3:
            continue
        area = sum(float(dist[r, c]) * 2.0 for (r, c) in full_path)
        avg_w = area / len(full_path) if full_path else 0.0
        branch_data.append((tip, full_path, found_attach, area, avg_w))

    # Filter: dead twigs (width/area) + noise (short complete branches)
    WIDTH_RATIO = 0.70
    AREA_RATIO = 0.25
    MIN_BRANCHES = 4
    MIN_BRANCH_LEN = 10  # denoising: complete branches shorter than this are noise
    rejected_pixels = set()
    rejected_branches = []

    # Step A: remove very short complete branches (denoising)
    for item in branch_data:
        if len(item[1]) < MIN_BRANCH_LEN:
            rejected_branches.append(item)
            rejected_pixels.update(item[1])
    remaining = [item for item in branch_data if item not in rejected_branches]
    noise_count = len(rejected_branches)

    # Step B: adaptive width/area filtering for dead twigs
    if remaining and len(remaining) >= MIN_BRANCHES:
        all_areas = np.array([d[3] for d in remaining])
        all_widths = np.array([d[4] for d in remaining])
        median_area = float(np.median(all_areas))
        median_width = float(np.median(all_widths))
        area_fence = median_area * AREA_RATIO
        width_fence = median_width * WIDTH_RATIO

        for item in remaining:
            area_val = item[3]
            avg_w_val = item[4]
            if area_val >= area_fence and avg_w_val >= width_fence:
                pass  # keep
            else:
                rejected_branches.append(item)
                rejected_pixels.update(item[1])
        twig_count = len(rejected_branches) - noise_count
        print(f"  [PodStalk] Pass1: branches={len(branch_data)}, "
              f"noise(<{MIN_BRANCH_LEN}px)={noise_count}, "
              f"med_area={median_area:.0f} fence={area_fence:.0f}, "
              f"med_w={median_width:.1f} fence={width_fence:.1f}, "
              f"twigs={twig_count}")
    else:
        print(f"  [PodStalk] Pass1: branches={len(branch_data)}, "
              f"noise(<{MIN_BRANCH_LEN}px)={noise_count}, "
              f"too few to filter twigs (MIN_BRANCHES={MIN_BRANCHES})")

    # Step image: Pass 1 filter
    p1_vis = crop_bgr.copy()
    for p in main_path:
        cv2.circle(p1_vis, (p[1], p[0]), 2, (255, 128, 0), -1)
    for item in rejected_branches:
        for p in item[1]:
            cv2.circle(p1_vis, (p[1], p[0]), 1, (200, 80, 200), -1)
        cv2.circle(p1_vis, (item[0][1], item[0][0]), 4, (200, 80, 200), -1)
    # Show kept branches in gray
    kept_p1 = [item for item in branch_data if item not in rejected_branches]
    for item in kept_p1:
        for p in item[1]:
            cv2.circle(p1_vis, (p[1], p[0]), 1, (180, 180, 180), -1)
    step_images.append((
        p1_vis,
        f"Pass1 枯枝过滤：粉色=枯枝({len(rejected_branches)})，灰色=保留({len(kept_p1)})"
    ))

    # ══════════════════════════════════════════════════════════════════
    # PASS 2 — Remove dead twigs, rebuild graph, count stalks
    # ══════════════════════════════════════════════════════════════════
    # Clean skeleton: remove rejected pixels (but keep main stem)
    cleaned_skel_set = skel_set - rejected_pixels

    # Rebuild topology graph on cleaned skeleton
    nodes_p2, edges_p2 = _build_topo_graph(cleaned_skel_set, main_set)

    if not nodes_p2:
        pod_count = 0
        stalk_info = []
    else:
        # Every non-main outgoing edge from an attach node = 1 fruit stalk.
        # No filtering here — all filtering was done in Pass 1.
        adj_p2 = _graph_adjacency(nodes_p2, edges_p2)

        stalk_info = []  # (attach_pt, other_pt, path_pixels, path_length)
        seen_edges = set()

        for node, ntype in nodes_p2.items():
            if ntype != 'attach':
                continue
            for nb, ei in adj_p2.get(node, []):
                if ei in seen_edges:
                    continue
                if nb in main_set:
                    continue  # skip edges along main stem
                seen_edges.add(ei)
                path = edges_p2[ei][2]
                stalk_info.append((node, nb, path, len(path)))

        pod_count = len(stalk_info)

    print(f"  [PodStalk] Pass2: stalks={len(stalk_info)}, pod_count={pod_count}")

    # ── Debug visualization ──
    debug = crop_bgr.copy()
    # Layer 1: Main stem in orange
    for p in main_path:
        cv2.circle(debug, (p[1], p[0]), 2, (255, 128, 0), -1)
    # Layer 2: Cleaned non-main skeleton in dark gray
    non_main_clean = cleaned_skel_set - main_set
    for p in non_main_clean:
        cv2.circle(debug, (p[1], p[0]), 1, (80, 80, 80), -1)
    # Layer 3: Rejected branches in purple
    for item in rejected_branches:
        for p in item[1]:
            cv2.circle(debug, (p[1], p[0]), 1, (200, 80, 200), -1)
    # Layer 4: Valid stalks in red, attach in white, tips in cyan
    for idx_s, (attach_pt, tip_pt, path_px, path_len) in enumerate(stalk_info, 1):
        for p in path_px:
            cv2.circle(debug, (p[1], p[0]), 1, (0, 0, 255), -1)
        cv2.circle(debug, (attach_pt[1], attach_pt[0]), 5, (255, 255, 255), -1)
        cv2.circle(debug, (tip_pt[1], tip_pt[0]), 5, (255, 255, 0), -1)
        cv2.putText(debug, str(idx_s), (tip_pt[1] + 6, tip_pt[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    # Stem endpoints
    for p in [start_pt, end_pt]:
        cv2.circle(debug, (p[1], p[0]), 6, (0, 255, 0), 2)
    # Count text
    fs = max(0.5, min(h, w) / 500)
    cv2.putText(debug, f"Pods: {pod_count} (stalk)", (5, int(25 * fs) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), max(1, int(fs * 2)))
    cv2.putText(debug,
                f"stalks={len(stalk_info)} "
                f"reject={len(rejected_branches)}",
                (5, int(50 * fs) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (180, 180, 180), 1)

    # Step image: stalk counting result
    step_images.append((debug.copy(),
                        f"果柄计数结果：{pod_count} 个（果柄方案）"))

    # ── Standardized markers for downstream VLM verification ──
    markers = []
    for idx_s, (attach_pt, tip_pt, path_px, path_len) in enumerate(stalk_info, 1):
        markers.append({"id": idx_s, "type": "pod",
                        "x_norm": tip_pt[1] / w, "y_norm": tip_pt[0] / h})
    for j, item in enumerate(rejected_branches, len(stalk_info) + 1):
        tip = item[0]
        markers.append({"id": j, "type": "filtered",
                        "x_norm": tip[1] / w, "y_norm": tip[0] / h})

    return {
        "pod_count": pod_count,
        "main_stem_length": stem_len,
        "fork_count": len(stalk_info),
        "skeleton_pixels": len(skel_set),
        "stem_source": stem_source,
        "debug_image": debug,
        "step_images": step_images,
        "markers": markers,
    }
