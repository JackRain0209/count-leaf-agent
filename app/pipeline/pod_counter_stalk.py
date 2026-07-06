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

Filtering rule:
  Candidate paths that pass through/near non-main junctions are protected from
  adaptive width/area twig filtering. Very short noise is still removed first.
"""

import cv2
import numpy as np
from collections import deque

from .pod_counter_common import (
    _binarize,
    _get_skeleton,
    _remove_ruler_region,
    _select_center_component,
    _find_main_stem_numba,
    _nb_find_endpoints,
    _nb_bfs_farthest,
    _nb_bfs_path,
    _neighbors,
    _build_topo_graph,
    _graph_adjacency,
)


def _dedupe_path(path_pixels):
    """Preserve path order while removing duplicated pixels."""
    return list(dict.fromkeys(path_pixels))


def _orient_edge_path(path_pixels, start_node):
    """Return edge pixels ordered so start_node is first when possible."""
    if not path_pixels:
        return []
    if path_pixels[0] == start_node:
        return path_pixels
    if path_pixels[-1] == start_node:
        return path_pixels[::-1]
    return path_pixels


def _collect_candidate_path(start_node, first_node, edge_index, adj, edges,
                            attach_nodes, max_pixels=600):
    """
    Collect root edge plus nearby downstream pixels for Pass2 validation.

    Counting still happens per attach root edge, but validation needs a little
    downstream context so true pods that merge near their tips are not rejected
    just because the root edge is short.
    """
    root_path = _orient_edge_path(edges[edge_index][2], start_node)
    if not root_path:
        return [], []

    pixels = list(root_path)
    visited_edges = {edge_index}
    visited_nodes = {start_node}
    queue = deque([first_node])

    while queue and len(pixels) < max_pixels:
        node = queue.popleft()
        if node in visited_nodes:
            continue
        visited_nodes.add(node)

        # Do not borrow pixels from a different main-stem attachment.
        if node in attach_nodes and node != start_node:
            continue

        for nb, ei in adj.get(node, []):
            if ei in visited_edges:
                continue
            if nb in attach_nodes and nb != start_node:
                continue

            edge_path = _orient_edge_path(edges[ei][2], node)
            if not edge_path:
                continue
            visited_edges.add(ei)
            pixels.extend(edge_path[1:])

            if len(pixels) >= max_pixels:
                break
            if nb not in visited_nodes:
                queue.append(nb)

    return root_path, _dedupe_path(pixels[:max_pixels])


def _dedupe_terminal_paths(paths, endpoint_radius=6):
    kept = []
    endpoints = []
    for path in sorted(paths, key=len, reverse=True):
        if len(path) < 3:
            continue
        end = path[-1]
        too_close = any(
            (end[0] - prev[0]) ** 2 + (end[1] - prev[1]) ** 2 <= endpoint_radius ** 2
            for prev in endpoints
        )
        if too_close:
            continue
        endpoints.append(end)
        kept.append(path)
    return kept


def _collect_near_root_split_paths(start_node, first_node, edge_index, adj,
                                   edges, attach_nodes, nodes,
                                   max_prefix_len=34, max_branch_len=120,
                                   max_children=4):
    """
    Split only the first fork that appears very close to the main-stem attach.

    This avoids recursively splitting pod-body contours far away from the stem,
    while still rescuing close stalks that share a short root at the stem.
    """
    root_path = _orient_edge_path(edges[edge_index][2], start_node)
    if not root_path:
        return [], []

    prefix = list(root_path)
    node = first_node
    visited = {edge_index}
    fork_edges = []

    while len(prefix) <= max_prefix_len:
        if node in attach_nodes and node != start_node:
            return root_path, []

        next_edges = []
        for nb, ei in adj.get(node, []):
            if ei in visited:
                continue
            if nb in attach_nodes and nb != start_node:
                continue
            edge_path = _orient_edge_path(edges[ei][2], node)
            if edge_path:
                next_edges.append((nb, ei, edge_path))

        if nodes.get(node) == "junction" and 2 <= len(next_edges) <= max_children:
            fork_edges = next_edges
            break

        if len(next_edges) != 1:
            return root_path, []

        nb, ei, edge_path = next_edges[0]
        prefix.extend(edge_path[1:])
        visited.add(ei)
        node = nb

    if not fork_edges:
        return root_path, []

    split_paths = []
    for nb, ei, edge_path in fork_edges:
        path = prefix + edge_path[1:]
        branch_node = nb
        branch_visited = visited | {ei}

        while len(path) < max_branch_len:
            if nodes.get(branch_node) in ("tip", "junction"):
                break
            next_edges = []
            for next_nb, next_ei in adj.get(branch_node, []):
                if next_ei in branch_visited:
                    continue
                if next_nb in attach_nodes and next_nb != start_node:
                    continue
                next_path = _orient_edge_path(edges[next_ei][2], branch_node)
                if next_path:
                    next_edges.append((next_nb, next_ei, next_path))

            if len(next_edges) != 1:
                break
            branch_node, next_ei, next_path = next_edges[0]
            path.extend(next_path[1:])
            branch_visited.add(next_ei)

        split_paths.append(_dedupe_path(path[:max_branch_len]))

    return root_path, _dedupe_terminal_paths(split_paths, endpoint_radius=8)


def _plant_median_v(crop_bgr, mask):
    plant_pixels = crop_bgr[mask > 0]
    if plant_pixels.size == 0:
        return 1.0
    hsv = cv2.cvtColor(plant_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
    return max(1.0, float(np.median(hsv[:, 0, 2])))


def _path_metrics(path_pixels, dist, crop_bgr, mask, plant_med_v):
    path_pixels = _dedupe_path(path_pixels)
    if not path_pixels:
        return {
            "length": 0, "area": 0.0, "avg_w": 0.0, "p10_w": 0.0,
            "p90_w": 0.0, "width_cv": 0.0, "bulge_ratio": 0.0,
            "dark_ratio": 1.0, "brown_dark_ratio": 1.0,
            "green_yellow_ratio": 0.0, "mean_v_ratio": 0.0,
        }

    h, w = mask.shape[:2]
    widths = np.array([float(dist[r, c]) * 2.0 for r, c in path_pixels],
                      dtype=np.float32)
    length = len(path_pixels)
    area = float(widths.sum())
    avg_w = float(widths.mean()) if length else 0.0
    p10_w = float(np.percentile(widths, 10)) if length else 0.0
    p90_w = float(np.percentile(widths, 90)) if length else 0.0
    width_cv = float(widths.std() / avg_w) if avg_w > 0 else 0.0
    bulge_ratio = float(p90_w / max(p10_w, 1.0))

    sample_mask = np.zeros((h, w), dtype=np.uint8)
    for r, c in path_pixels:
        if 0 <= r < h and 0 <= c < w:
            sample_mask[r, c] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sample_mask = cv2.dilate(sample_mask, kernel, iterations=1)
    sample_mask = cv2.bitwise_and(sample_mask, mask)
    sample_pixels = crop_bgr[sample_mask > 0]
    if sample_pixels.size == 0:
        sample_pixels = crop_bgr[[r for r, _ in path_pixels],
                                 [c for _, c in path_pixels]]

    hsv = cv2.cvtColor(sample_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
    h_vals = hsv[:, 0, 0].astype(np.float32)
    s_vals = hsv[:, 0, 1].astype(np.float32)
    v_vals = hsv[:, 0, 2].astype(np.float32)
    mean_v = float(v_vals.mean()) if v_vals.size else 0.0
    mean_v_ratio = mean_v / max(plant_med_v, 1.0)

    bright = v_vals > plant_med_v * 0.70
    green_yellow = ((h_vals >= 18) & (h_vals <= 95) &
                    (s_vals > 35) & bright)
    dark = v_vals < plant_med_v * 0.65
    brown_dark = ((h_vals >= 5) & (h_vals <= 30) &
                  (v_vals < plant_med_v * 0.85))
    denom = max(1, int(v_vals.size))

    return {
        "length": length,
        "area": area,
        "avg_w": avg_w,
        "p10_w": p10_w,
        "p90_w": p90_w,
        "width_cv": width_cv,
        "bulge_ratio": bulge_ratio,
        "dark_ratio": float(dark.sum()) / denom,
        "brown_dark_ratio": float(brown_dark.sum()) / denom,
        "green_yellow_ratio": float(green_yellow.sum()) / denom,
        "mean_v_ratio": mean_v_ratio,
    }


def _score_stalk_candidate(metrics, median_area, median_width, median_len,
                           crossing_related):
    """Decide whether a cleaned Pass2 attach edge still looks like a pod stalk."""
    length = metrics["length"]
    avg_w = metrics["avg_w"]
    p90_w = metrics["p90_w"]
    area = metrics["area"]
    width_cv = metrics["width_cv"]
    bulge_ratio = metrics["bulge_ratio"]
    dark_ratio = metrics["dark_ratio"]
    brown_dark_ratio = metrics["brown_dark_ratio"]
    green_yellow_ratio = metrics["green_yellow_ratio"]
    mean_v_ratio = metrics["mean_v_ratio"]
    root_len = metrics.get("root_len", length)

    area_ratio = area / max(median_area, 1.0)
    width_ratio = avg_w / max(median_width, 1.0)
    len_ratio = length / max(median_len, 1.0)
    metrics.update({
        "area_ratio": area_ratio,
        "width_ratio": width_ratio,
        "len_ratio": len_ratio,
    })

    if length < 10:
        return False, "short_noise", -99.0
    if root_len <= 3 and length < 16:
        return False, "short_root_noise", -99.0
    if avg_w < median_width * 0.45 and p90_w < median_width * 0.65:
        return False, "very_thin", -99.0
    if dark_ratio > 0.55 and bulge_ratio < 1.20 and width_cv < 0.20:
        return False, "dark_uniform", -99.0
    if area_ratio < 0.18 and width_ratio < 0.75:
        return False, "small_area_thin", -99.0

    if green_yellow_ratio >= 0.25 and length >= 14:
        return True, "bright_green", 99.0
    if bulge_ratio >= 1.35 and p90_w >= median_width * 0.85:
        return True, "bulged", 99.0
    if width_ratio >= 0.85 and area_ratio >= 0.45 and length >= 12:
        return True, "normal_size", 99.0

    score = 0.0
    if length >= max(14.0, median_len * 0.30):
        score += 1.0
    if length >= max(22.0, median_len * 0.45):
        score += 1.0
    if width_ratio >= 0.70:
        score += 1.0
    if p90_w >= median_width * 0.85:
        score += 1.0
    if bulge_ratio >= 1.25:
        score += 1.0
    if width_cv >= 0.22:
        score += 1.0
    if green_yellow_ratio >= 0.20:
        score += 1.0
    if mean_v_ratio >= 0.85:
        score += 1.0
    if dark_ratio >= 0.55:
        score -= 2.0
    if brown_dark_ratio >= 0.45:
        score -= 2.0
    if bulge_ratio < 1.15 and width_cv < 0.18:
        score -= 2.0
    if area_ratio < 0.30:
        score -= 1.0
    if root_len <= 4 and length < 20:
        score -= 1.0
    if crossing_related:
        score += 0.5

    threshold = 2.0 if crossing_related else 2.5
    reason = f"score_{score:.1f}_ge_{threshold:.1f}"
    return score >= threshold, reason, score


def _snap_local_point_to_skeleton(point, skel_arr, max_distance=None):
    """Snap an x/y local hint to the nearest skeleton pixel."""
    if not point or len(point) != 2:
        return None, None
    h, w = skel_arr.shape
    try:
        x = int(round(float(point[0])))
        y = int(round(float(point[1])))
    except (TypeError, ValueError):
        return None, None
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))

    if max_distance is None:
        max_distance = max(24, int(max(h, w) * 0.12))
    max_distance = int(max_distance)

    best = None
    best_d2 = None
    r1 = max(0, y - max_distance)
    r2 = min(h, y + max_distance + 1)
    c1 = max(0, x - max_distance)
    c2 = min(w, x + max_distance + 1)
    ys, xs = np.nonzero(skel_arr[r1:r2, c1:c2])
    if len(xs):
        ys = ys + r1
        xs = xs + c1
        d2 = (ys - y) * (ys - y) + (xs - x) * (xs - x)
        idx = int(np.argmin(d2))
        best = (int(ys[idx]), int(xs[idx]))
        best_d2 = float(d2[idx])

    if best is None:
        ys, xs = np.nonzero(skel_arr)
        if not len(xs):
            return None, None
        d2 = (ys - y) * (ys - y) + (xs - x) * (xs - x)
        idx = int(np.argmin(d2))
        best = (int(ys[idx]), int(xs[idx]))
        best_d2 = float(d2[idx])

    dist = best_d2 ** 0.5
    if dist > max_distance * 1.75:
        return None, dist
    return best, dist


def _find_main_stem_from_hints(skel_arr, start_local, end_local):
    """Use VLM-provided x/y endpoints as anchors, then trace the skeleton path."""
    h, w = skel_arr.shape
    snap_limit = max(30, int(max(h, w) * 0.16))
    start_pt, start_dist = _snap_local_point_to_skeleton(
        start_local, skel_arr, max_distance=snap_limit)
    end_pt, end_dist = _snap_local_point_to_skeleton(
        end_local, skel_arr, max_distance=snap_limit)
    if start_pt is None or end_pt is None or start_pt == end_pt:
        return [], 0, 0.0, (start_dist, end_dist)

    path_arr = _nb_bfs_path(skel_arr, start_pt[0], start_pt[1],
                            end_pt[0], end_pt[1])
    if path_arr.shape[0] < 20:
        return [], 0, 0.0, (start_dist, end_dist)

    path = [(int(path_arr[k, 0]), int(path_arr[k, 1]))
            for k in range(path_arr.shape[0])]
    dy = float(path[-1][0] - path[0][0])
    dx = float(path[-1][1] - path[0][1])
    chord = (dy * dy + dx * dx) ** 0.5
    straightness = chord / max(len(path), 1)
    if straightness < 0.25:
        return [], 0, straightness, (start_dist, end_dist)

    return path, len(path), straightness, (start_dist, end_dist)


def count_pods_on_branch(
    crop_bgr: np.ndarray,
    stem_start_local=None,
    stem_end_local=None,
    min_fork_gap: int = 15,
    skip_ruler_rejection: bool = False,
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
    step_images.append((mask_vis, "二值化分割：黑底排除 + Otsu，白色=植物区域"))
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

    # 3. Find main stem. Prefer VLM endpoint anchors when available, then
    # fall back to pure CV scoring.
    MAX_RULER_RETRIES = 2
    main_path = []
    stem_len = 0
    stem_score = 0.0
    stem_source = "scored"
    stem_hint_dists = None
    if stem_start_local is not None and stem_end_local is not None:
        main_path, stem_len, stem_score, stem_hint_dists = _find_main_stem_from_hints(
            skel_arr, stem_start_local, stem_end_local)
        if main_path:
            stem_source = "vlm_hint"

    for _ruler_attempt in range(MAX_RULER_RETRIES + 1):
        if not main_path:
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
                if stem_source != "vlm_hint" and not skip_ruler_rejection and _cv < 0.15:
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
    stem_desc = f"主干识别：橙色=主干路径（{stem_len}px），绿色=端点"
    if stem_source == "vlm_hint":
        if stem_hint_dists:
            stem_desc += (
                f"，VLM端点引导（吸附距离 {stem_hint_dists[0]:.1f}/"
                f"{stem_hint_dists[1]:.1f}px）"
            )
        else:
            stem_desc += "，VLM端点引导"
    step_images.append((stem_vis, stem_desc))

    start_pt, end_pt = main_path[0], main_path[-1]
    main_set = set(main_path)

    # Extend main path toward root endpoint in the bottom third.
    # The straightness-based scoring may stop short of the true root tip
    # if the root bends. A short low-curvature extension is safe to add.
    ROOT_ZONE = h * 2 // 3   # bottom third of the crop
    root_endpoints = [
        (er, ec) for er, ec in _nb_find_endpoints(skel_arr)
        if er >= ROOT_ZONE and (er, ec) not in main_set
    ]
    if root_endpoints and stem_source != "vlm_hint":
        best_ext = None
        best_ext_len = 0
        # Pick the root endpoint closest to the current main_path bottom end.
        # Use the endpoint (end_pt or start_pt) that is lower in the image.
        anchor = end_pt if end_pt[0] > start_pt[0] else start_pt
        for er, ec in root_endpoints:
            sub_path = _nb_bfs_path(skel_arr, anchor[0], anchor[1], er, ec)
            sub_len = sub_path.shape[0]
            if sub_len < 6 or sub_len > stem_len * 0.6:
                continue
            sy = float(sub_path[-1, 0] - sub_path[0, 0])
            sx = float(sub_path[-1, 1] - sub_path[0, 1])
            sub_dist = (sy * sy + sx * sx) ** 0.5
            straight = sub_dist / float(sub_len)
            if straight < 0.30:
                continue
            if sub_len > best_ext_len:
                best_ext_len = sub_len
                best_ext = [(int(sub_path[k, 0]), int(sub_path[k, 1]))
                            for k in range(1, sub_path.shape[0])]  # skip anchor (already in main_path)
        if best_ext:
            if anchor == end_pt:
                main_path.extend(best_ext)
            else:
                # Prepend — main_path needs to keep start→end order.
                # If we extended from start_pt, reverse so the extension heads further out.
                best_ext.reverse()
                main_path = best_ext + main_path
            stem_len = len(main_path)
            print(f"  [PodStalk] Root extended +{best_ext_len}px → stem_len={stem_len}")

    main_set = set(main_path)  # rebuild in case main_path was extended

    # ══════════════════════════════════════════════════════════════════
    # PASS 1 — topology-based dead twig identification.
    # ══════════════════════════════════════════════════════════════════
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    nodes_p1, edges_p1 = _build_topo_graph(skel_set, main_set)
    if not nodes_p1:
        return {"pod_count": 0, "debug_image": crop_bgr.copy(),
                "step_images": step_images, "note": "no graph nodes"}

    adj_p1 = _graph_adjacency(nodes_p1, edges_p1)
    attach_nodes_p1 = set(n for n, t in nodes_p1.items() if t == 'attach')
    tip_nodes_p1 = [n for n, t in nodes_p1.items() if t == 'tip']
    junction_nodes_p1 = set(n for n, t in nodes_p1.items() if t == 'junction')
    CROSSING_GUARD_RADIUS = 3
    junction_guard_mask = None
    if junction_nodes_p1:
        junction_guard_mask = np.zeros((h, w), dtype=np.uint8)
        for r, c in junction_nodes_p1:
            junction_guard_mask[r, c] = 255
        k = CROSSING_GUARD_RADIUS * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        junction_guard_mask = cv2.dilate(junction_guard_mask, kernel, iterations=1)

    def _is_crossing_related(path_pixels):
        if junction_guard_mask is None:
            return False
        return any(junction_guard_mask[r, c] > 0 for r, c in path_pixels)

    # BFS from each tip to the nearest main-stem attachment.
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
        crossing_related = _is_crossing_related(full_path)
        branch_data.append((tip, full_path, found_attach, area, avg_w, crossing_related))

    # Filter: dead twigs (width/area) + noise (short complete branches)
    WIDTH_RATIO = 0.70
    AREA_RATIO = 0.25
    MIN_BRANCHES = 4
    MIN_BRANCH_LEN = 10  # denoising: complete branches shorter than this are noise
    rejected_pixels = set()
    rejected_branches = []
    crossing_guard_count = 0
    crossing_protected_ids = set()
    median_area = 0.0
    median_width = 0.0
    median_len = 0.0

    # Step A: remove very short complete branches (denoising)
    for item in branch_data:
        if len(item[1]) < MIN_BRANCH_LEN:
            rejected_branches.append(item)
            rejected_pixels.update(item[1])
    remaining = [item for item in branch_data if item not in rejected_branches]
    noise_count = len(rejected_branches)

    reference_branches = remaining if remaining else branch_data
    if reference_branches:
        median_area = float(np.median([d[3] for d in reference_branches]))
        median_width = float(np.median([d[4] for d in reference_branches]))
        median_len = float(np.median([len(d[1]) for d in reference_branches]))

    # Step B: adaptive width/area filtering for dead twigs
    if remaining and len(remaining) >= MIN_BRANCHES:
        area_fence = median_area * AREA_RATIO
        width_fence = median_width * WIDTH_RATIO

        for item in remaining:
            area_val = item[3]
            avg_w_val = item[4]
            crossing_related = item[5]
            if area_val >= area_fence and avg_w_val >= width_fence:
                pass  # keep
            elif crossing_related:
                crossing_guard_count += 1
                crossing_protected_ids.add(id(item))
            else:
                rejected_branches.append(item)
                rejected_pixels.update(item[1])
        twig_count = len(rejected_branches) - noise_count
        print(f"  [PodStalk] Pass1: branches={len(branch_data)}, "
              f"noise(<{MIN_BRANCH_LEN}px)={noise_count}, "
              f"med_area={median_area:.0f} fence={area_fence:.0f}, "
              f"med_w={median_width:.1f} fence={width_fence:.1f}, "
              f"cross_guard={crossing_guard_count}, "
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
    guarded_p1 = [item for item in kept_p1 if id(item) in crossing_protected_ids]
    for item in guarded_p1:
        for p in item[1]:
            cv2.circle(p1_vis, (p[1], p[0]), 1, (0, 215, 255), -1)
        cv2.circle(p1_vis, (item[0][1], item[0][0]), 4, (0, 215, 255), -1)
    step_images.append((
        p1_vis,
        f"Pass1 枯枝过滤：粉色=枯枝({len(rejected_branches)})，"
        f"黄色=交叉保护保留({len(guarded_p1)})，灰色=其他保留({len(kept_p1) - len(guarded_p1)})"
    ))

    # ══════════════════════════════════════════════════════════════════
    # PASS 2 — Remove dead twigs, rebuild graph, count stalks
    # ══════════════════════════════════════════════════════════════════
    # Clean skeleton: remove rejected pixels (but keep main stem)
    cleaned_skel_set = skel_set - rejected_pixels
    plant_med_v = _plant_median_v(crop_bgr, mask)

    # Rebuild topology graph on cleaned skeleton
    nodes_p2, edges_p2 = _build_topo_graph(cleaned_skel_set, main_set)
    stalk_info = []
    p2_rejected = []  # (marker_pt, path_pixels, reason, metrics)
    stalk_candidates = []
    split_group_count = 0
    split_extra_count = 0

    if not nodes_p2:
        pod_count = 0
    else:
        adj_p2 = _graph_adjacency(nodes_p2, edges_p2)
        attach_nodes_p2 = set(n for n, t in nodes_p2.items() if t == 'attach')
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
                root_path, candidate_path = _collect_candidate_path(
                    node, nb, ei, adj_p2, edges_p2, attach_nodes_p2)
                if not candidate_path:
                    continue

                split_root, split_paths = _collect_near_root_split_paths(
                    node, nb, ei, adj_p2, edges_p2, attach_nodes_p2, nodes_p2)
                root_len = len(root_path)
                should_split = (
                    len(split_paths) >= 2
                    and root_len <= 24
                    and any(len(path) >= 16 for path in split_paths)
                )
                if should_split:
                    split_group_count += 1
                    split_extra_count += len(split_paths) - 1
                    for split_idx, split_path in enumerate(split_paths, 1):
                        metrics = _path_metrics(split_path, dist, crop_bgr, mask, plant_med_v)
                        metrics["root_len"] = root_len
                        metrics["split_candidate"] = True
                        metrics["split_group_size"] = len(split_paths)
                        metrics["split_index"] = split_idx
                        crossing_related = _is_crossing_related(split_path)
                        metrics["crossing_related"] = crossing_related
                        stalk_candidates.append(
                            (node, nb, split_root, split_path, metrics, crossing_related)
                        )
                else:
                    metrics = _path_metrics(candidate_path, dist, crop_bgr, mask, plant_med_v)
                    metrics["root_len"] = root_len
                    metrics["split_candidate"] = False
                    crossing_related = _is_crossing_related(candidate_path)
                    metrics["crossing_related"] = crossing_related
                    stalk_candidates.append(
                        (node, nb, root_path, candidate_path, metrics, crossing_related)
                    )

        candidate_metrics = [c[4] for c in stalk_candidates if c[4]["length"] >= 3]
        p2_median_area = median_area or (
            float(np.median([m["area"] for m in candidate_metrics]))
            if candidate_metrics else 1.0)
        p2_median_width = median_width or (
            float(np.median([m["avg_w"] for m in candidate_metrics]))
            if candidate_metrics else 1.0)
        p2_median_len = median_len or (
            float(np.median([m["length"] for m in candidate_metrics]))
            if candidate_metrics else 1.0)

        # Count only attach edges that pass morphology/color validation.
        for node, nb, root_path, candidate_path, metrics, crossing_related in stalk_candidates:
            keep, reason, score = _score_stalk_candidate(
                metrics, p2_median_area, p2_median_width, p2_median_len,
                crossing_related)
            metrics["decision"] = reason
            metrics["score"] = score

            marker_path = (
                candidate_path
                if metrics.get("split_candidate")
                else (root_path if root_path else candidate_path)
            )
            marker_pt = marker_path[-1]
            if keep:
                stalk_info.append((node, marker_pt, marker_path, len(marker_path), metrics))
            else:
                p2_rejected.append((marker_pt, candidate_path, reason, metrics))

        pod_count = len(stalk_info)

    print(f"  [PodStalk] Pass2: candidates={len(stalk_candidates)}, "
          f"reject={len(p2_rejected)}, stalks={len(stalk_info)}, "
          f"pod_count={pod_count}, split_groups={split_group_count}, "
          f"split_extra={split_extra_count}")

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
    # Layer 3b: Pass2 rejected attach candidates in darker purple
    for marker_pt, path_px, reason, metrics in p2_rejected:
        for p in path_px:
            cv2.circle(debug, (p[1], p[0]), 1, (130, 40, 180), -1)
        cv2.circle(debug, (marker_pt[1], marker_pt[0]), 4, (130, 40, 180), -1)
    # Layer 4: Valid stalks in red, attach in white, tips in cyan
    for idx_s, (attach_pt, tip_pt, path_px, path_len, metrics) in enumerate(stalk_info, 1):
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
                f"p1_reject={len(rejected_branches)} p2_reject={len(p2_rejected)} "
                f"split+{split_extra_count}",
                (5, int(50 * fs) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (180, 180, 180), 1)

    # Step image: stalk counting result
    step_images.append((debug.copy(),
                        f"果柄计数结果：{pod_count} 个（Pass2 过滤 {len(p2_rejected)} 个候选，"
                        f"近距分叉拆分 +{split_extra_count}）"))

    # ── Standardized markers for downstream VLM verification ──
    # Build in priority order, then dedupe so no two markers overlap visually.
    MIN_MARKER_DIST_PX = 8  # min pixel distance between any two marker tips

    markers = []
    used_tips = []  # (r, c) in pixel coords

    def _too_close(tip_pt):
        r, c = tip_pt
        for ur, uc in used_tips:
            if (r - ur) ** 2 + (c - uc) ** 2 < MIN_MARKER_DIST_PX ** 2:
                return True
        return False

    # Pod markers first (highest priority)
    for idx_s, (attach_pt, tip_pt, path_px, path_len, metrics) in enumerate(stalk_info, 1):
        if _too_close(tip_pt):
            continue
        markers.append({"id": idx_s, "type": "pod",
                        "x_norm": tip_pt[1] / w, "y_norm": tip_pt[0] / h})
        used_tips.append(tip_pt)

    # P1 rejected branches
    for j, item in enumerate(rejected_branches):
        tip = item[0]
        if _too_close(tip):
            continue
        mid = len(stalk_info) + j + 1
        markers.append({"id": mid, "type": "filtered",
                        "x_norm": tip[1] / w, "y_norm": tip[0] / h})
        used_tips.append(tip)

    # P2 rejected candidates
    for j, (tip, path_px, reason, metrics) in enumerate(p2_rejected):
        if _too_close(tip):
            continue
        mid = len(stalk_info) + len(rejected_branches) + j + 1
        markers.append({"id": mid, "type": "filtered",
                        "x_norm": tip[1] / w, "y_norm": tip[0] / h})
        used_tips.append(tip)

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
