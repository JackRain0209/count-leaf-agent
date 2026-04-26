"""
Pod (角果) counting via skeleton → topology graph.

Algorithm:
  1. Binarize + skeletonize  (shared with pod_counter.py)
  2. Find main stem (Numba-accelerated longest-straight path)
  3. Build topology graph from non-main skeleton:
     - Nodes = key pixels (tips, junctions, attachment points)
     - Edges = pixel paths between consecutive nodes
  4. From each attachment node, traverse graph edges to reach tip nodes
     Each tip = 1 pod.  Edges can be shared (no pixel-level blocking).
  5. Filter branches by adaptive median-ratio thresholds on width/area
"""

import cv2
import numpy as np
from collections import deque
from skimage.morphology import skeletonize

# Reuse common utilities from the skeleton-based counter
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


def _build_topo_graph(skel_set, main_set):
    """Build topology graph from non-main skeleton pixels.

    Returns:
        nodes: dict  pixel -> 'tip' | 'junction' | 'attach'
        edges: list of (node_a, node_b, path_pixels)
               path_pixels includes endpoints (node_a and node_b).
    """
    non_main = skel_set - main_set

    # Classify every non-main pixel
    # degree = number of non-main skeleton neighbors (including attach neighbors)
    degree_map = {}
    for p in non_main:
        deg = sum(1 for nb in _neighbors(p, skel_set) if nb in non_main)
        degree_map[p] = deg

    # Identify key nodes
    nodes = {}
    for p in non_main:
        adj_main = any(nb in main_set for nb in _neighbors(p, skel_set))
        deg = degree_map[p]
        if adj_main:
            nodes[p] = 'attach'
        elif deg == 1:
            nodes[p] = 'tip'
        elif deg >= 3:
            nodes[p] = 'junction'
        # deg == 0 → isolated pixel, ignore
        # deg == 2 → pass-through pixel, part of an edge

    # Trace edges: from each node, walk along degree-2 pixels until hitting
    # another node.  Each edge is traced once.
    edges = []
    traced_pairs = set()  # (frozenset of endpoint pair) to avoid duplicates

    for start in list(nodes):
        # Get non-main neighbors of this node
        nbs = [nb for nb in _neighbors(start, skel_set) if nb in non_main]
        for first_step in nbs:
            if first_step in nodes:
                # Direct node-to-node edge (no intermediate pixels)
                pair = frozenset((start, first_step))
                if pair not in traced_pairs:
                    traced_pairs.add(pair)
                    edges.append((start, first_step, [start, first_step]))
                continue

            # Walk along degree-2 pixels
            path = [start, first_step]
            visited = {start, first_step}
            cur = first_step
            while True:
                next_pixels = [nb for nb in _neighbors(cur, skel_set)
                               if nb in non_main and nb not in visited]
                if not next_pixels:
                    # Dead end without hitting a node — shouldn't normally happen
                    # but treat cur as a pseudo-tip
                    if cur not in nodes:
                        nodes[cur] = 'tip'
                    break
                if len(next_pixels) == 1:
                    nxt = next_pixels[0]
                    path.append(nxt)
                    visited.add(nxt)
                    if nxt in nodes:
                        break  # reached another node
                    cur = nxt
                else:
                    # Multiple exits — cur should be a junction but wasn't classified
                    # (can happen due to skeleton noise)
                    if cur not in nodes:
                        nodes[cur] = 'junction'
                    break

            end = path[-1]
            pair = frozenset((start, end))
            if pair not in traced_pairs:
                traced_pairs.add(pair)
                edges.append((start, end, path))

    return nodes, edges


def _graph_adjacency(nodes, edges):
    """Build adjacency list: node -> [(neighbor_node, edge_index), ...]"""
    adj = {n: [] for n in nodes}
    for i, (a, b, _) in enumerate(edges):
        if a in adj:
            adj[a].append((b, i))
        if b in adj:
            adj[b].append((a, i))
    return adj


def count_pods_on_branch(
    crop_bgr: np.ndarray,
    stem_start_local=None,
    stem_end_local=None,
    min_fork_gap: int = 15,
) -> dict:
    """Count pods on a single branch crop via skeleton topology graph.

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
                    print(f"  [PodGraph] Ruler rejected: CV={_cv:.3f}")
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

    # ── 4. Build topology graph ──
    non_main_set = skel_set - main_set
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    nodes, edges = _build_topo_graph(skel_set, main_set)

    if not nodes:
        return {"pod_count": 0, "debug_image": crop_bgr.copy(),
                "step_images": step_images, "note": "no graph nodes"}

    adj = _graph_adjacency(nodes, edges)

    # Pre-compute edge metrics
    edge_metrics = []  # parallel to edges: (length, area, avg_width)
    for (a, b, path) in edges:
        length = len(path)
        area = sum(float(dist[r, c]) * 2.0 for (r, c) in path)
        avg_w = area / length if length > 0 else 0.0
        edge_metrics.append((length, area, avg_w))

    # ── 5. BFS from each tip → find shortest path to nearest attach node ──
    #    This is O(V+E) per tip, much faster than DFS enumerate-all-paths.
    #    Edges can be shared across different tips' paths.
    attach_nodes = set(n for n, t in nodes.items() if t == 'attach')
    tip_nodes = [n for n, t in nodes.items() if t == 'tip']

    branch_data = []  # (tip, path_pixels, attach_pt, area, avg_w)

    for tip in tip_nodes:
        # BFS on graph from tip to any attach node
        parent = {tip: None}   # node → (parent_node, edge_index)
        queue = deque([tip])
        found_attach = None
        while queue:
            cur = queue.popleft()
            if cur in attach_nodes:
                found_attach = cur
                break
            for nb, ei in adj[cur]:
                if nb not in parent:
                    parent[nb] = (cur, ei)
                    queue.append(nb)

        if found_attach is None:
            continue

        # Reconstruct node path: tip → ... → attach
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
        # node_path is attach → ... → tip, reverse to tip → ... → attach
        node_path.reverse()
        edge_indices.reverse()

        # Build full pixel path
        full_path = []
        for k in range(len(node_path) - 1):
            n1, n2 = node_path[k], node_path[k + 1]
            ei = edge_indices[k]
            ep = edges[ei][2]
            # Orient: n1 → n2
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

    tip_endpoints = [d[0] for d in branch_data]

    # ── 6. Filter: adaptive outlier removal on width and area ──
    WIDTH_RATIO = 0.6
    AREA_RATIO = 0.25
    MIN_BRANCHES = 4
    real_pods = []
    rejected = []

    if not branch_data:
        pass
    elif len(branch_data) < MIN_BRANCHES:
        real_pods = list(branch_data)
    else:
        all_areas = np.array([d[3] for d in branch_data])
        all_widths = np.array([d[4] for d in branch_data])
        median_area = float(np.median(all_areas))
        median_width = float(np.median(all_widths))
        area_fence = median_area * AREA_RATIO
        width_fence = median_width * WIDTH_RATIO

        for item in branch_data:
            area = item[3]
            avg_w = item[4]
            if area >= area_fence and avg_w >= width_fence:
                real_pods.append(item)
            else:
                rejected.append(item)
        print(f"  [PodGraph] branches={len(branch_data)}, "
              f"med_area={median_area:.0f} fence={area_fence:.0f}, "
              f"med_w={median_width:.1f} fence={width_fence:.1f}, "
              f"keep={len(real_pods)}, reject={len(rejected)}")

    pod_count = len(real_pods)

    # ── 7. Debug visualization ──
    debug = crop_bgr.copy()
    # Layer 1: Main stem in orange
    for p in main_path:
        cv2.circle(debug, (p[1], p[0]), 2, (255, 128, 0), -1)
    # Layer 2: Non-main skeleton in dark gray
    for p in non_main_set:
        cv2.circle(debug, (p[1], p[0]), 1, (80, 80, 80), -1)
    # Layer 3: Graph edges in dark cyan
    for (a, b, path) in edges:
        for p in path:
            cv2.circle(debug, (p[1], p[0]), 1, (128, 128, 0), -1)
    # Layer 4: Rejected in purple
    for tip, bp, ap, a, w_ in rejected:
        for p in bp:
            cv2.circle(debug, (p[1], p[0]), 1, (200, 80, 200), -1)
        cv2.circle(debug, (tip[1], tip[0]), 4, (200, 80, 200), -1)
    # Layer 5: Counted pods in red, tips in cyan
    for i, (tip, bp, ap, a, w_) in enumerate(real_pods, 1):
        for p in bp:
            cv2.circle(debug, (p[1], p[0]), 1, (0, 0, 255), -1)
        cv2.circle(debug, (tip[1], tip[0]), 5, (255, 255, 0), -1)
        cv2.circle(debug, (ap[1], ap[0]), 4, (255, 255, 255), -1)
        cv2.putText(debug, str(i), (tip[1] + 6, tip[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    # Stem endpoints
    for p in [start_pt, end_pt]:
        cv2.circle(debug, (p[1], p[0]), 6, (0, 255, 0), 2)
    # Count text
    fs = max(0.5, min(h, w) / 500)
    cv2.putText(debug, f"Pods: {pod_count} (graph)", (5, int(25 * fs) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), max(1, int(fs * 2)))
    cv2.putText(debug,
                f"tips={len(tip_endpoints)} reject={len(rejected)} "
                f"nodes={len(nodes)} edges={len(edges)}",
                (5, int(50 * fs) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (180, 180, 180), 1)

    # Step image A: filter vis
    filter_vis = crop_bgr.copy()
    for p in main_path:
        cv2.circle(filter_vis, (p[1], p[0]), 2, (255, 128, 0), -1)
    for p in non_main_set:
        cv2.circle(filter_vis, (p[1], p[0]), 1, (100, 100, 100), -1)
    for tip, bp, ap, a, w_ in real_pods:
        cv2.circle(filter_vis, (tip[1], tip[0]), 5, (0, 255, 0), -1)
        txt = f"a={a:.0f} w={w_:.1f} l={len(bp)}"
        cv2.putText(filter_vis, txt, (tip[1] + 6, tip[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    for tip, bp, ap, a, w_ in rejected:
        cv2.circle(filter_vis, (tip[1], tip[0]), 4, (0, 0, 255), -1)
        txt = f"a={a:.0f} w={w_:.1f} l={len(bp)}"
        cv2.putText(filter_vis, txt, (tip[1] + 6, tip[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    step_images.append((
        filter_vis,
        f"拓扑图过滤：绿=角果({len(real_pods)})，红=枯枝({len(rejected)})"
    ))

    step_images.append((debug.copy(),
                        f"角果计数结果：{pod_count} 个（拓扑图方案）"))

    print(f"  [PodGraph:{stem_source}] stem={stem_len}px, nodes={len(nodes)}, "
          f"edges={len(edges)}, tips={len(tip_endpoints)}, "
          f"reject={len(rejected)}, pods={pod_count}")

    return {
        "pod_count": pod_count,
        "main_stem_length": stem_len,
        "fork_count": len(tip_endpoints),
        "skeleton_pixels": len(skel_set),
        "stem_source": stem_source,
        "debug_image": debug,
        "step_images": step_images,
    }
