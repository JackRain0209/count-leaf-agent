"""
Pod (角果) counting via skeleton traversal.

Algorithm for each branch crop:
  1. Binarize (plant on dark background)
  2. Skeletonize → 1px-wide structure
  3. Find main stem: longest path between two furthest endpoints (2-pass BFS)
  4. Walk main stem, each side-branch fork = 1 pod
  5. Cluster nearby forks to avoid double-counting
"""

import cv2
import numpy as np
from collections import deque
from skimage.morphology import skeletonize
from numba import njit


def _binarize(crop_bgr: np.ndarray) -> np.ndarray:
    """Gaussian blur + Otsu + CLOSE + DILATE/ERODE to bridge broken thin stems."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    # Bridge small gaps at branch roots: dilate then erode (= CLOSE with larger kernel)
    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, k_bridge, iterations=1)
    mask = cv2.erode(mask, k_bridge, iterations=1)
    return mask


def _get_skeleton(mask: np.ndarray) -> np.ndarray:
    """Get 1px-wide skeleton."""
    skel = skeletonize(mask > 0)
    return (skel.astype(np.uint8)) * 255


# ─── Numba-accelerated BFS on 2D boolean array ───

@njit(cache=True)
def _nb_bfs_path(skel_arr, sr, sc, er, ec):
    """BFS shortest path on 2D bool array. Returns path as Nx2 int32 array."""
    H, W = skel_arr.shape
    parent_r = np.full((H, W), -1, dtype=np.int32)
    parent_c = np.full((H, W), -1, dtype=np.int32)
    visited = np.zeros((H, W), dtype=np.bool_)
    visited[sr, sc] = True
    # Manual queue using pre-allocated arrays
    queue_r = np.empty(H * W, dtype=np.int32)
    queue_c = np.empty(H * W, dtype=np.int32)
    head = 0
    tail = 0
    queue_r[tail] = sr
    queue_c[tail] = sc
    tail += 1
    found = False
    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        head += 1
        if r == er and c == ec:
            found = True
            break
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr = r + dr
                nc = c + dc
                if 0 <= nr < H and 0 <= nc < W and skel_arr[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    parent_r[nr, nc] = r
                    parent_c[nr, nc] = c
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1
    if not found:
        return np.empty((0, 2), dtype=np.int32)
    # Trace back
    path_r = np.empty(H * W, dtype=np.int32)
    path_c = np.empty(H * W, dtype=np.int32)
    length = 0
    cr_, cc_ = er, ec
    while cr_ != -1:
        path_r[length] = cr_
        path_c[length] = cc_
        length += 1
        pr = parent_r[cr_, cc_]
        pc = parent_c[cr_, cc_]
        cr_, cc_ = pr, pc
    # Reverse
    result = np.empty((length, 2), dtype=np.int32)
    for i in range(length):
        result[i, 0] = path_r[length - 1 - i]
        result[i, 1] = path_c[length - 1 - i]
    return result


@njit(cache=True)
def _nb_bfs_farthest(skel_arr, sr, sc):
    """BFS to find farthest point from (sr, sc). Returns (fr, fc, dist)."""
    H, W = skel_arr.shape
    visited = np.zeros((H, W), dtype=np.bool_)
    visited[sr, sc] = True
    queue_r = np.empty(H * W, dtype=np.int32)
    queue_c = np.empty(H * W, dtype=np.int32)
    queue_d = np.empty(H * W, dtype=np.int32)
    head = 0
    tail = 0
    queue_r[tail] = sr
    queue_c[tail] = sc
    queue_d[tail] = 0
    tail += 1
    fr, fc, max_d = sr, sc, 0
    while head < tail:
        r = queue_r[head]
        c = queue_c[head]
        d = queue_d[head]
        head += 1
        if d > max_d:
            max_d = d
            fr, fc = r, c
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr = r + dr
                nc = c + dc
                if 0 <= nr < H and 0 <= nc < W and skel_arr[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    queue_d[tail] = d + 1
                    tail += 1
    return fr, fc, max_d


@njit(cache=True)
def _nb_find_main_stem(skel_arr, endpoints, min_straightness=0.5):
    """
    Enumerate all endpoint pairs, score each path, pick best.
    score = path_length² × straightness⁶
    Higher straightness exponent penalizes curvy end-branches that are long but bendy.
    Returns index of best_start, best_end in endpoints, and best_score.
    """
    n = endpoints.shape[0]
    best_i = -1
    best_j = -1
    best_score = 0.0
    H, W = skel_arr.shape
    for i in range(n):
        for j in range(i + 1, n):
            sr, sc = endpoints[i, 0], endpoints[i, 1]
            er, ec = endpoints[j, 0], endpoints[j, 1]
            path = _nb_bfs_path(skel_arr, sr, sc, er, ec)
            L = path.shape[0]
            if L < 10:
                continue
            dy = float(path[L - 1, 0] - path[0, 0])
            dx = float(path[L - 1, 1] - path[0, 1])
            D = (dy * dy + dx * dx) ** 0.5
            S = D / float(L)
            if S < min_straightness:
                continue
            # straightness^6 — heavier curvature penalty
            S2 = S * S
            S6 = S2 * S2 * S2
            score = float(L) * float(L) * S6
            if score > best_score:
                best_score = score
                best_i = i
                best_j = j
    return best_i, best_j, best_score


@njit(cache=True)
def _nb_neighbor_count(skel_arr, r, c):
    """Count 8-connected skeleton neighbors."""
    H, W = skel_arr.shape
    cnt = 0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < H and 0 <= nc < W and skel_arr[nr, nc]:
                cnt += 1
    return cnt


@njit(cache=True)
def _nb_find_endpoints(skel_arr):
    """Find all endpoint pixels (1 neighbor) in skeleton."""
    H, W = skel_arr.shape
    pts_r = np.empty(H * W, dtype=np.int32)
    pts_c = np.empty(H * W, dtype=np.int32)
    count = 0
    for r in range(H):
        for c in range(W):
            if skel_arr[r, c] and _nb_neighbor_count(skel_arr, r, c) == 1:
                pts_r[count] = r
                pts_c[count] = c
                count += 1
    result = np.empty((count, 2), dtype=np.int32)
    for i in range(count):
        result[i, 0] = pts_r[i]
        result[i, 1] = pts_c[i]
    return result


# ─── Python wrappers (kept for fork detection / side-branch counting) ───

def _neighbors(p, skel_set):
    """8-connected neighbors that are in skeleton."""
    r, c = p
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr or dc) and (r + dr, c + dc) in skel_set]


def _find_main_stem_numba(skel_arr, min_straightness=0.5):
    """
    Use Numba-accelerated functions to find best main stem path.
    Returns (path_as_list_of_tuples, length, score).
    """
    endpoints = _nb_find_endpoints(skel_arr)
    if endpoints.shape[0] < 2:
        return [], 0, 0.0

    best_i, best_j, best_score = _nb_find_main_stem(skel_arr, endpoints, min_straightness)
    if best_i < 0:
        return [], 0, 0.0

    # Retrieve the best path
    path_arr = _nb_bfs_path(skel_arr, endpoints[best_i, 0], endpoints[best_i, 1],
                            endpoints[best_j, 0], endpoints[best_j, 1])
    path_list = [(int(path_arr[k, 0]), int(path_arr[k, 1])) for k in range(path_arr.shape[0])]
    return path_list, len(path_list), best_score


def _remove_ruler_region(mask: np.ndarray, density_thresh: float = 0.25,
                         max_scan: float = 0.40, smooth_win: int = 15) -> np.ndarray:
    """Remove ruler strips by scanning inward from each border with smoothed
    column/row density.

    Unlike a naive per-column check, this uses a sliding-window average so
    ruler scale markings (which create brief density dips) don't stop the scan
    prematurely.

    For each border (left/right/top/bottom):
      1. Compute white-pixel ratio per column (or row)
      2. Smooth with a moving average window
      3. Scan inward: while smoothed density > threshold, mark for removal
      4. Zero out marked columns/rows in the mask
    Max scan depth is capped at max_scan fraction of image size.
    """
    mh, mw = mask.shape

    def _smooth(arr, win):
        if len(arr) < win:
            return arr
        kernel = np.ones(win) / win
        return np.convolve(arr, kernel, mode='same')

    # Column density (for left/right scan)
    col_density = np.count_nonzero(mask, axis=0).astype(float) / mh
    col_smooth = _smooth(col_density, smooth_win)

    # Row density (for top/bottom scan)
    row_density = np.count_nonzero(mask, axis=1).astype(float) / mw
    row_smooth = _smooth(row_density, smooth_win)

    max_cols = max(2, int(mw * max_scan))
    max_rows = max(2, int(mh * max_scan))

    # Left
    cut_left = 0
    for c in range(max_cols):
        if col_smooth[c] > density_thresh:
            cut_left = c + 1
        else:
            break
    if cut_left > 0:
        mask[:, :cut_left] = 0
        print(f"  [Ruler] Left strip removed: cols 0~{cut_left} (smoothed density > {density_thresh})")

    # Right
    cut_right = mw
    for c in range(mw - 1, mw - 1 - max_cols, -1):
        if col_smooth[c] > density_thresh:
            cut_right = c
        else:
            break
    if cut_right < mw:
        mask[:, cut_right:] = 0
        print(f"  [Ruler] Right strip removed: cols {cut_right}~{mw} (smoothed density > {density_thresh})")

    # Top
    cut_top = 0
    for r in range(max_rows):
        if row_smooth[r] > density_thresh:
            cut_top = r + 1
        else:
            break
    if cut_top > 0:
        mask[:cut_top, :] = 0
        print(f"  [Ruler] Top strip removed: rows 0~{cut_top}")

    # Bottom
    cut_bot = mh
    for r in range(mh - 1, mh - 1 - max_rows, -1):
        if row_smooth[r] > density_thresh:
            cut_bot = r
        else:
            break
    if cut_bot < mh:
        mask[cut_bot:, :] = 0
        print(f"  [Ruler] Bottom strip removed: rows {cut_bot}~{mh}")

    return mask


def _select_center_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the connected component whose centroid is closest to the image
    center, among those with area >= 5% of total white pixels.
    This filters out rulers, stray objects, and bleed-in from adjacent parts."""
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 2:
        return mask  # 0=bg + 1 component, nothing to filter

    total_white = cv2.countNonZero(mask)
    min_area = max(50, int(total_white * 0.05))
    cy_center = mask.shape[0] / 2.0
    cx_center = mask.shape[1] / 2.0

    best_label = -1
    best_dist = float('inf')
    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[lbl]
        d = (cx - cx_center) ** 2 + (cy - cy_center) ** 2
        if d < best_dist:
            best_dist = d
            best_label = lbl

    if best_label < 0:
        return mask  # fallback: keep original

    out = np.zeros_like(mask)
    out[labels == best_label] = 255
    return out


def count_pods_on_branch(
    crop_bgr: np.ndarray,
    stem_start_local=None,  # kept for compat, unused
    stem_end_local=None,
    min_fork_gap: int = 15,
) -> dict:
    """Count pods on a single branch crop via skeleton traversal."""
    h, w = crop_bgr.shape[:2]

    # 0. Auto-upscale low-res images to ensure skeleton quality
    MIN_DIM = 800
    max_side = max(h, w)
    if max_side < MIN_DIM:
        scale = MIN_DIM / max_side
        crop_bgr = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        h, w = crop_bgr.shape[:2]

    step_images = []  # list of (image, description)

    # 1. Binarize (Gaussian blur + Otsu)
    mask = _binarize(crop_bgr)
    # --- Step image: binary mask ---
    mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    step_images.append((mask_vis, "二值化分割：Otsu + 暗色恢复，白色=植物区域"))
    if cv2.countNonZero(mask) < 50:
        return {"pod_count": 0, "debug_image": crop_bgr.copy(), "step_images": step_images, "note": "mask too small"}

    # 1b. Ruler removal: smoothed column/row density scan from borders.
    mask = _remove_ruler_region(mask)

    # 1c. Center-priority connected component selection
    mask = _select_center_component(mask)

    # 2. Skeletonize
    skeleton = _get_skeleton(mask)
    skel_arr = (skeleton > 0).astype(np.bool_)
    skel_set = set(map(tuple, np.argwhere(skel_arr)))
    # --- Step image: skeleton ---
    skel_vis = crop_bgr.copy()
    skel_vis[skel_arr] = (0, 255, 255)
    step_images.append((skel_vis, f"骨架提取：1px 宽骨架线（黄色），共 {len(skel_set)} 像素"))
    if len(skel_set) < 20:
        return {"pod_count": 0, "debug_image": crop_bgr.copy(), "step_images": step_images, "note": "skeleton too short"}

    # 3. Find main stem (Numba-accelerated), with ruler rejection loop
    #    If the best main-stem candidate has very uniform width (CV < 0.15),
    #    it's likely a ruler — remove that component from the mask and retry.
    MAX_RULER_RETRIES = 2
    for _ruler_attempt in range(MAX_RULER_RETRIES + 1):
        main_path, stem_len, stem_score = _find_main_stem_numba(skel_arr)
        stem_source = "scored"

        # Fallback: plain longest path (2-pass BFS, also Numba)
        if not main_path:
            endpoints = _nb_find_endpoints(skel_arr)
            if endpoints.shape[0] >= 1:
                fr, fc, _ = _nb_bfs_farthest(skel_arr, endpoints[0, 0], endpoints[0, 1])
                fr2, fc2, _ = _nb_bfs_farthest(skel_arr, fr, fc)
                path_arr = _nb_bfs_path(skel_arr, fr, fc, fr2, fc2)
                main_path = [(int(path_arr[k, 0]), int(path_arr[k, 1])) for k in range(path_arr.shape[0])]
                stem_len = len(main_path)
            stem_source = "bfs_fallback"

        if not main_path:
            break

        # Ruler check: measure width uniformity along main stem
        if _ruler_attempt < MAX_RULER_RETRIES and stem_len > 30:
            _rdist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            _widths = np.array([_rdist[p[0], p[1]] for p in main_path])
            _mean_w = _widths.mean()
            if _mean_w > 0:
                _cv = _widths.std() / _mean_w  # coefficient of variation
                if _cv < 0.15:
                    # Very uniform width → likely ruler, erase its connected component
                    mid = main_path[len(main_path) // 2]
                    _flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
                    cv2.floodFill(mask, _flood_mask, (mid[1], mid[0]), 0)
                    print(f"  [PodCounter] Ruler rejected: CV={_cv:.3f}, erasing component at ({mid[1]},{mid[0]})")
                    skeleton = _get_skeleton(mask)
                    skel_arr = (skeleton > 0).astype(np.bool_)
                    skel_set = set(map(tuple, np.argwhere(skel_arr)))
                    if len(skel_set) < 20:
                        main_path = []
                        break
                    continue
        break  # Passed ruler check or no retry needed

    if not main_path:
        return {"pod_count": 0, "debug_image": crop_bgr.copy(), "step_images": step_images, "note": "no main path"}

    # --- Step image: main stem ---
    stem_vis = crop_bgr.copy()
    for p in main_path:
        cv2.circle(stem_vis, (p[1], p[0]), 2, (255, 128, 0), -1)
    cv2.circle(stem_vis, (main_path[0][1], main_path[0][0]), 6, (0, 255, 0), 2)
    cv2.circle(stem_vis, (main_path[-1][1], main_path[-1][0]), 6, (0, 255, 0), 2)
    step_images.append((stem_vis, f"主干识别：橙色=主干路径（{stem_len}px），绿色=端点"))

    start_pt, end_pt = main_path[0], main_path[-1]
    main_set = set(main_path)

    # 5. Pre-compute junction pixels in non-main skeleton
    #    Junction = non-main pixel with ≥3 non-main skeleton neighbors
    #    These are crossing points where two pods overlap
    non_main = skel_set - main_set
    junctions = set()
    for p in non_main:
        non_main_nbs = sum(1 for nb in _neighbors(p, skel_set) if nb not in main_set)
        if non_main_nbs >= 3:
            junctions.add(p)

    # 6. Curvature-aware branch traversal from main stem.
    #    From each attachment point, walk outward following smooth curvature.
    #    At junctions, pick the candidate with smallest angle difference to
    #    the incoming direction. All walked pixels are globally marked to
    #    prevent loops. No junction count limit.
    non_main_set = skel_set - main_set
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    DIR_WINDOW = 8        # pixels to look back for direction estimation
    ANGLE_THRESH = 60.0   # max deflection (degrees) to count as "smooth"

    # 6a. Find attachment points: non-main pixels adjacent to main stem
    attach_points = []
    for p in non_main_set:
        if any(nb in main_set for nb in _neighbors(p, skel_set)):
            attach_points.append(p)

    # Cluster nearby attachment points (within 10px)
    def _cluster_points(pts, radius=10):
        if not pts:
            return []
        used = set()
        clusters = []
        for i, p in enumerate(pts):
            if i in used:
                continue
            group = [p]
            used.add(i)
            for j, q in enumerate(pts):
                if j in used:
                    continue
                if abs(p[0]-q[0]) + abs(p[1]-q[1]) <= radius:
                    group.append(q)
                    used.add(j)
            clusters.append(group)
        return clusters

    attach_clusters = _cluster_points(attach_points, radius=10)

    def _direction_vector(path_segment):
        """Compute direction vector from a path segment (list of (r,c)).
        Returns (dr, dc) normalized, or None if too short."""
        if len(path_segment) < 2:
            return None
        r0, c0 = path_segment[0]
        r1, c1 = path_segment[-1]
        dr, dc = float(r1 - r0), float(c1 - c0)
        mag = (dr*dr + dc*dc) ** 0.5
        if mag < 1e-6:
            return None
        return (dr / mag, dc / mag)

    def _angle_between(d1, d2):
        """Angle in degrees between two direction vectors."""
        if d1 is None or d2 is None:
            return 180.0  # unknown → treat as sharp turn
        dot = d1[0]*d2[0] + d1[1]*d2[1]
        dot = max(-1.0, min(1.0, dot))
        return np.degrees(np.arccos(dot))

    # 6b. Curvature-aware walk from each attachment cluster
    #     Global visited prevents any walk from entering pixels already
    #     claimed by a previous walk → no loops, no double-counting.
    global_visited = set(main_set)  # main stem is off-limits
    branch_data = []  # (tip, path_pixels, attach_pt, area, avg_w)

    for cluster in attach_clusters:
        attach_set = set(cluster)
        # Representative attachment point
        ap_r = sum(p[0] for p in cluster) // len(cluster)
        ap_c = sum(p[1] for p in cluster) // len(cluster)
        attach_pt = (ap_r, ap_c)

        # Find initial outward neighbors from cluster (non-main, non-cluster)
        start_pixels = []
        for ap in cluster:
            for nb in _neighbors(ap, skel_set):
                if nb in non_main_set and nb not in attach_set and nb not in global_visited:
                    start_pixels.append((nb, ap))

        # For each outward start, do curvature-aware walk
        for start_px, from_px in start_pixels:
            if start_px in global_visited:
                continue  # another walk from this cluster already claimed it

            path = [from_px, start_px]
            path_set = {from_px, start_px}
            cur = start_px

            while True:
                # Candidates: non-main neighbors not on this path, not globally visited
                candidates = [nb for nb in _neighbors(cur, skel_set)
                              if nb in non_main_set
                              and nb not in path_set
                              and nb not in global_visited
                              and nb not in attach_set]

                if not candidates:
                    break  # dead end (tip)

                if len(candidates) == 1:
                    nxt = candidates[0]
                    path.append(nxt)
                    path_set.add(nxt)
                    cur = nxt
                    continue

                # Junction: pick candidate with smallest angle diff to incoming dir
                window = path[-min(DIR_WINDOW, len(path)):]
                in_dir = _direction_vector(window)

                best_nb = None
                best_angle = 999.0
                for nb in candidates:
                    out_dir = (float(nb[0] - cur[0]), float(nb[1] - cur[1]))
                    mag = (out_dir[0]**2 + out_dir[1]**2) ** 0.5
                    if mag > 0:
                        out_dir = (out_dir[0]/mag, out_dir[1]/mag)
                    else:
                        continue
                    angle = _angle_between(in_dir, out_dir)
                    if angle < best_angle:
                        best_angle = angle
                        best_nb = nb

                if best_nb is None or best_angle > ANGLE_THRESH:
                    break  # no smooth continuation

                path.append(best_nb)
                path_set.add(best_nb)
                cur = best_nb

            # Mark all pixels on this path as globally visited (prevent loops)
            global_visited.update(path_set)

            # Path must have at least a few pixels beyond attachment
            if len(path) < 4:
                continue

            branch_path = path[1:]  # skip from_px (attachment pixel)
            tip = branch_path[-1]

            area = sum(float(dist[r, c]) * 2.0 for (r, c) in branch_path)
            avg_w = area / len(branch_path) if branch_path else 0.0
            branch_data.append((tip, branch_path, attach_pt, area, avg_w))

    tip_endpoints = [d[0] for d in branch_data]

    # Filter: adaptive outlier removal on width and area.
    WIDTH_RATIO = 0.6     # avg_w  < 60% of median → dead twig
    AREA_RATIO = 0.25     # area   < 25% of median → dead twig
    MIN_BRANCHES = 4      # need enough samples for meaningful median
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
        print(f"  [PodFilter] branches={len(branch_data)}, "
              f"med_area={median_area:.0f} fence={area_fence:.0f}, "
              f"med_w={median_width:.1f} fence={width_fence:.1f}, "
              f"keep={len(real_pods)}, reject={len(rejected)}")

    pod_count = len(real_pods)

    # 7. Debug visualization
    debug = crop_bgr.copy()
    h, w = debug.shape[:2]
    # Layer 1: Main stem in orange
    for p in main_path:
        cv2.circle(debug, (p[1], p[0]), 2, (255, 128, 0), -1)
    # Layer 2: Non-main skeleton in dark gray (background)
    for p in non_main_set:
        cv2.circle(debug, (p[1], p[0]), 1, (80, 80, 80), -1)
    # Layer 3: Rejected branch paths in purple
    for tip, bp, ap, a, w_ in rejected:
        for p in bp:
            cv2.circle(debug, (p[1], p[0]), 1, (200, 80, 200), -1)
        cv2.circle(debug, (tip[1], tip[0]), 4, (200, 80, 200), -1)
    # Layer 4: Counted branch paths in red, tips in cyan
    for i, (tip, bp, ap, a, w_) in enumerate(real_pods, 1):
        for p in bp:
            cv2.circle(debug, (p[1], p[0]), 1, (0, 0, 255), -1)
        cv2.circle(debug, (tip[1], tip[0]), 5, (255, 255, 0), -1)   # cyan tip
        cv2.circle(debug, (ap[1], ap[0]), 4, (255, 255, 255), -1)   # white attach
        cv2.putText(debug, str(i), (tip[1] + 6, tip[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    # Stem endpoints
    for p in [start_pt, end_pt]:
        cv2.circle(debug, (p[1], p[0]), 6, (0, 255, 0), 2)
    # Count text
    fs = max(0.5, min(h, w) / 500)
    cv2.putText(debug, f"Pods: {pod_count} ({stem_source})", (5, int(25 * fs) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), max(1, int(fs * 2)))
    cv2.putText(debug,
                f"tips={len(tip_endpoints)} reject={len(rejected)} "
                f"junc={len(junctions)}",
                (5, int(50 * fs) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (180, 180, 180), 1)

    # --- Step image A: filter visualization ---
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
        f"自适应过滤：绿=角果({len(real_pods)})，红=枯枝({len(rejected)})"
    ))

    # --- Step image B: final result ---
    step_images.append((debug.copy(), f"角果计数结果：{pod_count} 个（青=计入尖端，紫=被过滤，橙=主干）"))

    print(f"  [PodCounter:{stem_source}] stem={stem_len}px, tips={len(tip_endpoints)}, "
          f"reject={len(rejected)}, pods={pod_count}")

    # ── Standardized markers for downstream VLM verification ──
    # Coordinates normalized to [0, 1] relative to working image dimensions
    # to avoid mismatch when auto-upscale changes resolution.
    markers = []
    for i, (tip, bp, ap, a, w_) in enumerate(real_pods, 1):
        markers.append({"id": i, "type": "pod",
                        "x_norm": tip[1] / w, "y_norm": tip[0] / h})
    for j, (tip, bp, ap, a, w_) in enumerate(rejected, len(real_pods) + 1):
        markers.append({"id": j, "type": "filtered",
                        "x_norm": tip[1] / w, "y_norm": tip[0] / h})

    return {
        "pod_count": pod_count,
        "main_stem_length": stem_len,
        "fork_count": len(tip_endpoints),
        "skeleton_pixels": len(skel_set),
        "stem_source": stem_source,
        "debug_image": debug,
        "step_images": step_images,
        "markers": markers,
    }
