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
    """Gaussian blur + Otsu + CLOSE (no OPEN, to preserve thin stems)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
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

    # 6. Find ALL side-branch segments along the main stem
    #    At junctions: continue in the most-aligned direction (no sharp U-turns),
    #    instead of stopping.  Junction pixels are NOT globally locked, so
    #    multiple branches can pass through the same crossing from different angles.
    branch_comps = []  # list of (comp_pixels, attach_point)
    visited_branches = set()   # non-junction pixels claimed by a branch

    for idx, p in enumerate(main_path):
        for nb in _neighbors(p, skel_set):
            if nb not in main_set and nb not in visited_branches:
                comp = []
                comp_set = set()
                # Queue entries: (pixel, came_from) — came_from used for heading
                queue = deque([(nb, p)])
                while queue:
                    sp, came_from = queue.popleft()
                    # Non-junction pixels are exclusive; junction pixels can be shared
                    if sp not in junctions and sp in visited_branches:
                        continue
                    if sp in junctions:
                        if sp in comp_set:
                            continue
                    else:
                        visited_branches.add(sp)
                    comp.append(sp)
                    comp_set.add(sp)

                    if sp in junctions:
                        # Direction-guided: pick the neighbor most aligned with heading
                        dr = sp[0] - came_from[0]
                        dc = sp[1] - came_from[1]
                        h_len = (dr * dr + dc * dc) ** 0.5
                        if h_len > 0:
                            best_nb = None
                            best_cos = -2.0
                            for n2 in _neighbors(sp, skel_set):
                                if n2 in main_set:
                                    continue
                                if n2 not in junctions and n2 in visited_branches:
                                    continue
                                if n2 in comp_set:
                                    continue
                                n_dr = n2[0] - sp[0]
                                n_dc = n2[1] - sp[1]
                                n_len = (n_dr * n_dr + n_dc * n_dc) ** 0.5
                                if n_len == 0:
                                    continue
                                cos_val = (dr * n_dr + dc * n_dc) / (h_len * n_len)
                                if cos_val > best_cos:
                                    best_cos = cos_val
                                    best_nb = n2
                            # Only continue if angle < 90° (cos > 0)
                            if best_nb is not None and best_cos > 0:
                                queue.append((best_nb, sp))
                    else:
                        # Normal pixel: expand all unvisited non-main neighbors
                        for n2 in _neighbors(sp, skel_set):
                            if n2 not in main_set and n2 not in visited_branches:
                                queue.append((n2, sp))
                if comp:
                    branch_comps.append((comp, p))

    # 6. Filter stage A: remove tiny noise (size filter)
    #    Keep branches that touch a junction (truncated by crossing → short but real)
    MIN_SIDE_PIXELS = 3
    threshold = MIN_SIDE_PIXELS
    size_passed = []  # [(comp, attach_pt), ...]
    size_rejected = []
    for comp, attach_pt in branch_comps:
        touches_junction = any(p in junctions for p in comp)
        if touches_junction or len(comp) >= threshold:
            size_passed.append((comp, attach_pt))
        else:
            size_rejected.append((comp, attach_pt))

    # 6b. Filter stage B: width expansion (pedicel → silique morphology)
    #     Real pod: narrow near the main stem, then clearly widens along path or at tip.
    #     Dead twig/leaf residue: uniform thin width all the way.
    EXPAND_RATIO = 1.6      # peak must be ≥ 1.6× base
    EXPAND_ABS_PX = 2.0     # OR peak − base ≥ 2 px (for thin scales)
    TAIL_SEARCH_RADIUS = 6  # px neighborhood around branch tail to capture
                            # pod body that was cut off at junction

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)  # half-width at each pixel
    H_img, W_img = dist.shape

    def _branch_widths(comp, attach_pt):
        """Order comp via BFS from entry pixel (neighbor of attach in comp),
        return (widths_along_path, tail_pixel) — widths are full widths (2*dist)."""
        comp_set = set(comp)
        # Entry pixel = comp member 8-adjacent to attach_pt (pick nearest)
        entry = None
        ar, ac = attach_pt
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: continue
                if (ar + dr, ac + dc) in comp_set:
                    entry = (ar + dr, ac + dc)
                    break
            if entry is not None: break
        if entry is None:
            # Fallback: pick any member
            entry = next(iter(comp_set))
        # BFS, tracking depth; tail = deepest pixel
        depth = {entry: 0}
        queue = deque([entry])
        ordered = []
        tail = entry
        max_d = 0
        while queue:
            p = queue.popleft()
            ordered.append(p)
            if depth[p] > max_d:
                max_d = depth[p]
                tail = p
            for nb in _neighbors(p, skel_set):
                if nb in comp_set and nb not in depth:
                    depth[nb] = depth[p] + 1
                    queue.append(nb)
        widths = [float(dist[r, c]) * 2.0 for (r, c) in ordered]
        return widths, tail, ordered

    def _tail_peak_width(tail):
        """Scan mask in a small neighborhood around tail to catch pod body that was
        truncated at a crossing junction."""
        tr, tc = tail
        r0, r1 = max(0, tr - TAIL_SEARCH_RADIUS), min(H_img, tr + TAIL_SEARCH_RADIUS + 1)
        c0, c1 = max(0, tc - TAIL_SEARCH_RADIUS), min(W_img, tc + TAIL_SEARCH_RADIUS + 1)
        patch = dist[r0:r1, c0:c1]
        return float(patch.max()) * 2.0 if patch.size else 0.0

    real_pods = []      # branches passing width filter → counted
    width_rejected = [] # passed size but failed width → dead twig
    branch_stats = []   # (attach_pt, w_base, w_peak, is_real, tail) for debug

    for comp, attach_pt in size_passed:
        touches_junction = any(p in junctions for p in comp)

        if touches_junction:
            # Overlapping / incomplete branch — skip width filter, count directly
            widths, tail, ordered = _branch_widths(comp, attach_pt)
            w_base = float(np.median(widths[:max(3, len(widths))])) if widths else 0.0
            w_peak = max(widths) if widths else 0.0
            real_pods.append((comp, attach_pt))
            branch_stats.append((attach_pt, w_base, w_peak, True, tail))
            continue

        # Complete (non-overlapping) branch — apply width expansion filter
        widths, tail, ordered = _branch_widths(comp, attach_pt)
        if not widths:
            width_rejected.append((comp, attach_pt))
            branch_stats.append((attach_pt, 0.0, 0.0, False, tail))
            continue
        # Base = median of first 20% (or first min(3, N))
        n_base = max(3, int(len(widths) * 0.2))
        w_base = float(np.median(widths[:min(n_base, len(widths))]))
        # Peak = max of (90th pct along path, tail neighborhood peak on mask)
        w_peak_path = float(np.quantile(widths, 0.9))
        w_peak_tail = _tail_peak_width(tail)
        w_peak = max(w_peak_path, w_peak_tail)
        # Decision
        is_real = (w_peak >= w_base * EXPAND_RATIO) or (w_peak - w_base >= EXPAND_ABS_PX)
        if is_real:
            real_pods.append((comp, attach_pt))
        else:
            width_rejected.append((comp, attach_pt))
        branch_stats.append((attach_pt, w_base, w_peak, is_real, tail))

    pod_count = len(real_pods)

    # 7. Debug visualization
    debug = crop_bgr.copy()
    # Layer 1: Main stem in orange (bottom layer)
    for p in main_path:
        cv2.circle(debug, (p[1], p[0]), 2, (255, 128, 0), -1)
    # Layer 2: Size-rejected branches (dark gray dots)
    for comp, _ in size_rejected:
        for p in comp:
            cv2.circle(debug, (p[1], p[0]), 1, (80, 80, 80), -1)
    # Layer 3: Width-rejected branches (枯枝) in purple
    for comp, _ in width_rejected:
        for p in comp:
            cv2.circle(debug, (p[1], p[0]), 1, (200, 80, 200), -1)
    # Layer 4: Counted pods in bright red
    for comp, _ in real_pods:
        for p in comp:
            cv2.circle(debug, (p[1], p[0]), 1, (0, 0, 255), -1)
    # Layer 5: Labels for counted pods
    for i, (comp, attach_pt) in enumerate(real_pods, 1):
        cv2.circle(debug, (attach_pt[1], attach_pt[0]), 5, (255, 255, 255), -1)
        cv2.putText(debug, str(i), (attach_pt[1] + 7, attach_pt[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    # Stem endpoints in green circles
    for p in [start_pt, end_pt]:
        cv2.circle(debug, (p[1], p[0]), 6, (0, 255, 0), 2)
    # Count text
    fs = max(0.5, min(h, w) / 500)
    cv2.putText(debug, f"Pods: {pod_count} ({stem_source})", (5, int(25 * fs) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), max(1, int(fs * 2)))
    cv2.putText(debug,
                f"size_pass={len(size_passed)} width_kill={len(width_rejected)} "
                f"junc={len(junctions)}",
                (5, int(50 * fs) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (180, 180, 180), 1)

    # --- Step image A: width-filter visualization with per-branch stats ---
    width_vis = crop_bgr.copy()
    for p in main_path:
        cv2.circle(width_vis, (p[1], p[0]), 2, (255, 128, 0), -1)
    for (attach_pt, wb, wp, is_real, tail) in branch_stats:
        color = (0, 0, 255) if is_real else (200, 80, 200)
        cv2.circle(width_vis, (attach_pt[1], attach_pt[0]), 3, color, -1)
        cv2.circle(width_vis, (tail[1], tail[0]), 3, (0, 255, 255), 1)
        txt = f"{wb:.1f}->{wp:.1f}"
        cv2.putText(width_vis, txt, (attach_pt[1] + 4, attach_pt[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    step_images.append((
        width_vis,
        f"宽度过滤：沿分支测量宽度（base→peak px），红=角果({len(real_pods)})，"
        f"紫=枯枝被剔除({len(width_rejected)})，阈值 ratio≥{EXPAND_RATIO} 或 Δ≥{EXPAND_ABS_PX}px"
    ))

    # --- Step image B: final result ---
    step_images.append((debug.copy(), f"角果计数结果：{pod_count} 个（红=计入，紫=枯枝，灰=过短，橙=主干）"))

    print(f"  [PodCounter:{stem_source}] stem={stem_len}px, branches={len(branch_comps)}, "
          f"size_pass={len(size_passed)}, width_kill={len(width_rejected)}, "
          f"pods={pod_count}")

    return {
        "pod_count": pod_count,
        "main_stem_length": stem_len,
        "fork_count": len(branch_comps),
        "skeleton_pixels": len(skel_set),
        "stem_source": stem_source,
        "debug_image": debug,
        "step_images": step_images,
    }
