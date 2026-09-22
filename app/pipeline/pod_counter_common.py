"""
Shared skeleton and topology helpers used by the stalk counter.

This module intentionally contains no standalone counting algorithm; it only
keeps the image-processing utilities that the main stalk method depends on.
"""

import cv2
import numpy as np
from numba import njit
from skimage.morphology import skeletonize


def _binarize(crop_bgr: np.ndarray) -> np.ndarray:
    """Otsu + explicit black-background exclusion (no HSV color gating)."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect black background level from border pixels
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    border = max(2, min(crop_bgr.shape[:2]) // 30)
    border_v = np.concatenate([
        v[:border, :].reshape(-1),
        v[-border:, :].reshape(-1),
        v[:, :border].reshape(-1),
        v[:, -border:].reshape(-1),
    ])
    bg_v = float(np.percentile(border_v, 90)) if border_v.size else 0.0

    # Simple rule: anything near black-background level → background
    black_thresh = max(25.0, bg_v + 10.0)
    non_black = v > black_thresh

    # Intersection: Otsu foreground AND not-black
    mask = (otsu > 0) & non_black

    # 兜底：边框被亮物体（直尺、白色标签纸）占据时，border_v 的 90 分位数会被
    # 拉爆，black_thresh 可能超过 255，于是 non_black 恒为空、mask 全灭，调用方
    # 会直接判定为 "mask too small" 返回 0。实测一张被复核放大的主枝裁剪图
    # （4799×1700，左边框正好切到黄色直尺）：bg_v=246 → black_thresh=256 → mask=0。
    # 这种情况下退回纯 Otsu 结果。
    if np.count_nonzero(mask) < 50 and np.count_nonzero(otsu) >= 50:
        mask = otsu > 0

    mask = mask.astype(np.uint8) * 255

    # ── Dark root recovery ──
    # Roots are often dim and brown — Otsu misses them. Expand Otsu mask
    # by a few pixels and recover dark-brown pixels ONLY in that narrow band,
    # so we don't create false foreground on the black background.
    k_nbr = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    nbr = cv2.dilate(mask, k_nbr, iterations=1)  # 5 px neighbourhood
    dark_root = (
        (h >= 10) & (h <= 40) &
        (s >= 12) & (s <= 60) &
        (v >= 18) & (v <= 50)
    )
    mask = cv2.bitwise_or(mask, cv2.bitwise_and(nbr, dark_root.astype(np.uint8) * 255))

    # Morphology: close to bridge thin breaks, then bridge small gaps
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    k_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, k_bridge, iterations=1)
    mask = cv2.erode(mask, k_bridge, iterations=1)
    return mask


def _get_skeleton(mask: np.ndarray) -> np.ndarray:
    skel = skeletonize(mask > 0)
    return skel.astype(np.uint8) * 255


@njit(cache=True)
def _nb_bfs_path(skel_arr, sr, sc, er, ec):
    """BFS shortest path on a 2D bool skeleton array."""
    height, width = skel_arr.shape
    parent_r = np.full((height, width), -1, dtype=np.int32)
    parent_c = np.full((height, width), -1, dtype=np.int32)
    visited = np.zeros((height, width), dtype=np.bool_)
    visited[sr, sc] = True

    queue_r = np.empty(height * width, dtype=np.int32)
    queue_c = np.empty(height * width, dtype=np.int32)
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
                if 0 <= nr < height and 0 <= nc < width and skel_arr[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    parent_r[nr, nc] = r
                    parent_c[nr, nc] = c
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    tail += 1

    if not found:
        return np.empty((0, 2), dtype=np.int32)

    path_r = np.empty(height * width, dtype=np.int32)
    path_c = np.empty(height * width, dtype=np.int32)
    length = 0
    cr_, cc_ = er, ec
    while cr_ != -1:
        path_r[length] = cr_
        path_c[length] = cc_
        length += 1
        pr = parent_r[cr_, cc_]
        pc = parent_c[cr_, cc_]
        cr_, cc_ = pr, pc

    result = np.empty((length, 2), dtype=np.int32)
    for i in range(length):
        result[i, 0] = path_r[length - 1 - i]
        result[i, 1] = path_c[length - 1 - i]
    return result


@njit(cache=True)
def _nb_bfs_farthest(skel_arr, sr, sc):
    """BFS to find the farthest skeleton point from a start point."""
    height, width = skel_arr.shape
    visited = np.zeros((height, width), dtype=np.bool_)
    visited[sr, sc] = True

    queue_r = np.empty(height * width, dtype=np.int32)
    queue_c = np.empty(height * width, dtype=np.int32)
    queue_d = np.empty(height * width, dtype=np.int32)
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
                if 0 <= nr < height and 0 <= nc < width and skel_arr[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue_r[tail] = nr
                    queue_c[tail] = nc
                    queue_d[tail] = d + 1
                    tail += 1
    return fr, fc, max_d


@njit(cache=True)
def _nb_find_main_stem(skel_arr, endpoints, min_straightness=0.5):
    """
    Enumerate endpoint pairs and pick the longest sufficiently straight path.
    score = path_length^2 * straightness^6.
    """
    n = endpoints.shape[0]
    best_i = -1
    best_j = -1
    best_score = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            sr, sc = endpoints[i, 0], endpoints[i, 1]
            er, ec = endpoints[j, 0], endpoints[j, 1]
            path = _nb_bfs_path(skel_arr, sr, sc, er, ec)
            length = path.shape[0]
            if length < 10:
                continue
            dy = float(path[length - 1, 0] - path[0, 0])
            dx = float(path[length - 1, 1] - path[0, 1])
            distance = (dy * dy + dx * dx) ** 0.5
            straightness = distance / float(length)
            if straightness < min_straightness:
                continue
            s2 = straightness * straightness
            s8 = s2 * s2 * s2 * s2   # straightness⁸ — heavily penalize curvature
            score = float(length) * float(length) * s8
            if score > best_score:
                best_score = score
                best_i = i
                best_j = j
    return best_i, best_j, best_score


@njit(cache=True)
def _nb_neighbor_count(skel_arr, r, c):
    height, width = skel_arr.shape
    count = 0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and skel_arr[nr, nc]:
                count += 1
    return count


@njit(cache=True)
def _nb_find_endpoints(skel_arr):
    """Find all endpoint pixels, defined as skeleton pixels with one neighbor."""
    height, width = skel_arr.shape
    pts_r = np.empty(height * width, dtype=np.int32)
    pts_c = np.empty(height * width, dtype=np.int32)
    count = 0
    for r in range(height):
        for c in range(width):
            if skel_arr[r, c] and _nb_neighbor_count(skel_arr, r, c) == 1:
                pts_r[count] = r
                pts_c[count] = c
                count += 1
    result = np.empty((count, 2), dtype=np.int32)
    for i in range(count):
        result[i, 0] = pts_r[i]
        result[i, 1] = pts_c[i]
    return result


def _neighbors(p, skel_set):
    """8-connected neighbors that are in the provided skeleton set."""
    r, c = p
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr or dc) and (r + dr, c + dc) in skel_set
    ]


def _find_main_stem_numba(skel_arr, min_straightness=0.5):
    """Return the best main-stem path as (path, length, score)."""
    endpoints = _nb_find_endpoints(skel_arr)
    if endpoints.shape[0] < 2:
        return [], 0, 0.0

    best_i, best_j, best_score = _nb_find_main_stem(skel_arr, endpoints, min_straightness)
    if best_i < 0:
        return [], 0, 0.0

    path_arr = _nb_bfs_path(
        skel_arr,
        endpoints[best_i, 0],
        endpoints[best_i, 1],
        endpoints[best_j, 0],
        endpoints[best_j, 1],
    )
    path_list = [(int(path_arr[k, 0]), int(path_arr[k, 1])) for k in range(path_arr.shape[0])]
    return path_list, len(path_list), best_score


def _remove_ruler_region(
    mask: np.ndarray,
    density_thresh: float = 0.25,
    max_scan: float = 0.40,
    smooth_win: int = 15,
) -> np.ndarray:
    """Remove ruler-like strips by scanning inward from the image borders."""
    mh, mw = mask.shape

    def _smooth(arr, win):
        if len(arr) < win:
            return arr
        kernel = np.ones(win) / win
        return np.convolve(arr, kernel, mode="same")

    col_smooth = _smooth(np.count_nonzero(mask, axis=0).astype(float) / mh, smooth_win)
    row_smooth = _smooth(np.count_nonzero(mask, axis=1).astype(float) / mw, smooth_win)
    max_cols = max(2, int(mw * max_scan))
    max_rows = max(2, int(mh * max_scan))

    cut_left = 0
    for c in range(max_cols):
        if col_smooth[c] > density_thresh:
            cut_left = c + 1
        else:
            break
    if cut_left > 0:
        mask[:, :cut_left] = 0
        print(f"  [Ruler] Left strip removed: cols 0~{cut_left}")

    cut_right = mw
    for c in range(mw - 1, mw - 1 - max_cols, -1):
        if col_smooth[c] > density_thresh:
            cut_right = c
        else:
            break
    if cut_right < mw:
        mask[:, cut_right:] = 0
        print(f"  [Ruler] Right strip removed: cols {cut_right}~{mw}")

    cut_top = 0
    for r in range(max_rows):
        if row_smooth[r] > density_thresh:
            cut_top = r + 1
        else:
            break
    if cut_top > 0:
        mask[:cut_top, :] = 0
        print(f"  [Ruler] Top strip removed: rows 0~{cut_top}")

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
    """Keep the large connected component whose centroid is closest to center."""
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 2:
        return mask

    total_white = cv2.countNonZero(mask)
    min_area = max(50, int(total_white * 0.05))
    cy_center = mask.shape[0] / 2.0
    cx_center = mask.shape[1] / 2.0

    best_label = -1
    best_dist = float("inf")
    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[lbl]
        dist = (cx - cx_center) ** 2 + (cy - cy_center) ** 2
        if dist < best_dist:
            best_dist = dist
            best_label = lbl

    if best_label < 0:
        return mask

    out = np.zeros_like(mask)
    out[labels == best_label] = 255
    return out


def _build_topo_graph(skel_set, main_set):
    """Build a topology graph from non-main skeleton pixels."""
    non_main = skel_set - main_set
    degree_map = {}
    for p in non_main:
        degree_map[p] = sum(1 for nb in _neighbors(p, skel_set) if nb in non_main)

    nodes = {}
    for p in non_main:
        adj_main = any(nb in main_set for nb in _neighbors(p, skel_set))
        degree = degree_map[p]
        if adj_main:
            nodes[p] = "attach"
        elif degree == 1:
            nodes[p] = "tip"
        elif degree >= 3:
            nodes[p] = "junction"

    edges = []
    traced_pairs = set()
    for start in list(nodes):
        nbs = [nb for nb in _neighbors(start, skel_set) if nb in non_main]
        for first_step in nbs:
            if first_step in nodes:
                pair = frozenset((start, first_step))
                if pair not in traced_pairs:
                    traced_pairs.add(pair)
                    edges.append((start, first_step, [start, first_step]))
                continue

            path = [start, first_step]
            visited = {start, first_step}
            cur = first_step
            while True:
                next_pixels = [
                    nb
                    for nb in _neighbors(cur, skel_set)
                    if nb in non_main and nb not in visited
                ]
                if not next_pixels:
                    if cur not in nodes:
                        nodes[cur] = "tip"
                    break
                if len(next_pixels) == 1:
                    nxt = next_pixels[0]
                    path.append(nxt)
                    visited.add(nxt)
                    if nxt in nodes:
                        break
                    cur = nxt
                else:
                    if cur not in nodes:
                        nodes[cur] = "junction"
                    break

            end = path[-1]
            pair = frozenset((start, end))
            if pair not in traced_pairs:
                traced_pairs.add(pair)
                edges.append((start, end, path))

    return nodes, edges


def _graph_adjacency(nodes, edges):
    """Build adjacency list: node -> [(neighbor_node, edge_index), ...]."""
    adj = {n: [] for n in nodes}
    for i, (a, b, _) in enumerate(edges):
        if a in adj:
            adj[a].append((b, i))
        if b in adj:
            adj[b].append((a, i))
    return adj
