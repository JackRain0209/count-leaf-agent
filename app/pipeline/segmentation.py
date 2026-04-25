"""
Plant segmentation & CV utilities.

Pipeline:
  1. crop_black_region() — detect black cloth, crop to it
  2. segment_and_find_components() — green mask → connected components → numbered image
  3. (VLM classifies numbered components — see vlm_counter.py)
  4. clean_part_crop() — binarize a part's mask region, keep largest CC, denoise
  5. count_siliques_by_skeleton_walk() — walk skeleton, count side branches
"""

import cv2
import numpy as np
from collections import deque
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label, regionprops

# Distinct colors for numbering parts
_LABEL_COLORS = [
    (0, 0, 255),    (0, 165, 255),  (0, 255, 255),  (0, 255, 0),
    (255, 255, 0),  (255, 0, 0),    (255, 0, 255),   (128, 0, 255),
    (128, 255, 128),(255, 128, 0),
]


# ═══════════════════════════════════════════════════════════════════════
# Step 1: Crop the black cloth region
# ═══════════════════════════════════════════════════════════════════════

def crop_black_region(image: np.ndarray, dark_thresh: int = 50,
                      min_area_ratio: float = 0.15) -> tuple:
    """
    Detect the black cloth area and crop the image to it.
    Plants are placed on black cloth; edges may have table/floor.

    Returns: (cropped_image, (x, y, w, h) of crop in original coords)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Threshold: dark pixels = cloth
    _, dark_mask = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)

    # Close small gaps in the dark region
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, k)

    # Find contours of dark regions
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("[Crop] No dark region found, using full image")
        return image, (0, 0, w, h)

    # Pick the largest dark contour
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Must cover a significant portion of the image
    if area < h * w * min_area_ratio:
        print(f"[Crop] Dark region too small ({area}/{h*w}), using full image")
        return image, (0, 0, w, h)

    x, y, rw, rh = cv2.boundingRect(largest)
    # Small padding inward to exclude cloth edges
    pad = 10
    x, y = x + pad, y + pad
    rw, rh = rw - 2 * pad, rh - 2 * pad
    x, y = max(0, x), max(0, y)
    rw = min(rw, w - x)
    rh = min(rh, h - y)

    cropped = image[y:y+rh, x:x+rw]
    print(f"[Crop] Black region: ({x},{y}) {rw}x{rh} from {w}x{h}")
    return cropped, (x, y, rw, rh)


# ═══════════════════════════════════════════════════════════════════════
# Step 2: Segment green plant parts → find components → number them
# ═══════════════════════════════════════════════════════════════════════

def make_ruler_mask(image: np.ndarray) -> np.ndarray:
    """
    Detect the ruler (yellow elongated strip) and return a mask.
    Uses shape filtering: ruler is yellow + very elongated (aspect ratio > 5).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Yellow pixels
    yellow = cv2.inRange(hsv, np.array([15, 80, 120]), np.array([35, 255, 255]))

    # Find contours of yellow regions
    contours, _ = cv2.findContours(yellow, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    ruler_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / (min(w, h) + 1)
        area = cv2.contourArea(cnt)
        # Ruler: elongated (aspect > 5) and not too small
        if aspect > 5 and area > 2000:
            # Dilate the contour region a bit
            cv2.drawContours(ruler_mask, [cnt], -1, 255, -1)

    if np.sum(ruler_mask > 0) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        ruler_mask = cv2.dilate(ruler_mask, k, iterations=1)
        print(f"[Ruler mask] {np.sum(ruler_mask > 0)} pixels masked")
    else:
        print("[Ruler mask] No ruler detected")
    return ruler_mask


def segment_plant(image: np.ndarray, ruler_mask: np.ndarray = None,
                  min_area: int = 500) -> np.ndarray:
    """
    Segment green plant from black background using HSV.
    Broad hue range (20-95) to capture green, yellow-green, and brown-green.
    Excludes ruler via ruler_mask.
    Returns binary mask (uint8, 0/255).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green range: hue 20-95 covers yellow-green to teal
    mask = cv2.inRange(hsv, np.array([20, 20, 30]), np.array([95, 255, 255]))

    # Exclude ruler
    if ruler_mask is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(ruler_mask))

    # Light cleanup — preserve thin stems
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # Remove small noise
    bool_mask = mask.astype(bool)
    bool_mask = remove_small_objects(bool_mask, min_size=min_area)
    result = (bool_mask.astype(np.uint8)) * 255
    print(f"[Segment] Plant pixels: {np.sum(result > 0):,}")
    return result


def find_stems(mask: np.ndarray, min_stem_area: int = 500) -> np.ndarray:
    """
    Erode the binary mask aggressively to keep only thick stems.
    Thin features (siliques, leaves) disappear — only root/backbone remains.
    This naturally avoids merging because stems don't touch each other.
    Returns: eroded binary mask of stems only.
    """
    h, w = mask.shape[:2]
    # Erosion kernel size proportional to image size
    # For a ~5000px image, use ~15px kernel
    ks = max(5, min(h, w) // 350)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    eroded = cv2.erode(mask, k, iterations=2)

    # Re-dilate slightly to restore stem width for visibility
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks//2+1, ks//2+1))
    stems = cv2.dilate(eroded, k_small, iterations=1)

    # Remove small fragments
    bool_mask = stems.astype(bool)
    bool_mask = remove_small_objects(bool_mask, min_size=min_stem_area)
    stems = (bool_mask.astype(np.uint8)) * 255

    print(f"[Stems] kernel={ks}px, stem pixels: {np.sum(stems > 0):,}")
    return stems


def label_plants_by_stems(full_mask: np.ndarray, stem_mask: np.ndarray,
                          min_stem_area: int = 300) -> list:
    """
    Use stems as anchors to split the full plant mask into individual plants.

    1. Find connected components in stem_mask → each = one plant unit
    2. Use each stem as a seed marker
    3. Grow (watershed-like) on full_mask to assign all plant pixels to nearest stem
    4. Return list of complete plant parts (full pixels, not just stems)
    """
    from scipy.ndimage import label as nd_label, distance_transform_edt

    # Find stem components
    stem_labeled, n_stems = nd_label(stem_mask > 0)
    if n_stems == 0:
        return []

    # Filter small stem fragments
    stem_regions = regionprops(stem_labeled)
    valid_stems = [r for r in stem_regions if r.area >= min_stem_area]
    if not valid_stems:
        return []

    print(f"[LabelByStems] {len(valid_stems)} stem anchors found")

    # Create marker image: each stem gets a unique label (1, 2, 3...)
    markers = np.zeros(full_mask.shape[:2], dtype=np.int32)
    for i, r in enumerate(valid_stems):
        markers[stem_labeled == r.label] = i + 1

    # Grow markers to fill full_mask using distance-based nearest assignment
    # For each unlabeled plant pixel, assign to nearest stem
    plant_pixels = full_mask > 0
    labeled_pixels = markers > 0

    # Pixels that need assignment: plant pixels not yet labeled
    unlabeled_plant = plant_pixels & ~labeled_pixels

    if np.any(unlabeled_plant):
        # For each stem, compute distance transform
        # Then assign each unlabeled pixel to the closest stem
        n = len(valid_stems)
        dist_maps = np.full((n,) + full_mask.shape[:2], np.inf, dtype=np.float32)

        for i in range(n):
            stem_i = (markers == (i + 1)).astype(np.uint8)
            # Distance from each pixel to this stem
            dist_maps[i] = distance_transform_edt(stem_i == 0)

        # For unlabeled plant pixels, assign to nearest stem
        nearest = np.argmin(dist_maps, axis=0) + 1  # 1-indexed
        markers[unlabeled_plant] = nearest[unlabeled_plant]

    # Build component list from assigned labels
    components = []
    for i, r in enumerate(valid_stems):
        lbl = i + 1
        comp_mask = np.zeros_like(full_mask)
        comp_mask[(markers == lbl) & plant_pixels] = 255

        if np.sum(comp_mask > 0) < 100:
            continue

        # Compute properties
        comp_regions = regionprops(label(comp_mask > 0))
        if not comp_regions:
            continue
        cr = comp_regions[0]

        # Also store the stem mask for this component
        stem_only = np.zeros_like(full_mask)
        stem_only[stem_labeled == r.label] = 255

        components.append({
            "label": lbl,
            "area": int(np.sum(comp_mask > 0)),
            "stem_area": r.area,
            "bbox": cr.bbox,
            "mask": comp_mask,
            "stem_mask": stem_only,
            "centroid": (int(cr.centroid[0]), int(cr.centroid[1])),
        })

    components.sort(key=lambda c: c["area"], reverse=True)
    for i, c in enumerate(components):
        print(f"  Plant #{i+1}: total={c['area']:,}px, stem={c['stem_area']:,}px")
    return components


def find_components(mask: np.ndarray, min_area: int = 200) -> list:
    """
    Find all connected components from binary mask.
    Returns list of dicts sorted by area descending.
    """
    labeled = label(mask > 0)
    if labeled.max() == 0:
        return []

    regions = regionprops(labeled)
    components = []
    for r in regions:
        if r.area < min_area:
            continue
        comp_mask = np.zeros_like(mask)
        comp_mask[labeled == r.label] = 255
        components.append({
            "label": r.label,
            "area": r.area,
            "bbox": r.bbox,  # (min_row, min_col, max_row, max_col)
            "mask": comp_mask,
            "centroid": (int(r.centroid[0]), int(r.centroid[1])),
        })
    components.sort(key=lambda c: c["area"], reverse=True)
    return components


def separate_branches_and_siliques(components: list) -> tuple:
    """
    Separate large parts (stem/branches) from small parts (siliques)
    using the biggest ratio gap in sorted areas.
    Returns: (branch_parts, silique_parts)
    """
    if len(components) <= 1:
        return components, []

    areas = [c["area"] for c in components]
    best_gap_idx, best_ratio = 0, 1.0
    for i in range(len(areas) - 1):
        ratio = areas[i] / areas[i + 1]
        if ratio > best_ratio:
            best_ratio = ratio
            best_gap_idx = i

    if best_ratio >= 3.0:
        branches = components[:best_gap_idx + 1]
        siliques = components[best_gap_idx + 1:]
    else:
        branches = components
        siliques = []

    print(f"[Separate] {len(branches)} branch-level, "
          f"{len(siliques)} silique-level, gap={best_ratio:.1f}x")
    return branches, siliques


def draw_numbered_parts(image: np.ndarray, parts: list) -> np.ndarray:
    """
    Draw numbered bounding boxes + colored overlay on the original color image.
    Returns annotated image for debug viewing.
    """
    vis = image.copy()
    overlay = image.copy()
    for i, part in enumerate(parts):
        color = _LABEL_COLORS[i % len(_LABEL_COLORS)]
        overlay[part["mask"] > 0] = color
        r0, c0, r1, c1 = part["bbox"]
        cv2.rectangle(vis, (c0, r0), (c1, r1), color, 3)
        lbl = f"#{i + 1}"
        font_scale, thickness = 1.5, 4
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        cv2.rectangle(vis, (c0, r0 - th - 10), (c0 + tw + 10, r0), color, -1)
        cv2.putText(vis, lbl, (c0 + 5, r0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
    return vis


def draw_numbered_bw(mask: np.ndarray, parts: list) -> np.ndarray:
    """
    Draw numbered labels on the B&W binary image.
    Each component gets a different gray level + a number label.
    This is the image sent to VLM for classification.
    Returns: BGR image (black bg, white plants, colored numbers).
    """
    h, w = mask.shape[:2]
    # Start with black bg, draw all plant pixels in white
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for i, part in enumerate(parts):
        color = _LABEL_COLORS[i % len(_LABEL_COLORS)]
        # Tint each component with a unique color so VLM can distinguish
        vis[part["mask"] > 0] = color

        r0, c0, r1, c1 = part["bbox"]
        # Draw bbox
        cv2.rectangle(vis, (c0, r0), (c1, r1), color, 3)

        # Number label — scale based on image size
        font_scale = max(1.0, min(h, w) / 1500)
        thickness = max(2, int(font_scale * 2.5))
        lbl = f"#{i + 1}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        # Place label above bbox
        lx = c0 + 5
        ly = max(r0 - 10, th + 10)
        cv2.rectangle(vis, (lx - 2, ly - th - 6), (lx + tw + 6, ly + 6),
                      (0, 0, 0), -1)
        cv2.putText(vis, lbl, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return vis


# ═══════════════════════════════════════════════════════════════════════
# Step 3: Clean a cropped part — binarize, keep main body, denoise
# ═══════════════════════════════════════════════════════════════════════

def clean_part_crop(crop: np.ndarray, min_area: int = 300) -> np.ndarray:
    """
    Given a VLM-cropped region of a single plant part:
      1. Convert to grayscale, threshold to separate from black background
      2. Keep only the largest connected component (the plant part itself)
      3. Morphological denoise

    Returns: binary mask (uint8, 0/255) of the cleaned plant part.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Broad range: any non-black pixel with some saturation or brightness
    # This captures green, brown, dried — everything on black background
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    # Plant pixels: either have color (S > 30) or brightness (V > 60) on dark bg
    mask = ((v_channel > 50) & (s_channel > 20)).astype(np.uint8) * 255

    # Morphological cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # Keep only the largest connected component
    labeled = label(mask > 0)
    if labeled.max() == 0:
        return mask

    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    clean = np.zeros_like(mask)
    clean[labeled == largest.label] = 255

    # Final denoise
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, k, iterations=1)

    return clean


# ═══════════════════════════════════════════════════════════════════════
# Step 4 & 5: Skeleton walk — count siliques as side branches
# ═══════════════════════════════════════════════════════════════════════

def extract_skeleton(mask: np.ndarray) -> np.ndarray:
    """Skeletonize binary mask. Returns bool array."""
    return skeletonize(mask > 0)


def _build_skeleton_graph(skeleton: np.ndarray) -> dict:
    """
    Build adjacency list from skeleton pixels.
    Returns: {(r, c): [(nr, nc), ...], ...}
    """
    skel = skeleton.astype(np.uint8)
    coords = np.argwhere(skel > 0)
    coord_set = set(map(tuple, coords))

    graph = {tuple(c): [] for c in coords}
    for r, c in coord_set:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in coord_set:
                    graph[(r, c)].append(nb)
    return graph


def _find_longest_path(graph: dict) -> list:
    """
    Find the longest path in the skeleton graph (the main axis).
    Uses BFS from each endpoint, picks the pair with max distance.
    Returns list of (r, c) along the main axis.
    """
    if not graph:
        return []

    # Find endpoints (degree 1) and junctions (degree >= 3)
    endpoints = [n for n, nbs in graph.items() if len(nbs) == 1]

    if len(endpoints) < 2:
        # No clear endpoints — pick arbitrary start
        endpoints = list(graph.keys())[:2]
    if len(endpoints) < 2:
        return list(graph.keys())

    # BFS from first endpoint to find farthest point
    def bfs_farthest(start):
        visited = {start}
        queue = deque([(start, [start])])
        farthest = start
        longest_path = [start]
        while queue:
            node, path = queue.popleft()
            if len(path) > len(longest_path):
                longest_path = path
                farthest = node
            for nb in graph[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return farthest, longest_path

    # Two-pass BFS to find diameter (longest path)
    far1, _ = bfs_farthest(endpoints[0])
    far2, main_path = bfs_farthest(far1)

    return main_path


def count_siliques_by_skeleton_walk(mask: np.ndarray) -> dict:
    """
    Count siliques by walking the skeleton:
      1. Skeletonize the cleaned binary mask
      2. Find the main axis (longest path)
      3. Walk along main axis, at each junction count side branches
      4. Each side branch = 1 silique

    Returns:
      {
        "silique_count": int,
        "main_axis_length_px": float,
        "branch_points": [(r, c), ...],
        "skeleton_pixels": int,
      }
    """
    skeleton = extract_skeleton(mask)
    skel_pixels = int(np.sum(skeleton))

    if skel_pixels < 10:
        return {"silique_count": 0, "main_axis_length_px": 0,
                "branch_points": [], "skeleton_pixels": skel_pixels}

    graph = _build_skeleton_graph(skeleton)
    main_path = _find_longest_path(graph)
    main_set = set(main_path)

    # Find branch points: main-path pixels with neighbors NOT on main path
    branch_points = []
    for node in main_path:
        side_branches = [nb for nb in graph[node] if nb not in main_set]
        if side_branches:
            branch_points.append(node)

    # Cluster nearby branch points (within 5px) — same junction
    clustered = []
    used = set()
    for bp in branch_points:
        if bp in used:
            continue
        cluster = [bp]
        used.add(bp)
        for other in branch_points:
            if other not in used:
                dist = abs(bp[0] - other[0]) + abs(bp[1] - other[1])
                if dist <= 5:
                    cluster.append(other)
                    used.add(other)
        # Use centroid of cluster
        cr = int(np.mean([p[0] for p in cluster]))
        cc = int(np.mean([p[1] for p in cluster]))
        clustered.append((cr, cc))

    # Measure main axis length (approximate by path pixel count)
    main_length = 0.0
    for i in range(1, len(main_path)):
        r0, c0 = main_path[i - 1]
        r1, c1 = main_path[i]
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        main_length += 1.414 if (dr + dc == 2) else 1.0

    return {
        "silique_count": len(clustered),
        "main_axis_length_px": round(main_length, 1),
        "branch_points": clustered,
        "skeleton_pixels": skel_pixels,
    }


def measure_skeleton_length_px(skeleton: np.ndarray) -> float:
    """Measure total skeleton path length in pixels."""
    skel = skeleton.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(skel, -1, kernel)
    diag_kernel = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)
    diag_count = cv2.filter2D(skel, -1, diag_kernel)
    diag_connections = np.sum(diag_count[skel > 0])
    ortho_connections = np.sum(neighbor_count[skel > 0]) - diag_connections
    return (ortho_connections / 2) * 1.0 + (diag_connections / 2) * 1.414
