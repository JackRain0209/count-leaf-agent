"""
Branch structure analysis: identify main stem, main branch, and sub-branches.
Uses skeleton graph analysis.
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

from .segmentation import (
    find_branch_points,
    find_endpoints,
    measure_skeleton_length_px,
)


def cluster_nearby_points(points: list, radius: int = 10) -> list:
    """Merge nearby junction points into cluster centers."""
    if not points:
        return []
    points = np.array(points)
    clusters = []
    used = set()
    for i, p in enumerate(points):
        if i in used:
            continue
        group = [p]
        used.add(i)
        for j, q in enumerate(points):
            if j in used:
                continue
            if np.linalg.norm(p - q) < radius:
                group.append(q)
                used.add(j)
        center = np.mean(group, axis=0).astype(int)
        clusters.append(center.tolist())
    return clusters


def trace_branch_from_junction(skeleton: np.ndarray, start: tuple,
                                junctions_set: set, max_steps: int = 5000) -> list:
    """
    Trace a skeleton path from a junction point until hitting another junction or endpoint.
    Returns the list of (row, col) pixels along the path.
    """
    skel = skeleton.astype(np.uint8)
    visited = set()
    visited.add(tuple(start))
    path = [tuple(start)]
    current = tuple(start)

    for _ in range(max_steps):
        r, c = current
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1]:
                    if skel[nr, nc] > 0 and (nr, nc) not in visited:
                        neighbors.append((nr, nc))

        if len(neighbors) == 0:
            break  # endpoint

        # Pick the first unvisited neighbor
        next_pt = neighbors[0]
        visited.add(next_pt)
        path.append(next_pt)

        # Stop if we've reached another junction
        if next_pt in junctions_set and len(path) > 2:
            break

        current = next_pt

    return path


def analyze_branches(skeleton: np.ndarray) -> dict:
    """
    Analyze the skeleton to find:
    - main_stem (longest path from bottom)
    - main_branch (longest branch off main stem)
    - sub_branches (other branches)

    Returns dict with branch info including pixel paths and lengths.
    """
    junctions = find_branch_points(skeleton)
    endpoints = find_endpoints(skeleton)

    # Cluster nearby junction points
    junctions = cluster_nearby_points(junctions, radius=15)
    junctions_set = set(tuple(j) for j in junctions)

    # Find all paths between junctions/endpoints
    all_nodes = junctions + endpoints
    if len(all_nodes) < 2:
        # Simple plant with no branches: entire skeleton is the main stem
        length_px = measure_skeleton_length_px(skeleton)
        return {
            "main_stem_length_px": length_px,
            "main_branch_length_px": 0,
            "branch_count": 0,
            "branches": [],
            "junctions": junctions,
            "endpoints": endpoints,
        }

    # Trace branches from each junction
    branches = []
    skel_copy = skeleton.copy().astype(np.uint8)

    for junc in junctions:
        r, c = junc
        # Look at each direction from the junction
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (0 <= nr < skel_copy.shape[0] and
                    0 <= nc < skel_copy.shape[1] and
                    skel_copy[nr, nc] > 0):
                    path = trace_branch_from_junction(
                        skeleton, (nr, nc), junctions_set
                    )
                    if len(path) > 5:  # ignore very short spurs
                        # Calculate path length (with diagonal correction)
                        length = 0
                        for i in range(1, len(path)):
                            pr, pc = path[i - 1]
                            cr_, cc = path[i]
                            if abs(pr - cr_) + abs(pc - cc) == 2:
                                length += 1.414
                            else:
                                length += 1.0
                        branches.append({
                            "path": path,
                            "length_px": length,
                            "start": path[0],
                            "end": path[-1],
                        })

    # Deduplicate branches (same path traced from both ends)
    unique_branches = []
    seen_pairs = set()
    for b in branches:
        key = (min(b["start"], b["end"]), max(b["start"], b["end"]))
        if key not in seen_pairs:
            seen_pairs.add(key)
            unique_branches.append(b)

    # Sort by length descending
    unique_branches.sort(key=lambda b: b["length_px"], reverse=True)

    # Identification rules:
    #   main_stem  = longest path (the thickest, longest stalk)
    #   main_branch = second longest (adjacent to main stem)
    #   sub_branches = everything else
    main_stem_length = unique_branches[0]["length_px"] if unique_branches else 0
    main_branch_length = unique_branches[1]["length_px"] if len(unique_branches) > 1 else 0
    sub_branches = unique_branches[2:] if len(unique_branches) > 2 else []

    return {
        "main_stem_length_px": main_stem_length,
        "main_branch_length_px": main_branch_length,
        "branch_count": len(sub_branches),  # only sub-branches, exclude main stem & main branch
        "sub_branches": sub_branches,
        "all_branches": unique_branches,
        "junctions": junctions,
        "endpoints": endpoints,
    }
