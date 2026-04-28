#!/usr/bin/env python3
"""
Benchmark: compare pod-counting algorithm against ground-truth Excel data.

Two-phase design:
  Phase A (VLM) — call Doubao Vision to identify + crop plant parts; cache results.
  Phase B (CV)  — run pod_counter on cached crops; compare with Excel ground truth.

Usage:
  # Full run (VLM + CV):
  python scripts/benchmark.py

  # CV-only (reuse cached VLM crops, skip API calls):
  python scripts/benchmark.py --cv-only

  # Limit to first N images (for quick testing):
  python scripts/benchmark.py --limit 5
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openpyxl
import glob as _glob

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pipeline.vlm_counter import vlm_label_plants
from app.pipeline.pod_counter import count_pods_on_branch as _skeleton_counter
from app.pipeline.pod_counter_graph import count_pods_on_branch as _graph_counter
from app.pipeline.pod_counter_plantcv import count_pods_on_branch as _plantcv_counter

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "online_uploads" / "农生院卢坤-合川考种照片-整理20260414"
PHOTO_ROOT = DATA_ROOT / "合川考种照片整理-20260414"
EXCEL_PATH = DATA_ROOT / "合川考种数据-20260414终版.xlsx"
CACHE_DIR = PROJECT_ROOT / "benchmark_cache"
RESULT_CSV = PROJECT_ROOT / "benchmark_results.csv"
SKIPPED_CSV = PROJECT_ROOT / "benchmark_skipped.csv"
SUMMARY_TXT = PROJECT_ROOT / "benchmark_summary.txt"


def _safe_int(val):
    """Convert a value to int, handling strings like '=53+31+32' or None."""
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip()
    if not s:
        return 0
    # Try eval for simple arithmetic expressions like "=53+31+32+60"
    s = s.lstrip("=")
    try:
        return int(eval(s))
    except Exception:
        return 0


def load_ground_truth():
    """Read Excel → list of dicts keyed by photo filename (without ext)."""
    wb = openpyxl.load_workbook(str(EXCEL_PATH), read_only=True)
    ws = wb["Sheet1"]
    gt = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        conc, rep, variety, photo_num = row[0], row[1], row[2], row[3]
        fname = f"{conc}.{rep}.{variety}.{photo_num}"
        remark = str(row[8]).strip() if row[8] is not None else ""
        gt[fname] = {
            "氮肥浓度": conc,
            "重复": rep,
            "品种编号": variety,
            "照片编号": photo_num,
            "分枝数": row[4],
            "主花序角果数": row[5],   # F
            "分枝角果数": row[6],     # G
            "单株总角果数": row[7],   # H
            "备注": remark,           # I — non-empty means bad data
        }
    wb.close()
    return gt


def find_photo(fname: str):
    """Find the actual photo file matching a base name like 'N0.1.1.1'."""
    for ext in (".JPG", ".jpg", ".jpeg", ".png", ".PNG"):
        # Search all subdirectories
        for root, _, files in os.walk(PHOTO_ROOT):
            if fname + ext in files:
                return Path(root) / (fname + ext)
    return None


def run_vlm_phase(fname: str, photo_path: Path, cache_dir: Path):
    """Run VLM on a photo, save crops + metadata to cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "vlm_result.json"

    if meta_path.exists():
        print(f"  [VLM] cache hit, skip")
        with open(meta_path) as f:
            return json.load(f)

    image = cv2.imread(str(photo_path))
    if image is None:
        print(f"  [VLM] ERROR: cannot read {photo_path}")
        return None

    # Retry with backoff on rate limit (429) errors
    vlm_result = None
    for attempt in range(5):
        try:
            vlm_result = vlm_label_plants(image)
            break
        except Exception as e:
            if "429" in str(e) or "TooManyRequests" in str(e) or "SetLimitExceeded" in str(e):
                wait = 30 * (2 ** attempt)  # 30s, 60s, 120s, 240s, 480s
                print(f"  [VLM] Rate limited, waiting {wait}s before retry ({attempt+1}/5)...")
                time.sleep(wait)
            else:
                print(f"  [VLM] ERROR: {e}")
                return None
    if vlm_result is None:
        print(f"  [VLM] Failed after 5 retries")
        return None
    plants = vlm_result.get("plants", [])
    crops = vlm_result.get("crops", {})

    # Save crops as images
    crop_files = {}
    for pid, crop_img in crops.items():
        # Find label
        label = "unknown"
        for p in plants:
            if p.get("id") == pid:
                label = p.get("label", "unknown")
                break
        crop_fname = f"crop_{pid}_{label}.jpg"
        cv2.imwrite(str(cache_dir / crop_fname), crop_img)
        crop_files[str(pid)] = crop_fname

    # Save metadata (without numpy arrays)
    meta = {
        "plants": plants,
        "crop_files": crop_files,
        "raw_response": vlm_result.get("raw_response", ""),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def _clean_debug_images(cache_dir: Path):
    """Remove old debug_*.jpg files from cache dir, keep crop_* and vlm_result.json."""
    for f in cache_dir.glob("debug_*.jpg"):
        f.unlink()


def run_cv_phase(cache_dir: Path, counter_fn=None):
    """Run pod_counter on cached crops. Returns per-plant pod counts."""
    if counter_fn is None:
        counter_fn = _graph_counter
    meta_path = cache_dir / "vlm_result.json"
    if not meta_path.exists():
        return None

    _clean_debug_images(cache_dir)

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    plants = meta.get("plants", [])
    crop_files = meta.get("crop_files", {})

    results = []  # list of (pid, label, pod_count)
    for p in plants:
        pid = p.get("id")
        label = p.get("label", "")
        if label == "主干":
            results.append((pid, label, 0))
            continue
        crop_fname = crop_files.get(str(pid))
        if not crop_fname:
            results.append((pid, label, 0))
            continue
        crop_path = cache_dir / crop_fname
        if not crop_path.exists():
            results.append((pid, label, 0))
            continue

        crop_img = cv2.imread(str(crop_path))
        if crop_img is None:
            results.append((pid, label, 0))
            continue

        pod_result = counter_fn(
            crop_img,
            stem_start_local=p.get("stem_start_local"),
            stem_end_local=p.get("stem_end_local"),
        )
        pod_count = pod_result.get("pod_count", 0)
        results.append((pid, label, pod_count))

        # Save debug image
        debug_img = pod_result.get("debug_image")
        if debug_img is not None:
            cv2.imwrite(str(cache_dir / f"debug_{pid}_{label}.jpg"), debug_img)

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark pod counting")
    parser.add_argument("--cv-only", action="store_true",
                        help="Skip VLM, reuse cached crops")
    parser.add_argument("--method", choices=["skeleton", "graph", "plantcv"], default="graph",
                        help="CV algorithm: skeleton, graph, or plantcv (default: graph)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N images (0 = all)")
    args = parser.parse_args()

    if args.method == "skeleton":
        counter_fn = _skeleton_counter
    elif args.method == "plantcv":
        counter_fn = _plantcv_counter
    else:
        counter_fn = _graph_counter

    print("=" * 60)
    print(f"  Pod Counting Benchmark (method={args.method})")
    print("=" * 60)

    # Load ground truth
    gt = load_ground_truth()
    print(f"Ground truth: {len(gt)} rows from Excel")

    # Build photo list (ordered by Excel row order)
    wb = openpyxl.load_workbook(str(EXCEL_PATH), read_only=True)
    ws = wb["Sheet1"]
    ordered_names = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is None:
            continue
        fname = f"{row[0]}.{row[1]}.{row[2]}.{row[3]}"
        ordered_names.append(fname)
    wb.close()

    if args.limit > 0:
        ordered_names = ordered_names[:args.limit]
    print(f"Processing: {len(ordered_names)} images")

    # CSV output
    csv_rows = []
    skipped_rows = []  # samples skipped due to 备注 or missing photo
    total_processed = 0
    total_skipped = 0

    for idx, fname in enumerate(ordered_names):
        print(f"\n[{idx+1}/{len(ordered_names)}] {fname}")

        # Skip samples with non-empty 备注 (bad data)
        truth = gt.get(fname, {})
        remark = truth.get("备注", "")
        if remark:
            print(f"  SKIP: 备注='{remark}' (data quality issue)")
            skipped_rows.append({"photo": fname, "reason": f"备注: {remark}"})
            total_skipped += 1
            continue

        photo_path = find_photo(fname)
        if photo_path is None:
            print(f"  SKIP: photo not found")
            skipped_rows.append({"photo": fname, "reason": "photo not found"})
            total_skipped += 1
            continue

        cache_dir = CACHE_DIR / fname

        # Phase A: VLM
        if not args.cv_only:
            t0 = time.time()
            meta = run_vlm_phase(fname, photo_path, cache_dir)
            vlm_time = time.time() - t0
            if meta is None:
                total_skipped += 1
                continue
            print(f"  [VLM] {len(meta.get('plants', []))} parts, {vlm_time:.1f}s")
        else:
            if not (cache_dir / "vlm_result.json").exists():
                print(f"  SKIP: no VLM cache")
                total_skipped += 1
                continue

        # Phase B: CV
        t0 = time.time()
        plant_results = run_cv_phase(cache_dir, counter_fn=counter_fn)
        cv_time = time.time() - t0

        if plant_results is None:
            total_skipped += 1
            continue

        # Aggregate
        # 主花序角果 = 主枝 only (VLM label), 分枝角果 = 分枝 only, 主干不计角果
        main_pods = 0      # 主枝 → F列 主花序角果数
        branch_pods = 0    # 分枝 → G列 分枝角果数
        for pid, label, count in plant_results:
            if label == "主枝":
                main_pods += count
            elif label == "分枝":
                branch_pods += count
            # 主干: skip, no pods
        total_pods = main_pods + branch_pods

        # Ground truth
        truth = gt.get(fname, {})
        gt_main = _safe_int(truth.get("主花序角果数", 0))
        gt_branch = _safe_int(truth.get("分枝角果数", 0))
        gt_total = _safe_int(truth.get("单株总角果数", 0))

        # Per-sample accuracy
        def _pct_acc(algo, gt_val):
            if gt_val == 0:
                return 100.0 if algo == 0 else 0.0
            return max(0.0, (1 - abs(algo - gt_val) / gt_val)) * 100

        acc_main = _pct_acc(main_pods, gt_main)
        acc_branch = _pct_acc(branch_pods, gt_branch)
        acc_total = _pct_acc(total_pods, gt_total)

        print(f"  [CV] main={main_pods} branch={branch_pods} total={total_pods} "
              f"(truth: F={gt_main} G={gt_branch} H={gt_total})  {cv_time:.1f}s")
        print(f"  [ACC] 主花序={acc_main:.1f}% 分枝={acc_branch:.1f}% 总计={acc_total:.1f}%")

        csv_rows.append({
            "photo": fname,
            "algo_main_pods": main_pods,
            "algo_branch_pods": branch_pods,
            "algo_total_pods": total_pods,
            "gt_main_pods": gt_main,
            "gt_branch_pods": gt_branch,
            "gt_total_pods": gt_total,
            "err_main": main_pods - gt_main,
            "err_branch": branch_pods - gt_branch,
            "err_total": total_pods - gt_total,
            "parts_detected": len(plant_results),
            "gt_branches": truth.get("分枝数", 0) or 0,
            "acc_main_pct": round(acc_main, 1),
            "acc_branch_pct": round(acc_branch, 1),
            "acc_total_pct": round(acc_total, 1),
            "cv_time_s": round(cv_time, 2),
        })
        total_processed += 1

    # Write CSV
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(RESULT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nResults saved to: {RESULT_CSV}")

    # Write skipped CSV
    if skipped_rows:
        with open(SKIPPED_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["photo", "reason"])
            writer.writeheader()
            writer.writerows(skipped_rows)
        print(f"Skipped samples saved to: {SKIPPED_CSV}")

    # Compute summary stats
    if csv_rows:
        def _stats(key_algo, key_gt):
            errs = []
            pct_errs = []
            for r in csv_rows:
                e = r[key_algo] - r[key_gt]
                errs.append(e)
                if r[key_gt] > 0:
                    pct_errs.append(abs(e) / r[key_gt] * 100)
            errs = np.array(errs)
            mae = np.mean(np.abs(errs))
            me = np.mean(errs)
            mape = np.mean(pct_errs) if pct_errs else 0
            rmse = np.sqrt(np.mean(errs ** 2))
            algo_vals = np.array([r[key_algo] for r in csv_rows], dtype=float)
            gt_vals = np.array([r[key_gt] for r in csv_rows], dtype=float)
            # R²
            ss_res = np.sum((algo_vals - gt_vals) ** 2)
            ss_tot = np.sum((gt_vals - np.mean(gt_vals)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            return mae, me, mape, rmse, r2

        lines = []
        lines.append("=" * 60)
        lines.append("  Benchmark Summary")
        lines.append("=" * 60)
        lines.append(f"Processed: {total_processed}, Skipped: {total_skipped}")
        lines.append("")

        for name, ka, kg, acc_key in [
            ("主花序角果 (F列)", "algo_main_pods", "gt_main_pods", "acc_main_pct"),
            ("分枝角果 (G列)", "algo_branch_pods", "gt_branch_pods", "acc_branch_pct"),
            ("单株总角果 (H列)", "algo_total_pods", "gt_total_pods", "acc_total_pct"),
        ]:
            mae, me, mape, rmse, r2 = _stats(ka, kg)
            avg_acc = np.mean([r[acc_key] for r in csv_rows])
            lines.append(f"--- {name} ---")
            lines.append(f"  平均准确率 = {avg_acc:.1f}%")
            lines.append(f"  MAE  = {mae:.1f}")
            lines.append(f"  ME   = {me:+.1f}  (正=高估, 负=低估)")
            lines.append(f"  MAPE = {mape:.1f}%")
            lines.append(f"  RMSE = {rmse:.1f}")
            lines.append(f"  R²   = {r2:.4f}")
            lines.append("")

        summary = "\n".join(lines)
        print(f"\n{summary}")

        with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary saved to: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
