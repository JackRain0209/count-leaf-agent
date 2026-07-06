#!/usr/bin/env python3
"""实时对比批量测试结果与 ground truth，持续刷新直到全部完成。"""
import csv
import json
import os
import time
from pathlib import Path

GT_CSV = "benchmark_results.csv"
ARTIFACT_DIR = Path("benchmark_cache")
LEGACY_RESULTS_DIR = Path("results")
ALLOW_LEGACY_RESULTS = os.environ.get("ALLOW_LEGACY_RESULTS") == "1"

def load_gt():
    gt = {}
    with open(GT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gt[row["photo"]] = row
    return gt


def load_result(photo):
    """Load the current benchmark artifact for a sample.

    The old results/<photo>/result.json layout is intentionally not used by
    default because result directories are now per-run. Reading the old fixed
    path can silently compare against stale output from another run.
    """
    artifact = ARTIFACT_DIR / photo / "experiment_result.json"
    if artifact.exists():
        with artifact.open(encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("result") or {}, artifact

    if ALLOW_LEGACY_RESULTS:
        legacy = LEGACY_RESULTS_DIR / photo / "result.json"
        if legacy.exists():
            with legacy.open(encoding="utf-8") as f:
                return json.load(f), legacy

    return None, None


def check_once(gt):
    rows = []
    for photo, g in sorted(gt.items()):
        try:
            r, source = load_result(photo)
        except Exception:
            continue
        if not r:
            continue

        algo_total = r.get("total_pods", 0)
        gt_total = int(g["gt_total_pods"])
        err = algo_total - gt_total
        acc = max(0, round((1 - abs(err) / gt_total) * 100, 1)) if gt_total else 0

        plants = r.get("plants", [])
        main_pods = sum(p.get("pod_count", 0) for p in plants if p.get("label") == "主枝")
        branch_pods = sum(p.get("pod_count", 0) for p in plants if p.get("label") == "分枝")
        gt_main = int(g["gt_main_pods"])
        gt_branch = int(g["gt_branch_pods"])
        parts = len([p for p in plants if p.get("label") != "主干"])

        rows.append({
            "photo": photo,
            "gt_total": gt_total, "algo_total": algo_total,
            "err": err, "acc": acc,
            "gt_main": gt_main, "algo_main": main_pods,
            "gt_branch": gt_branch, "algo_branch": branch_pods,
            "parts": parts,
            "source": str(source),
        })
    return rows

def print_table(rows, total_gt_photos):
    os.system("clear" if os.name != "nt" else "cls")
    print(f"{'Photo':<12} {'GT':>4} {'Algo':>5} {'Err':>5} {'Acc%':>6}  "
          f"{'GT_M':>4} {'Al_M':>5} {'GT_B':>5} {'Al_B':>5} {'Parts':>5}")
    print("-" * 75)

    sum_gt = sum_algo = 0
    accs = []
    for r in rows:
        sum_gt += r["gt_total"]
        sum_algo += r["algo_total"]
        accs.append(r["acc"])
        print(f"{r['photo']:<12} {r['gt_total']:>4} {r['algo_total']:>5} "
              f"{r['err']:>+5} {r['acc']:>5.1f}%  "
              f"{r['gt_main']:>4} {r['algo_main']:>5} "
              f"{r['gt_branch']:>5} {r['algo_branch']:>5} {r['parts']:>5}")

    print("-" * 75)
    overall_err = sum_algo - sum_gt
    overall_acc = max(0, round((1 - abs(overall_err) / sum_gt) * 100, 1)) if sum_gt else 0
    avg_acc = round(sum(accs) / len(accs), 1) if accs else 0
    print(f"{'TOTAL':<12} {sum_gt:>4} {sum_algo:>5} {overall_err:>+5} {overall_acc:>5.1f}%")
    print(f"\n完成: {len(rows)}/{total_gt_photos}  |  "
          f"逐图平均准确率: {avg_acc}%  |  "
          f"总体准确率: {overall_acc}%")
    print(f"\n(每5秒刷新，Ctrl+C 退出)")

if __name__ == "__main__":
    gt = load_gt()
    while True:
        rows = check_once(gt)
        print_table(rows, len(gt))
        if len(rows) >= len(gt):
            print("\n✅ 全部完成!")
            break
        time.sleep(5)
