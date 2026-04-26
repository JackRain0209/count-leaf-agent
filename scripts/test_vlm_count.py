#!/usr/bin/env python3
"""One-off: send first image to VLM + CV pipeline, print pod counts."""
import sys, cv2
sys.path.insert(0, "/Users/jack/cv-agent")
from app.pipeline.vlm_counter import vlm_label_plants
from app.pipeline.pod_counter_graph import count_pods_on_branch

IMG = "/Users/jack/cv-agent/online_uploads/农生院卢坤-合川考种照片-整理20260414/合川考种照片整理-20260414/N0-1（孔）/N0.1.1.1.JPG"
img = cv2.imread(IMG)
print(f"Image: {img.shape[1]}x{img.shape[0]}")

result = vlm_label_plants(img)
plants = result.get("plants", [])
crops = result.get("crops", {})

main_pods = 0
branch_pods = 0
for p in plants:
    pid = p.get("id")
    label = p.get("label", "")
    if label == "主干" or pid not in crops:
        print(f"  #{pid} {label}: skip")
        continue
    pod_result = count_pods_on_branch(
        crops[pid],
        stem_start_local=p.get("stem_start_local"),
        stem_end_local=p.get("stem_end_local"),
    )
    cnt = pod_result.get("pod_count", 0)
    if label == "主枝":
        main_pods += cnt
    else:
        branch_pods += cnt
    print(f"  #{pid} {label}: {cnt} pods")

print(f"\n=== Result ===")
print(f"主花序角果: {main_pods}  (truth: 67)")
print(f"分枝角果:   {branch_pods}  (truth: 159)")
print(f"总角果:     {main_pods+branch_pods}  (truth: 226)")
