"""
VLM-based pod counting verification — post-processor module.

Decoupled from any specific counting algorithm (skeleton / graph / plantcv).
Takes the crop image, debug image with markers, and standardized marker list,
sends them to the VLM for verification, and returns adjustments.

Standardized marker format (input):
    [
        {"id": 1, "type": "pod",      "x_norm": float, "y_norm": float},
        {"id": 2, "type": "filtered", "x_norm": float, "y_norm": float},
        ...
    ]

    x_norm / y_norm are in [0, 1] range, relative to the counter's working
    image.  This avoids coordinate mismatch when counters auto-upscale
    low-res crops internally.

VLM verification tasks:
    1. Detect missed pods (due to crossing / overlap)
    2. Detect false positives (dead twigs incorrectly marked as pods)

Returns:
    {
        "missed_count": int,
        "false_positive_ids": [int, ...],
        "adjusted_pod_count": int,
        "reason": str,
        "verified_image": np.ndarray,
    }
"""

import cv2
import json
import numpy as np
from pathlib import Path

from .vlm_counter import _get_client, _image_to_base64, _resize_for_vlm, _parse_json
from app.config import VLM_MODEL

# ─── Reference images for few-shot VLM prompting ─────────────────────────────

_SUPPORT_DIR = Path(__file__).resolve().parent.parent / "support"
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def _load_reference_images(max_dim: int = 512) -> list:
    """Load and encode reference images from app/support/ as base64."""
    if not _SUPPORT_DIR.is_dir():
        return []
    refs = []
    for f in sorted(_SUPPORT_DIR.iterdir()):
        if f.suffix.lower() not in _IMG_EXTS:
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        img = _resize_for_vlm(img, max_dim=max_dim)
        refs.append(_image_to_base64(img))
    return refs


# ─── Build annotated image for VLM ───────────────────────────────────────────

def _norm_to_pixel(markers: list, w: int, h: int) -> list:
    """Convert normalized [0,1] markers to pixel coordinates for a given image size."""
    out = []
    for m in markers:
        px = int(m["x_norm"] * w)
        py = int(m["y_norm"] * h)
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))
        out.append({**m, "x": px, "y": py})
    return out


def _draw_markers_for_vlm(crop_bgr: np.ndarray, markers: list) -> np.ndarray:
    """
    Draw numbered markers on a clean copy of the crop for VLM consumption.
    Green circles + number = pod (valid)
    Red circles + number = filtered (dead twig)

    markers must already have pixel-space 'x','y' (call _norm_to_pixel first).
    """
    vis = crop_bgr.copy()
    h, w = vis.shape[:2]
    fs = max(0.4, min(h, w) / 800)

    for m in markers:
        mid = m["id"]
        mx, my = int(m["x"]), int(m["y"])
        mtype = m["type"]

        if mtype == "pod":
            color = (0, 255, 0)       # green
            label = f"P{mid}"
        else:
            color = (0, 0, 255)       # red
            label = f"F{mid}"

        cv2.circle(vis, (mx, my), 6, color, -1)
        cv2.circle(vis, (mx, my), 8, (255, 255, 255), 1)
        cv2.putText(vis, label, (mx + 10, my - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, max(1, int(fs * 2)))

    return vis


# ─── Build marker summary text ───────────────────────────────────────────────

def _markers_to_text(markers: list) -> str:
    """Convert marker list to text description for prompt.

    markers must already have pixel-space 'x','y' (call _norm_to_pixel first).
    """
    pods = [m for m in markers if m["type"] == "pod"]
    filtered = [m for m in markers if m["type"] == "filtered"]
    lines = []
    lines.append(f"有效角果标记（绿色，P 开头）共 {len(pods)} 个：")
    for m in pods:
        lines.append(f"  P{m['id']}: 坐标 ({m['x']}, {m['y']})")
    lines.append(f"被过滤标记（红色，F 开头）共 {len(filtered)} 个：")
    for m in filtered:
        lines.append(f"  F{m['id']}: 坐标 ({m['x']}, {m['y']})")
    return "\n".join(lines)


# ─── VLM verification call ───────────────────────────────────────────────────

_POD_KNOWLEDGE = """【有效角果 vs 枯枝 判别标准】

✅ 有效角果特征（三个关键词：亮色、饱满、有宽度变化）：
1. **颜色鲜亮**：青绿、黄绿、黄色，与深色背景有强烈对比，一眼就能看到是亮色的
2. **纺锤/椭圆轮廓**：中段微微鼓起膨大、两端收窄，沿长度方向有明显宽度变化
3. 果身充实饱满，有一定厚度感，不是扁平的线条
4. 长度正常，从主干自然伸出

❌ 枯枝特征（三个关键词：暗色、细瘦、等宽）：
1. **颜色暗沉**：褐色、暗棕、发黑，与亮黄绿色的正常角果相比明显偏暗偏深
2. 不会有一个形态上的宽度的突变
3. **非常纤细**：比正常角果细很多，宽度远小于旁边的健康角果
4. 通常**较短**，可能弯曲、卷曲、蜷缩
5. 看起来像脱落了果实只剩果柄，或者是未发育的干瘪残枝

判断时请综合颜色、粗细、形态三方面，与旁边的健康角果对比来判断。"""


def verify_pod_markers(
    crop_bgr: np.ndarray,
    debug_image: np.ndarray,
    markers: list,
    pod_count: int,
) -> dict:
    """
    Send crop + annotated debug image + marker data to VLM for verification.

    Args:
        crop_bgr:    Original crop image (no markers)
        debug_image: Debug image from counter (with colored overlays)
        markers:     Standardized marker list from counter
        pod_count:   Original pod count from counter

    Returns:
        {
            "missed_count": int,
            "false_positive_ids": [int, ...],
            "adjusted_pod_count": int,
            "reason": str,
            "verified_image": np.ndarray,
        }
    """
    if not markers:
        return {
            "missed_count": 0,
            "false_positive_ids": [],
            "adjusted_pod_count": pod_count,
            "reason": "无标记点，跳过验证",
            "verified_image": debug_image.copy(),
        }

    client = _get_client()

    # Convert normalized coords to pixel space of the crop image
    h_crop, w_crop = crop_bgr.shape[:2]
    px_markers = _norm_to_pixel(markers, w_crop, h_crop)

    # Build annotated image with clear numbered markers
    annotated = _draw_markers_for_vlm(crop_bgr, px_markers)
    marker_text = _markers_to_text(px_markers)

    # Resize for API payload
    annotated_resized = _resize_for_vlm(annotated, max_dim=1536)
    crop_resized = _resize_for_vlm(crop_bgr, max_dim=1536)
    b64_annotated = _image_to_base64(annotated_resized)
    b64_crop = _image_to_base64(crop_resized)

    pod_markers = [m for m in px_markers if m["type"] == "pod"]
    filtered_markers = [m for m in px_markers if m["type"] == "filtered"]

    # Load reference images for dead twig examples
    ref_images = _load_reference_images()
    ref_note = ""
    if ref_images:
        ref_note = f"""\n\n📎 **参考图（枯枝样例）**：以下 {len(ref_images)} 张图片展示了典型的枯枝外观。
请仔细观察这些枯枝的特征（暗色、细瘦、等宽、无膨大），在审核时以此为参照判断标注点是否为枯枝。\n"""

    prompt = f"""你是油菜角果计数的审核专家。下面有多张图：
- 第一张：**原始截图**（干净无标记，用于观察植株真实外观）
- 第二张：**标注图**（带有编号标记点）
{f'- 后面 {len(ref_images)} 张：**枯枝参考图**（展示典型枯枝外观，供你对比参照）' if ref_images else ''}
{ref_note}

标记图说明：
- **绿色圆点 + P编号**：算法判定为"有效角果"的位置（标在角果尖端）
- **红色圆点 + F编号**：算法判定为"枯枝/噪声"已过滤的位置

当前标记详情：
{marker_text}

算法计数结果：有效角果 {len(pod_markers)} 个，已过滤 {len(filtered_markers)} 个

{_POD_KNOWLEDGE}

⚠️ 重要约束：
- 只关注一件事：**混在 P 标记里的枯枝**
- 已标记的正常 P 标记点不需要你确认
- 不需要检查漏数，只检查误判

**审核任务 — 找枯枝：**
检查绿色 P 标记点中，有没有实际上是**枯枝**却没有被过滤掉的？
{f'请对比前面的枯枝参考图，' if ref_images else ''}判断方法：沿着该标记点往主干方向回溯，观察这条分支的外形——
- 如果全程**粗细均匀、没有椭圆形膨大**，而且较短 → 枯枝
- 如果中段有明显**鼓起/膨大**（纺锤形轮廓） → 有效角果，不需要报告
如果有枯枝，请列出这些 P 编号。

⚠️ 严格按以下 JSON 返回，不要其他文字：
```json
{{
  "false_positive_ids": [],
  "reason": "简短说明审核结论"
}}
```

注意：
- false_positive_ids：被误判为角果的 P 编号列表，如 [1, 3, 7]（没有误判就填空列表 []）
- 宁可保守（少报误判），也不要过度纠正
- 如果图片模糊看不清，保持原判定即可"""

    print(f"[VLM PodVerify] Sending {len(markers)} markers ({len(pod_markers)} pods, "
          f"{len(filtered_markers)} filtered) for verification...")

    raw = ""
    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64_crop}"}},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64_annotated}"}},
                        *[
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{ref_b64}"}}
                            for ref_b64 in ref_images
                        ],
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        usage = getattr(response, 'usage', None)
        if usage:
            print(f"[VLM PodVerify] Tokens: prompt={usage.prompt_tokens}, "
                  f"completion={usage.completion_tokens}")
        print(f"[VLM PodVerify] Response: {raw[:400]}")
    except Exception as e:
        print(f"[VLM PodVerify] API error: {e}")
        return {
            "missed_count": 0,
            "false_positive_ids": [],
            "adjusted_pod_count": pod_count,
            "reason": f"VLM 调用失败: {e}",
            "verified_image": debug_image.copy(),
        }

    # Parse response
    parsed = _parse_json(raw)
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        print(f"[VLM PodVerify] Parse failed, keeping original count")
        return {
            "missed_count": 0,
            "false_positive_ids": [],
            "adjusted_pod_count": pod_count,
            "reason": "VLM 返回解析失败，保持原计数",
            "verified_image": debug_image.copy(),
        }

    raw_fp = parsed.get("false_positive_ids", [])
    if not isinstance(raw_fp, list):
        raw_fp = []
    false_positive_ids = []
    for x in raw_fp:
        if isinstance(x, (int, float)):
            false_positive_ids.append(int(x))
        elif isinstance(x, str):
            # Handle "P8", "P 8", "8" etc.
            cleaned = x.strip().lstrip("PpFf").strip()
            try:
                false_positive_ids.append(int(cleaned))
            except ValueError:
                pass
    reason = parsed.get("reason", "")

    # Calculate adjusted count (only subtract false positives, no missed_count)
    adjusted = pod_count - len(false_positive_ids)
    adjusted = max(0, adjusted)

    print(f"[VLM PodVerify] false_pos={false_positive_ids}, "
          f"original={pod_count} → adjusted={adjusted}")

    # Build verified image (use pixel-space markers)
    verified_img = _build_verified_image(
        crop_bgr, px_markers, false_positive_ids, adjusted)

    return {
        "missed_count": 0,
        "false_positive_ids": false_positive_ids,
        "adjusted_pod_count": adjusted,
        "reason": reason,
        "verified_image": verified_img,
    }


# ─── Build final verified visualization ──────────────────────────────────────

def _build_verified_image(
    crop_bgr: np.ndarray,
    markers: list,
    false_positive_ids: list,
    adjusted_count: int,
) -> np.ndarray:
    """
    Regenerate the debug image after VLM verification:
    - Green = confirmed valid pods
    - Red strikethrough = false positives (VLM says not a pod)
    - Filtered markers stay in gray
    - Text overlay with adjusted count
    """
    vis = crop_bgr.copy()
    h, w = vis.shape[:2]
    fs = max(0.4, min(h, w) / 800)
    fp_set = set(false_positive_ids)

    confirmed = 0
    for m in markers:
        mid = m["id"]
        mx, my = int(m["x"]), int(m["y"])
        mtype = m["type"]

        if mtype == "pod":
            if mid in fp_set:
                # False positive — red X
                cv2.circle(vis, (mx, my), 7, (0, 0, 255), 2)
                cv2.line(vis, (mx - 5, my - 5), (mx + 5, my + 5), (0, 0, 255), 2)
                cv2.line(vis, (mx - 5, my + 5), (mx + 5, my - 5), (0, 0, 255), 2)
                cv2.putText(vis, f"X{mid}", (mx + 10, my - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), max(1, int(fs * 2)))
            else:
                # Confirmed pod — green
                confirmed += 1
                cv2.circle(vis, (mx, my), 5, (0, 255, 0), -1)
                cv2.putText(vis, str(confirmed), (mx + 8, my - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 0.9, (0, 255, 0), max(1, int(fs * 2)))
        else:
            # Filtered — gray
            cv2.circle(vis, (mx, my), 3, (150, 150, 150), -1)

    # Overlay adjusted count
    fs2 = max(0.5, min(h, w) / 500)
    cv2.putText(vis, f"VLM Verified: {adjusted_count}", (5, int(25 * fs2) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs2, (0, 255, 255), max(1, int(fs2 * 2)))
    detail = f"confirmed={confirmed} false_pos=-{len(fp_set)}"
    cv2.putText(vis, detail, (5, int(50 * fs2) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs2 * 0.6, (180, 180, 180), 1)

    return vis
