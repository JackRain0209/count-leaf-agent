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
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .vlm_counter import _get_client, _image_to_base64, _resize_for_vlm, _parse_json
from app.config import VLM_MODEL, VLM_FP_CONFIDENCE_THRESHOLD

# ─── Reference images for few-shot VLM prompting ─────────────────────────────

_SUPPORT_DIR = Path(__file__).resolve().parent.parent / "support"
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_DESC_EXTS = {".md", ".markdown", ".txt"}


@dataclass
class _SupportExample:
    name: str
    type: str  # "positive" or "negative"
    description: str
    image_b64: str


def _parse_support_descriptions() -> Dict[str, Tuple[str, str]]:
    """Parse description markdown/txt files in support directory.

    Returns mapping from basename (without suffix) to (type, description).
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    if not _SUPPORT_DIR.is_dir():
        return mapping

    for desc_file in sorted(_SUPPORT_DIR.iterdir()):
        if desc_file.suffix.lower() not in _DESC_EXTS:
            continue
        try:
            text = desc_file.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[SupportExamples] Failed to read {desc_file}: {exc}")
            continue

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, raw_desc = line.split(":", 1)
            key = key.strip()
            desc = raw_desc.strip()
            if not key or not desc:
                continue

            lower_key = key.lower()
            if "反例" in lower_key or "negative" in lower_key:
                ex_type = "negative"
            elif "正例" in lower_key or "positive" in lower_key:
                ex_type = "positive"
            else:
                # default to negative examples if keyword missing
                ex_type = "positive"
            mapping[key] = (ex_type, desc)

    return mapping


def _load_support_examples(max_dim: int = 512) -> Tuple[List[_SupportExample], List[_SupportExample]]:
    """Load labeled support examples (positive & negative) for VLM prompting."""
    positives: List[_SupportExample] = []
    negatives: List[_SupportExample] = []

    if not _SUPPORT_DIR.is_dir():
        return positives, negatives

    desc_map = _parse_support_descriptions()
    if not desc_map:
        print("[SupportExamples] No description mapping found; skipping support images")
        return positives, negatives

    for img_path in sorted(_SUPPORT_DIR.iterdir()):
        if img_path.suffix.lower() not in _IMG_EXTS:
            continue

        stem = img_path.stem
        if stem not in desc_map:
            print(f"[SupportExamples] Description missing for {stem}, skipping")
            continue

        ex_type, desc = desc_map[stem]

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[SupportExamples] Failed to read image {img_path}")
            continue
        img = _resize_for_vlm(img, max_dim=max_dim)
        b64 = _image_to_base64(img)

        ex = _SupportExample(name=stem, type=ex_type, description=desc, image_b64=b64)
        if ex_type == "negative":
            negatives.append(ex)
        else:
            positives.append(ex)

    return positives, negatives


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

    # Load support examples (positives = 枯枝, negatives = 正常角果)
    positive_examples, negative_examples = _load_support_examples()

    support_text_blocks: List[str] = []
    if positive_examples:
        lines = [f"📎 **枯枝正例（应当过滤）**：共 {len(positive_examples)} 张参考图。"]
        for idx, ex in enumerate(positive_examples, 1):
            lines.append(f"  - 正例 {idx}（{ex.name}）：{ex.description}")
        support_text_blocks.append("\n".join(lines))
    if negative_examples:
        lines = [f"📎 **枯枝反例 / 正常角果（不应过滤）**：共 {len(negative_examples)} 张参考图。"]
        for idx, ex in enumerate(negative_examples, 1):
            lines.append(f"  - 反例 {idx}（{ex.name}）：{ex.description}")
        support_text_blocks.append("\n".join(lines))

    if support_text_blocks:
        support_intro = "- 后面的参考图包含枯枝正例与正常角果反例，请结合描述对比判断。\n"
        support_note = support_intro + "\n\n".join(support_text_blocks)
    else:
        support_note = ""

    prompt = f"""你是油菜角果计数的审核专家。下面有多张图：
- 第一张：**原始截图**（干净无标记，用于观察植株真实外观）
- 第二张：**标注图**（带有编号标记点）
{support_note}

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

⚠️ **代价不对称（极其重要）**：
- 误删 1 个真角果 = 漏过 5 个枯枝的代价
- 默认应当**保留**，只有当你**非常确信（≥80%）**这是枯枝时才列入 false_positive_ids
- 任何模糊、不确定、看不清、被遮挡的 P 点 → 一律保留，不要报告
- 单独孤立的 P 点（不在密集簇里）默认保留，它们很可能是真角果

**容易误判的情况（这些请保留，不要剔除）**：
- 细长且偏暗的真角果（果荚干瘪但仍是角果）
- 处于图像边缘 / 被遮挡 / 局部模糊的 P 点
- 短小但中段有轻微膨大轮廓的真角果
- 颜色与枯枝相近、但形态是纺锤形的角果

**真正的枯枝特征（同时满足才算）**：
- 沿 P 点回溯整条分支，**从基部到尖端粗细完全均匀**
- **完全没有任何椭圆/纺锤形膨大轮廓**
- 通常较短、笔直、像一根光秃秃的小棍

**审核任务 — 找枯枝：**
检查绿色 P 标记点中，有没有实际上是**枯枝**却没有被过滤掉的？
{('请对比前面的枯枝正例 / 反例参考图，' if support_text_blocks else '')}

请按以下两步完成：

**Step 1 — 逐点形态分析**（先写出来强迫你看清楚）：
对你怀疑是枯枝的每个 P 编号，用一句话描述其分支形态，例如：
- "P3：分支细长笔直、粗细均匀、无膨大 → 枯枝（置信度 0.9）"
- "P7：中段有轻微膨大但不明显 → 不确定，保留"

**Step 2 — 输出 JSON**：
基于 Step 1，把**置信度 ≥ 0.8** 的枯枝列入返回。

⚠️ 严格按以下 JSON 返回（Step 1 的分析写在 reason 里）：
```json
{{
  "false_positives": [
    {{"id": 3, "confidence": 0.9}},
    {{"id": 7, "confidence": 0.85}}
  ],
  "reason": "P3 笔直无膨大；P7 短小且粗细均匀..."
}}
```

注意：
- false_positives：每项必须是 {{"id": int, "confidence": float}} 的对象
- confidence ∈ [0,1]，表示你判定该点为枯枝的把握程度
- 没有误判就填空列表 []
- 宁可保守（漏报误判），绝不要过度纠正
- 兼容旧格式：如果你坚持只能输出整数 id 列表，请确保只输出你 ≥0.9 把握的"""

    print(f"[VLM PodVerify] Sending {len(markers)} markers ({len(pod_markers)} pods, "
          f"{len(filtered_markers)} filtered) for verification...")

    raw = ""
    try:
        message_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64_crop}"}},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64_annotated}"}},
        ]

        for ex in positive_examples:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{ex.image_b64}"}
            })
        for ex in negative_examples:
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{ex.image_b64}"}
            })

        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": message_content,
                }
            ],
            max_tokens=1200,
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

    # 支持两种格式：
    #   新版：false_positives = [{"id": 3, "confidence": 0.9}, ...]
    #   旧版：false_positive_ids = [3, 7, ...]（无 confidence，按 0.9 处理）
    raw_fp_new = parsed.get("false_positives", None)
    raw_fp_old = parsed.get("false_positive_ids", [])

    def _coerce_id(x):
        if isinstance(x, (int, float)):
            return int(x)
        if isinstance(x, str):
            cleaned = x.strip().lstrip("PpFf").strip()
            try:
                return int(cleaned)
            except ValueError:
                return None
        return None

    # 收集 (id, confidence) 对
    fp_pairs: List[Tuple[int, float]] = []
    if isinstance(raw_fp_new, list) and raw_fp_new:
        for item in raw_fp_new:
            if isinstance(item, dict):
                _id = _coerce_id(item.get("id"))
                _conf = item.get("confidence", 0.9)
                try:
                    _conf = float(_conf)
                except (TypeError, ValueError):
                    _conf = 0.9
                if _id is not None:
                    fp_pairs.append((_id, _conf))
            else:
                # 退化为裸 id
                _id = _coerce_id(item)
                if _id is not None:
                    fp_pairs.append((_id, 0.9))
    elif isinstance(raw_fp_old, list):
        for item in raw_fp_old:
            _id = _coerce_id(item)
            if _id is not None:
                fp_pairs.append((_id, 0.9))

    # 应用置信度阈值过滤
    threshold = VLM_FP_CONFIDENCE_THRESHOLD
    kept_pairs = [(i, c) for (i, c) in fp_pairs if c >= threshold]
    dropped_pairs = [(i, c) for (i, c) in fp_pairs if c < threshold]
    false_positive_ids = [i for (i, _) in kept_pairs]
    false_positive_confidences = {str(i): c for (i, c) in kept_pairs}

    if dropped_pairs:
        print(f"[VLM PodVerify] Dropped low-confidence FPs (<{threshold}): "
              f"{dropped_pairs}")
    if kept_pairs:
        print(f"[VLM PodVerify] Kept FPs (>={threshold}): {kept_pairs}")

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
        "false_positive_confidences": false_positive_confidences,
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
    - Red F markers = dead twigs already filtered by the CV algorithm
    - Text overlay with adjusted count
    """
    vis = crop_bgr.copy()
    h, w = vis.shape[:2]
    fs = max(0.4, min(h, w) / 800)
    fp_set = set(false_positive_ids)

    confirmed = 0
    cv_filtered = 0
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
            # CV-filtered dead twig — red F marker
            cv_filtered += 1
            cv2.circle(vis, (mx, my), 5, (0, 0, 255), -1)
            cv2.circle(vis, (mx, my), 7, (255, 255, 255), 1)
            cv2.putText(vis, f"F{mid}", (mx + 8, my - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, fs * 0.85, (0, 0, 255), max(1, int(fs * 2)))

    # Overlay adjusted count
    fs2 = max(0.5, min(h, w) / 500)
    cv2.putText(vis, f"VLM Verified: {adjusted_count}", (5, int(25 * fs2) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs2, (0, 255, 255), max(1, int(fs2 * 2)))
    detail = f"confirmed={confirmed} vlm_fp=-{len(fp_set)} cv_filtered={cv_filtered}"
    cv2.putText(vis, detail, (5, int(50 * fs2) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, fs2 * 0.6, (180, 180, 180), 1)

    return vis
