"""
VLM plant labeling using Doubao (豆包) Vision model.

Grid-based approach:
  1. Draw grid (512px cells) on image with row/col labels (A1, B2, ...)
  2. Send gridded image to Doubao
  3. Doubao returns which grid cells each plant occupies
  4. Convert grid coords → pixel bbox → draw labels
"""

import base64
import io
import json
import re

import cv2
import numpy as np
from PIL import Image
from openai import OpenAI
from typing import Optional

from app import trace
from app.config import (
    ARK_API_KEY, ARK_API_BASE, VLM_MODEL,
    VLM_THINKING, VLM_TIMEOUT_S, VLM_MAX_RETRIES,
)

_COLORS = [
    (0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 165, 255),
    (255, 0, 255), (0, 255, 255), (255, 255, 0),
    (128, 0, 255), (255, 128, 0), (0, 128, 255),
]

# Row labels: A, B, C, ... ; Col labels: 1, 2, 3, ...
_ROW_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class _ThinkingOverride:
    """给每次 create 调用附加 thinking 参数，其余全部透传。"""

    def __init__(self, client, extra_body):
        self._client = client
        self._extra_body = extra_body
        self.chat = self
        self.completions = self

    def create(self, *args, **kwargs):
        if self._extra_body:
            merged = dict(kwargs.get("extra_body") or {})
            merged.update(self._extra_body)
            kwargs["extra_body"] = merged
        return self._client.chat.completions.create(*args, **kwargs)


def _get_client(phase: str = "unknown") -> OpenAI:
    """创建 OpenAI 客户端。所有 VLM 调用都要经过这里，所以埋点也放在这里。"""
    client = OpenAI(
        api_key=ARK_API_KEY,
        base_url=ARK_API_BASE,
        timeout=VLM_TIMEOUT_S,
        max_retries=VLM_MAX_RETRIES,
    )
    if VLM_THINKING == "disabled":
        client = _ThinkingOverride(client, {"thinking": {"type": "disabled"}})
    return trace.wrap_openai_client(client, phase)


def _image_to_base64(image: np.ndarray) -> str:
    """Convert OpenCV image (BGR) to base64 JPEG string."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    print(f"[VLM] Payload: {len(b64)//1024}KB")
    return b64


def _resize_for_vlm(image: np.ndarray, max_dim: int = 2048) -> np.ndarray:
    """Resize if too large."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    print(f"[VLM] Resized {w}x{h} → {new_w}x{new_h}")
    return resized


def draw_grid(image: np.ndarray, cell_size: int = 512):
    """
    Draw grid on image with row/col labels.
    Returns: (gridded_image, n_rows, n_cols, cell_size)
    """
    h, w = image.shape[:2]
    n_cols = (w + cell_size - 1) // cell_size
    n_rows = (h + cell_size - 1) // cell_size

    gridded = image.copy()

    # Draw grid lines
    for i in range(1, n_cols):
        x = i * cell_size
        cv2.line(gridded, (x, 0), (x, h), (0, 255, 255), 2)
    for j in range(1, n_rows):
        y = j * cell_size
        cv2.line(gridded, (0, y), (w, y), (0, 255, 255), 2)

    # Label each cell
    fs = max(0.6, cell_size / 600)
    th = max(1, int(fs * 2))
    for r in range(n_rows):
        for c in range(n_cols):
            label = f"{_ROW_LABELS[r]}{c+1}"
            cx = c * cell_size + 5
            cy = r * cell_size + int(25 * fs) + 5
            # Background
            cv2.rectangle(gridded, (cx, cy - int(20*fs)),
                         (cx + int(45*fs), cy + 5), (0, 0, 0), -1)
            cv2.putText(gridded, label, (cx, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), th)

    print(f"[Grid] {n_rows} rows x {n_cols} cols, cell={cell_size}px")
    return gridded, n_rows, n_cols, cell_size


_MISSING_COMMA = re.compile(r"(?<=\d)\s+(?=\d)")


def _repair_json_text(text: str) -> str:
    """修补模型常见的 JSON 格式错误：bbox 里少了逗号。

    模型偶尔把 bbox 写成 `[597 142,903 187]`，本该是 `[597,142,903,187]`。
    这种整体解析会失败，退化到逐对象解析时这些条目又被静默丢弃——
    实测一张图 13 个部件里丢了 6 个，其中 3 个"主枝"全在内，
    直接导致"主花序"和"主干+主枝"两列归零。
    """
    if not text:
        return text
    return _MISSING_COMMA.sub(",", text)


def _extract_json_array(content: str) -> Optional[str]:
    """Return the first complete JSON array substring from a model response."""
    start = content.find("[")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(content)):
        ch = content[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return content[start:idx + 1]
    return None


def _extract_json_objects_from_array(content: str):
    """Salvage complete top-level objects from a truncated JSON array."""
    arr_start = content.find("[")
    if arr_start < 0:
        return []

    objects = []
    depth = 0
    obj_start = None
    in_string = False
    escaped = False
    for idx in range(arr_start, len(content)):
        ch = content[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                obj_start = idx
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and obj_start is not None:
                candidate = content[obj_start:idx + 1]
                try:
                    obj = json.loads(candidate)
                except json.JSONDecodeError:
                    try:
                        obj = json.loads(_repair_json_text(candidate))
                    except json.JSONDecodeError:
                        obj_start = None
                        continue
                if isinstance(obj, dict):
                    objects.append(obj)
                obj_start = None
    return objects


def _parse_json(content: str):
    """Extract plant JSON from VLM response."""
    raw = content or ""
    candidates = []
    if "```json" in raw:
        candidates.append(raw.split("```json", 1)[1].split("```", 1)[0].strip())
    if "```" in raw:
        candidates.append(raw.split("```", 1)[1].split("```", 1)[0].strip())
    candidates.append(raw.strip())

    embedded_array = _extract_json_array(raw)
    if embedded_array:
        candidates.append(embedded_array)

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        # Return full parsed result — callers extract plants/ruler/label_tag as needed.
        return parsed

    # 整体解析失败时先试着修补再重试。缺逗号是模型最常见的格式错误，
    # 不在这里补救的话，下面退化的逐对象提取会把这些部件静默丢掉。
    for candidate in candidates:
        repaired = _repair_json_text(candidate)
        if not repaired or repaired == candidate:
            continue
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            continue

    partial_objects = _extract_json_objects_from_array(raw)
    if partial_objects:
        print(f"[VLM] JSON array incomplete, salvaged {len(partial_objects)} objects")
        return partial_objects

    print(f"[VLM] JSON parse failed: {raw[:300]}")
    return None


def _grid_to_pixel(cell_str: str, cell_size: int):
    """Convert grid coordinate like 'A1' to pixel (x, y) of cell's top-left corner."""
    cell_str = cell_str.strip().upper()
    row_char = cell_str[0]
    col_num = int(cell_str[1:]) - 1
    row = _ROW_LABELS.index(row_char)
    x = col_num * cell_size
    y = row * cell_size
    return x, y


def _build_spatial_context(current_plant: dict, all_plants: list,
                           crop_bbox: list) -> str:
    """
    Build spatial context text describing where the main stem and neighboring
    plants are relative to the current crop region.

    Args:
        current_plant: The plant dict for the current crop.
        all_plants: All plant dicts from the first VLM call.
        crop_bbox: [ox1, oy1, ox2, oy2] pixel bbox of the current crop.

    Returns: A Chinese text block to inject into the VLM prompt.
    """
    cx1, cy1, cx2, cy2 = crop_bbox
    cw = cx2 - cx1
    ch = cy2 - cy1
    if cw <= 0 or ch <= 0:
        return ""

    lines = []

    # --- Main stem info ---
    trunk = None
    for p in all_plants:
        if p.get("label") == "主干":
            trunk = p
            break

    if trunk and "pixel_bbox" in trunk:
        tx1, ty1, tx2, ty2 = trunk["pixel_bbox"]
        # Convert trunk bbox to normalized coords within current crop (0-1000)
        def _to_crop_norm(px, py):
            nx = int((px - cx1) / cw * 1000)
            ny = int((py - cy1) / ch * 1000)
            return nx, ny

        tnx1, tny1 = _to_crop_norm(tx1, ty1)
        tnx2, tny2 = _to_crop_norm(tx2, ty2)

        # Describe direction relative to crop
        trunk_cx = (tx1 + tx2) / 2
        trunk_cy = (ty1 + ty2) / 2
        crop_cx = (cx1 + cx2) / 2
        crop_cy = (cy1 + cy2) / 2
        dirs = []
        if trunk_cy < crop_cy:
            dirs.append("上方")
        elif trunk_cy > crop_cy:
            dirs.append("下方")
        if trunk_cx < crop_cx:
            dirs.append("左侧")
        elif trunk_cx > crop_cx:
            dirs.append("右侧")
        dir_str = "".join(dirs) if dirs else "附近"

        lines.append(f"【主干位置】整株植物的主干位于当前裁剪区域的{dir_str}，"
                     f"主干在裁剪区域坐标系中的大致范围：({tnx1},{tny1}) 到 ({tnx2},{tny2})。"
                     f"当前植株({current_plant.get('label','')})应该是从主干延伸出来的，"
                     f"其走势与主干方向相关。")

    # --- Neighboring plants ---
    current_id = current_plant.get("id")
    neighbors = []
    for p in all_plants:
        pid = p.get("id")
        if pid == current_id or "pixel_bbox" not in p:
            continue
        px1, py1, px2, py2 = p["pixel_bbox"]
        # Check if this plant's bbox overlaps or is close to current crop
        margin = max(cw, ch) * 0.3  # 30% proximity threshold
        if (px2 >= cx1 - margin and px1 <= cx2 + margin and
                py2 >= cy1 - margin and py1 <= cy2 + margin):
            # Compute direction
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2
            crop_cx = (cx1 + cx2) / 2
            crop_cy = (cy1 + cy2) / 2
            dirs = []
            if pcy < crop_cy - ch * 0.1:
                dirs.append("上")
            elif pcy > crop_cy + ch * 0.1:
                dirs.append("下")
            if pcx < crop_cx - cw * 0.1:
                dirs.append("左")
            elif pcx > crop_cx + cw * 0.1:
                dirs.append("右")
            dir_str = "".join(dirs) if dirs else "重叠"
            neighbors.append(f"#{pid} {p.get('label','')} 在{dir_str}方向")

    if neighbors:
        lines.append(f"【邻近植株】以下植株与当前裁剪区域相邻或部分重叠，"
                     f"它们的枝叶可能从边缘伸入：{', '.join(neighbors)}。"
                     f"从这些方向伸入的、与当前植株主体不相连的枝叶很可能是外来的。")

    return "\n".join(lines)


def vlm_detect_adhesion(crop_bgr: np.ndarray, plant_id: int = 0,
                        label: str = "", all_plants: list = None,
                        current_plant: dict = None) -> dict:
    """
    Send a single crop image to Doubao VLM to detect foreign plant parts
    from OTHER plants that leaked into this crop region.

    VLM identifies regions that belong to neighboring plants (not the current
    one) and returns closed polygon areas to be masked out (filled black).

    Args:
        crop_bgr: The crop image (BGR).
        plant_id: ID of the current plant part.
        label: Label of the current plant part (主枝/分枝).
        all_plants: Full list of plant dicts from the first VLM call.
        current_plant: The current plant's dict (with pixel_bbox etc.).

    Returns:
        {
            "has_foreign": bool,
            "regions": [{"points": [[x1,y1],[x2,y2],...], "description": str}, ...],
            "raw_response": str,
        }
    """
    client = _get_client(phase="adhesion")
    h, w = crop_bgr.shape[:2]

    vlm_img = _resize_for_vlm(crop_bgr, max_dim=1536)
    b64 = _image_to_base64(vlm_img)

    # Build spatial context if available
    spatial_ctx = ""
    if all_plants and current_plant and "pixel_bbox" in current_plant:
        spatial_ctx = _build_spatial_context(
            current_plant, all_plants, current_plant["pixel_bbox"])
        if spatial_ctx:
            spatial_ctx = f"\n\n以下是空间位置参考信息：\n{spatial_ctx}\n"

    prompt = f"""这是一张油菜植株裁剪图（#{plant_id} {label}），植株的各个部分被拆开后平铺在黑色背景上。

这张裁剪图是从整株植物照片中按区域裁剪出来的，裁剪时可能把**相邻其他植株**的枝叶/角果也框进来了，甚至与当前植株发生粘连（互相接触、重叠）。
{spatial_ctx}
**区分当前植株 vs 外来植株的核心方法 — 生长方向判断：**

当前植株（#{plant_id} {label}）是一根从主干上拆下来的枝条。它的结构特点是：
- 有一个**根部/基部**（靠近主干方向的一端），所有枝叶从这个基部**向外生长、发散**
- 枝条、角果的走势是**从基部向图片四周扩散**的

外来植株的特点完全相反：
- 它们**从图片的边缘进入画面**，一端连接在图片边框上
- 它们的走势是**从图片边缘向画面内部延伸**
- 它们可能与当前植株发生粘连（接触、重叠），但生长方向与当前植株不同

**判断规则：**
1. 从主干方向（基部）向外延伸出去的枝叶 → **当前植株本体，不要标记**
2. 从图片上/下/左/右边缘伸进来的枝叶，一端连在图片边框上 → **外来植株，需要标记**
3. 即使外来枝叶与当前植株粘连（接触/重叠）了，也要把外来部分标记出来
4. 在粘连处，多边形应沿两株植株的**交界处**切割，尽量只框住外来部分
5. 粘连切割时，宁可多保留一点当前植株（少切），不要多切到当前植株（多切）

**请你按以下步骤分析：**

第一步：找到当前植株的基部（靠近主干的一端），确定其主轴走势方向
第二步：沿主轴方向延伸，识别所有从基部生长出去的枝叶 = 当前植株
第三步：检查图片四周边缘，是否有枝叶从边框处伸入画面 = 疑似外来植株
第四步：对每个疑似外来部分，确认它的走势是否是"从边缘向内"而非"从基部向外"
第五步：如果外来枝叶与当前植株有粘连，在交界处划定切割边界

如果存在外来植株部分，请用**封闭多边形区域**将其框出。

请使用归一化坐标系：
- 左上角 = (0, 0)，右下角 = (1000, 1000)
- x 从左到右 0→1000，y 从上到下 0→1000

严格按以下JSON格式返回，不要输出其他文字：
```json
{{
    "has_foreign": true/false,
    "regions": [
        {{
            "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
            "description": "从[方向]边缘伸入的外来枝条/角果，与当前植株[是否粘连]"
        }}
    ]
}}
```

- 如果没有外来植株混入，返回 {{"has_foreign": false, "regions": []}}
- points 是封闭多边形的顶点列表（至少3个点），按顺序连接后围成要涂黑的区域
- 在粘连处，多边形边界沿交界切割；非粘连处，多边形紧贴外来枝叶轮廓
- **切割原则：宁可少切（保留外来残余）也不要多切（伤到当前植株）**"""

    print(f"[VLM Adhesion] #{plant_id} {label}, crop {w}x{h}")
    if spatial_ctx:
        print(f"[VLM Adhesion] Spatial context:\n{spatial_ctx[:300]}")
    raw_response = ""
    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.1,
        )
        raw_response = response.choices[0].message.content.strip()
        usage = getattr(response, 'usage', None)
        if usage:
            print(f"[VLM Adhesion] Tokens: prompt={usage.prompt_tokens}, "
                  f"completion={usage.completion_tokens}")
        print(f"[VLM Adhesion] Response ({len(raw_response)} chars):\n{raw_response[:500]}")
    except Exception as e:
        raw_response = f"Doubao API error: {e}"
        print(f"[VLM Adhesion] {raw_response}")

    parsed = _parse_json(raw_response)
    if parsed is None:
        # Try parsing as dict directly
        try:
            import re as _re
            m = _re.search(r'\{.*\}', raw_response, _re.DOTALL)
            if m:
                parsed = json.loads(m.group())
        except Exception:
            pass

    if not parsed or not isinstance(parsed, dict):
        return {"has_foreign": False, "regions": [], "raw_response": raw_response}

    has_foreign = parsed.get("has_foreign", False)
    raw_regions = parsed.get("regions", [])

    # Validate regions: each must have a "points" list with ≥3 coordinate pairs
    valid_regions = []
    for region in raw_regions:
        pts = region.get("points", [])
        if isinstance(pts, list) and len(pts) >= 3:
            # Validate each point is [x, y]
            ok = all(isinstance(p, list) and len(p) == 2 for p in pts)
            if ok:
                valid_regions.append({
                    "points": pts,
                    "description": region.get("description", ""),
                })

    print(f"[VLM Adhesion] #{plant_id}: has_foreign={has_foreign}, "
          f"regions={len(valid_regions)}")

    return {
        "has_foreign": has_foreign,
        "regions": valid_regions,
        "raw_response": raw_response,
    }


def mask_adhesion_regions(crop_bgr: np.ndarray, regions: list) -> tuple:
    """
    Fill closed polygon regions with black to remove foreign plant parts.

    Args:
        crop_bgr: The crop image (BGR).
        regions: List of {"points": [[nx1,ny1], ...], "description": str}
                 in 0-1000 normalized coords.

    Returns:
        (masked_image, debug_image)
        - masked_image: crop with foreign regions filled black (for pod counting)
        - debug_image:  crop with foreign regions highlighted in red (for visualization)
    """
    h, w = crop_bgr.shape[:2]
    masked = crop_bgr.copy()
    debug = crop_bgr.copy()

    for region in regions:
        pts_norm = region["points"]
        # Normalize 0-1000 → pixel coords
        pts_px = []
        for nx, ny in pts_norm:
            px = int(max(0, min(w - 1, float(nx) / 1000.0 * w)))
            py = int(max(0, min(h - 1, float(ny) / 1000.0 * h)))
            pts_px.append([px, py])
        polygon = np.array(pts_px, dtype=np.int32)

        # Black fill on masked image (for processing)
        cv2.fillPoly(masked, [polygon], (0, 0, 0))
        # Semi-transparent red overlay + red border on debug image
        overlay = debug.copy()
        cv2.fillPoly(overlay, [polygon], (0, 0, 255))
        cv2.addWeighted(overlay, 0.4, debug, 0.6, 0, debug)
        cv2.polylines(debug, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)

    print(f"[Mask Adhesion] Filled {len(regions)} polygon regions with black")
    return masked, debug


def vlm_label_plants(image: np.ndarray) -> dict:
    """
    Send image to Doubao. Tell VLM the actual pixel size it sees.
    VLM returns bbox in those pixel coords. Scale back to original.
    """
    client = _get_client(phase="annotate")
    h, w = image.shape[:2]

    # Resize for API payload only
    vlm_img = _resize_for_vlm(image)
    b64 = _image_to_base64(vlm_img)

    prompt = """请你帮我给图片里面的每个单体植株部分打标签，同时检测黄色直尺和白色标签纸。

图片中油菜植株的各个部分被拆开后平铺在黑色背景上。

═══════════════════════════════════════
一、植株识别规则
═══════════════════════════════════════

1. 最长，枝叶（角果）最少的那根是"主干"
2. 离主干最近且枝叶最茂盛，角果数量最多的是"主枝"
3. 其余的都是"分枝"

⚠️ 重要：一个分枝/主枝可能本身就有多个大的侧枝分叉，只有这些分叉明确共享同一根连续主轴时，才视为同一个植株部件，**必须用一个bbox完整框住**，不要把同一个部件的不同分叉拆成多个bbox。

⚠️ 但如果两根植株部件只是因为摆放时接触、压叠、交叉、端点碰到一起，或者视觉上短暂连在一起，不能合并成一个 bbox。判断标准：
- 看到两条不同方向的主轴/中轴，且各自有独立角果排列 → 必须拆成两个 bbox
- 一根长而少角果的主干碰到或压住一根有角果的分枝/主枝 → 主干和分枝/主枝必须分开框
- bbox 不要为了包含接触点而横跨两个明显不同的植株主体
- 黑色背景上平铺的样本允许接触，但"接触/交叉"不等于"同一个植株部件"

请使用归一化坐标系来描述位置：
- 左上角 = (0, 0)
- 右下角 = (1000, 1000)
- x 从左到右 0→1000，y 从上到下 0→1000
- 图片正中心 = (500, 500)

请识别每个独立的植株部分，返回它的包围框，并为每个"主枝/分枝"标出该部件主轴的两个端点。
不要把同一根连续主轴上的小侧枝、角果串或局部末梢单独拆成多个 bbox；它们应该跟所属主枝/分枝合并在同一个 bbox 里。

⚠️ bbox 必须**完整包住**该植株部件的所有枝叶、角果，包括末梢和尖端，不能截断任何部分。宁可bbox稍大一点，也不要漏掉植株的边缘。

主轴端点规则：
- stem_start / stem_end 都使用整张原图的归一化坐标 0~1000，不是 bbox 内局部坐标。
- 对"主枝/分枝"：stem_start 是该枝条主轴靠近基部/较粗连接端的一端；stem_end 是沿该枝条主轴延伸到远端的一端。
- 主轴指承载角果的连续中心茎，不是旁边伸出的角果、侧枝或枯枝。不要把单根角果尖端当成主轴端点。
- 如果主轴弯曲，只需要给主轴两端，后续 CV 会沿骨架自动找路径。
- 对"主干"可以省略 stem_start/stem_end，或给主干两端。

- bbox: [左上角x, 左上角y, 右下角x, 右下角y]，数值范围 0~1000。
- stem_start/stem_end: [x,y]，数值范围 0~1000；如果确实无法判断可省略。
- label 只能是：主干、主枝、分枝
- 不要返回 description、reason、comment 等解释性字段。

═══════════════════════════════════════
二、黄色直尺
═══════════════════════════════════════

图片中可能有一把黄色直尺（黄色部分长度正好是 1 米 = 100 cm）。请检测并返回：
- ruler: 黄色直尺的包围框 bbox [x1,y1,x2,y2]（归一化 0~1000），要框住整个黄色尺身
- 如果图中没有黄色直尺，设 ruler 为 null
- 不需要给端点，后续会用 CV 精确测量

═══════════════════════════════════════
三、白色标签纸
═══════════════════════════════════════

图片中可能有一张白色标签纸，上面有 3 行字符/数字。请检测并返回：
- label_tag.bbox: 标签纸的包围框 [x1,y1,x2,y2]（归一化 0~1000）
- label_tag.is_flipped: 如果标签文字的方向与图片方向一致（正常可读）填 false；如果文字是倒置的（需要旋转 180° 才能正常阅读）填 true
- label_tag.rows: 按正常阅读方向（从上到下），每行从左到右读取的字符，共 3 个字符串。若某行看不到或读不出，填空字符串 ""
- 如果图中没有白色标签纸，设 label_tag 为 null

═══════════════════════════════════════
输出格式
═══════════════════════════════════════

严格按以下 JSON 格式返回，不要输出其他文字。请输出紧凑单行 JSON，不要缩进：
```json
{
    "plants": [
        {
            "id": 1,
            "label": "主干/主枝/分枝",
            "bbox": [x1, y1, x2, y2],
            "stem_start": [x, y],
            "stem_end": [x, y]
        }
    ],
    "ruler": {"bbox": [x1, y1, x2, y2]},
    "label_tag": {"bbox": [x1, y1, x2, y2], "is_flipped": false, "rows": ["xx", "yy", "zz"]}
}
```"""

    print(f"[VLM] Original {w}x{h}, using 0-1000 normalized coords...")
    raw_response = ""
    api_error = None
    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            }
                        }
                    ]
                }
            ],
            max_tokens=8000,
            temperature=0.1,
        )
        raw_response = response.choices[0].message.content.strip()
        usage = getattr(response, 'usage', None)
        if usage:
            print(f"[VLM] Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
        print(f"[VLM] Response ({len(raw_response)} chars):\n{raw_response[:800]}")
    except Exception as e:
        # 这里是把「API 失败」和「模型真没识别到」区分开的唯一位置：
        # raw_response 一旦被压成字符串，下游只能看到一句笼统的解析失败。
        api_error = e
        trace.event(
            "vlm_annotate_failed",
            phase="annotate",
            error_type=type(e).__name__,
            error_message=str(e)[:500],
            http_status=getattr(e, "status_code", None),
        )
        raw_response = f"Doubao API error: {e}"
        print(f"[VLM] {raw_response}")

    if api_error is not None:
        # 不能去解析错误信息。API 的报错原文里可能带方括号，例如
        # "...Your account [2122781975] has reached..."，_extract_json_array
        # 会把它当成 plants 数组解析成 [2122781975]，于是 plants 非空、
        # 绕过下游的空值守卫，最终在 p.get("bbox") 处抛 AttributeError。
        return {
            "labeled_image": image.copy(),
            "crops": {},
            "plants": [],
            "raw_response": raw_response,
            "parse_error": True,
            "partial_json": False,
            "api_error": type(api_error).__name__,
            "api_error_message": str(api_error)[:500],
            "api_status": getattr(api_error, "status_code", None),
        }

    # Parse
    parsed = _parse_json(raw_response)
    ruler = None
    label_tag = None
    sample_id = ""

    if isinstance(parsed, dict):
        plants = parsed.get("plants", [])
        ruler = parsed.get("ruler")
        label_tag = parsed.get("label_tag")
    elif isinstance(parsed, list):
        # Backward compat: old array-only format
        plants = parsed
    else:
        plants = None

    # 只保留真正的部件字典。模型偶尔会返回 [1,2,3] 这类数组，
    # 下游按 dict 用（p.get）会直接崩。
    if isinstance(plants, list):
        plants = [p for p in plants if isinstance(p, dict)]

    partial_json = bool(parsed and _extract_json_array(raw_response) is None)
    if not plants or not isinstance(plants, list):
        return {
            "labeled_image": image.copy(),
            "crops": {},
            "plants": [],
            "raw_response": raw_response,
            "parse_error": True,
            "partial_json": False,
        }

    # Normalize ruler / label_tag
    if isinstance(ruler, dict) and "bbox" in ruler and len(ruler["bbox"]) == 4:
        ruler["found"] = True
    else:
        ruler = {"found": False}

    if isinstance(label_tag, dict) and "bbox" in label_tag and len(label_tag["bbox"]) == 4:
        label_tag["found"] = True
        rows = label_tag.get("rows", [])
        if isinstance(rows, list):
            rows = [str(r).strip() for r in rows[:3]]
            # Drop empty trailing rows
            while rows and rows[-1] == "":
                rows.pop()
        else:
            rows = []
        label_tag["rows"] = rows
        label_tag["is_flipped"] = bool(label_tag.get("is_flipped", False))
        if label_tag["is_flipped"]:
            rows.reverse()
            label_tag["rows"] = rows
        sample_id = "-".join(rows) if rows else ""
    else:
        label_tag = {"found": False, "rows": ["", "", ""], "is_flipped": False}

    # 0-1000 normalized coord → original pixel coord
    labeled = image.copy()
    crops = {}  # id -> cropped image
    for i, p in enumerate(plants):
        bbox = p.get("bbox", [])
        if len(bbox) != 4:
            continue
        nx1, ny1, nx2, ny2 = [float(v) for v in bbox]
        # Raw pixel coords with 2% padding to ensure complete plant coverage
        rx1 = nx1 / 1000.0 * w
        ry1 = ny1 / 1000.0 * h
        rx2 = nx2 / 1000.0 * w
        ry2 = ny2 / 1000.0 * h
        pad_x = (rx2 - rx1) * 0.02
        pad_y = (ry2 - ry1) * 0.02
        ox1 = max(0, int(rx1 - pad_x))
        oy1 = max(0, int(ry1 - pad_y))
        ox2 = min(w, int(rx2 + pad_x))
        oy2 = min(h, int(ry2 + pad_y))

        # Store pixel bbox for downstream use
        p["pixel_bbox"] = [ox1, oy1, ox2, oy2]

        # Convert stem endpoints (0-1000 norm) → full-image px, then crop-local px
        def _norm_to_px(pt):
            if not pt or len(pt) != 2:
                return None
            fx = max(0, min(w - 1, int(float(pt[0]) / 1000.0 * w)))
            fy = max(0, min(h - 1, int(float(pt[1]) / 1000.0 * h)))
            return (fx, fy)

        s_full = _norm_to_px(p.get("stem_start"))
        e_full = _norm_to_px(p.get("stem_end"))
        if s_full and e_full:
            p["stem_start_px"] = list(s_full)
            p["stem_end_px"] = list(e_full)
            # Crop-local: clamp into bbox then subtract origin
            def _to_local(pt):
                lx = max(ox1, min(ox2 - 1, pt[0])) - ox1
                ly = max(oy1, min(oy2 - 1, pt[1])) - oy1
                return (lx, ly)
            p["stem_start_local"] = list(_to_local(s_full))
            p["stem_end_local"] = list(_to_local(e_full))

        # Crop from original
        if ox2 > ox1 and oy2 > oy1:
            crop = image[oy1:oy2, ox1:ox2].copy()
            pid = p.get('id', i + 1)
            crops[pid] = crop

        color = _COLORS[i % len(_COLORS)]
        label = f"#{p.get('id', i+1)} {p.get('label', '?')}"

        lw = max(3, min(h, w) // 1000)
        cv2.rectangle(labeled, (ox1, oy1), (ox2, oy2), color, lw)

        fs = max(0.9, min(h, w) / 2000)
        th_t = max(2, int(fs * 2.5))
        (tw, thl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th_t)
        cv2.rectangle(labeled, (ox1, oy1 - thl - 14), (ox1 + tw + 8, oy1), color, -1)
        cv2.putText(labeled, label, (ox1 + 4, oy1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th_t)

        print(f"  #{p.get('id')}: {p.get('label')} norm({nx1},{ny1})-({nx2},{ny2}) "
              f"→ px ({ox1},{oy1})-({ox2},{oy2})")

    return {
        "labeled_image": labeled,
        "crops": crops,
        "plants": plants,
        "raw_response": raw_response,
        "partial_json": partial_json,
        "ruler": ruler,
        "label_tag": label_tag,
        "sample_id": sample_id,
    }


# ─────────────────────────────────────────────────────────────────────────
# Crop verification: send original + crop + metadata back to VLM,
# ask it to verify if each crop is a complete, single, independent plant.
# ─────────────────────────────────────────────────────────────────────────

def vlm_verify_crop(original_image: np.ndarray,
                     crop_image: np.ndarray,
                     plant: dict,
                     all_plants: list) -> dict:
    """
    Ask VLM to verify the quality of a single crop.

    Args:
        original_image: full original image (BGR)
        crop_image: the cropped region (BGR)
        plant: dict with id, label, bbox (0-1000 normalized)
        all_plants: list of all plants for context

    Returns:
        {
          "action": "keep" | "delete" | "recrop",
          "reason": "...",
          "new_bbox": [x1,y1,x2,y2]   # only if action=="recrop", in 0-1000 norm
        }
    """
    client = _get_client(phase="verify_crop")
    pid = plant.get("id", "?")
    label = plant.get("label", "")
    bbox = plant.get("bbox", [])

    # Build a concise list of other plants for context
    others = []
    for p in all_plants:
        if p.get("id") == pid:
            continue
        others.append(f"#{p.get('id')} {p.get('label','?')} bbox={p.get('bbox',[])}")
    others_str = "\n".join(others) if others else "（无其他植株）"

    # Resize both for API payload
    orig_resized = _resize_for_vlm(original_image)
    crop_resized = _resize_for_vlm(crop_image, max_dim=1024)
    b64_orig = _image_to_base64(orig_resized)
    b64_crop = _image_to_base64(crop_resized)

    prompt = f"""你是植物表型分析助手。下面有两张图：
- 第一张：**原图**（已划定多个植株部件，每个都用彩色框标注）
- 第二张：**当前要审核的截图** — 来自第一张图中 #{pid} {label} 的 bbox 区域

当前植株信息：
- ID: #{pid}
- 标签: {label}
- bbox（归一化 0-1000）: {bbox}

原图中其他植株部件：
{others_str}

⚠️ 请仔细审核第二张截图，回答以下问题：
1. 截图中是否是**一个完整且独立的单体植株**？（即一根完整的主干、主枝或分枝，从头到尾）
2. 截图是否**截全了**？还是有枝叶/角果被切到框外？
3. 截图是否**只是包含两个植株中间的间隔区域**或者**多个植株的杂乱混合**（即不是一个独立完整的植株主体）？
4. 截图是否把两根只是接触/交叉/压叠的不同植株部件合进了一个框？如果能通过缩小 bbox 保留当前主体、排除另一根植株，则选择 recrop；如果无法确定当前主体，选择 delete。

根据审核结果，给出以下三种处理方案之一：

**方案 A — keep（正常）**：截图是一个完整、独立、截全的单体植株，无需调整。

**方案 B — delete（删除）**：截图不是独立完整的植株主体，比如：
  - 是多个植株的枝叶杂乱集合
  - 主要是空白/间隔区域，没有主体植株
  - 与其他 bbox 严重重叠重复

**方案 C — recrop（重新截）**：截图确实是一个独立完整的单体植株，但 bbox 截得不全，有部分枝叶/角果被切到框外。请给出新的 bbox（归一化 0-1000）以完整框住该植株。
也可以在当前 bbox 过大、把接触/交叉的另一根植株部件包含进来时使用 recrop：新的 bbox 应只框住当前 #{pid} {label} 主体，排除无关植株。

⚠️ 严格按以下 JSON 返回，不要其他文字：
```json
{{
  "action": "keep",
  "reason": "简短说明"
}}
```
或
```json
{{
  "action": "delete",
  "reason": "简短说明"
}}
```
或
```json
{{
  "action": "recrop",
  "reason": "简短说明",
  "new_bbox": [x1, y1, x2, y2]
}}
```

注意：
- new_bbox 是相对于**原图**的归一化坐标（0-1000）
- 宁可保守保留（keep），也不要轻易 delete 真正的植株
- 只有非常确信是杂乱混合或纯空白时才 delete"""

    print(f"[VLM Verify] Checking #{pid} {label}...")
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
                         "image_url": {"url": f"data:image/jpeg;base64,{b64_orig}"}},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64_crop}"}},
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        print(f"[VLM Verify] #{pid} response: {raw[:300]}")
    except Exception as e:
        print(f"[VLM Verify] #{pid} error: {e}")
        return {"action": "keep", "reason": f"verify failed: {e}"}

    # Parse — _parse_json returns either a list or dict
    parsed = _parse_json(raw)
    if isinstance(parsed, list) and parsed:
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        return {"action": "keep", "reason": "parse failed, default keep"}

    action = parsed.get("action", "keep")
    if action not in ("keep", "delete", "recrop"):
        action = "keep"

    result = {"action": action, "reason": parsed.get("reason", "")}
    if action == "recrop":
        nb = parsed.get("new_bbox", [])
        if isinstance(nb, list) and len(nb) == 4:
            result["new_bbox"] = [float(v) for v in nb]
        else:
            # Fallback: invalid new_bbox, downgrade to keep
            print(f"[VLM Verify] #{pid} recrop missing valid new_bbox, falling back to keep")
            result["action"] = "keep"
            result["reason"] = "recrop requested but new_bbox invalid"
    return result


def apply_recrop(original_image: np.ndarray,
                 plant: dict,
                 new_bbox_norm: list,
                 padding: float = 0.02) -> tuple:
    """
    Re-crop the original image using a new normalized bbox (0-1000).

    Returns: (new_crop, new_pixel_bbox)
    """
    h, w = original_image.shape[:2]
    nx1, ny1, nx2, ny2 = [float(v) for v in new_bbox_norm]
    rx1 = nx1 / 1000.0 * w
    ry1 = ny1 / 1000.0 * h
    rx2 = nx2 / 1000.0 * w
    ry2 = ny2 / 1000.0 * h
    pad_x = (rx2 - rx1) * padding
    pad_y = (ry2 - ry1) * padding
    ox1 = max(0, int(rx1 - pad_x))
    oy1 = max(0, int(ry1 - pad_y))
    ox2 = min(w, int(rx2 + pad_x))
    oy2 = min(h, int(ry2 + pad_y))
    new_crop = original_image[oy1:oy2, ox1:ox2].copy()
    new_pixel_bbox = [ox1, oy1, ox2, oy2]
    # Update plant bbox/pixel_bbox
    plant["bbox"] = [nx1, ny1, nx2, ny2]
    plant["pixel_bbox"] = new_pixel_bbox
    return new_crop, new_pixel_bbox


# ─────────────────────────────────────────────────────────────────────────
# Yellow ruler measurement via CV (HSV color segmentation)
# ─────────────────────────────────────────────────────────────────────────

def measure_ruler_yellow(ruler_crop: np.ndarray) -> dict:
    """
    Detect yellow portion of the ruler using HSV segmentation and return
    pixel length of the yellow segment.

    The yellow ruler portion is 1 meter (100 cm). Returns px_per_cm.

    Args:
        ruler_crop: BGR image crop containing the ruler (from VLM bbox).

    Returns:
        {"px_per_cm": float, "yellow_length_px": float, "detected": bool,
         "debug_image": np.ndarray}
    """
    if ruler_crop is None or ruler_crop.size == 0:
        return {"px_per_cm": 0.0, "yellow_length_px": 0.0,
                "detected": False, "debug_image": None}

    hsv = cv2.cvtColor(ruler_crop, cv2.COLOR_BGR2HSV)

    # Yellow hue range (reused from deleted ruler.py)
    lower_yellow = np.array([15, 80, 120])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Morphology: close small gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, k, iterations=1)

    # Find largest yellow connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        yellow_mask, connectivity=8)
    if n_labels <= 1:
        return {"px_per_cm": 0.0, "yellow_length_px": 0.0,
                "detected": False, "debug_image": cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)}

    # Skip background label 0
    largest_label = 1
    largest_area = stats[1, cv2.CC_STAT_AREA]
    for lbl in range(2, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest_label = lbl

    yellow_region = (labels == largest_label).astype(np.uint8) * 255

    # Find contours of the yellow region
    contours, _ = cv2.findContours(yellow_region, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"px_per_cm": 0.0, "yellow_length_px": 0.0,
                "detected": False, "debug_image": cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)}

    # Merge all contours
    all_pts = np.vstack(contours)

    # Fit rotated rectangle to get the ruler axis
    rect = cv2.minAreaRect(all_pts)
    (cx, cy), (rw, rh), angle = rect

    # The longer edge is the ruler length
    yellow_length_px = max(rw, rh)

    # 1 meter = 100 cm
    px_per_cm = yellow_length_px / 100.0 if yellow_length_px > 0 else 0.0

    # Debug: draw minAreaRect and endpoints
    debug = ruler_crop.copy()
    box = cv2.boxPoints(rect)
    box = np.int32(box)  # numpy 2 compat: np.intp
    cv2.drawContours(debug, [box], 0, (0, 255, 0), 2)

    # Draw the two endpoints of the longer axis
    if rw >= rh:
        # Width is the longer side — endpoints along the width axis
        angle_rad = np.deg2rad(angle)
        dx = (rw / 2) * np.cos(angle_rad)
        dy = (rw / 2) * np.sin(angle_rad)
    else:
        # Height is longer — endpoints along height axis (perpendicular to width)
        angle_rad = np.deg2rad(angle + 90)
        dx = (rh / 2) * np.cos(angle_rad)
        dy = (rh / 2) * np.sin(angle_rad)

    p1 = (int(cx + dx), int(cy + dy))
    p2 = (int(cx - dx), int(cy - dy))
    cv2.circle(debug, p1, 5, (0, 0, 255), -1)
    cv2.circle(debug, p2, 5, (255, 0, 0), -1)
    cv2.line(debug, p1, p2, (255, 255, 0), 2)

    return {
        "px_per_cm": px_per_cm,
        "yellow_length_px": yellow_length_px,
        "detected": True,
        "debug_image": debug,
    }
