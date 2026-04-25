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

from app.config import ARK_API_KEY, ARK_API_BASE, VLM_MODEL

_COLORS = [
    (0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 165, 255),
    (255, 0, 255), (0, 255, 255), (255, 255, 0),
    (128, 0, 255), (255, 128, 0), (0, 128, 255),
]

# Row labels: A, B, C, ... ; Col labels: 1, 2, 3, ...
_ROW_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _get_client() -> OpenAI:
    return OpenAI(api_key=ARK_API_KEY, base_url=ARK_API_BASE)


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


def _parse_json(content: str):
    """Extract JSON array from VLM response."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"[VLM] JSON parse failed: {content[:300]}")
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


def vlm_label_plants(image: np.ndarray) -> dict:
    """
    Send image to Doubao. Tell VLM the actual pixel size it sees.
    VLM returns bbox in those pixel coords. Scale back to original.
    """
    client = _get_client()
    h, w = image.shape[:2]

    # Resize for API payload only
    vlm_img = _resize_for_vlm(image)
    b64 = _image_to_base64(vlm_img)

    prompt = """请你帮我给图片里面的每个单体植株部分打一个标签。

图片中油菜植株的各个部分被拆开后平铺在黑色背景上。

识别规则：
1. 最长，枝叶（角果）最少的那根是"主干"
2. 离主干最近且枝叶最茂盛的是"主枝"
3. 其余的都是"分枝"

请使用归一化坐标系来描述位置：
- 左上角 = (0, 0)
- 右下角 = (1000, 1000)
- x 从左到右 0→1000，y 从上到下 0→1000
- 图片正中心 = (500, 500)

请识别每个独立的植株部分，返回它的包围框。

严格按以下JSON格式返回，不要输出其他文字：
```json
[
    {
        "id": 1,
        "label": "主干/主枝/分枝",
        "bbox": [x1, y1, x2, y2],
        "description": "简短描述"
    }
]
```

- bbox: [左上角x, 左上角y, 右下角x, 右下角y]，数值范围 0~1000。
- label 只能是：主干、主枝、分枝
- 忽略卡尺、标签纸等非植物物体。"""

    print(f"[VLM] Original {w}x{h}, using 0-1000 normalized coords...")
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
            max_tokens=3000,
            temperature=0.1,
        )
        raw_response = response.choices[0].message.content.strip()
        usage = getattr(response, 'usage', None)
        if usage:
            print(f"[VLM] Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
        print(f"[VLM] Response ({len(raw_response)} chars):\n{raw_response[:800]}")
    except Exception as e:
        raw_response = f"Doubao API error: {e}"
        print(f"[VLM] {raw_response}")

    # Parse
    plants = _parse_json(raw_response)
    if not plants or not isinstance(plants, list):
        return {
            "labeled_image": image.copy(),
            "plants": [],
            "raw_response": raw_response,
        }

    # 0-1000 normalized coord → original pixel coord
    labeled = image.copy()
    crops = {}  # id -> cropped image
    for i, p in enumerate(plants):
        bbox = p.get("bbox", [])
        if len(bbox) != 4:
            continue
        nx1, ny1, nx2, ny2 = [float(v) for v in bbox]
        # Raw pixel coords
        rx1 = nx1 / 1000.0 * w
        ry1 = ny1 / 1000.0 * h
        rx2 = nx2 / 1000.0 * w
        ry2 = ny2 / 1000.0 * h
        # Add 10% padding to compensate for VLM bbox imprecision
        pad_x = (rx2 - rx1) * 0.10
        pad_y = (ry2 - ry1) * 0.10
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
    }
