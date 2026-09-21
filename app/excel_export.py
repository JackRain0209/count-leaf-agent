"""考种结果 Excel 导出模块。实现 REQUIREMENTS.md 的 R3/R4/R5/R6。"""
from __future__ import annotations

import json
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Iterable

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


SUCCESS = "成功"
FAILED_PREFIX = "失败"

_ERROR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Cannot read image|无法读取.*图片|读取.*失败", re.I), "读取失败"),
    (re.compile(r"没有返回任何|结构识别没有返回|VLM.*空|空返回", re.I), "VLM 空返回"),
    (re.compile(r"没有任何有效\s*bbox|无.*bbox", re.I), "VLM 无有效框"),
    (re.compile(r"timeout|timed out|超时", re.I), "VLM 超时"),
    (re.compile(r"429|rate.?limit|too many requests|限流", re.I), "接口限流"),
    (re.compile(r"connect|connection|network|网络", re.I), "网络异常"),
]


def classify_error(error_message: str | None) -> str:
    if not error_message:
        return "未知原因"
    for pattern, label in _ERROR_PATTERNS:
        if pattern.search(error_message):
            return label
    return "内部异常"

# ========== R5 低质量判定（可调阈值集中在这里） ==========
QUALITY_NORMAL = "正常"
QUALITY_FAILED = "失败"

# 低质量判定阈值，可按实际数据调整
QUALITY_MAX_MIN_PLANTS = 1        # 部件数 <= 此值 → 可疑
QUALITY_REQUIRE_BRANCH = True     # 分枝数为 0 → 可疑
QUALITY_FLAG_ZERO_PODS = True     # 总角果数为 0 → 可疑
QUALITY_FLAG_MISSING_STEM = False # 主干长度为 空 → 可疑（默认关闭，因为没标尺时常见）


def assess_quality(record: dict[str, Any]) -> str:
    """返回数据质量：正常 / 可疑：xxx / 失败。

    只在 status=completed 时判定；失败记录直接返回"失败"。
    """
    if _is_failed(record):
        return QUALITY_FAILED

    result = _parse_result(record)
    total = result.get("total_pods")
    plant_count = result.get("plant_count")
    branch_count = result.get("branch_count")
    stem_len = result.get("main_stem_length_cm")

    if QUALITY_FLAG_ZERO_PODS and total == 0:
        return "可疑：角果数为 0"
    if plant_count is not None and plant_count <= QUALITY_MAX_MIN_PLANTS:
        return "可疑：部件数过少"
    if QUALITY_REQUIRE_BRANCH and branch_count == 0:
        return "可疑：无分枝"
    if QUALITY_FLAG_MISSING_STEM and stem_len is None:
        return "可疑：无主干长度"
    return QUALITY_NORMAL


SUMMARY_HEADERS = [
    "图片编号", "分枝数", "主枝角果数", "分枝角果数",
    "总角果数", "主干长度(cm)", "状态", "质量", "是否修订",
]

DETAIL_HEADERS = ["图片编号", "部件编号", "部件类型", "角果数"]


def _parse_result(record: dict[str, Any]) -> dict[str, Any]:
    """取当前结果（含人工修订）。兼容 _row_to_detail 与原始行两种形态。"""
    result = record.get("result")
    if isinstance(result, dict):
        return result
    raw = record.get("current_result_json") or record.get("original_result_json") or {}
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _get_label(record: dict[str, Any]) -> str:
    """编号：优先 sample_id（图上标签），回退文件名（去掉路径和扩展名），再回退 run_id。"""
    result = _parse_result(record)
    sample = (record.get("sample_id") or result.get("sample_id") or "").strip()
    if sample:
        return sample

    source = (record.get("source_label") or "").strip()
    # 去掉路径前缀（正反斜杠都处理）
    if "/" in source:
        source = source.rsplit("/", 1)[-1]
    if "\\" in source:
        source = source.rsplit("\\", 1)[-1]
    # 去掉扩展名
    stem = source.rsplit(".", 1)[0] if "." in source else source

    return stem or (record.get("run_id") or "").strip()
def _is_failed(record: dict[str, Any]) -> bool:
    return (record.get("status") or "").lower() in {"failed", "error"}


def _sort_key(record: dict[str, Any]) -> str:
    return record.get("completed_at") or record.get("created_at") or ""


def dedup_by_label(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Q7 默认方案：同编号保留最新一条。"""
    ordered = sorted(records, key=_sort_key)
    latest: dict[str, dict[str, Any]] = {}
    for r in ordered:
        latest[_get_label(r)] = r
    return [latest[k] for k in sorted(latest.keys())]


def build_summary_row(record: dict[str, Any]) -> list[Any]:
    result = _parse_result(record)
    label = _get_label(record)
    quality = assess_quality(record)

    if _is_failed(record):
        reason = classify_error(record.get("error_message"))
        return [label, None, None, None, None, None,
                f"{FAILED_PREFIX}：{reason}", quality, ""]

    return [
        label,
        result.get("branch_count"),
        result.get("main_inflorescence_pod_count"),
        result.get("branch_pod_count"),
        result.get("total_pods"),
        result.get("main_stem_length_cm"),
        SUCCESS,
        quality,
        "是" if record.get("is_modified") else "否",
    ]

def build_detail_rows(record: dict[str, Any]) -> list[list[Any]]:
    result = _parse_result(record)
    label = _get_label(record)
    rows: list[list[Any]] = []
    for plant in result.get("plants") or []:
        rows.append([
            label,
            plant.get("id"),
            plant.get("label"),
            plant.get("pod_count"),
        ])
    return rows


_HEADER_FILL = PatternFill("solid", fgColor="D9E1F2")
_HEADER_FONT = Font(bold=True, name="微软雅黑", size=11)
_CELL_FONT = Font(name="微软雅黑", size=10)
_CENTER = Alignment(horizontal="center", vertical="center")
_THIN = Side(style="thin", color="BFBFBF")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


def _write_sheet(ws, headers: list[str], rows: list[list[Any]]) -> None:
    ws.append(headers)
    for c in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=c)
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = _CENTER
        cell.border = _BORDER

    for row in rows:
        ws.append(row)

    for r in range(2, ws.max_row + 1):
        for c in range(1, len(headers) + 1):
            cell = ws.cell(row=r, column=c)
            cell.font = _CELL_FONT
            cell.alignment = _CENTER
            cell.border = _BORDER

    for c in range(1, len(headers) + 1):
        max_len = len(str(headers[c - 1]))
        for r in range(2, ws.max_row + 1):
            v = ws.cell(row=r, column=c).value
            if v is not None:
                max_len = max(max_len, len(str(v)))
        ws.column_dimensions[get_column_letter(c)].width = min(max(max_len + 4, 10), 40)

    ws.freeze_panes = "A2"


def build_workbook(records: Iterable[dict[str, Any]]) -> Workbook:
    records = dedup_by_label(list(records))
    wb = Workbook()

    ws_summary = wb.active
    ws_summary.title = "汇总"
    _write_sheet(ws_summary, SUMMARY_HEADERS, [build_summary_row(r) for r in records])

    ws_detail = wb.create_sheet("部件明细")
    detail_rows: list[list[Any]] = []
    for r in records:
        detail_rows.extend(build_detail_rows(r))
    _write_sheet(ws_detail, DETAIL_HEADERS, detail_rows)

    return wb


def workbook_to_bytes(wb: Workbook) -> bytes:
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def make_filename(now: datetime | None = None, tag: str = "") -> str:
    now = now or datetime.now()
    ts = now.strftime("%Y%m%d_%H%M")
    suffix = f"_{tag}" if tag else ""
    return f"考种汇总{suffix}_{ts}.xlsx"