"""SQLite-backed history storage and manual result revision helpers."""

from __future__ import annotations

import copy
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ALLOWED_LABELS = {"主干", "主枝", "分枝"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} 必须是非负整数")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是非负整数") from exc
    if number < 0:
        raise ValueError(f"{field} 必须是非负整数")
    if number > 100_000:
        raise ValueError(f"{field} 不能超过 100000")
    return number


def recalculate_result(result: dict[str, Any]) -> dict[str, Any]:
    """Rebuild all derived counts from the current component list."""
    updated = copy.deepcopy(result)
    plants = updated.get("plants") or []

    trunk_pods = sum(
        _as_non_negative_int(p.get("pod_count", 0), "pod_count")
        for p in plants
        if p.get("label") == "主干"
    )
    main_pods = sum(
        _as_non_negative_int(p.get("pod_count", 0), "pod_count")
        for p in plants
        if p.get("label") == "主枝"
    )
    branch_plants = [p for p in plants if p.get("label") == "分枝"]
    branch_pods = sum(
        _as_non_negative_int(p.get("pod_count", 0), "pod_count")
        for p in branch_plants
    )
    total_pods = trunk_pods + main_pods + branch_pods
    branch_breakdown = [
        {
            "id": p.get("id"),
            "label": p.get("label"),
            "pod_count": _as_non_negative_int(p.get("pod_count", 0), "pod_count"),
        }
        for p in branch_plants
    ]

    updated["branch_count"] = len(branch_plants)
    updated["main_inflorescence_pod_count"] = main_pods
    updated["main_stem_plus_main_pod_count"] = trunk_pods + main_pods
    updated["branch_pod_count"] = branch_pods
    updated["branch_total_pod_count"] = branch_pods
    updated["total_pods"] = total_pods

    summary = copy.deepcopy(updated.get("summary") or {})
    summary.update({
        "branch_count": len(branch_plants),
        "predicted_branch_count": len(branch_plants),
        "main_inflorescence_pod_count": main_pods,
        "main_stem_plus_main_pod_count": trunk_pods + main_pods,
        "branch_pod_count": branch_pods,
        "branch_total_pod_count": branch_pods,
        "total_pod_count": total_pods,
        "branch_pod_breakdown": branch_breakdown,
    })
    updated["summary"] = summary
    return updated


def apply_manual_edits(
    current_result: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Validate an editor payload while retaining non-editable analysis metadata."""
    if not isinstance(payload, dict):
        raise ValueError("修改内容格式错误")
    incoming_plants = payload.get("plants")
    if not isinstance(incoming_plants, list):
        raise ValueError("plants 必须是数组")
    if len(incoming_plants) > 500:
        raise ValueError("部件数量不能超过 500")

    existing = {
        str(p.get("id")): p
        for p in current_result.get("plants", [])
        if p.get("id") is not None
    }
    seen_ids: set[str] = set()
    next_id = 1
    numeric_ids = [
        int(p.get("id"))
        for p in current_result.get("plants", [])
        if str(p.get("id", "")).isdigit()
    ]
    if numeric_ids:
        next_id = max(numeric_ids) + 1

    plants: list[dict[str, Any]] = []
    for index, incoming in enumerate(incoming_plants, 1):
        if not isinstance(incoming, dict):
            raise ValueError(f"第 {index} 个部件格式错误")
        plant_id = incoming.get("id")
        if plant_id in (None, ""):
            while str(next_id) in seen_ids or str(next_id) in existing:
                next_id += 1
            plant_id = next_id
            next_id += 1
        id_key = str(plant_id)
        if id_key in seen_ids:
            raise ValueError(f"部件编号重复：{plant_id}")
        seen_ids.add(id_key)

        label = str(incoming.get("label", "")).strip()
        if label not in ALLOWED_LABELS:
            raise ValueError(f"不支持的部件类型：{label}")
        pod_count = _as_non_negative_int(incoming.get("pod_count", 0), "角果数量")

        base = copy.deepcopy(existing.get(id_key) or {})
        base.update({
            "id": plant_id,
            "label": label,
            "pod_count": pod_count,
        })
        if id_key not in existing:
            base["manual"] = True
            base["description"] = "用户新增"
        plants.append(base)

    updated = copy.deepcopy(current_result)
    updated["plants"] = plants
    updated["plant_count"] = _as_non_negative_int(
        payload.get("plant_count", len(plants)),
        "植株数量",
    )
    updated["manual_edited_at"] = _now_iso()
    return recalculate_result(updated)


class HistoryStore:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    run_id TEXT PRIMARY KEY,
                    source_label TEXT NOT NULL,
                    source_path TEXT,
                    source_kind TEXT NOT NULL DEFAULT 'upload',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    error_message TEXT,
                    original_result_json TEXT,
                    current_result_json TEXT,
                    steps_json TEXT NOT NULL DEFAULT '[]',
                    revision INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_analysis_history_created_at
                ON analysis_history(created_at DESC);
                """
            )

    def start(
        self,
        run_id: str,
        source_label: str,
        source_path: str,
        source_kind: str = "upload",
    ) -> None:
        now = _now_iso()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO analysis_history (
                    run_id, source_label, source_path, source_kind,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'processing', ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    source_label=excluded.source_label,
                    source_path=excluded.source_path,
                    source_kind=excluded.source_kind,
                    status='processing',
                    updated_at=excluded.updated_at,
                    error_message=NULL
                """,
                (run_id, source_label, source_path, source_kind, now, now),
            )

    def complete(
        self,
        run_id: str,
        result: dict[str, Any],
        steps: list[dict[str, Any]],
    ) -> dict[str, Any]:
        normalized = recalculate_result(result)
        now = _now_iso()
        result_json = json.dumps(normalized, ensure_ascii=False, default=str)
        steps_json = json.dumps(steps, ensure_ascii=False, default=str)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE analysis_history SET
                    status='completed', completed_at=?, updated_at=?,
                    error_message=NULL, original_result_json=?,
                    current_result_json=?, steps_json=?, revision=0
                WHERE run_id=?
                """,
                (now, now, result_json, result_json, steps_json, run_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"历史记录不存在：{run_id}")
        return normalized

    def fail(
        self,
        run_id: str,
        message: str,
        steps: list[dict[str, Any]] | None = None,
    ) -> None:
        now = _now_iso()
        steps_json = json.dumps(steps or [], ensure_ascii=False, default=str)
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE analysis_history SET
                    status='failed', error_message=?, updated_at=?, steps_json=?
                WHERE run_id=? AND status='processing'
                """,
                (message, now, steps_json, run_id),
            )

    def fail_stale_processing(
        self,
        message: str = "分析中断（连接终止或服务重启）",
    ) -> int:
        """把残留的 processing 记录标记为失败，返回处理条数。

        客户端断开时 SSE 生成器是被「抛弃」而不是「关闭」的：finally 必须等到
        生成器被回收才执行，时机不确定；服务重启则完全不会执行。
        所以启动时扫一遍，避免历史里留下永远处于「进行中」的记录。
        """
        now = _now_iso()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE analysis_history SET
                    status='failed', error_message=?, updated_at=?
                WHERE status='processing'
                """,
                (message, now),
            )
            return cursor.rowcount

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM analysis_history WHERE run_id=?",
                (run_id,),
            ).fetchone()
        return self._row_to_detail(row) if row else None

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        query: str = "",
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), 500))
        offset = max(0, int(offset))
        where = ""
        params: list[Any] = []
        if query.strip():
            where = "WHERE source_label LIKE ?"
            params.append(f"%{query.strip()}%")

        with self._connection() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM analysis_history {where}",
                params,
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT * FROM analysis_history {where}
                ORDER BY created_at DESC LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
        return {
            "count": total,
            "items": [self._row_to_summary(row) for row in rows],
        }

    def update(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM analysis_history WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if not row or not row["current_result_json"]:
                raise KeyError(f"可编辑结果不存在：{run_id}")
            current = json.loads(row["current_result_json"])
            updated = apply_manual_edits(current, payload)
            revision = int(row["revision"] or 0) + 1
            updated["manual_revision"] = revision
            now = _now_iso()
            connection.execute(
                """
                UPDATE analysis_history SET
                    current_result_json=?, revision=?, updated_at=?
                WHERE run_id=?
                """,
                (
                    json.dumps(updated, ensure_ascii=False, default=str),
                    revision,
                    now,
                    run_id,
                ),
            )
        return updated

    def restore(self, run_id: str) -> dict[str, Any]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM analysis_history WHERE run_id=?",
                (run_id,),
            ).fetchone()
            if not row or not row["original_result_json"]:
                raise KeyError(f"原始结果不存在：{run_id}")
            restored = json.loads(row["original_result_json"])
            revision = int(row["revision"] or 0) + 1
            now = _now_iso()
            connection.execute(
                """
                UPDATE analysis_history SET
                    current_result_json=?, revision=?, updated_at=?
                WHERE run_id=?
                """,
                (
                    json.dumps(restored, ensure_ascii=False, default=str),
                    revision,
                    now,
                    run_id,
                ),
            )
        return restored

    def _row_to_summary(self, row: sqlite3.Row) -> dict[str, Any]:
        current = json.loads(row["current_result_json"]) if row["current_result_json"] else {}
        original = row["original_result_json"] or ""
        current_json = row["current_result_json"] or ""
        return {
            "run_id": row["run_id"],
            "source_label": row["source_label"],
            "source_kind": row["source_kind"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
            "error_message": row["error_message"],
            "revision": row["revision"],
            "is_modified": bool(original and current_json and original != current_json),
            "thumbnail_url": f"/results/{row['run_id']}/step_01.png",
            "sample_id": current.get("sample_id", ""),
            "plant_count": current.get("plant_count"),
            "branch_count": current.get("branch_count"),
            "total_pods": current.get("total_pods"),
            "processing_time_s": current.get("processing_time_s"),
        }

    def _row_to_detail(self, row: sqlite3.Row) -> dict[str, Any]:
        detail = self._row_to_summary(row)
        detail.update({
            "source_path": row["source_path"],
            "result": (
                json.loads(row["current_result_json"])
                if row["current_result_json"] else None
            ),
            "original_result": (
                json.loads(row["original_result_json"])
                if row["original_result_json"] else None
            ),
            "steps": json.loads(row["steps_json"] or "[]"),
        })
        return detail
