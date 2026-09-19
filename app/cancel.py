"""运行取消标记。

前端点"停止"时，SSE 连接断开，但 pipeline 正阻塞在 VLM 调用里，
GeneratorExit 没有机会投递。这里用一个显式的标记让 pipeline 在
阶段之间自查，从而停止发起后续的 VLM 请求（一次请求仍然会跑完，
但已经受 VLM_TIMEOUT_S 约束）。

pipeline 逐张串行处理，所以简单的集合足够。
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_cancelled: set[str] = set()


def request_cancel(run_id: str | None) -> None:
    if not run_id:
        return
    with _lock:
        _cancelled.add(run_id)


def is_cancelled(run_id: str | None) -> bool:
    if not run_id:
        return False
    with _lock:
        return run_id in _cancelled


def clear(run_id: str | None) -> None:
    if not run_id:
        return
    with _lock:
        _cancelled.discard(run_id)
