"""诊断埋点：把 pipeline 的真实行为写成结构化 JSONL，供事后分析。

设计原则：只记录，不改变任何控制流。所有写入都包在 try/except 里，
埋点自身绝不抛异常，也不能影响返回值。
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime

from app.config import RESULT_DIR

TRACE_DIR = RESULT_DIR / "_trace"
TRACE_FILE = TRACE_DIR / "trace.jsonl"

_lock = threading.Lock()
_run_id = None

# print 摘要时优先展示的字段
_SUMMARY_KEYS = (
    "phase", "label", "status", "fail_reason", "duration_ms",
    "ok", "error_type", "reasoning_tokens", "n_calls",
)


def set_run(run_id):
    """设置当前 run_id。

    pipeline 逐张串行处理，所以模块级变量足够。若将来改成并发处理多张图，
    这里会串号，需要改成 contextvars 显式传下去。
    """
    global _run_id
    _run_id = run_id


def get_run():
    return _run_id


def _summary(kind, fields):
    parts = []
    for k in _SUMMARY_KEYS:
        v = fields.get(k)
        if v is not None:
            parts.append(f"{k}={v}")
    return f"[Trace] {kind} " + " ".join(parts)


def event(kind, **fields):
    """追加一行 trace。永不抛异常。"""
    try:
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "kind": kind,
            "run_id": _run_id,
        }
        record.update(fields)
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            with open(TRACE_FILE, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception:
        return
    try:
        print(_summary(kind, fields))
    except Exception:
        pass


@contextmanager
def phase(name, **fields):
    """记录一个阶段的耗时。异常照常向上抛，耗时照样记录。"""
    t0 = time.perf_counter()
    error_type = None
    try:
        yield
    except BaseException as exc:
        error_type = type(exc).__name__
        raise
    finally:
        event(
            "phase_end",
            phase=name,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            error_type=error_type,
            **fields,
        )


# ---------------------------------------------------------------------------
# OpenAI 客户端包装
# ---------------------------------------------------------------------------


class _Proxy:
    """把 target 上指定的属性替换掉，其余一律透传。"""

    __slots__ = ("_target", "_overrides")

    def __init__(self, target, **overrides):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_overrides", overrides)

    def __getattr__(self, name):
        overrides = object.__getattribute__(self, "_overrides")
        if name in overrides:
            return overrides[name]
        return getattr(object.__getattribute__(self, "_target"), name)


def _count_images(kwargs):
    total = 0
    for message in kwargs.get("messages") or []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    total += 1
    return total


def _make_traced_create(inner_create, phase_name):
    def create(*args, **kwargs):
        model = kwargs.get("model")
        n_images = _count_images(kwargs)
        max_tokens = kwargs.get("max_tokens")
        t0 = time.perf_counter()

        def _elapsed_ms():
            return round((time.perf_counter() - t0) * 1000, 1)

        try:
            response = inner_create(*args, **kwargs)
        except Exception as exc:
            event(
                "vlm_call",
                phase=phase_name,
                model=model,
                n_images=n_images,
                max_tokens=max_tokens,
                duration_ms=_elapsed_ms(),
                ok=False,
                error_type=type(exc).__name__,
                error_message=str(exc)[:500],
                http_status=getattr(exc, "status_code", None),
            )
            raise

        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        try:
            content = response.choices[0].message.content
        except Exception:
            content = None
        event(
            "vlm_call",
            phase=phase_name,
            model=model,
            n_images=n_images,
            max_tokens=max_tokens,
            duration_ms=_elapsed_ms(),
            ok=True,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
            response_chars=len(content) if content else 0,
        )
        return response

    return create


def wrap_openai_client(client, phase_name):
    """返回一个 client 代理：chat.completions.create 被埋点，其余透传。"""
    try:
        completions = _Proxy(
            client.chat.completions,
            create=_make_traced_create(client.chat.completions.create, phase_name),
        )
        chat = _Proxy(client.chat, completions=completions)
        return _Proxy(client, chat=chat)
    except Exception:
        return client
