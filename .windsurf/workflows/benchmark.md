---
description: 启动和清理批量测试(benchmark)进程，避免僵尸进程
---

# 批量测试(Benchmark)管理

## 启动前：清理残留进程

每次启动 benchmark 前，必须先检查并清理残留的 Python 进程：

// turbo
1. 检查是否有残留 benchmark 或 multiprocessing 进程：
```bash
ps aux | grep -E "benchmark|test_vlm|multiprocessing" | grep -v grep
```

2. 如果有残留进程，强制杀掉（普通 kill 对 multiprocessing 子进程无效，必须 kill -9）：
```bash
pkill -9 -f "benchmark.py"
pkill -9 -f "test_vlm_count.py"
pkill -9 -f "multiprocessing.spawn"
pkill -9 -f "multiprocessing.resource_tracker"
```

// turbo
3. 再次确认已清理干净：
```bash
ps aux | grep -E "benchmark|test_vlm|multiprocessing" | grep -v grep
```

## 启动 benchmark

4. 确认后端 uvicorn 不受影响（应该仍在运行）：
```bash
ps aux | grep uvicorn | grep -v grep
```

5. 启动 benchmark（根据需要选择模式和算法）：

支持参数：
- `--method skeleton` 或 `--method graph`（默认 graph）
- `--cv-only`：跳过 VLM，使用缓存裁剪图
- `--limit N`：只跑前 N 张

**图论算法 + CV-only：**
```bash
/Users/jack/cv-agent/venv/bin/python -u /Users/jack/cv-agent/scripts/benchmark.py --cv-only --method graph 2>&1 | tee /Users/jack/cv-agent/benchmark_log.txt
```

**骨架算法 + CV-only：**
```bash
/Users/jack/cv-agent/venv/bin/python -u /Users/jack/cv-agent/scripts/benchmark.py --cv-only --method skeleton 2>&1 | tee /Users/jack/cv-agent/benchmark_log.txt
```

**全量测试（VLM + CV，图论）：**
```bash
/Users/jack/cv-agent/venv/bin/python -u /Users/jack/cv-agent/scripts/benchmark.py --method graph 2>&1 | tee /Users/jack/cv-agent/benchmark_log.txt
```

**限量测试（前 N 张）：**
```bash
/Users/jack/cv-agent/venv/bin/python -u /Users/jack/cv-agent/scripts/benchmark.py --cv-only --method graph --limit 10 2>&1 | tee /Users/jack/cv-agent/benchmark_log.txt
```

注意：启动命令用 `Blocking: false` + `WaitMsBeforeAsync: 15000`，观察前 15 秒输出确认无报错。

## 停止 benchmark

6. 停止时必须杀掉整个进程树，不能只杀主进程：
```bash
pkill -9 -f "benchmark.py"
pkill -9 -f "multiprocessing.spawn"
```

// turbo
7. 确认已全部停止：
```bash
ps aux | grep -E "benchmark|multiprocessing.spawn" | grep -v grep
```

## 清理 VLM 缓存（可选）

如果需要让 VLM 重新标注（prompt 改过之后）：
```bash
rm -rf /Users/jack/cv-agent/benchmark_cache/*/vlm_result.json /Users/jack/cv-agent/benchmark_cache/*/crop_*.jpg
```

## 查看结果

// turbo
8. 查看最终汇总：
```bash
cat /Users/jack/cv-agent/benchmark_summary.txt
```

// turbo
9. 查看详细 CSV：
```bash
head -5 /Users/jack/cv-agent/benchmark_results.csv
```

## 注意事项

- `benchmark.py` 中的 `from app.pipeline.pod_counter_graph import count_pods_on_branch` 决定了用哪个 CV 算法，改之前确认 import 路径
- Numba JIT 首次编译会 fork 子进程，如果主进程被异常终止，这些子进程会变成僵尸进程并占满 CPU
- 始终用 `pkill -9` 而非 `kill` 来终止，确保子进程也被清理
