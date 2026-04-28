# CV Agent — 油菜表型分析 开发文档

> 最后更新：2026-04-27

## 项目概述

基于 VLM（视觉语言模型）+ CV（计算机视觉）的油菜植株表型分析系统。上传油菜植株照片后，自动识别主干/主枝/分枝，计数各部位角果数量。

## 技术栈

- **后端**：FastAPI + Uvicorn（SSE 流式返回分析步骤）
- **前端**：单页 HTML（原生 JS，暗色主题）
- **VLM**：豆包视觉模型（Doubao Vision），通过 API 调用
- **CV**：OpenCV + scikit-image（骨架化）+ Numba（加速 BFS/路径搜索）
- **数据**：Excel 真值 + 900 张油菜考种照片

## 功能模块

### 核心流水线 (`app/pipeline/`)

| 文件 | 功能 | 说明 |
|------|------|------|
| `vlm_counter.py` | VLM 标注 | 调用豆包 Vision API，识别主干/主枝/分枝并返回 bbox，裁剪各部件 |
| `analyzer.py` | 流水线编排 | 串联 VLM 标注 → CV 计数，支持 `method` 参数切换骨架/图论方案 |
| `pod_counter.py` | 骨架 CV 算法 | 曲率感知遍历：二值化→骨架化→找主干→从 attach 出发沿曲率走到 tip→自适应过滤 |
| `pod_counter_graph.py` | 图论 CV 算法 | 拓扑图方案：骨架→建图(tip/junction/attach 节点+边)→BFS from tip to attach→自适应过滤 |
| `segmentation.py` | 图像分割 | 二值化、形态学处理等底层工具 |
| `ruler.py` | 尺子去除 | 检测并移除照片边缘的直尺区域 |
| `branch_analysis.py` | 分支分析 | 分支相关的辅助分析功能 |

### 后端 API (`app/main.py`)

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/analyze` | POST | 上传图片分析（支持 `method` 参数：skeleton/graph） |
| `/api/analyze_online` | POST | 从在线数据库选图分析（支持 `method` 参数） |
| `/api/online_images` | GET | 列出在线数据库中的所有图片 |
| `/api/health` | GET | 健康检查 |

### 前端 (`app/static/index.html`)

- 拖拽/点击上传图片
- 两个在线数据库入口按钮：**骨架分析**（蓝色）/ **图论分析**（紫色）
- SSE 实时展示分析流水线步骤（带 lightbox 大图预览）
- 分析结果卡片：部件数、角果总数、处理步骤、耗时

### 测试脚本 (`scripts/`)

| 文件 | 功能 |
|------|------|
| `benchmark.py` | 批量测试：VLM+CV 全流水线 or CV-only，对比 Excel 真值，输出准确率报告 |
| `test_vlm_count.py` | 单张图片一次性 VLM+CV 测试 |

## 两套 CV 算法对比

### 骨架方案 (`pod_counter.py`)

1. 二值化 + 骨架化
2. Numba 加速找主干（最长+最直路径）
3. 沿主干找 attach 点，从 attach 出发曲率感知遍历
4. 遇到 junction 时选角度差最小的方向继续
5. 全局 visited 防止回路
6. 自适应中位数比例过滤（area/width/length）

### 图论方案 (`pod_counter_graph.py`)

1. 二值化 + 骨架化（共用）
2. 找主干（共用）
3. 非主干骨架像素建拓扑图：tip(degree=1)、junction(degree≥3)、attach(邻主干) 为节点，degree=2 像素压缩为边
4. 从每个 tip 节点 BFS 到最近 attach 节点，重建像素路径
5. 边可被多条路径共享（不存在像素级阻塞）
6. 自适应中位数比例过滤（area/width）

### 当前精度（21 张有缓存样本，CV-only）

| 指标 | 骨架方案 | 图论方案 |
|------|----------|----------|
| 主花序准确率 | ~75% | 75.4% |
| 分枝准确率 | ~70% | 62.0% |
| 总准确率 | ~73% | 72.1% |
| 分枝 ME（偏差） | - | +69.9（高估） |

## 未解决问题

### 1. 角果尖端粘连问题

当两个角果的尖端在骨架上粘连时，原本的 degree=1 端点变成 degree=2 或 degree≥3，不再被识别为 tip 节点，导致**漏数**。

- 尝试过"短桥接边删除"方案（删除连接两个非 attach junction 的短边，恢复 tip），但误删过多导致算法崩溃
- 尝试过"补回 attach 直连未计数边"方案，但引入大量误报，误差更大
- **待探索方向**：更精准的粘连检测条件，或基于角果形态（长宽比、曲率）的额外验证

### 2. VLM 标注无程序级校验

当前完全信任 VLM 返回的 bbox 和标签，没有做任何程序上的限制或校验：

- VLM 可能漏标、错标（如把分枝标成主干、bbox 严重偏移）
- 没有对 bbox 合理性的检查（如面积过小/过大、严重重叠）
- 没有对标签逻辑一致性的校验（如出现多个主干）
- VLM 回答格式异常时缺乏健壮的 fallback 机制
- **影响**：VLM 错误会直接传导到 CV 阶段，导致角果计数完全偏离

## 版本记录

| Tag | 日期 | 说明 |
|-----|------|------|
| `v1.0-stable-vlm` | 2026-04-27 | 稳定 VLM 方案：骨架+图论双 CV 算法，前端双按钮 |
