# CV Agent 项目现状总结

> 最后更新：2026-04-29
> 目的：记录项目完整架构、各模块职责、历史改动、已知问题，以便快速恢复开发上下文。

---

## 一、项目概述

**油菜植株表型分析系统**：上传油菜考种照片 → VLM 识别各部件（主干/主枝/分枝）→ CV 算法逐个部件计数角果 → VLM 复核修正 → 输出总角果数。

- **数据**：~900 张油菜考种照片（黑布背景平铺），Excel 真值包含主枝角果数、分枝角果数、总角果数
- **运行端口**：`localhost:8501`
- **VLM**：豆包视觉模型（Doubao Vision），通过 OpenAI 兼容 API 调用

---

## 二、技术栈

| 层 | 技术 |
|---|---|
| 后端 | FastAPI + Uvicorn，SSE 流式返回分析步骤 |
| 前端 | 单页 HTML（原生 JS，暗色主题），无框架 |
| VLM | 豆包 Vision（`ARK_API_KEY` + `ARK_API_BASE`） |
| CV | OpenCV + scikit-image（骨架化）+ Numba（加速 BFS） |
| 依赖 | `requirements.txt`：fastapi, opencv-python-headless, scikit-image, scipy, numba, plantcv, openai, Pillow, pandas |

---

## 三、目录结构

```
cv-agent/
├── app/
│   ├── config.py                  # 配置：路径、API key、端口
│   ├── main.py                    # FastAPI 应用、API 路由、批量测试逻辑
│   ├── pipeline/
│   │   ├── analyzer.py            # 核心流水线编排（串联 VLM+CV 全流程）
│   │   ├── vlm_counter.py         # VLM 部件识别 + 裁剪验证
│   │   ├── pod_counter.py         # CV 算法 1：骨架遍历法
│   │   ├── pod_counter_graph.py   # CV 算法 2：拓扑图法
│   │   ├── pod_counter_stalk.py   # CV 算法 3：果柄法（当前主力）⭐
│   │   ├── pod_counter_plantcv.py # CV 算法 4：PlantCV 法（实验性）
│   │   ├── pod_verify_vlm.py      # VLM 角果复核（枯枝检测）
│   │   ├── segmentation.py        # 底层 CV 工具（二值化、连通域、骨架）
│   │   ├── ruler.py               # 直尺检测 + px/cm 换算
│   │   └── branch_analysis.py     # 分支结构辅助分析
│   ├── static/
│   │   └── index.html             # 前端单页应用（711行）
│   └── support/                   # VLM 枯枝参考图（few-shot）
│       ├── 微信图片_..._97_19.png
│       ├── 微信图片_..._98_19.png
│       └── 微信图片_..._99_19.png
├── scripts/
│   ├── benchmark.py               # 独立批量测试脚本（VLM+CV 或 CV-only）
│   └── test_vlm_count.py          # 单张快速测试
├── check_accuracy.py              # 实时准确率对比工具（读 result.json vs GT CSV）
├── online_uploads/                # 在线图片库（~900张考种照 + Excel 真值）
├── uploads/                       # 用户上传的临时图片
├── results/                       # 分析结果（step 图 + result.json）
├── benchmark_cache/               # VLM 缓存（已验证裁剪，供 CV-only 重跑）
├── requirements.txt
├── DEVELOPMENT.md                 # 旧版开发文档
├── .env                           # 实际环境变量（ARK_API_KEY 等）
└── .env.example                   # 环境变量示例
```

---

## 四、核心流水线详解

### 入口：`analyzer.py` → `analyze_plant_image_stream()`

这是一个 **Generator**，通过 SSE 逐步 yield 每个处理阶段的结果，前端实时展示。

#### 完整处理流程（7 大阶段）

```
1. 原始图片加载
       ↓
2. VLM 部件识别（vlm_label_plants）
   → 豆包 Vision 识别主干/主枝/分枝，返回 0-1000 归一化 bbox
   → 转换为像素坐标，裁剪各部件
       ↓
3. VLM 裁剪验证（vlm_verify_crop）— 并行执行
   → 对每个非主干部件，发送原图+裁剪图给 VLM
   → 判断：keep / delete / recrop
   → 剔除无效裁剪，重截不完整的
       ↓
4. 缓存已验证裁剪（benchmark_cache/）
   → 保存 crop 图 + vlm_result.json，供后续 CV-only 模式复用
       ↓
5. CV 角果计数（per branch, 顺序执行）
   → 根据 method 参数选择算法：skeleton / graph / stalk / plantcv
   → 当前批量测试默认使用 **stalk**（果柄法）
       ↓
6. VLM 角果复核（verify_pod_markers）— 并行执行
   → 发送原始裁剪 + 标注图 + 枯枝参考图给 VLM
   → 只检查 false positive（误判为角果的枯枝），不检查漏数
   → 返回 false_positive_ids → 调整 pod_count
       ↓
7. 汇总结果
   → 输出 total_pods, plants 数组（含 pod_count、vlm_verify 等）
```

#### 关键设计决策

- **VLM 并行化**：裁剪验证 和 角果复核 两个阶段均用 `ThreadPoolExecutor` 并行调用 VLM
- **缓存机制**：第一次 VLM 标注+验证后，裁剪图缓存到 `benchmark_cache/`，后续可用 `--cv-only` 模式跳过 VLM 阶段
- **步骤图 cache-busting**：URL 加 `?t=timestamp` 防止浏览器缓存旧图
- **每次分析前清空旧 step 图**：`debug_dir.glob("step_*.png")` unlink

---

## 五、四套 CV 计数算法

### 1. 骨架遍历法 `pod_counter.py`（method="skeleton"）

- 二值化 → 骨架化 → Numba BFS 找主干（最长+最直路径）
- 从 attach 点出发，曲率感知遍历侧枝
- junction 处选角度差最小方向
- 自适应中位数比例过滤（area/width/length）

### 2. 拓扑图法 `pod_counter_graph.py`（method="graph"）

- 骨架上建拓扑图：tip(degree=1)、junction(degree≥3)、attach(邻主干) 为节点
- degree=2 像素压缩为边
- 从 tip BFS 到 attach，每条路径 = 1 个角果
- 边可被多路径共享（无像素阻塞）
- 核心函数：`_build_topo_graph()`, `_graph_adjacency()` — 被 stalk 方案复用

### 3. 果柄法 `pod_counter_stalk.py`（method="stalk"）⭐ 当前主力

**两遍算法**：
- **Pass 1**：复用图论拓扑 → 识别枯枝（width/area 自适应中位数过滤）+ 短分支去噪
- **Pass 2**：删除枯枝像素后重建拓扑图 → 数主干上的 attach 边数 = 角果数

**优势**：角果尖端粘连时，骨架 tip 合并导致漏数；但果柄（attach 点）仍独立，因此果柄法不受 tip-merging 影响。

**关键参数**：
- `WIDTH_RATIO = 0.70`（枯枝宽度 < 中位数×0.7 → 过滤）
- `AREA_RATIO = 0.25`（枯枝面积 < 中位数×0.25 → 过滤）
- `MIN_BRANCH_LEN = 10`（<10px 的完整分支直接当噪声）
- `MIN_BRANCHES = 4`（分支数太少不做自适应过滤）
- `MIN_DIM = 800`（低分辨率图自动放大）

### 4. PlantCV 法 `pod_counter_plantcv.py`（method="plantcv"）

- 使用 PlantCV 4.4 库的形态学分析
- 实验性质，精度不稳定

---

## 六、VLM 模块详解

### `vlm_counter.py`（764 行）

**核心函数**：

| 函数 | 功能 |
|---|---|
| `vlm_label_plants()` | 发送整图给 VLM，识别各部件 bbox（0-1000 归一化坐标）→ 转像素 bbox → 裁剪 |
| `vlm_verify_crop()` | 发送原图+裁剪图，判断 keep/delete/recrop |
| `apply_recrop()` | 根据 VLM 返回的新 bbox 重新裁剪 |
| `vlm_detect_adhesion()` | 异株粘连检测（**当前已禁用**，注释掉了） |
| `mask_adhesion_regions()` | 粘连区域遮涂（**当前已禁用**） |

**VLM 配置**：
- 模型：`VLM_MODEL`（env 配置，当前为豆包 Vision endpoint）
- `max_tokens=3000`（标注），`500`（验证）
- `temperature=0.1`

**坐标系**：0-1000 归一化 → 像素坐标，bbox 加 2% padding

### `pod_verify_vlm.py`（394 行）

**核心函数**：`verify_pod_markers()`

- 输入：裁剪图 + 计数 debug 图 + 标准化 marker 列表
- marker 格式：`{"id": int, "type": "pod"|"filtered", "x_norm": float, "y_norm": float}`
- 发送 VLM 的图片（按顺序）：原始裁剪 → 标注图 → 枯枝参考图（3张，来自 `app/support/`）
- **VLM 只做一件事**：检查绿色 P 标记中是否有枯枝（false positive）
- **不检查漏数**（之前有 task 1 检查漏数，已删除）
- 返回 `false_positive_ids` → 从 pod_count 中减去

**枯枝判别标准**（写在 prompt 中）：
- ✅ 有效角果：亮色、饱满、纺锤形有宽度变化
- ❌ 枯枝：暗色、细瘦、等宽无膨大

---

## 七、API 接口

| 端点 | 方法 | 功能 |
|---|---|---|
| `/api/analyze` | POST | 上传图片分析（multipart，支持 method 参数） |
| `/api/analyze_online` | POST | 选择在线图片分析（`{"path":"...", "method":"stalk"}`) |
| `/api/online_images` | GET | 列出 online_uploads 中所有图片 |
| `/api/batch_test` | POST | 批量测试全部在线图片（SSE 流），默认 stalk |
| `/api/batch_stop` | POST | 紧急停止批量测试 |
| `/api/results/{id}` | GET | 获取已保存的分析结果 |
| `/api/health` | GET | 健康检查 |

### 批量测试 `/api/batch_test` 详细逻辑

1. 从 Excel（`*终版*.xlsx`）加载 ground truth
2. 遍历 `online_uploads/` 所有图片
3. 逐张调用 `analyze_plant_image_stream(method="stalk")`
4. 与 GT 对比：计算逐图 accuracy、主枝/分枝分别计算
5. 每 10 张输出阶段性汇总（MAE, RMSE, 平均 Acc, 主枝 Acc, 分枝 Acc）
6. 最后输出总体汇总

**GT 匹配逻辑**：
- Excel 列：A=处理编号(N0/N1...)，B=重复，C=品种，D=照片号，E=分枝数，F=主枝角果，G=分枝角果，H=总角果，I=备注
- 照片名拼接：`{A}.{B}.{C}.{D}`（如 `N0.1.1.1`）
- I 列非空的行（备注=脏数据）跳过

---

## 八、前端 `index.html`

- **暗色主题**（`#0f172a` 背景）
- 上传区：拖拽/点击上传
- 在线图片选择器：可浏览 `online_uploads/` 中的图片
- **method 选择**：骨架/图论/PlantCV/果柄 四种
- SSE 实时展示分析步骤：每个 step 显示缩略图 + 描述
- Lightbox 大图预览
- 批量测试面板：启动/停止按钮，日志滚动显示
- 结果卡片：部件数、角果总数、步骤数、耗时

---

## 九、Git 提交历史

```
e215788 (HEAD) feat: 新增果柄计数算法 + VLM枯枝参考图 + 批量测试改用stalk
cf7b313        feat: VLM角果复核 + 批量测试 + 缓存逻辑更新
1cf9056 (v1.0) feat: 图论CV算法+骨架/图论双按钮前端+VLM prompt优化
bacc486        feat: 直尺检测(平滑列密度扫描) + VLM提示词优化 + benchmark脚本增强
```

---

## 十、历史改动记录（按时间倒序）

### 最近一轮改动（e215788）

1. **新增果柄计数算法** `pod_counter_stalk.py`
   - 两遍法：Pass1 枯枝识别 → Pass2 果柄计数
   - 复用 `pod_counter_graph.py` 的 `_build_topo_graph` 和 `_graph_adjacency`

2. **VLM 角果复核简化** `pod_verify_vlm.py`
   - 删除 task 1（漏检检测），只保留 task 2（枯枝识别）
   - 删除 `missed_count` 相关逻辑
   - 添加枯枝参考图（few-shot）：`app/support/` 下 3 张参考图随 prompt 一起发送
   - 图片发送顺序调整为：原始裁剪 → 标注图 → 参考图

3. **批量测试增强** `main.py`
   - 改用 `stalk` method
   - 从 Excel 加载 GT（替代旧 CSV 方案）
   - 逐图输出准确率（总、主枝、分枝）
   - 每 10 张阶段汇总
   - 流式输出关键步骤描述（VLM/计数等进度）
   - GT 图名匹配修复：数值列正确转 int

4. **VLM 并行化** `analyzer.py`
   - `vlm_verify_crop` 并行：ThreadPoolExecutor
   - `verify_pod_markers` 并行：ThreadPoolExecutor
   - 步骤图 cache-busting（URL 加 timestamp）
   - 每次分析前清空旧 step 图

### 上一轮改动（cf7b313）

- 新增 VLM 角果复核模块 `pod_verify_vlm.py`
- 批量测试从 CSV 改为内嵌逻辑
- 缓存逻辑更新

### 稳定版（v1.0-stable-vlm, 1cf9056）

- 图论 CV 算法 `pod_counter_graph.py`
- 前端双按钮（骨架/图论）
- VLM prompt 优化

### 初始版（bacc486）

- 直尺检测 `ruler.py`
- VLM 提示词优化
- benchmark 脚本增强

---

## 十一、已知问题与待解决

### P0 — 角果尖端粘连

两个角果尖端在骨架上合并 → tip 消失 → 漏数。果柄法（stalk）部分缓解此问题，但仍非完美。

### P1 — VLM 标注无程序级校验

- VLM 返回的 bbox/标签完全信任，无合理性检查
- 可能漏标、错标、bbox 偏移
- 没有多主干检测、bbox 重叠检测

### P2 — 异株粘连检测已禁用

`analyzer.py` 中 `vlm_detect_adhesion` 调用被注释掉（暂时测试 tight crop 效果），但代码仍保留在 `vlm_counter.py` 中。

### P3 — 批量测试无法即时取消

VLM 调用在 ThreadPoolExecutor 中运行时，`_batch_cancel` 信号无法中断正在执行的 future，需要等当前并行批次全部完成才能停止。

### P4 — 枯枝过滤参数硬编码

`WIDTH_RATIO=0.70`, `AREA_RATIO=0.25` 等是经验值，不同品种/拍照条件可能需要不同阈值。

---

## 十二、开发环境快速恢复

```bash
cd /Users/jack/cv-agent
source venv/bin/activate
python -m app.main
# 访问 http://localhost:8501

# 批量测试（前端操作或 curl）
curl -X POST http://localhost:8501/api/batch_test

# 手动停止
curl -X POST http://localhost:8501/api/batch_stop
```

### 关键环境变量（`.env`）

```
ARK_API_KEY=xxx           # 豆包 API Key
ARK_API_BASE=https://ark.cn-beijing.volces.com/api/v3
VLM_MODEL=ep-2026042...   # 豆包 Vision 模型 endpoint
HOST=0.0.0.0
PORT=8501
```

---

## 十三、精度概况

### 早期精度（21 张缓存样本，CV-only）

| 指标 | 骨架方案 | 图论方案 |
|---|---|---|
| 主花序准确率 | ~75% | 75.4% |
| 分枝准确率 | ~70% | 62.0% |
| 总准确率 | ~73% | 72.1% |

### 果柄方案（stalk）

正在进行批量测试验证中，预期解决 tip-merging 漏数问题后有提升。

---

## 十四、代码依赖关系图

```
main.py
 └── analyzer.py（流水线编排）
      ├── vlm_counter.py
      │    ├── vlm_label_plants()      — 部件识别
      │    ├── vlm_verify_crop()       — 裁剪验证
      │    └── vlm_detect_adhesion()   — 粘连检测（已禁用）
      ├── pod_counter.py               — 骨架法（+ 共用工具函数）
      │    ├── _binarize()
      │    ├── _get_skeleton()
      │    ├── _remove_ruler_region()
      │    ├── _select_center_component()
      │    ├── _find_main_stem_numba()
      │    └── _nb_bfs_*()             — Numba 加速 BFS
      ├── pod_counter_graph.py         — 图论法
      │    ├── _build_topo_graph()     — 拓扑图构建（被 stalk 复用）
      │    └── _graph_adjacency()      — 邻接表（被 stalk 复用）
      ├── pod_counter_stalk.py         — 果柄法 ⭐
      │    └── 复用 pod_counter.py + pod_counter_graph.py
      ├── pod_counter_plantcv.py       — PlantCV 法
      └── pod_verify_vlm.py           — VLM 角果复核
           └── 复用 vlm_counter.py 的工具函数
```
