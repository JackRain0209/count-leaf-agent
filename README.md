# 油菜考种工作台

油菜植株表型分析工具。用户在本机浏览器中上传照片，系统通过豆包视觉模型识别主干、主枝和分枝，再用果柄法统计角果数量，并展示标尺换算后的主干长度。

当前部署形态是单用户本机 Docker 服务，默认访问地址为 `http://localhost:8501`。

## 功能

- 单张图片上传并实时查看分析步骤。
- 选择本地文件夹，将图片加入队列后逐张识别。
- 通过数据目录选择 `online_uploads/` 中的图片。
- SQLite 保存识别历史、原始结果和人工修订。
- 修改植株数量、部件类型和角果数量。
- 新增或删除主干、主枝、分枝，保存后自动重算分枝数和角果总数。
- 恢复某条记录的原始识别结果。
- 保存中间步骤图、VLM 原始响应和验证裁剪，便于复核。

> 当前编辑器修改的是现有算法的 `pod_count`，界面显示为“角果数量”。如果业务中的“叶子数量”指叶片而不是角果，需要另行增加叶片识别指标；本版本不会把角果数量冒充叶片数量。

## 处理流程

```text
上传图片
  → 豆包视觉模型识别主干/主枝/分枝、标尺和标签
  → VLM 复核裁剪范围
  → 果柄法进行二值化、骨架化和拓扑计数
  → VLM 复核枯枝误报与被过滤的候选
  → 返回步骤图、部件明细、角果汇总和主干长度
```

## Docker 启动

### 1. 准备配置

```bash
cp .env.example .env
```

编辑 `.env`，至少填写：

```dotenv
ARK_API_KEY=你的豆包 Ark API Key
ARK_API_BASE=https://ark.cn-beijing.volces.com/api/v3
VLM_MODEL=你的视觉模型 Endpoint
```

### 2. 启动服务

```bash
docker compose up -d --build
```

打开 `http://localhost:8501`。

查看日志和状态：

```bash
docker compose logs -f cv-agent
docker compose ps
```

停止服务：

```bash
docker compose down
```

### 3. 持久化目录

`docker-compose.yml` 将以下目录挂载到宿主机项目目录：

| 目录 | 内容 |
| --- | --- |
| `data/` | SQLite 历史库 `history.sqlite3` |
| `uploads/` | 用户上传的原图 |
| `results/` | 每次运行的步骤图、调试文件和 `result.json` |
| `online_uploads/` | 可从数据目录入口选择的图片 |
| `benchmark_cache/` | 已验证裁剪和 VLM 元数据缓存 |

这些目录不会被提交到 Git。重建镜像不会删除宿主机中的历史记录和结果文件。

## 本地开发启动

Python 3.11 或更高版本：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m app.main
```

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m app.main
```

## API

分析接口返回 `text/event-stream`，每行格式为 `data: {JSON}`。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/analyze` | 上传一张图片并启动分析 |
| `POST` | `/api/analyze_online` | 分析 `online_uploads/` 下的相对路径 |
| `GET` | `/api/online_images` | 列出数据目录中的图片 |
| `GET` | `/api/history` | 查询历史记录，支持 `limit`、`offset`、`q` |
| `GET` | `/api/history/{run_id}` | 获取历史详情、当前结果和步骤元数据 |
| `PUT` | `/api/history/{run_id}` | 保存人工修改后的部件和数量 |
| `POST` | `/api/history/{run_id}/restore` | 恢复原始识别结果 |
| `GET` | `/api/results/{run_id}` | 获取当前持久化结果 |
| `GET` | `/api/health` | 服务健康检查 |

上传示例：

```bash
curl -N -X POST http://localhost:8501/api/analyze \
  -F "file=@plant_photo.jpg"
```

最终 `result` 事件包含：

- `run_id`：运行标识。
- `plants`：部件明细，每项包含 `id`、`label`、`pod_count`。
- `branch_count`：分枝数量。
- `main_inflorescence_pod_count`：主枝角果数量。
- `branch_pod_count`：分枝角果数量。
- `total_pods`：角果总数。
- `main_stem_length_cm`：检测到标尺时的主干长度。
- `result_dir`：步骤图和调试文件目录。

当前只支持 `stalk` 果柄法；其他旧版 `skeleton`、`graph`、`plantcv` 方法已移除。

## 输入照片要求

- 使用黑色背景布。
- 左侧放置已知长度的黄色标尺。
- 植株尽量平铺展开，减少分枝重叠。
- 俯拍并保持光线均匀。
- 保证主干、主枝和分枝末端都在照片内。

## 测试与验证

运行后端和 API 测试：

```bash
python -m unittest discover -s tests -v
```

验证 Docker 配置：

```bash
docker compose config --quiet
```

## 相关文档

- [DOCKER.md](DOCKER.md)：镜像构建、迁移和挂载目录说明。
- [DEVELOPMENT.md](DEVELOPMENT.md)：历史开发记录，部分内容保留为背景资料；当前行为以本 README 和代码为准。
