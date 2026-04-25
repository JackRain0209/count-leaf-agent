# CV Agent — 油菜植株表型分析

基于 OpenCV + PlantCV + GPT-4o Vision 的油菜（Rapeseed）植株表型自动分析工具。

## 功能

- **分支计数**：骨架化 + 节点分析自动识别分支数
- **角果计数**：GPT-4o Vision 识别各分支角果（silique）数量
- **长度测量**：沿弯曲骨架测量主干/主支长度，标尺自动校准像素→厘米
- **可视化**：生成分析过程中间图（分割、骨架、节点标注）

## 技术架构

```
照片上传 → 标尺检测（OpenCV） → 植株分割（HSV阈值）
        → 骨架提取（skimage）  → 分支结构分析
        → VLM角果计数（GPT-4o） → 汇总输出
```

## 快速开始

```bash
# 1. 创建虚拟环境
cd cv-agent
python -m venv venv
source venv/bin/activate   # macOS/Linux

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY（与 lab-agent 相同）

# 4. 启动服务
python -m app.main
# 访问 http://localhost:8501
```

## API

### POST /api/analyze

上传图片进行分析。

```bash
curl -X POST http://localhost:8501/api/analyze \
  -F "file=@plant_photo.jpg"
```

返回 JSON 包含：
- `branch_count_cv` / `branch_count_vlm`：分支数（CV 和 VLM 双验证）
- `main_stem_length_cm`：主干长度（厘米）
- `main_branch_length_cm`：主支长度（厘米）
- `silique_counts`：VLM 角果计数详情
- `debug_images`：中间过程图片 URL

## 拍摄要求

- 黑色背景布
- 左侧放置已知长度的标尺
- 植株平铺展开，分支尽量不重叠
- 俯拍，光线均匀

## 后续扩展

如果 VLM 计数精度不够，可以：
1. 标注角果数据 → 训练 YOLOv8 检测模型
2. 将 `vlm_counter.py` 替换为 YOLO 推理模块
