# Docker 使用说明

这个项目现在可以用 Docker 运行和打包。镜像里只包含代码和 Python 依赖，不会把 `uploads/`、`results/`、`online_uploads/`、`benchmark_cache/`、历史实验目录、zip 数据包等大文件打进去。

## 1. 准备配置

复制示例配置：

```bash
cp .env.example .env
```

编辑 `.env`，至少填写：

```bash
ARK_API_KEY=你的 API Key
```

如果你使用的不是默认豆包 Ark 地址和模型，也一起改：

```bash
ARK_API_BASE=https://ark.cn-beijing.volces.com/api/v3
VLM_MODEL=ep-xxxxxxxx
```

## 2. 本机启动

```bash
docker compose up --build
```

访问：

```text
http://localhost:8501
```

后台启动：

```bash
docker compose up -d --build
```

查看日志：

```bash
docker compose logs -f
```

停止：

```bash
docker compose down
```

## 3. 数据目录

`docker-compose.yml` 会把这些目录挂载到容器里：

```text
uploads/          上传图片
results/          分析结果和中间图
online_uploads/   在线图片库
benchmark_cache/  分析缓存
data/             SQLite 历史记录
```

这些目录保留在宿主机项目目录下，不在镜像里。这样镜像体积小，也方便你决定是否单独拷贝数据。

## 4. 打包到另一台电脑

### 方式 A：推荐，在目标电脑重新构建

适合两台电脑 CPU 架构不同的情况，比如一台是 Apple Silicon，另一台是 Intel/AMD。

在当前电脑打一个源码包：

```bash
tar \
  --exclude='./.git' \
  --exclude='./venv' \
  --exclude='./.venv' \
  --exclude='./.env' \
  --exclude='./uploads/*' \
  --exclude='./results/*' \
  --exclude='./online_uploads' \
  --exclude='./benchmark_cache' \
  --exclude='./benchmark_cache_backup_*' \
  --exclude='./old_experiment_artifacts_*' \
  --exclude='./first10_*_backup*' \
  --exclude='./reports' \
  --exclude='./logs' \
  --exclude='./*.log' \
  --exclude='./cv-agent-*.screen.log' \
  --exclude='./.DS_Store' \
  --exclude='./.windsurf' \
  --exclude='./.playwright-mcp' \
  --exclude='./*.zip' \
  --exclude='./*.tar' \
  --exclude='./*.tar.gz' \
  --exclude='./*.tgz' \
  -czf cv-agent-docker-source.tar.gz .
```

把 `cv-agent-docker-source.tar.gz` 拷到目标电脑后：

```bash
tar -xzf cv-agent-docker-source.tar.gz
cd cv-agent
cp .env.example .env
# 编辑 .env
docker compose up --build
```

### 方式 B：保存 Docker 镜像

适合同架构电脑之间迁移。

当前电脑构建并保存镜像：

```bash
docker compose build
docker save cv-agent:latest -o cv-agent-image.tar
```

目标电脑加载镜像：

```bash
docker load -i cv-agent-image.tar
```

然后在项目目录里准备 `.env`，启动：

```bash
docker compose up
```

注意：`docker save` 保存的是当前 CPU 架构的镜像。Apple Silicon 上保存的 ARM64 镜像通常不能直接在 Intel/AMD 机器上跑，反过来也一样。
