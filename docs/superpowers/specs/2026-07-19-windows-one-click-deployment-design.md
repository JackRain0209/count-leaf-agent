# Windows 一键部署设计规格

日期：2026-07-19

状态：已由用户确认，等待规格文档审阅

项目：油菜考种工作台（cv-agent）

## 1. 背景与目标

当前项目是基于 FastAPI、OpenCV、Numba 和豆包视觉模型的单用户本机服务，已经具备 `Dockerfile`、`docker-compose.yml`、健康检查和宿主机数据目录挂载。

本次工作目标是为可信用户提供一个 Windows 交付包。用户自行根据 PDF 教程安装并启动 Docker Desktop，随后解压交付包并双击一个批处理文件，即可完成项目构建、启动、健康检查和浏览器打开。用户不需要安装 Python、不需要编辑文件、不需要填写环境变量，也不需要理解 Docker 命令。

## 2. 已确认的约束与决策

- 目标电脑允许安装 Docker Desktop，并具备完成安装所需的管理员权限。
- 目标电脑部署时可以稳定联网，并能访问 Docker 镜像源、Python 软件包源和豆包 Ark API。
- Docker Desktop、WSL2 和虚拟化环境由用户按照 PDF 教程自行安装和配置；项目部署脚本不负责修改 Windows 系统功能。
- 采用“目标 Windows 电脑根据源码现场构建 Docker 镜像”的方案，不交付由 Mac 构建的 ARM64 镜像。
- 所有项目环境变量使用提供方的配置，用户不填写配置。
- 目标用户可信，接受 API Key 部署到对方电脑后理论上可被本机管理员提取的安全边界。
- 真实环境变量只能进入最终可信交付包，不得提交到 Git，不得进入普通日志或诊断包。
- 默认服务地址固定为 `http://localhost:8501`，第一版不自动选择其他端口。
- 用户数据必须在重复部署、停止、启动和镜像重建时保留。

## 3. 方案比较与选择

### 3.1 方案 A：源码包在目标电脑现场构建

用户解压源码交付包，部署脚本执行 Docker Compose 构建和启动。

优点：

- 不需要维护镜像仓库。
- 自动适配目标 Windows 电脑的 Docker 架构。
- 复用项目现有 Dockerfile 和 Compose 配置，改动范围小。
- 交付包体积小，后续版本容易重新打包。

代价：

- 首次构建需要下载基础镜像和 Python 依赖，预计需要 5～20 分钟。
- 构建过程依赖目标电脑的网络状态。

### 3.2 方案 B：从镜像仓库拉取预构建镜像

部署速度较快，但需要维护公共或私有镜像仓库、版本和登录凭据。公共镜像还可能扩大代码暴露范围。

### 3.3 方案 C：交付离线镜像归档

不依赖构建时网络，但交付包可能达到 1～3 GB，每次更新都需要重新发送大型镜像文件。

### 3.4 最终选择

采用方案 A。目标电脑网络稳定，且不需要额外维护镜像仓库或大型离线包。

## 4. 仓库内的部署组件

仓库新增以下结构：

```text
deployment/
└─ windows/
   ├─ README-请先阅读.txt
   ├─ launchers/
   │  ├─ 02-一键部署.bat
   │  ├─ 03-启动系统.bat
   │  ├─ 04-停止系统.bat
   │  ├─ 05-查看运行状态.bat
   │  ├─ 06-打开系统.bat
   │  └─ 07-导出诊断日志.bat
   ├─ powershell/
   │  ├─ common.ps1
   │  ├─ deploy.ps1
   │  ├─ start.ps1
   │  ├─ stop.ps1
   │  ├─ status.ps1
   │  └─ diagnose.ps1
   └─ docs/
      └─ docker-desktop-install-guide.html

scripts/
└─ build_windows_release.sh
```

组件职责：

- `.bat` 文件提供可双击的中文入口，只负责设置终端编码、定位目录并使用本次进程级 `ExecutionPolicy Bypass` 调用 PowerShell，不永久修改系统策略。
- `common.ps1` 统一提供路径解析、中文输出、日志、Docker 检查、Compose 调用、健康检查、错误格式和秘密脱敏功能。
- `deploy.ps1` 负责首次检查、目录创建、配置验证、镜像构建、容器启动、健康检查和打开浏览器。
- `start.ps1` 只启动已有服务并执行健康检查，不重新构建镜像。
- `stop.ps1` 停止容器，但不删除容器数据、镜像或持久化目录。
- `status.ps1` 显示 Docker、Compose、容器、端口和 HTTP 健康状态。
- `diagnose.ps1` 生成不包含秘密与业务数据的脱敏诊断压缩包。
- `build_windows_release.sh` 从当前源码、部署模板和本机 `.env` 生成可信交付 ZIP；不得输出环境变量内容。

## 5. 最终交付包结构

发布构建脚本生成：

```text
dist/
└─ 油菜考种工作台-Windows版-YYYYMMDD/
   ├─ 01-Docker安装教程.pdf
   ├─ 02-一键部署.bat
   ├─ 03-启动系统.bat
   ├─ 04-停止系统.bat
   ├─ 05-查看运行状态.bat
   ├─ 06-打开系统.bat
   ├─ 07-导出诊断日志.bat
   ├─ README-请先阅读.txt
   ├─ user-data/
   │  ├─ data/
   │  ├─ uploads/
   │  ├─ results/
   │  ├─ online_uploads/
   │  └─ benchmark_cache/
   └─ resources/
      ├─ powershell/
      ├─ app/
      ├─ Dockerfile
      ├─ docker-compose.yml
      ├─ .dockerignore
      ├─ requirements.txt
      └─ config/
         └─ deployment.env
```

最终 ZIP 与解压目录位于 `dist/`。`dist/` 必须加入 `.gitignore`，防止包含真实凭据的交付物被提交。

最终包不包含 Git 历史、Mac 虚拟环境、开发日志、实验数据、旧缓存、大型备份、压缩数据集或其他本地无关文件。

## 6. 配置与秘密处理

发布构建脚本将项目根目录的本机 `.env` 完整复制为最终交付包的 `resources/config/deployment.env`。仓库只保留 `.env.example`，不新增任何包含真实秘密的受版本控制文件。

最终交付包中的 `.dockerignore` 必须显式排除 `config/deployment.env`。发布构建和测试必须验证该文件没有进入 Docker 构建上下文或镜像层；环境变量只能由 Compose 在容器启动时注入。

部署时 Compose 必须显式使用：

```text
docker compose --env-file config/deployment.env ...
```

部署脚本只校验必要变量存在且非空，不打印变量值。最低检查范围包括：

- `ARK_API_KEY`
- `ARK_API_BASE`
- `VLM_MODEL`
- `VLM_FP_CONFIDENCE_THRESHOLD`

最终配置文件可设置为 Windows 隐藏文件，以减少普通用户误操作；这不是加密措施，也不改变“本机管理员可以恢复凭据”的安全边界。

所有普通日志、部署日志和诊断包必须：

- 排除 `deployment.env`。
- 不保存完整展开后的 Compose 配置。
- 不列出容器完整环境变量。
- 使用实际秘密值替换规则，将日志中出现的秘密替换为 `***REDACTED***`。
- 不包含用户上传图片、结果图片、缓存内容或 SQLite 数据库。

## 7. 一键部署流程

### 7.1 检查部署包

部署脚本确认 `Dockerfile`、`docker-compose.yml`、`.dockerignore`、`requirements.txt`、`app/` 和 `config/deployment.env` 存在。若脚本疑似直接从 ZIP 预览目录运行，停止并要求用户先执行“全部解压”。

所有 Compose 调用都必须先把工作目录固定到交付包的 `resources`，并使用绝对路径定位脚本、日志和配置文件。不能依赖用户启动批处理文件时的当前目录。

### 7.2 检查 Docker

依次验证：

```text
docker --version
docker compose version
docker info
```

- 找不到 Docker CLI：提示阅读 PDF 并安装 Docker Desktop。
- Docker CLI 存在但 Engine 未运行：尝试从默认安装位置启动 Docker Desktop，并在有限时间内轮询；超时后给出人工处理步骤。
- Compose 不存在：提示升级 Docker Desktop。
- 脚本不自动安装 Docker、不启用 WSL2、不修改 BIOS 或 Windows 系统功能。

### 7.3 检查系统资源与端口

- 检查 Windows 版本、64 位架构和可用磁盘空间。
- 8501 未占用时继续。
- 8501 被当前 `cv-agent` 容器占用时视为重复部署并允许继续。
- 8501 被其他程序占用时停止，不自动切换端口。

### 7.4 准备数据目录

创建 `user-data/data`、`user-data/uploads`、`user-data/results`、`user-data/online_uploads` 和 `user-data/benchmark_cache`。只创建缺失目录，不清空、不覆盖已有业务数据。

Compose 挂载路径调整为最终交付包的 `user-data`，确保源码与镜像更新不影响历史记录和结果。

### 7.5 验证、构建和启动

先执行 Compose 配置校验，再构建启动：

```text
docker compose --env-file config/deployment.env config --quiet
docker compose --env-file config/deployment.env up -d --build
```

完整输出保存到 `deployment-logs/deploy-日期时间.log`。用户窗口只显示六个中文阶段：Docker 检查、配置检查、数据目录准备、镜像构建、服务启动、健康检查。

构建因临时网络错误失败时自动重试一次；再次失败后停止并保留日志，不进行无限重试。

### 7.6 健康检查与成功状态

容器启动后轮询 `http://127.0.0.1:8501/api/health`，只有收到 `status=ok` 且 `service=cv-agent` 才判定部署成功。

成功后显示日常使用说明，并打开默认浏览器访问 `http://localhost:8501`。

若容器存在但健康检查失败，自动收集容器状态和最近日志，写入脱敏部署日志并引导用户生成诊断包。

## 8. 幂等性与数据保护

`02-一键部署.bat` 必须可重复执行：

- 不删除或覆盖 SQLite 数据库。
- 不删除上传图片、结果文件、在线图库或缓存。
- 不调用 `docker compose down -v`。
- 容器已存在或正在运行时由 Compose 安全更新或复用。
- 失败时不执行清理用户数据的补偿操作。

日常入口规则：

- `03-启动系统.bat`：启动现有服务、检查健康状态并打开浏览器，不构建。
- `04-停止系统.bat`：执行 Compose stop，保留容器和数据。
- `05-查看运行状态.bat`：只读检查 Docker、容器、端口和健康接口。
- `06-打开系统.bat`：服务健康时打开浏览器；未运行时给出启动指引。
- `07-导出诊断日志.bat`：生成脱敏诊断 ZIP，不改变运行状态。

第一版不提供会删除 `user-data` 的一键卸载功能。

## 9. PDF 教程设计

交付文件名为 `01-Docker安装教程.pdf`。教程面向无开发经验用户，支持 Windows 10 22H2 64 位和 Windows 11 64 位，使用 Docker Desktop 的 WSL2 后端。

教程内容包括：

1. 安装前的系统、权限、内存、磁盘和网络要求。
2. 检查 Windows 版本。
3. 检查 CPU 虚拟化状态，并说明需要在 BIOS/UEFI 中启用的常见名称。
4. 安装或启用 WSL2，并处理系统重启。
5. 从官方来源安装 Docker Desktop。
6. 首次启动 Docker Desktop，并启用 WSL2 后端。
7. 确认 Docker Engine 和 Compose 正常。
8. 解压项目交付包并运行一键部署。
9. 日常启动、停止、打开系统和查看状态。
10. 常见错误与诊断日志导出方法。

最低建议资源：8 GB 内存、8～10 GB 可用磁盘和稳定网络。教程必须使用大字号步骤编号，明确展示每步的成功状态和失败处理。官方链接可点击并标注文档生成日期。

教程采用可维护的 HTML/CSS 或 Markdown 源文件生成 A4 PDF。源文件进入 Git，最终 PDF 进入可信发布包。

## 10. 错误处理规范

错误消息必须包含：结果标题、错误编号、普通用户可理解的原因、编号处理步骤和日志路径。禁止只显示原始 PowerShell 或 Docker 异常。

错误编号：

| 编号 | 含义 |
| --- | --- |
| D001 | 部署包未完整解压 |
| D002 | 未安装 Docker Desktop |
| D003 | Docker Engine 未启动 |
| D004 | Docker Compose 不可用 |
| D005 | 必要配置缺失 |
| D006 | 8501 端口被其他程序占用 |
| D007 | Compose 配置校验失败 |
| D008 | 镜像下载或构建失败 |
| D009 | 容器启动失败 |
| D010 | 服务健康检查超时 |
| D011 | 磁盘空间明显不足 |
| D012 | 网络连接异常 |
| D013 | Windows 或 CPU 架构不受支持 |
| D099 | 未分类异常 |

## 11. 诊断包设计

`07-导出诊断日志.bat` 生成 `diagnostics/cv-agent-diagnostics-日期时间.zip`。诊断包只包含：

- Windows 版本和 CPU 架构。
- Docker CLI、Compose 和 Engine 状态。
- 容器与健康状态。
- 8501 端口状态。
- 最近的脱敏容器日志和部署日志。
- 交付包文件完整性检查结果。
- 持久化目录是否存在及大小汇总，不包含目录内容。
- 健康接口响应。

生成压缩包前必须再次扫描并替换已知秘密值。

## 12. 测试策略

### 12.1 项目回归测试

```text
python -m unittest discover -s tests -v
docker compose config --quiet
```

### 12.2 PowerShell 脚本测试

覆盖：

- 中文和带空格路径。
- 完整配置与缺失配置。
- Docker CLI 缺失、Engine 未启动、Compose 缺失。
- 8501 未占用、被本项目占用、被其他程序占用。
- 健康成功、健康超时和容器退出。
- 日志目录异常时的降级行为。
- 秘密值脱敏。
- `deployment.env` 不进入 Docker 构建上下文和镜像层。
- 重复部署时持久化数据不丢失。

### 12.3 Docker 实际验证

验证镜像构建、容器启动、健康接口、首页访问、容器重启和停止再启动后的数据持久化。

### 12.4 真实 Windows 验收

Mac 上的静态检查不能代替 Windows 验证。发布前至少在一台 Windows 10 或 Windows 11 x64 电脑上验证：

- Docker 已运行和 Docker 未启动两种状态。
- 首次构建、重复部署、停止和重新启动。
- 中文与空格路径。
- 临时断网后的重试。
- 端口冲突提示。
- Docker Desktop 重启后的容器恢复。
- 使用真实图片完成一次豆包 VLM 分析。
- 重启后历史记录仍存在。
- 诊断包搜索不到真实 API Key。

## 13. 验收标准

1. Docker Desktop 安装完成后，用户不需要安装 Python 或任何 Python 依赖。
2. 用户不需要修改文件、填写 API Key 或输入 Docker 命令。
3. 用户解压后双击一次一键部署入口，可完成构建和启动。
4. 部署成功后自动打开 `http://localhost:8501`。
5. 首页与 `/api/health` 可访问。
6. 真实图片可以完成一次完整分析。
7. 重复部署、停止、启动和镜像重建不删除用户数据。
8. Docker 未安装、未启动、端口冲突或网络失败时显示明确中文处理步骤。
9. 诊断包可用于远程排查，且不包含 API Key 或业务数据。
10. 最终 ZIP 不包含本地开发环境、实验数据或无关大型文件。
11. 真实环境变量只存在于可信交付包，不进入 Git。
12. `deployment.env` 不进入 Docker 镜像层，镜像检查不能检出真实 API Key。

## 14. 非目标

第一版明确不包括：

- 自动安装 Docker Desktop、WSL2 或修改 BIOS/UEFI。
- 完全隐藏或加密目标电脑上的 API Key。
- 远程 API 代理或授权服务器。
- Docker 镜像仓库发布流程。
- 离线镜像交付。
- 自动选择非 8501 端口。
- 删除用户数据的一键卸载功能。
- 自动跨版本迁移数据库结构；如未来数据库结构变化，需要单独设计迁移方案。

## 15. 实施边界

本规格聚焦单个交付目标：生成 Windows Docker Desktop 环境下的源码一键部署包及配套 PDF。部署脚本、发布构建、PDF 源文件、Compose 路径调整、测试与文档属于同一个实施计划；远程代理、离线部署和自动系统环境安装不进入本次范围。
