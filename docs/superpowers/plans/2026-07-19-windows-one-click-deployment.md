# Windows One-Click Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a trusted Windows release ZIP that lets a non-technical user install Docker Desktop from a Chinese PDF, then deploy, start, stop, inspect, open, and diagnose cv-agent through numbered double-clickable batch files without entering configuration.

**Architecture:** Keep the existing developer Docker workflow unchanged and add an isolated `deployment/windows` release template. Thin `.bat` launchers call focused Windows PowerShell 5.1 scripts; a macOS/Linux release builder copies only production source, injects the local `.env` into an ignored release directory, renders the maintained HTML guide to PDF, and produces a ZIP whose persistent data lives outside the Docker build context under `user-data`.

**Tech Stack:** Bash, Windows PowerShell 5.1, Docker Desktop with WSL2, Docker Compose v2, Python `unittest`, HTML/CSS, headless Google Chrome PDF rendering, ZIP archives.

---

## Scope and execution constraints

- Implement from an isolated worktree created at execution time with `superpowers:using-git-worktrees`.
- Follow `superpowers:test-driven-development`: add each failing test before the matching production asset.
- Do not read, print, commit, or send the real `.env` contents. The release builder may copy the file byte-for-byte into ignored `dist/` output because the user explicitly authorized that trusted artifact.
- Do not claim Windows compatibility solely from macOS checks. Complete the automated local checks and leave the real Windows acceptance checklist explicitly pending until run on Windows 10/11 x64.
- During implementation of the PDF, re-check the official sources with `agent-reach` because Docker and Microsoft requirements can change.
- Do not add automatic Docker Desktop installation, a remote API proxy, offline images, automatic port switching, or destructive uninstall behavior.

## File map

### Files to create

- `tests/test_windows_deployment_assets.py` — repository-level contracts for Compose, ignore rules, launchers, PowerShell scripts, docs, and secret-handling invariants.
- `tests/test_windows_release_builder.py` — isolated release-builder tests using a fake environment file and temporary output directory.
- `deployment/windows/resources/docker-compose.yml` — Windows release Compose definition with `../user-data` bind mounts.
- `deployment/windows/resources/.dockerignore` — release build-context exclusions, including `config/deployment.env`.
- `deployment/windows/launchers/02-一键部署.bat` — first deployment entry point.
- `deployment/windows/launchers/03-启动系统.bat` — daily start entry point.
- `deployment/windows/launchers/04-停止系统.bat` — safe stop entry point.
- `deployment/windows/launchers/05-查看运行状态.bat` — read-only status entry point.
- `deployment/windows/launchers/06-打开系统.bat` — browser entry point.
- `deployment/windows/launchers/07-导出诊断日志.bat` — sanitized diagnostics entry point.
- `deployment/windows/powershell/common.ps1` — paths, dotenv parsing, redaction, logging, health polling, and user-facing error text.
- `deployment/windows/powershell/docker.ps1` — Docker CLI, Docker Desktop startup, Compose, port, and container helpers.
- `deployment/windows/powershell/deploy.ps1` — first-build orchestration.
- `deployment/windows/powershell/start.ps1` — daily start orchestration.
- `deployment/windows/powershell/stop.ps1` — safe stop orchestration.
- `deployment/windows/powershell/status.ps1` — read-only status report.
- `deployment/windows/powershell/open.ps1` — health check and browser open.
- `deployment/windows/powershell/diagnose.ps1` — sanitized diagnostic ZIP generation.
- `deployment/windows/tests/run-unit-tests.ps1` — dependency-free PowerShell 5.1 unit tests for pure helpers and mocked Docker results.
- `deployment/windows/docs/docker-desktop-install-guide.html` — maintained Chinese A4 tutorial source.
- `deployment/windows/README-请先阅读.txt` — the short user-facing workflow.
- `deployment/windows/WINDOWS-ACCEPTANCE.md` — real Windows acceptance checklist and evidence fields.
- `scripts/render_windows_guide_pdf.sh` — deterministic headless-browser PDF renderer.
- `scripts/build_windows_release.sh` — trusted release directory and ZIP builder.

### Files to modify

- `.gitignore` — ignore `dist/`, generated diagnostics, and generated deployment logs.
- `README.md` — document how maintainers build the Windows release.
- `DOCKER.md` — document Windows delivery, secret boundaries, and validation commands.

### Files intentionally left unchanged

- `docker-compose.yml` — preserve the existing local developer mounts and workflow.
- `Dockerfile` — reuse the current Python 3.11/OpenCV image definition.
- `app/**` — no application behavior changes are required for Windows packaging.
- `.env` — never edit or stage the real configuration.

---

### Task 1: Define the Windows release asset contract

**Files:**

- Create: `tests/test_windows_deployment_assets.py`
- Create: `deployment/windows/resources/docker-compose.yml`
- Create: `deployment/windows/resources/.dockerignore`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing repository asset tests**

Create `tests/test_windows_deployment_assets.py` with the initial contract:

```python
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ROOT / "deployment" / "windows"


class WindowsDeploymentAssetTests(unittest.TestCase):
    def test_release_compose_uses_parent_user_data(self):
        compose = (WINDOWS / "resources" / "docker-compose.yml").read_text(encoding="utf-8")
        expected_mounts = [
            "../user-data/uploads:/app/uploads",
            "../user-data/results:/app/results",
            "../user-data/online_uploads:/app/online_uploads",
            "../user-data/benchmark_cache:/app/benchmark_cache",
            "../user-data/data:/app/data",
        ]
        for mount in expected_mounts:
            self.assertIn(mount, compose)
        self.assertIn('"8501:8501"', compose)
        self.assertIn("restart: unless-stopped", compose)

    def test_release_compose_requires_injected_vlm_configuration(self):
        compose = (WINDOWS / "resources" / "docker-compose.yml").read_text(encoding="utf-8")
        self.assertIn("ARK_API_KEY: ${ARK_API_KEY}", compose)
        self.assertIn("ARK_API_BASE: ${ARK_API_BASE}", compose)
        self.assertIn("VLM_MODEL: ${VLM_MODEL}", compose)
        self.assertIn("- ./config/deployment.env", compose)
        self.assertIn(
            "VLM_FP_CONFIDENCE_THRESHOLD: ${VLM_FP_CONFIDENCE_THRESHOLD}",
            compose,
        )
        self.assertNotIn("your_api_key_here", compose)
        self.assertNotIn("ep-20260420011113-6jvmx", compose)

    def test_release_dockerignore_excludes_injected_configuration(self):
        dockerignore = (WINDOWS / "resources" / ".dockerignore").read_text(encoding="utf-8")
        entries = {
            line.strip()
            for line in dockerignore.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("config/deployment.env", entries)
        self.assertIn("user-data/", entries)
        self.assertIn("deployment-logs/", entries)
        self.assertIn("diagnostics/", entries)

    def test_generated_release_output_is_gitignored(self):
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("dist/", {line.strip() for line in gitignore})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm the expected failure**

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets -v
```

Expected: errors for missing `deployment/windows/resources/docker-compose.yml` and `.dockerignore`, plus a failure because `dist/` is not ignored.

- [ ] **Step 3: Add the release-specific Compose definition**

Create `deployment/windows/resources/docker-compose.yml`:

```yaml
name: cv-agent

services:
  cv-agent:
    build:
      context: .
      dockerfile: Dockerfile
    image: cv-agent:windows
    container_name: cv-agent
    env_file:
      - ./config/deployment.env
    environment:
      HOST: 0.0.0.0
      PORT: 8501
      ARK_API_KEY: ${ARK_API_KEY}
      ARK_API_BASE: ${ARK_API_BASE}
      VLM_MODEL: ${VLM_MODEL}
      VLM_FP_CONFIDENCE_THRESHOLD: ${VLM_FP_CONFIDENCE_THRESHOLD}
      DATA_DIR: /app/data
    ports:
      - "8501:8501"
    volumes:
      - ../user-data/uploads:/app/uploads
      - ../user-data/results:/app/results
      - ../user-data/online_uploads:/app/online_uploads
      - ../user-data/benchmark_cache:/app/benchmark_cache
      - ../user-data/data:/app/data
    restart: unless-stopped
    healthcheck:
      test:
        [
          "CMD-SHELL",
          "python -c \"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/api/health', timeout=3).read()\"",
        ]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s
```

- [ ] **Step 4: Add the release build-context exclusions**

Create `deployment/windows/resources/.dockerignore`:

```text
.git
.gitignore
.dockerignore
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.DS_Store
.env
.env.*
config/deployment.env
config/
powershell/
docker-compose.yml
user-data/
deployment-logs/
diagnostics/
venv/
.venv/
tests/
docs/
*.zip
*.tar
*.tar.gz
*.tgz
*.log
```

Append these exact entries to `.gitignore`:

```text
dist/
deployment-logs/
diagnostics/
```

- [ ] **Step 5: Run the asset tests and Compose parse check**

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets -v
ARK_API_KEY=test-key ARK_API_BASE=https://example.invalid VLM_MODEL=test-model VLM_FP_CONFIDENCE_THRESHOLD=0.8 \
  docker compose -f deployment/windows/resources/docker-compose.yml config --quiet
```

Expected: four tests pass and Compose exits with code 0. The Compose command may warn that the build context does not yet contain the Dockerfile; it must not report invalid YAML.

- [ ] **Step 6: Commit the release asset contract**

```bash
git add .gitignore tests/test_windows_deployment_assets.py deployment/windows/resources
git commit -m "test: define Windows deployment asset contract"
```

---

### Task 2: Build the shared PowerShell foundation

**Files:**

- Create: `deployment/windows/powershell/common.ps1`
- Create: `deployment/windows/tests/run-unit-tests.ps1`
- Modify: `tests/test_windows_deployment_assets.py`

- [ ] **Step 1: Extend the Python contract to require safe PowerShell primitives**

Add this method to `WindowsDeploymentAssetTests`:

```python
    def test_common_powershell_defines_required_safe_helpers(self):
        common = (WINDOWS / "powershell" / "common.ps1").read_text(encoding="utf-8")
        for function_name in [
            "Get-DeploymentContext",
            "Read-DotEnvFile",
            "Test-RequiredEnvironment",
            "Protect-SecretText",
            "Write-SafeLog",
            "Get-DeploymentErrorText",
            "Wait-CvAgentHealth",
        ]:
            self.assertIn(f"function {function_name}", common)
        self.assertNotIn("Write-Host $env:ARK_API_KEY", common)
        self.assertNotIn("Get-ChildItem Env:", common)
```

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_common_powershell_defines_required_safe_helpers -v
```

Expected: error because `common.ps1` does not exist.

- [ ] **Step 2: Create dependency-free PowerShell tests before the helpers**

Create `deployment/windows/tests/run-unit-tests.ps1` with assertions for path calculation, dotenv parsing, required variables, redaction, and error text:

```powershell
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$script:Failures = 0

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        $script:Failures += 1
        Write-Host "FAIL: $Message" -ForegroundColor Red
    }
}

function Assert-Equal {
    param($Expected, $Actual, [string]$Message)
    Assert-True -Condition ($Expected -eq $Actual) -Message "$Message (expected=$Expected actual=$Actual)"
}

$PowerShellRoot = Join-Path $PSScriptRoot "..\powershell"
. (Join-Path $PowerShellRoot "common.ps1")

$context = Get-DeploymentContext -PowerShellRoot "C:\可信目录\resources\powershell"
Assert-Equal "C:\可信目录" $context.PackageRoot "Package root is two levels above powershell"
Assert-Equal "C:\可信目录\resources" $context.ResourcesRoot "Resources root is resolved"
Assert-Equal "C:\可信目录\user-data" $context.UserDataRoot "User data lives outside build context"

$tempEnv = Join-Path ([System.IO.Path]::GetTempPath()) ("cv-agent-env-" + [guid]::NewGuid() + ".env")
try {
    [System.IO.File]::WriteAllText(
        $tempEnv,
        "# comment`r`nARK_API_KEY=secret-value`r`nARK_API_BASE=https://example.invalid`r`nVLM_MODEL=model-a`r`nVLM_FP_CONFIDENCE_THRESHOLD=0.8`r`n",
        (New-Object System.Text.UTF8Encoding($false))
    )
    $envMap = Read-DotEnvFile -Path $tempEnv
    Assert-Equal "secret-value" $envMap["ARK_API_KEY"] "dotenv parser reads API key"
    Assert-True (Test-RequiredEnvironment -Values $envMap) "required environment is accepted"
    $safe = Protect-SecretText -Text "token=secret-value" -Values $envMap
    Assert-Equal "token=***REDACTED***" $safe "API key is redacted"
    $missing = @{
        ARK_API_KEY = ""
        ARK_API_BASE = "https://example.invalid"
        VLM_MODEL = "model-a"
        VLM_FP_CONFIDENCE_THRESHOLD = "0.8"
    }
    Assert-True (-not (Test-RequiredEnvironment -Values $missing)) "blank required value is rejected"
    $errorText = Get-DeploymentErrorText -Code "D003" -Reason "Docker Engine 未启动" -LogPath "C:\log.txt"
    Assert-True ($errorText.Contains("错误编号：D003")) "error code is shown"
    Assert-True ($errorText.Contains("打开 Docker Desktop")) "actionable help is shown"
}
finally {
    Remove-Item -LiteralPath $tempEnv -Force -ErrorAction SilentlyContinue
}

if ($script:Failures -gt 0) {
    Write-Host "$script:Failures PowerShell tests failed." -ForegroundColor Red
    exit 1
}

Write-Host "All PowerShell unit tests passed." -ForegroundColor Green
exit 0
```

Do not run this file yet on macOS because neither `powershell` nor `pwsh` is installed. Its execution is mandatory in Task 10 on real Windows PowerShell 5.1.

- [ ] **Step 3: Implement the shared PowerShell helpers**

Create `deployment/windows/powershell/common.ps1` with these exact public interfaces:

```powershell
Set-StrictMode -Version Latest

function Get-DeploymentContext {
    param([string]$PowerShellRoot = $PSScriptRoot)

    $resourcesRoot = Split-Path -Parent $PowerShellRoot
    $packageRoot = Split-Path -Parent $resourcesRoot
    [pscustomobject]@{
        PackageRoot = $packageRoot
        ResourcesRoot = $resourcesRoot
        ComposeFile = Join-Path $resourcesRoot "docker-compose.yml"
        EnvFile = Join-Path $resourcesRoot "config\deployment.env"
        UserDataRoot = Join-Path $packageRoot "user-data"
        LogRoot = Join-Path $packageRoot "deployment-logs"
        DiagnosticsRoot = Join-Path $packageRoot "diagnostics"
        AppUrl = "http://127.0.0.1:8501"
        HealthUrl = "http://127.0.0.1:8501/api/health"
        ProjectName = "cv-agent"
    }
}

function Ensure-Directory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Read-DotEnvFile {
    param([Parameter(Mandatory = $true)][string]$Path)

    $values = @{}
    foreach ($line in [System.IO.File]::ReadAllLines($Path)) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) { continue }
        $separator = $trimmed.IndexOf("=")
        if ($separator -lt 1) { continue }
        $name = $trimmed.Substring(0, $separator).Trim()
        $value = $trimmed.Substring($separator + 1).Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        $values[$name] = $value
    }
    return $values
}

function Test-RequiredEnvironment {
    param([Parameter(Mandatory = $true)][hashtable]$Values)

    $required = @("ARK_API_KEY", "ARK_API_BASE", "VLM_MODEL", "VLM_FP_CONFIDENCE_THRESHOLD")
    foreach ($name in $required) {
        if (-not $Values.ContainsKey($name) -or [string]::IsNullOrWhiteSpace([string]$Values[$name])) {
            return $false
        }
    }
    return $true
}

function Get-SecretValues {
    param([Parameter(Mandatory = $true)][hashtable]$Values)

    $secrets = New-Object System.Collections.Generic.List[string]
    foreach ($name in $Values.Keys) {
        if ($name -match '(?i)(KEY|TOKEN|SECRET|PASSWORD)' -and
            -not [string]::IsNullOrWhiteSpace([string]$Values[$name]) -and
            ([string]$Values[$name]).Length -ge 4) {
            $secrets.Add([string]$Values[$name])
        }
    }
    return $secrets
}

function Protect-SecretText {
    param(
        [AllowEmptyString()][string]$Text,
        [Parameter(Mandatory = $true)][hashtable]$Values
    )

    $safe = [string]$Text
    foreach ($secret in (Get-SecretValues -Values $Values | Sort-Object Length -Descending)) {
        $safe = $safe.Replace($secret, "***REDACTED***")
    }
    return $safe
}

function New-DeploymentLog {
    param([Parameter(Mandatory = $true)]$Context, [string]$Prefix = "deploy")
    Ensure-Directory -Path $Context.LogRoot
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $path = Join-Path $Context.LogRoot "$Prefix-$stamp.log"
    [System.IO.File]::WriteAllText($path, "", (New-Object System.Text.UTF8Encoding($false)))
    return $path
}

function Write-SafeLog {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [AllowEmptyString()][string]$Text,
        [Parameter(Mandatory = $true)][hashtable]$Values
    )
    $safe = Protect-SecretText -Text $Text -Values $Values
    [System.IO.File]::AppendAllText(
        $Path,
        $safe + [Environment]::NewLine,
        (New-Object System.Text.UTF8Encoding($false))
    )
}

function Write-Stage {
    param([int]$Current, [int]$Total, [string]$Message)
    Write-Host "[$Current/$Total] $Message" -ForegroundColor Cyan
}

function Get-DeploymentErrorText {
    param([string]$Code, [string]$Reason, [string]$LogPath)

    $actions = @{
        D001 = "请右键压缩包并选择“全部解压”，确认 resources 目录完整后重试。"
        D002 = "请按照“01-Docker安装教程.pdf”安装 Docker Desktop。"
        D003 = "请打开 Docker Desktop，等待 Docker Engine 正常运行后重试。"
        D004 = "请升级 Docker Desktop，确认 docker compose version 可以运行。"
        D005 = "部署配置不完整，请联系提供方重新生成安装包。"
        D006 = "8501 端口被其他程序占用，请关闭占用程序后重试。"
        D007 = "Docker Compose 配置校验失败，请导出诊断日志并联系提供方。"
        D008 = "镜像构建失败。脚本已重试一次，请检查网络后重试。"
        D009 = "容器启动失败，请导出诊断日志并联系提供方。"
        D010 = "服务启动后未通过健康检查，请导出诊断日志。"
        D011 = "可用磁盘不足 4 GB，请释放磁盘空间后重试。"
        D012 = "下载依赖时网络异常，请确认网络可用后重试。"
        D013 = "当前版本只支持 Windows 10/11 x64。"
        D099 = "发生未分类错误，请导出诊断日志并联系提供方。"
    }
    $action = if ($actions.ContainsKey($Code)) { $actions[$Code] } else { $actions["D099"] }
    return "部署未完成`r`n`r`n错误编号：$Code`r`n原因：$Reason`r`n处理方法：$action`r`n详细日志：$LogPath"
}

function Test-CvAgentHealth {
    param([Parameter(Mandatory = $true)][string]$Url)
    try {
        $response = Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 5 -UseBasicParsing
        return ($response.status -eq "ok" -and $response.service -eq "cv-agent")
    }
    catch {
        return $false
    }
}

function Wait-CvAgentHealth {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [int]$TimeoutSeconds = 180,
        [int]$IntervalSeconds = 3
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-CvAgentHealth -Url $Url) { return $true }
        Start-Sleep -Seconds $IntervalSeconds
    }
    return $false
}
```

- [ ] **Step 4: Run the macOS static contract**

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets -v
```

Expected: all current Python tests pass. Record that the PowerShell runtime test remains pending for Windows.

- [ ] **Step 5: Commit the shared PowerShell foundation**

```bash
git add tests/test_windows_deployment_assets.py deployment/windows/powershell/common.ps1 deployment/windows/tests/run-unit-tests.ps1
git commit -m "feat: add shared Windows deployment helpers"
```

---

### Task 3: Add Docker and Compose adapters

**Files:**

- Create: `deployment/windows/powershell/docker.ps1`
- Modify: `deployment/windows/tests/run-unit-tests.ps1`
- Modify: `tests/test_windows_deployment_assets.py`

- [ ] **Step 1: Add failing static and mocked Docker tests**

Add this Python test method:

```python
    def test_docker_powershell_defines_adapters_without_printing_environment(self):
        docker_ps1 = (WINDOWS / "powershell" / "docker.ps1").read_text(encoding="utf-8")
        for function_name in [
            "Invoke-NativeCommand",
            "Invoke-DockerCompose",
            "Test-DockerEngine",
            "Start-DockerDesktop",
            "Get-Port8501Owner",
        ]:
            self.assertIn(f"function {function_name}", docker_ps1)
        self.assertNotIn("docker inspect", docker_ps1.lower())
        self.assertNotIn("Get-ChildItem Env:", docker_ps1)
```

Append these tests before the final failure check in `run-unit-tests.ps1`:

```powershell
. (Join-Path $PowerShellRoot "docker.ps1")

function Invoke-NativeCommand {
    param([string]$FilePath, [string[]]$Arguments)
    if ($Arguments -contains "info") {
        return [pscustomobject]@{ ExitCode = 0; Output = "engine-ready" }
    }
    return [pscustomobject]@{ ExitCode = 0; Output = "ok" }
}

Assert-True (Test-DockerEngine) "mocked Docker engine is detected"
$composeResult = Invoke-DockerCompose -Context $context -Arguments @("ps", "--format", "json")
Assert-Equal 0 $composeResult.ExitCode "Compose wrapper returns native exit code"
```

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_docker_powershell_defines_adapters_without_printing_environment -v
```

Expected: error because `deployment/windows/powershell/docker.ps1` is absent.

- [ ] **Step 2: Implement Docker adapters**

Create `deployment/windows/powershell/docker.ps1`:

```powershell
Set-StrictMode -Version Latest

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @()
    )
    try {
        $output = (& $FilePath @Arguments 2>&1 | Out-String)
        $exitCode = $LASTEXITCODE
        return [pscustomobject]@{ ExitCode = $exitCode; Output = $output.TrimEnd() }
    }
    catch {
        return [pscustomobject]@{ ExitCode = 1; Output = $_.Exception.Message }
    }
}

function Invoke-DockerCompose {
    param(
        [Parameter(Mandatory = $true)]$Context,
        [string[]]$Arguments = @()
    )
    $prefix = @(
        "compose",
        "-f", $Context.ComposeFile,
        "--env-file", $Context.EnvFile,
        "-p", $Context.ProjectName
    )
    Push-Location -LiteralPath $Context.ResourcesRoot
    try {
        return Invoke-NativeCommand -FilePath "docker" -Arguments ($prefix + $Arguments)
    }
    finally {
        Pop-Location
    }
}

function Test-DockerEngine {
    $result = Invoke-NativeCommand -FilePath "docker" -Arguments @("info")
    return ($result.ExitCode -eq 0)
}

function Start-DockerDesktop {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\DockerDesktop\Docker Desktop.exe"),
        (Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            Start-Process -FilePath $candidate | Out-Null
            return $true
        }
    }
    return $false
}

function Wait-DockerEngine {
    param([int]$TimeoutSeconds = 120, [int]$IntervalSeconds = 3)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-DockerEngine) { return $true }
        Start-Sleep -Seconds $IntervalSeconds
    }
    return $false
}

function Get-Port8501Owner {
    $listeners = @(Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -eq 0) { return "free" }
    $container = Invoke-NativeCommand -FilePath "docker" -Arguments @(
        "ps", "--filter", "name=^/cv-agent$", "--format", "{{.Ports}}"
    )
    if ($container.ExitCode -eq 0 -and $container.Output -match "8501") {
        return "cv-agent"
    }
    return "other"
}

function Test-CvAgentContainerExists {
    param([Parameter(Mandatory = $true)]$Context)
    $result = Invoke-DockerCompose -Context $Context -Arguments @("ps", "-q", "cv-agent")
    return ($result.ExitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($result.Output))
}
```

- [ ] **Step 3: Run static tests**

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets -v
```

Expected: all Python tests pass.

- [ ] **Step 4: Commit Docker adapters**

```bash
git add tests/test_windows_deployment_assets.py deployment/windows/powershell/docker.ps1 deployment/windows/tests/run-unit-tests.ps1
git commit -m "feat: add Windows Docker adapters"
```

---

### Task 4: Add numbered double-click launchers

**Files:**

- Create: `deployment/windows/launchers/02-一键部署.bat`
- Create: `deployment/windows/launchers/03-启动系统.bat`
- Create: `deployment/windows/launchers/04-停止系统.bat`
- Create: `deployment/windows/launchers/05-查看运行状态.bat`
- Create: `deployment/windows/launchers/06-打开系统.bat`
- Create: `deployment/windows/launchers/07-导出诊断日志.bat`
- Modify: `tests/test_windows_deployment_assets.py`

- [ ] **Step 1: Add failing launcher contract tests**

Add this method:

```python
    def test_launchers_are_thin_utf8_powershell_entrypoints(self):
        expected = {
            "02-一键部署.bat": "deploy.ps1",
            "03-启动系统.bat": "start.ps1",
            "04-停止系统.bat": "stop.ps1",
            "05-查看运行状态.bat": "status.ps1",
            "06-打开系统.bat": "open.ps1",
            "07-导出诊断日志.bat": "diagnose.ps1",
        }
        for launcher_name, script_name in expected.items():
            content = (WINDOWS / "launchers" / launcher_name).read_text(encoding="utf-8-sig")
            self.assertIn("chcp 65001", content)
            self.assertIn("-ExecutionPolicy Bypass", content)
            self.assertIn(f"resources\\powershell\\{script_name}", content)
            self.assertIn("%~dp0", content)
            self.assertNotIn("docker compose", content.lower())
```

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_launchers_are_thin_utf8_powershell_entrypoints -v
```

Expected: errors for the six missing `.bat` files.

- [ ] **Step 2: Create the six batch files**

Use this complete template for each file, replacing only `SCRIPT_NAME` with the mapped PowerShell filename:

```bat
@echo off
chcp 65001 >nul
setlocal
set "PACKAGE_ROOT=%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%PACKAGE_ROOT%resources\powershell\SCRIPT_NAME"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%
```

Save the `.bat` files as UTF-8 with BOM so Windows PowerShell and Explorer preserve Chinese filenames and text.

- [ ] **Step 3: Run launcher tests**

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_launchers_are_thin_utf8_powershell_entrypoints -v
```

Expected: pass for all six launchers.

- [ ] **Step 4: Commit the launchers**

```bash
git add tests/test_windows_deployment_assets.py deployment/windows/launchers
git commit -m "feat: add Windows deployment launchers"
```

---

### Task 5: Implement first-time deployment orchestration

**Files:**

- Create: `deployment/windows/powershell/deploy.ps1`
- Modify: `tests/test_windows_deployment_assets.py`

- [ ] **Step 1: Add a failing deploy-script contract**

Add this method:

```python
    def test_deploy_script_has_required_stages_and_never_deletes_data(self):
        deploy = (WINDOWS / "powershell" / "deploy.ps1").read_text(encoding="utf-8")
        for token in [
            "Test-RequiredEnvironment",
            "Get-Port8501Owner",
            'Ensure-Directory -Path (Join-Path $context.UserDataRoot "data")',
            'Invoke-DockerCompose -Context $context -Arguments @("config", "--quiet")',
            'Invoke-DockerCompose -Context $context -Arguments @("up", "-d", "--build")',
            "Wait-CvAgentHealth",
            "Start-Process -FilePath $context.AppUrl",
        ]:
            self.assertIn(token, deploy)
        forbidden = ["down -v", "Remove-Item $context.UserDataRoot", "docker system prune"]
        for token in forbidden:
            self.assertNotIn(token.lower(), deploy.lower())
```

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_deploy_script_has_required_stages_and_never_deletes_data -v
```

Expected: error because `deployment/windows/powershell/deploy.ps1` is absent.

- [ ] **Step 2: Implement `deploy.ps1` using the shared adapters**

Create `deployment/windows/powershell/deploy.ps1` with this flow and exact helper usage:

```powershell
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$OutputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)

. (Join-Path $PSScriptRoot "common.ps1")
. (Join-Path $PSScriptRoot "docker.ps1")

$context = Get-DeploymentContext
$emptyValues = @{}
Ensure-Directory -Path $context.LogRoot
$logPath = New-DeploymentLog -Context $context -Prefix "deploy"
$values = $emptyValues

function Stop-Deployment {
    param([string]$Code, [string]$Reason)
    $text = Get-DeploymentErrorText -Code $Code -Reason $Reason -LogPath $logPath
    Write-SafeLog -Path $logPath -Text $text -Values $values
    Write-Host $text -ForegroundColor Red
    exit 1
}

try {
    Write-Stage -Current 1 -Total 6 -Message "正在检查 Docker 与系统环境……"
    if ($env:OS -ne "Windows_NT" -or $env:PROCESSOR_ARCHITECTURE -ne "AMD64") {
        Stop-Deployment -Code "D013" -Reason "检测到的系统不是 Windows x64。"
    }

    $requiredFiles = @(
        $context.ComposeFile,
        (Join-Path $context.ResourcesRoot "Dockerfile"),
        (Join-Path $context.ResourcesRoot ".dockerignore"),
        (Join-Path $context.ResourcesRoot "requirements.txt"),
        (Join-Path $context.ResourcesRoot "app"),
        $context.EnvFile
    )
    foreach ($path in $requiredFiles) {
        if (-not (Test-Path -LiteralPath $path)) {
            Stop-Deployment -Code "D001" -Reason "部署包缺少文件：$path"
        }
    }

    $driveName = ([System.IO.Path]::GetPathRoot($context.PackageRoot)).TrimEnd("\\").TrimEnd(":")
    $freeBytes = (Get-PSDrive -Name $driveName).Free
    if ($freeBytes -lt 4GB) {
        Stop-Deployment -Code "D011" -Reason "当前磁盘可用空间不足 4 GB。"
    }
    if ($freeBytes -lt 10GB) {
        Write-Host "提示：建议至少保留 10 GB 可用磁盘空间。" -ForegroundColor Yellow
    }

    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Stop-Deployment -Code "D002" -Reason "未找到 docker 命令。"
    }
    $composeVersion = Invoke-NativeCommand -FilePath "docker" -Arguments @("compose", "version")
    Write-SafeLog -Path $logPath -Text $composeVersion.Output -Values $values
    if ($composeVersion.ExitCode -ne 0) {
        Stop-Deployment -Code "D004" -Reason "docker compose version 执行失败。"
    }
    if (-not (Test-DockerEngine)) {
        [void](Start-DockerDesktop)
        if (-not (Wait-DockerEngine -TimeoutSeconds 120)) {
            Stop-Deployment -Code "D003" -Reason "Docker Engine 在 120 秒内未就绪。"
        }
    }

    $portOwner = Get-Port8501Owner
    if ($portOwner -eq "other") {
        Stop-Deployment -Code "D006" -Reason "8501 端口已被其他程序占用。"
    }

    Write-Stage -Current 2 -Total 6 -Message "正在检查部署配置……"
    $values = Read-DotEnvFile -Path $context.EnvFile
    if (-not (Test-RequiredEnvironment -Values $values)) {
        Stop-Deployment -Code "D005" -Reason "ARK API 或模型配置缺失。"
    }

    Write-Stage -Current 3 -Total 6 -Message "正在准备数据目录……"
    foreach ($name in @("data", "uploads", "results", "online_uploads", "benchmark_cache")) {
        Ensure-Directory -Path (Join-Path $context.UserDataRoot $name)
    }

    $configResult = Invoke-DockerCompose -Context $context -Arguments @("config", "--quiet")
    Write-SafeLog -Path $logPath -Text $configResult.Output -Values $values
    if ($configResult.ExitCode -ne 0) {
        Stop-Deployment -Code "D007" -Reason "Docker Compose 配置校验失败。"
    }

    Write-Stage -Current 4 -Total 6 -Message "正在下载并构建运行环境，首次运行可能需要 5～20 分钟……"
    $buildResult = Invoke-DockerCompose -Context $context -Arguments @("up", "-d", "--build")
    Write-SafeLog -Path $logPath -Text $buildResult.Output -Values $values
    if ($buildResult.ExitCode -ne 0) {
        Write-Host "首次构建失败，正在自动重试一次……" -ForegroundColor Yellow
        $buildResult = Invoke-DockerCompose -Context $context -Arguments @("up", "-d", "--build")
        Write-SafeLog -Path $logPath -Text $buildResult.Output -Values $values
    }
    if ($buildResult.ExitCode -ne 0) {
        $networkPattern = "timeout|connection|network|TLS|proxy|resolve|download"
        $code = if ($buildResult.Output -match $networkPattern) { "D012" } else { "D008" }
        Stop-Deployment -Code $code -Reason "Docker 镜像构建在重试后仍失败。"
    }

    Write-Stage -Current 5 -Total 6 -Message "正在启动油菜考种工作台……"
    if (-not (Test-CvAgentContainerExists -Context $context)) {
        Stop-Deployment -Code "D009" -Reason "Compose 未创建 cv-agent 容器。"
    }

    Write-Stage -Current 6 -Total 6 -Message "正在检查系统是否可用……"
    if (-not (Wait-CvAgentHealth -Url $context.HealthUrl -TimeoutSeconds 180)) {
        $logs = Invoke-DockerCompose -Context $context -Arguments @("logs", "--tail", "200", "cv-agent")
        Write-SafeLog -Path $logPath -Text $logs.Output -Values $values
        Stop-Deployment -Code "D010" -Reason "服务在 180 秒内未通过健康检查。"
    }

    Write-SafeLog -Path $logPath -Text "Deployment completed successfully." -Values $values
    Write-Host "油菜考种工作台部署成功。" -ForegroundColor Green
    Write-Host "访问地址：http://localhost:8501"
    Write-Host "以后使用时，请先启动 Docker Desktop，再双击“03-启动系统.bat”。"
    Start-Process -FilePath $context.AppUrl | Out-Null
    exit 0
}
catch {
    Stop-Deployment -Code "D099" -Reason $_.Exception.Message
}
```

- [ ] **Step 3: Run the deploy static contract and full Python suite**

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_deploy_script_has_required_stages_and_never_deletes_data -v
python3 -m unittest discover -s tests -v
```

Expected: the deploy contract and all existing application tests pass.

- [ ] **Step 4: Commit first-time deployment**

```bash
git add tests/test_windows_deployment_assets.py deployment/windows/powershell/deploy.ps1
git commit -m "feat: add Windows one-click deployment flow"
```

---

### Task 6: Implement daily lifecycle scripts

**Files:**

- Create: `deployment/windows/powershell/start.ps1`
- Create: `deployment/windows/powershell/stop.ps1`
- Create: `deployment/windows/powershell/status.ps1`
- Create: `deployment/windows/powershell/open.ps1`
- Modify: `tests/test_windows_deployment_assets.py`

- [ ] **Step 1: Add failing lifecycle contracts**

Add this method:

```python
    def test_lifecycle_scripts_preserve_data_and_have_expected_commands(self):
        scripts = {
            "start.ps1": ['@("start")', "Wait-CvAgentHealth"],
            "stop.ps1": ['@("stop")'],
            "status.ps1": ['@("ps")', "Test-CvAgentHealth"],
            "open.ps1": ["Test-CvAgentHealth", "Start-Process -FilePath $context.AppUrl"],
        }
        for name, required in scripts.items():
            content = (WINDOWS / "powershell" / name).read_text(encoding="utf-8")
            for token in required:
                self.assertIn(token, content)
            self.assertNotIn("down -v", content.lower())
            self.assertNotIn("Remove-Item", content)
```

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_lifecycle_scripts_preserve_data_and_have_expected_commands -v
```

Expected: errors for missing `start.ps1`, `stop.ps1`, `status.ps1`, and `open.ps1`.

- [ ] **Step 2: Implement `start.ps1`**

```powershell
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$OutputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
. (Join-Path $PSScriptRoot "common.ps1")
. (Join-Path $PSScriptRoot "docker.ps1")

$context = Get-DeploymentContext
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host (Get-DeploymentErrorText -Code "D002" -Reason "未找到 docker 命令。" -LogPath "无") -ForegroundColor Red
    exit 1
}
if (-not (Test-DockerEngine)) {
    [void](Start-DockerDesktop)
    if (-not (Wait-DockerEngine -TimeoutSeconds 120)) {
        Write-Host (Get-DeploymentErrorText -Code "D003" -Reason "Docker Engine 未启动。" -LogPath "无") -ForegroundColor Red
        exit 1
    }
}
if (-not (Test-CvAgentContainerExists -Context $context)) {
    Write-Host "尚未完成首次部署，请先双击“02-一键部署.bat”。" -ForegroundColor Yellow
    exit 1
}
$result = Invoke-DockerCompose -Context $context -Arguments @("start")
if ($result.ExitCode -ne 0) {
    Write-Host (Get-DeploymentErrorText -Code "D009" -Reason $result.Output -LogPath "无") -ForegroundColor Red
    exit 1
}
if (-not (Wait-CvAgentHealth -Url $context.HealthUrl -TimeoutSeconds 120)) {
    Write-Host (Get-DeploymentErrorText -Code "D010" -Reason "启动后健康检查超时。" -LogPath "无") -ForegroundColor Red
    exit 1
}
Write-Host "系统已启动：http://localhost:8501" -ForegroundColor Green
Start-Process -FilePath $context.AppUrl | Out-Null
exit 0
```

- [ ] **Step 3: Implement `stop.ps1`, `status.ps1`, and `open.ps1`**

Create `stop.ps1`:

```powershell
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$OutputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
. (Join-Path $PSScriptRoot "common.ps1")
. (Join-Path $PSScriptRoot "docker.ps1")
$context = Get-DeploymentContext
if (-not (Test-DockerEngine)) {
    Write-Host "Docker Desktop 当前未运行，系统已经处于停止状态。" -ForegroundColor Yellow
    exit 0
}
$result = Invoke-DockerCompose -Context $context -Arguments @("stop")
if ($result.ExitCode -ne 0) {
    Write-Host (Get-DeploymentErrorText -Code "D009" -Reason $result.Output -LogPath "无") -ForegroundColor Red
    exit 1
}
Write-Host "系统已停止，历史记录和分析数据已保留。" -ForegroundColor Green
exit 0
```

Create `status.ps1`:

```powershell
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$OutputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
. (Join-Path $PSScriptRoot "common.ps1")
. (Join-Path $PSScriptRoot "docker.ps1")
$context = Get-DeploymentContext
$dockerCli = [bool](Get-Command docker -ErrorAction SilentlyContinue)
$engine = if ($dockerCli) { Test-DockerEngine } else { $false }
Write-Host ("Docker 命令：" + $(if ($dockerCli) { "正常" } else { "未安装" }))
Write-Host ("Docker Engine：" + $(if ($engine) { "运行中" } else { "未运行" }))
if ($engine) {
    $compose = Invoke-DockerCompose -Context $context -Arguments @("ps")
    Write-Host $compose.Output
}
$healthy = Test-CvAgentHealth -Url $context.HealthUrl
Write-Host ("系统健康状态：" + $(if ($healthy) { "正常" } else { "不可用" })) -ForegroundColor $(if ($healthy) { "Green" } else { "Yellow" })
if ($healthy) { exit 0 }
exit 1
```

Create `open.ps1`:

```powershell
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$OutputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
. (Join-Path $PSScriptRoot "common.ps1")
$context = Get-DeploymentContext
if (-not (Test-CvAgentHealth -Url $context.HealthUrl)) {
    Write-Host "系统尚未运行，请先双击“03-启动系统.bat”。" -ForegroundColor Yellow
    exit 1
}
Start-Process -FilePath $context.AppUrl | Out-Null
Write-Host "已打开油菜考种工作台。" -ForegroundColor Green
exit 0
```

- [ ] **Step 4: Run lifecycle tests**

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_lifecycle_scripts_preserve_data_and_have_expected_commands -v
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit lifecycle scripts**

```bash
git add tests/test_windows_deployment_assets.py deployment/windows/powershell/start.ps1 deployment/windows/powershell/stop.ps1 deployment/windows/powershell/status.ps1 deployment/windows/powershell/open.ps1
git commit -m "feat: add Windows service lifecycle commands"
```

---

### Task 7: Implement sanitized diagnostics

**Files:**

- Create: `deployment/windows/powershell/diagnose.ps1`
- Modify: `tests/test_windows_deployment_assets.py`

- [ ] **Step 1: Add a failing diagnostics security contract**

Add this method:

```python
    def test_diagnostics_collect_only_sanitized_operational_metadata(self):
        diagnose = (WINDOWS / "powershell" / "diagnose.ps1").read_text(encoding="utf-8")
        self.assertIn("Protect-SecretText", diagnose)
        self.assertIn("Compress-Archive", diagnose)
        self.assertIn('Arguments @("logs", "--tail", "300", "cv-agent")', diagnose)
        self.assertIn("Latest deployment log", diagnose)
        self.assertIn("Required file", diagnose)
        for forbidden in [
            "Copy-Item $context.EnvFile",
            "Get-ChildItem Env:",
            "history.sqlite3",
            "user-data\\uploads",
            "user-data\\results",
        ]:
            self.assertNotIn(forbidden, diagnose)
```

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_diagnostics_collect_only_sanitized_operational_metadata -v
```

Expected: error because `deployment/windows/powershell/diagnose.ps1` is absent.

- [ ] **Step 2: Implement `diagnose.ps1`**

Create a report-only staging directory, redact every collected string, compress it, and delete only the explicitly created staging directory:

```powershell
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$OutputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
. (Join-Path $PSScriptRoot "common.ps1")
. (Join-Path $PSScriptRoot "docker.ps1")

$context = Get-DeploymentContext
Ensure-Directory -Path $context.DiagnosticsRoot
$values = if (Test-Path -LiteralPath $context.EnvFile) { Read-DotEnvFile -Path $context.EnvFile } else { @{} }
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$staging = Join-Path $context.DiagnosticsRoot ("staging-" + [guid]::NewGuid().ToString("N"))
$zipPath = Join-Path $context.DiagnosticsRoot "cv-agent-diagnostics-$stamp.zip"
Ensure-Directory -Path $staging

try {
    $lines = New-Object System.Collections.Generic.List[string]
    $os = Get-CimInstance Win32_OperatingSystem
    $computer = Get-CimInstance Win32_ComputerSystem
    $lines.Add("Generated: $(Get-Date -Format o)")
    $lines.Add("Windows: $($os.Caption) $($os.Version) build $($os.BuildNumber)")
    $lines.Add("Architecture: $env:PROCESSOR_ARCHITECTURE")
    $lines.Add("MemoryGB: $([math]::Round($computer.TotalPhysicalMemory / 1GB, 2))")
    $lines.Add("Port8501: $(Get-Port8501Owner)")
    $lines.Add("Health: $(Test-CvAgentHealth -Url $context.HealthUrl)")

    if (Get-Command docker -ErrorAction SilentlyContinue) {
        foreach ($args in @(
            @("version"),
            @("compose", "version")
        )) {
            $result = Invoke-NativeCommand -FilePath "docker" -Arguments $args
            $lines.Add((Protect-SecretText -Text $result.Output -Values $values))
        }
        if (Test-DockerEngine) {
            $psResult = Invoke-DockerCompose -Context $context -Arguments @("ps")
            $logResult = Invoke-DockerCompose -Context $context -Arguments @("logs", "--tail", "300", "cv-agent")
            $lines.Add((Protect-SecretText -Text $psResult.Output -Values $values))
            $lines.Add((Protect-SecretText -Text $logResult.Output -Values $values))
        }
    }
    else {
        $lines.Add("Docker CLI: not found")
    }

    foreach ($name in @("data", "uploads", "results", "online_uploads", "benchmark_cache")) {
        $path = Join-Path $context.UserDataRoot $name
        if (Test-Path -LiteralPath $path) {
            $size = (Get-ChildItem -LiteralPath $path -File -Recurse -ErrorAction SilentlyContinue |
                Measure-Object -Property Length -Sum).Sum
            if ($null -eq $size) { $size = 0 }
            $lines.Add("Directory $name exists; bytes=$size")
        }
        else {
            $lines.Add("Directory $name missing")
        }
    }

    $requiredFiles = @(
        $context.ComposeFile,
        (Join-Path $context.ResourcesRoot "Dockerfile"),
        (Join-Path $context.ResourcesRoot ".dockerignore"),
        (Join-Path $context.ResourcesRoot "requirements.txt"),
        (Join-Path $context.ResourcesRoot "app"),
        $context.EnvFile
    )
    foreach ($requiredFile in $requiredFiles) {
        $lines.Add("Required file $requiredFile exists=$(Test-Path -LiteralPath $requiredFile)")
    }

    $latestLog = Get-ChildItem -LiteralPath $context.LogRoot -Filter "*.log" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -ne $latestLog) {
        $lines.Add("Latest deployment log: $($latestLog.Name)")
        $lines.Add((Protect-SecretText -Text ([System.IO.File]::ReadAllText($latestLog.FullName)) -Values $values))
    }

    $report = Protect-SecretText -Text ($lines -join [Environment]::NewLine) -Values $values
    $reportPath = Join-Path $staging "diagnostics.txt"
    [System.IO.File]::WriteAllText($reportPath, $report, (New-Object System.Text.UTF8Encoding($false)))
    Compress-Archive -LiteralPath $reportPath -DestinationPath $zipPath -CompressionLevel Optimal
    Write-Host "诊断包已生成：$zipPath" -ForegroundColor Green
    exit 0
}
catch {
    Write-Host "诊断包生成失败：$($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
finally {
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
}
```

- [ ] **Step 3: Run diagnostics tests**

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_diagnostics_collect_only_sanitized_operational_metadata -v
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit diagnostics**

```bash
git add tests/test_windows_deployment_assets.py deployment/windows/powershell/diagnose.ps1
git commit -m "feat: add sanitized Windows diagnostics"
```

---

### Task 8: Create and render the Docker Desktop PDF guide

**Files:**

- Create: `deployment/windows/docs/docker-desktop-install-guide.html`
- Create: `scripts/render_windows_guide_pdf.sh`
- Modify: `tests/test_windows_deployment_assets.py`

- [ ] **Step 1: Re-check official documentation before writing the guide**

Use `agent-reach` Exa search and restrict selected sources to the official pages below:

- `https://docs.docker.com/desktop/setup/install/windows-install/`
- `https://docs.docker.com/desktop/features/wsl/`
- `https://docs.docker.com/desktop/troubleshoot-and-support/troubleshoot/topics/`
- `https://learn.microsoft.com/en-us/windows/wsl/install`
- `https://learn.microsoft.com/en-us/windows/wsl/basic-commands`
- `https://support.microsoft.com/en-us/windows/experience/enable-virtualization-on-windows`

Record the verification date `2026-07-19` in the HTML. If official requirements have changed at implementation time, update the tutorial text and the specification evidence, but keep the deployment scope unchanged.

- [ ] **Step 2: Add failing guide and renderer tests**

Add these methods:

```python
    def test_docker_guide_contains_official_requirements_and_user_flow(self):
        guide = (WINDOWS / "docs" / "docker-desktop-install-guide.html").read_text(encoding="utf-8")
        for token in [
            "Windows 10 22H2",
            "Windows 11 23H2",
            "WSL 2.1.5",
            "8 GB",
            "wsl --install",
            "wsl --update",
            "docker compose version",
            "02-一键部署.bat",
            "2026-07-19",
            "docs.docker.com",
            "learn.microsoft.com",
        ]:
            self.assertIn(token, guide)

    def test_pdf_renderer_uses_headless_chrome_and_validates_pdf_header(self):
        renderer = (ROOT / "scripts" / "render_windows_guide_pdf.sh").read_text(encoding="utf-8")
        self.assertIn("--headless", renderer)
        self.assertIn("--print-to-pdf", renderer)
        self.assertIn("%PDF", renderer)
```

Run:

```bash
python3 -m unittest \
  tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_docker_guide_contains_official_requirements_and_user_flow \
  tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_pdf_renderer_uses_headless_chrome_and_validates_pdf_header \
  -v
```

Expected: errors because the guide HTML and renderer script are absent.

- [ ] **Step 3: Create the maintained HTML tutorial**

Create `deployment/windows/docs/docker-desktop-install-guide.html` as a standalone A4 document with inline CSS. It must contain these concrete sections and instructions:

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>油菜考种工作台：Docker Desktop 安装教程</title>
  <style>
    @page { size: A4; margin: 15mm; }
    * { box-sizing: border-box; }
    body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; color: #172033; line-height: 1.65; font-size: 14px; }
    h1 { font-size: 28px; color: #0b5cab; margin-bottom: 8px; }
    h2 { font-size: 21px; color: #0b5cab; border-bottom: 2px solid #dcecff; padding-bottom: 6px; break-after: avoid; }
    h3 { font-size: 17px; color: #173b67; break-after: avoid; }
    .cover { min-height: 245mm; display: flex; flex-direction: column; justify-content: center; }
    .step { border: 1px solid #c9d8eb; border-radius: 10px; padding: 14px 16px; margin: 12px 0; break-inside: avoid; }
    .ok { background: #ecf8f0; border-left: 5px solid #2e8b57; padding: 10px 12px; }
    .warn { background: #fff7e6; border-left: 5px solid #d98b00; padding: 10px 12px; }
    code, pre { font-family: Consolas, monospace; background: #f3f6fa; }
    pre { padding: 10px 12px; border-radius: 6px; overflow-wrap: anywhere; white-space: pre-wrap; }
    a { color: #0b5cab; word-break: break-all; }
    .page-break { break-before: page; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border: 1px solid #c9d8eb; padding: 8px; vertical-align: top; }
  </style>
</head>
<body>
  <section class="cover">
    <h1>油菜考种工作台</h1>
    <h2>Docker Desktop 安装与首次部署教程</h2>
    <p>适用对象：Windows 10/11、无开发环境经验的使用者</p>
    <p>文档核对日期：2026-07-19</p>
    <div class="ok">完成本教程后，只需双击“02-一键部署.bat”。无需安装 Python，也无需填写 API Key。</div>
  </section>

  <section class="page-break">
    <h2>1. 安装前检查</h2>
    <table>
      <tr><th>项目</th><th>要求</th></tr>
      <tr><td>系统</td><td>Windows 10 22H2 build 19045，或 Windows 11 23H2 build 22631 及以上的 64 位版本</td></tr>
      <tr><td>WSL</td><td>WSL 2.1.5 或更高版本，建议更新到最新版</td></tr>
      <tr><td>内存</td><td>8 GB 或更多</td></tr>
      <tr><td>磁盘</td><td>建议至少保留 10 GB 可用空间</td></tr>
      <tr><td>处理器</td><td>64 位 x64，并支持 SLAT 与硬件虚拟化</td></tr>
      <tr><td>网络</td><td>首次部署时保持稳定联网</td></tr>
    </table>
    <div class="step"><h3>检查系统版本</h3><p>按 Win + R，输入 <code>winver</code>，按回车。记录版本和操作系统内部版本。</p></div>
    <div class="step"><h3>检查虚拟化</h3><p>打开任务管理器 → 性能 → CPU，确认“虚拟化”显示“已启用”。若未启用，按电脑厂商说明进入 BIOS/UEFI，启用 Intel Virtualization Technology、VT-x、AMD-V 或 SVM Mode。</p></div>
  </section>

  <section class="page-break">
    <h2>2. 安装和更新 WSL2</h2>
    <div class="step"><h3>以管理员身份打开 PowerShell</h3><p>右键开始菜单，选择“终端（管理员）”或“Windows PowerShell（管理员）”。</p></div>
    <div class="step"><h3>安装 WSL</h3><pre>wsl --install</pre><p>命令完成后重启电脑。</p></div>
    <div class="step"><h3>更新并验证 WSL</h3><pre>wsl --update
wsl --version</pre><p>确认 WSL 版本不低于 2.1.5。</p></div>
    <div class="warn">如果下载停在 0.0%，参考 Microsoft 官方说明使用 web-download 方式；不要从未知网站下载 WSL 安装包。</div>
  </section>

  <section class="page-break">
    <h2>3. 安装 Docker Desktop</h2>
    <p>只使用 Docker 官方页面：<a href="https://docs.docker.com/desktop/setup/install/windows-install/">Docker Desktop for Windows 安装说明</a></p>
    <div class="step"><h3>下载安装程序</h3><p>从官方页面下载 Docker Desktop Installer。对于本项目，选择 WSL 2 后端和 Linux containers。</p></div>
    <div class="step"><h3>首次启动</h3><p>安装结束后从开始菜单打开 Docker Desktop，接受许可条款，等待界面显示 Docker Engine 已运行。</p></div>
    <div class="step"><h3>验证命令</h3><pre>docker version
docker compose version</pre><p>两条命令都能显示版本信息才继续。</p></div>
    <div class="ok">Docker Desktop 正常运行后，不需要在 WSL 中单独安装 Docker Engine。</div>
  </section>

  <section class="page-break">
    <h2>4. 部署油菜考种工作台</h2>
    <ol>
      <li>右键收到的 ZIP，选择“全部解压”。</li>
      <li>进入解压后的文件夹。</li>
      <li>双击“02-一键部署.bat”。</li>
      <li>首次构建可能需要 5～20 分钟，请不要关闭窗口或退出 Docker Desktop。</li>
      <li>看到“部署成功”后，浏览器会自动打开 http://localhost:8501。</li>
    </ol>
    <h3>日常使用</h3>
    <table>
      <tr><th>文件</th><th>用途</th></tr>
      <tr><td>03-启动系统.bat</td><td>启动已有系统并打开浏览器</td></tr>
      <tr><td>04-停止系统.bat</td><td>停止服务但保留全部数据</td></tr>
      <tr><td>05-查看运行状态.bat</td><td>查看 Docker、容器和健康状态</td></tr>
      <tr><td>06-打开系统.bat</td><td>系统运行时打开网页</td></tr>
      <tr><td>07-导出诊断日志.bat</td><td>生成可发送给技术人员的脱敏诊断包</td></tr>
    </table>
  </section>

  <section class="page-break">
    <h2>5. 常见问题</h2>
    <div class="step"><h3>Docker Engine 一直未启动</h3><p>先运行 <code>wsl --update</code>，重启电脑，确认任务管理器中的虚拟化已启用，再重新打开 Docker Desktop。</p></div>
    <div class="step"><h3>出现 Unexpected WSL error</h3><p>检查“Windows Subsystem for Linux”和“虚拟机平台”功能是否启用，并参考 Docker 官方排障页面。</p></div>
    <div class="step"><h3>部署下载失败</h3><p>确认浏览器可以联网，保持 Docker Desktop 运行，然后重新双击一键部署。脚本不会删除已下载内容和用户数据。</p></div>
    <div class="step"><h3>仍无法解决</h3><p>双击“07-导出诊断日志.bat”，把生成的 ZIP 发给技术人员。诊断包不包含 API Key、上传图片、结果图片或历史数据库。</p></div>
  </section>

  <section class="page-break">
    <h2>6. 官方资料</h2>
    <ul>
      <li><a href="https://docs.docker.com/desktop/setup/install/windows-install/">Docker Desktop Windows 安装与系统要求</a></li>
      <li><a href="https://docs.docker.com/desktop/features/wsl/">Docker Desktop WSL2 后端</a></li>
      <li><a href="https://docs.docker.com/desktop/troubleshoot-and-support/troubleshoot/topics/">Docker Desktop Windows 排障主题</a></li>
      <li><a href="https://learn.microsoft.com/en-us/windows/wsl/install">Microsoft WSL 安装</a></li>
      <li><a href="https://learn.microsoft.com/en-us/windows/wsl/basic-commands">Microsoft WSL 常用命令</a></li>
      <li><a href="https://support.microsoft.com/en-us/windows/experience/enable-virtualization-on-windows">Microsoft Windows 虚拟化说明</a></li>
    </ul>
  </section>
</body>
</html>
```

- [ ] **Step 4: Create the deterministic renderer**

Create `scripts/render_windows_guide_pdf.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
html_path="${1:-$repo_root/deployment/windows/docs/docker-desktop-install-guide.html}"
pdf_path="${2:-$repo_root/dist/01-Docker安装教程.pdf}"

if [[ ! -f "$html_path" ]]; then
  echo "Guide HTML not found: $html_path" >&2
  exit 1
fi

mkdir -p "$(dirname "$pdf_path")"

browser=""
for candidate in \
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" \
  "/Applications/Chromium.app/Contents/MacOS/Chromium" \
  "$(command -v google-chrome 2>/dev/null || true)" \
  "$(command -v chromium 2>/dev/null || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    browser="$candidate"
    break
  fi
done

if [[ -z "$browser" ]]; then
  echo "No supported Chrome/Edge/Chromium executable found." >&2
  exit 1
fi

"$browser" \
  --headless=new \
  --disable-gpu \
  --no-pdf-header-footer \
  --print-to-pdf="$pdf_path" \
  "file://$html_path"

if [[ "$(head -c 4 "$pdf_path")" != "%PDF" ]]; then
  echo "Generated file is not a PDF: $pdf_path" >&2
  exit 1
fi

echo "$pdf_path"
```

Make it executable:

```bash
chmod +x scripts/render_windows_guide_pdf.sh
```

- [ ] **Step 5: Run guide tests and render the PDF**

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets -v
./scripts/render_windows_guide_pdf.sh
test "$(head -c 4 dist/01-Docker安装教程.pdf)" = "%PDF"
```

Expected: Python tests pass, Chrome reports a PDF written, and the header check succeeds. Open the generated PDF and visually inspect page breaks, Chinese fonts, links, and code blocks.

- [ ] **Step 6: Commit guide source and renderer, not the generated PDF**

```bash
git add tests/test_windows_deployment_assets.py deployment/windows/docs/docker-desktop-install-guide.html scripts/render_windows_guide_pdf.sh
git commit -m "docs: add Windows Docker installation guide"
```

---

### Task 9: Build the trusted Windows release artifact

**Files:**

- Create: `scripts/build_windows_release.sh`
- Create: `tests/test_windows_release_builder.py`
- Create: `deployment/windows/README-请先阅读.txt`

- [ ] **Step 1: Write the failing release-builder test**

Create `tests/test_windows_release_builder.py`:

```python
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WindowsReleaseBuilderTests(unittest.TestCase):
    def test_builder_creates_minimal_release_and_confines_fake_secret(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            env_file = temp_path / "release.env"
            dist_dir = temp_path / "dist"
            fake_pdf = temp_path / "guide.pdf"
            fake_secret = "test-secret-release-only-12345"
            env_file.write_text(
                "\n".join(
                    [
                        f"ARK_API_KEY={fake_secret}",
                        "ARK_API_BASE=https://example.invalid/api/v3",
                        "VLM_MODEL=test-model",
                        "VLM_FP_CONFIDENCE_THRESHOLD=0.8",
                        "HOST=0.0.0.0",
                        "PORT=8501",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            fake_pdf.write_bytes(b"%PDF-1.4\n% test guide\n")
            process_env = os.environ.copy()
            process_env.update(
                {
                    "CV_AGENT_ENV_FILE": str(env_file),
                    "CV_AGENT_DIST_DIR": str(dist_dir),
                    "CV_AGENT_GUIDE_PDF": str(fake_pdf),
                    "CV_AGENT_RELEASE_TAG": "test-release",
                }
            )
            result = subprocess.run(
                [str(ROOT / "scripts" / "build_windows_release.sh")],
                cwd=ROOT,
                env=process_env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            package = dist_dir / "油菜考种工作台-Windows版-test-release"
            archive = dist_dir / "油菜考种工作台-Windows版-test-release.zip"
            self.assertTrue(package.is_dir())
            self.assertTrue(archive.is_file())
            self.assertTrue((package / "01-Docker安装教程.pdf").read_bytes().startswith(b"%PDF"))
            self.assertEqual(
                fake_secret,
                next(
                    line.split("=", 1)[1]
                    for line in (package / "resources" / "config" / "deployment.env").read_text(encoding="utf-8").splitlines()
                    if line.startswith("ARK_API_KEY=")
                ),
            )
            self.assertTrue((package / "resources" / "app" / "main.py").is_file())
            self.assertTrue((package / "02-一键部署.bat").is_file())
            self.assertFalse((package / ".git").exists())
            self.assertFalse((package / "venv").exists())

            secret_locations = []
            for path in package.rglob("*"):
                if path.is_file() and path.name != "deployment.env":
                    if fake_secret.encode() in path.read_bytes():
                        secret_locations.append(str(path.relative_to(package)))
            self.assertEqual([], secret_locations)

            with zipfile.ZipFile(archive) as zipped:
                names = zipped.namelist()
                self.assertTrue(any(name.endswith("resources/app/main.py") for name in names))
                self.assertFalse(any("/.git/" in name or "/venv/" in name for name in names))


if __name__ == "__main__":
    unittest.main()
```

Run:

```bash
python3 -m unittest tests.test_windows_release_builder -v
```

Expected: error because `scripts/build_windows_release.sh` is absent.

- [ ] **Step 2: Create the user-facing short README**

Create `deployment/windows/README-请先阅读.txt`:

```text
油菜考种工作台 Windows 版

首次使用：
1. 打开“01-Docker安装教程.pdf”。
2. 安装并启动 Docker Desktop。
3. 双击“02-一键部署.bat”。
4. 等待窗口显示部署成功，浏览器会自动打开系统。

日常使用：
- 启动：03-启动系统.bat
- 停止：04-停止系统.bat
- 查看状态：05-查看运行状态.bat
- 打开网页：06-打开系统.bat
- 出现问题：07-导出诊断日志.bat

请勿删除 user-data 文件夹，其中保存历史记录、上传图片和分析结果。
请勿把整个安装包或 resources/config/deployment.env 转发给不可信人员。
```

- [ ] **Step 3: Implement the release builder**

Create `scripts/build_windows_release.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
env_file="${CV_AGENT_ENV_FILE:-$repo_root/.env}"
dist_dir="${CV_AGENT_DIST_DIR:-$repo_root/dist}"
release_tag="${CV_AGENT_RELEASE_TAG:-$(date +%Y%m%d-%H%M%S)}"
release_name="油菜考种工作台-Windows版-$release_tag"
guide_override="${CV_AGENT_GUIDE_PDF:-}"

if [[ ! -f "$env_file" ]]; then
  echo "Environment file not found: $env_file" >&2
  exit 1
fi

for variable_name in ARK_API_KEY ARK_API_BASE VLM_MODEL VLM_FP_CONFIDENCE_THRESHOLD; do
  if ! awk -F= -v name="$variable_name" '$1 == name && length(substr($0, index($0, "=") + 1)) > 0 { found=1 } END { exit !found }' "$env_file"; then
    echo "Required environment variable is missing: $variable_name" >&2
    exit 1
  fi
done

mkdir -p "$dist_dir"
package_dir="$dist_dir/$release_name"
archive_path="$dist_dir/$release_name.zip"
if [[ -e "$package_dir" || -e "$archive_path" ]]; then
  echo "Release already exists: $release_name" >&2
  exit 1
fi

staging_root="$(mktemp -d "${TMPDIR:-/tmp}/cv-agent-windows-release.XXXXXX")"
trap 'rm -rf "$staging_root"' EXIT
staging_package="$staging_root/$release_name"

mkdir -p \
  "$staging_package/resources/powershell" \
  "$staging_package/resources/config" \
  "$staging_package/user-data/data" \
  "$staging_package/user-data/uploads" \
  "$staging_package/user-data/results" \
  "$staging_package/user-data/online_uploads" \
  "$staging_package/user-data/benchmark_cache"

cp "$repo_root/deployment/windows/README-请先阅读.txt" "$staging_package/"
cp "$repo_root/deployment/windows/launchers/"*.bat "$staging_package/"
cp "$repo_root/deployment/windows/powershell/"*.ps1 "$staging_package/resources/powershell/"
cp "$repo_root/deployment/windows/resources/docker-compose.yml" "$staging_package/resources/"
cp "$repo_root/deployment/windows/resources/.dockerignore" "$staging_package/resources/"
cp "$repo_root/Dockerfile" "$staging_package/resources/"
cp "$repo_root/requirements.txt" "$staging_package/resources/"
mkdir -p "$staging_package/resources/app"
rsync -a \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  "$repo_root/app/" "$staging_package/resources/app/"
cp "$env_file" "$staging_package/resources/config/deployment.env"
chmod 600 "$staging_package/resources/config/deployment.env"

if [[ -n "$guide_override" ]]; then
  cp "$guide_override" "$staging_package/01-Docker安装教程.pdf"
else
  "$repo_root/scripts/render_windows_guide_pdf.sh" \
    "$repo_root/deployment/windows/docs/docker-desktop-install-guide.html" \
    "$staging_package/01-Docker安装教程.pdf" >/dev/null
fi

if [[ "$(head -c 4 "$staging_package/01-Docker安装教程.pdf")" != "%PDF" ]]; then
  echo "Guide PDF validation failed." >&2
  exit 1
fi

cp -R "$staging_package" "$dist_dir/"
(
  cd "$staging_root"
  zip -qr "$archive_path" "$release_name"
)

echo "$package_dir"
echo "$archive_path"
```

Make it executable:

```bash
chmod +x scripts/build_windows_release.sh
```

- [ ] **Step 4: Run the release-builder test**

Run:

```bash
python3 -m unittest tests.test_windows_release_builder -v
```

Expected: pass. The fake API key appears only in the temporary release's `resources/config/deployment.env` and nowhere else in extracted plaintext files.

- [ ] **Step 5: Run the complete Python suite**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: all existing history/API tests and all Windows deployment tests pass.

- [ ] **Step 6: Commit release generation**

```bash
git add tests/test_windows_release_builder.py deployment/windows/README-请先阅读.txt scripts/build_windows_release.sh
git commit -m "feat: build trusted Windows release package"
```

---

### Task 10: Add maintainer docs and real-Windows acceptance checklist

**Files:**

- Modify: `README.md`
- Modify: `DOCKER.md`
- Create: `deployment/windows/WINDOWS-ACCEPTANCE.md`
- Modify: `tests/test_windows_deployment_assets.py`

- [ ] **Step 1: Add a failing documentation contract**

Add this method:

```python
    def test_maintainer_docs_describe_windows_release_and_secret_boundary(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        docker_doc = (ROOT / "DOCKER.md").read_text(encoding="utf-8")
        acceptance = (WINDOWS / "WINDOWS-ACCEPTANCE.md").read_text(encoding="utf-8")
        self.assertIn("build_windows_release.sh", readme)
        self.assertIn("dist/", readme)
        self.assertIn("deployment.env", docker_doc)
        self.assertIn("不得提交", docker_doc)
        self.assertIn("Windows PowerShell 5.1", acceptance)
        self.assertIn("真实 API Key 未出现在诊断包", acceptance)
```

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_maintainer_docs_describe_windows_release_and_secret_boundary -v
```

Expected: error for the missing acceptance file and failures for the missing README/DOCKER sections.

- [ ] **Step 2: Add the Windows release section to `README.md`**

Add a section after the existing Docker startup instructions:

````markdown
## Windows 可信交付包

Windows 用户先按照交付包中的 `01-Docker安装教程.pdf` 安装并启动 Docker Desktop，然后双击 `02-一键部署.bat`。用户不需要安装 Python、编辑 `.env` 或输入 Docker 命令。

维护者在 macOS/Linux 上生成交付包：

```bash
./scripts/build_windows_release.sh
```

脚本从本机 `.env` 注入可信配置，生成名称形如 `dist/油菜考种工作台-Windows版-20260719-153000/` 的目录和同名 ZIP。`dist/` 不受 Git 管理，交付物包含真实 API Key，只能发送给已确认的可信用户。

部署源码、PowerShell 脚本、PDF 源文件和 Windows 验收清单位于 `deployment/windows/`。
````

- [ ] **Step 3: Add secret and validation details to `DOCKER.md`**

Append:

````markdown
## Windows 一键部署包

Windows 发布包使用 `deployment/windows/resources/docker-compose.yml`，持久化目录位于交付包根目录的 `user-data/`。重复部署和镜像重建不会删除这些目录。

发布命令：

```bash
./scripts/build_windows_release.sh
```

安全边界：

- `scripts/build_windows_release.sh` 会把本机 `.env` 复制为发布包中的 `resources/config/deployment.env`。
- `dist/` 和 `.env` 均不得提交到 Git。
- 发布专用 `.dockerignore` 必须排除 `config/deployment.env`，真实 API Key 不得进入镜像层。
- 普通日志和诊断包不得包含环境文件、容器完整环境、上传图片、结果图片或 SQLite 数据库。
- 本方案只适用于可信接收方；本机管理员理论上可以读取交付包中的凭据。

验证命令：

```bash
python3 -m unittest discover -s tests -v
./scripts/render_windows_guide_pdf.sh
./scripts/build_windows_release.sh
```
````

- [ ] **Step 4: Create the Windows acceptance checklist**

Create `deployment/windows/WINDOWS-ACCEPTANCE.md`:

```markdown
# Windows 一键部署验收记录

测试电脑必须为 Windows 10 22H2 x64 或 Windows 11 23H2+ x64。

测试 PowerShell 必须为 Windows PowerShell 5.1。

验收证据必须附上 `winver`、`docker version`、`docker compose version` 和测试日期的实际输出。

## 自动化检查

- [ ] 在仓库根目录运行 `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\windows\tests\run-unit-tests.ps1`，输出 `All PowerShell unit tests passed.`。开发测试不进入用户发布 ZIP。
- [ ] `docker version` 同时显示 Client 和 Server。
- [ ] `docker compose version` 成功。

## 首次部署

- [ ] Docker Desktop 已运行时，双击 `02-一键部署.bat` 可完成构建。
- [ ] 安装目录包含中文和空格时仍可部署。
- [ ] 浏览器自动打开 `http://localhost:8501`。
- [ ] `http://localhost:8501/api/health` 返回 `status=ok` 和 `service=cv-agent`。
- [ ] 使用真实油菜图片完成一次豆包 VLM 分析。

## 异常路径

- [ ] Docker Desktop 未启动时，脚本会尝试启动并等待。
- [ ] Docker 未安装时显示 D002。
- [ ] 8501 被其他程序占用时显示 D006。
- [ ] 临时断网时构建只重试一次并保存日志。
- [ ] 配置缺失时显示 D005，但不打印配置值。

## 数据保护

- [ ] 在 `user-data/data` 写入测试标记后重复部署，标记仍存在。
- [ ] 停止并重新启动后历史数据库仍存在。
- [ ] 脚本未执行 `docker compose down -v`、`docker system prune` 或删除 `user-data`。

## 诊断与秘密

- [ ] `07-导出诊断日志.bat` 成功生成 ZIP。
- [ ] 诊断包不包含 `deployment.env`。
- [ ] 诊断包不包含上传图片、结果图片或 `history.sqlite3`。
- [ ] 真实 API Key 未出现在诊断包、部署日志或容器日志副本中。
- [ ] 使用 `docker history --no-trunc cv-agent:windows` 和临时容器文件检查，真实 API Key 未进入镜像层。

## 结论

- [ ] 通过，可交付。
- [ ] 未通过；记录错误编号、日志路径和复现步骤。
```

- [ ] **Step 5: Run documentation contract and full tests**

Run:

```bash
python3 -m unittest tests.test_windows_deployment_assets.WindowsDeploymentAssetTests.test_maintainer_docs_describe_windows_release_and_secret_boundary -v
python3 -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit maintainer documentation**

```bash
git add README.md DOCKER.md deployment/windows/WINDOWS-ACCEPTANCE.md tests/test_windows_deployment_assets.py
git commit -m "docs: document Windows release validation"
```

---

### Task 11: Perform final local verification and build the trusted artifact

**Files:**

- Verify only: all files created above
- Generate but do not commit: `dist/01-Docker安装教程.pdf`
- Generate but do not commit: the newest directory matching `dist/油菜考种工作台-Windows版-*/`
- Generate but do not commit: the same release name with a `.zip` suffix

- [ ] **Step 1: Invoke the verification-before-completion workflow**

Before making any success claim, use `superpowers:verification-before-completion` and run every command below from a clean implementation worktree.

- [ ] **Step 2: Run source and test verification**

```bash
git diff --check
python3 -m unittest discover -s tests -v
ARK_API_KEY=test-key ARK_API_BASE=https://example.invalid VLM_MODEL=test-model VLM_FP_CONFIDENCE_THRESHOLD=0.8 \
  docker compose -f deployment/windows/resources/docker-compose.yml config --quiet
```

Expected: no diff errors, all tests pass, and Compose config exits 0 without printing the fake key.

- [ ] **Step 3: Render and visually inspect the real PDF**

```bash
./scripts/render_windows_guide_pdf.sh
test "$(head -c 4 dist/01-Docker安装教程.pdf)" = "%PDF"
```

Open `dist/01-Docker安装教程.pdf` and verify A4 pagination, Chinese glyphs, clickable official links, commands, and absence of clipped text.

- [ ] **Step 4: Start Docker Desktop locally and build the production image**

The current macOS Docker CLI is installed but the daemon was not running during planning. Start Docker Desktop, then run:

```bash
docker info
docker build -t cv-agent:windows-verification \
  -f Dockerfile \
  .
verification_container="cv-agent-verification-$$"
docker run --rm -d --name "$verification_container" -p 18501:8501 \
  -e ARK_API_KEY=test-key \
  -e ARK_API_BASE=https://example.invalid \
  -e VLM_MODEL=test-model \
  -e VLM_FP_CONFIDENCE_THRESHOLD=0.8 \
  cv-agent:windows-verification
```

Poll `http://127.0.0.1:18501/api/health` until it returns `status=ok`, then stop only the explicit verification container:

```bash
curl --fail --silent http://127.0.0.1:18501/api/health
docker stop "$verification_container"
```

Expected: health JSON contains `"status":"ok"` and `"service":"cv-agent"`.

- [ ] **Step 5: Build the trusted release using the real `.env` without displaying it**

```bash
./scripts/build_windows_release.sh
```

Expected: the script prints only the generated directory and ZIP paths. It must not print any environment value.

- [ ] **Step 6: Inspect the release contents without displaying the secret**

Resolve the newest generated ZIP without writing a placeholder path, then inspect it:

```bash
release_archive="$(find dist -maxdepth 1 -type f -name '油菜考种工作台-Windows版-*.zip' -print | sort | tail -n 1)"
test -n "$release_archive"
release_dir="${release_archive%.zip}"
test -d "$release_dir"
unzip -l "$release_archive"
```

Verify the archive contains the numbered files, `resources/app`, scripts, Compose, Dockerfile, requirements, `.dockerignore`, the PDF, empty `user-data` directories, and exactly one `resources/config/deployment.env`. Verify it excludes `.git`, `venv`, `.venv`, logs, results, benchmark backups, datasets, existing uploads, and experimental artifacts.

Read the real key into a shell variable without printing it, and fail if it appears in tracked files:

```bash
release_secret="$(awk -F= '$1 == "ARK_API_KEY" { print substr($0, index($0, "=") + 1); exit }' .env)"
test -n "$release_secret"
if git grep -lF -- "$release_secret"; then
  echo "Secret leaked into tracked content." >&2
  exit 1
fi
if rg -lF -- "$release_secret" "$release_dir" -g '!deployment.env'; then
  echo "Secret leaked outside deployment.env in the release directory." >&2
  exit 1
fi

docker build -t cv-agent:windows-release-verification "$release_dir/resources"
if docker history --no-trunc cv-agent:windows-release-verification | grep -Fq -- "$release_secret"; then
  echo "Secret leaked into Docker image history." >&2
  exit 1
fi
docker run --rm --entrypoint sh cv-agent:windows-release-verification \
  -c 'test ! -e /app/config/deployment.env'
unset release_secret
```

Expected: neither source files, other release files, Docker history, nor the image filesystem expose the real API key; only `resources/config/deployment.env` contains it.

- [ ] **Step 7: Verify repository state and commit any final non-generated fixes**

```bash
git status --short
git diff --check
git log --oneline --decorate -12
```

Expected: only ignored `dist/` artifacts are generated; no uncommitted tracked changes remain. If verification required a source fix, implement it test-first, rerun the affected checks, and commit that focused fix before continuing.

- [ ] **Step 8: Run the real Windows acceptance checklist before external delivery**

On a Windows 10 22H2 x64 or Windows 11 23H2+ x64 computer:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\deployment\windows\tests\run-unit-tests.ps1
```

Then complete every checkbox in `deployment/windows/WINDOWS-ACCEPTANCE.md`. Do not mark Windows deployment fully validated until this evidence exists. If a Windows machine is unavailable, hand off the ZIP as “locally verified; Windows acceptance pending” and list the unchecked items.

---

## Plan self-review coverage

| Specification requirement | Implemented by |
| --- | --- |
| Source-built Windows Docker deployment | Tasks 1, 5, 9, 11 |
| Numbered double-click entry points | Tasks 4–7 |
| Preconfigured trusted environment | Tasks 2, 5, 9 |
| Secret excluded from Git, logs, diagnostics, and image context | Tasks 1, 2, 7, 9, 11 |
| Persistent `user-data` and idempotency | Tasks 1, 5, 6, 10, 11 |
| Docker/Compose/port/disk checks and D001–D099 messaging | Tasks 2, 3, 5 |
| Docker Desktop PDF tutorial | Task 8 |
| Sanitized diagnostics ZIP | Task 7 |
| Maintainer documentation | Task 10 |
| Automated macOS checks and real Windows acceptance | Tasks 2, 10, 11 |

The plan intentionally does not add Docker auto-installation, remote secret proxying, offline images, automatic port selection, destructive uninstall, or database migrations.
