<# 
.SYNOPSIS
    Speechos Launcher for Windows
.DESCRIPTION
    Detects hardware and launches the appropriate Docker stack.
.EXAMPLE
    .\start.ps1              # Auto-detect
    .\start.ps1 -Mode cpu    # Force CPU-only
    .\start.ps1 -Mode gpu    # Force GPU
    .\start.ps1 -Tier gpu-32gb  # Force tier
    .\start.ps1 -Dev         # Development mode (no Docker)
#>

param(
    [ValidateSet("auto", "cpu", "gpu")]
    [string]$Mode = "auto",
    
    [string]$Tier = "",

    [switch]$Dev
)

$ErrorActionPreference = "Stop"

Write-Host "╔══════════════════════════════════════╗" -ForegroundColor Blue
Write-Host "║         Speechos Launcher            ║" -ForegroundColor Blue
Write-Host "╚══════════════════════════════════════╝" -ForegroundColor Blue
Write-Host ""

# Detect GPU
function Test-GpuAvailable {
    try {
        $gpu = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
        if ($gpu) {
            $parts = $gpu -split ","
            $gpuName = $parts[0].Trim()
            $gpuVram = $parts[1].Trim()
            Write-Host "GPU detected: $gpuName ($gpuVram)" -ForegroundColor Green
            return $true
        }
    } catch {}
    Write-Host "No GPU detected" -ForegroundColor Yellow
    return $false
}

# Detect RAM
$ramGb = [math]::Round((Get-CimInstance -ClassName Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Write-Host "System RAM: ${ramGb} GB" -ForegroundColor Green

# Determine compose file
$gpuAvailable = Test-GpuAvailable

if ($Mode -eq "auto") {
    if ($gpuAvailable) {
        $composeFile = "docker/docker-compose.gpu.yml"
        Write-Host "Mode: GPU (auto-detected)" -ForegroundColor Green
    } else {
        $composeFile = "docker/docker-compose.cpu.yml"
        Write-Host "Mode: CPU-only (auto-detected)" -ForegroundColor Green
    }
} elseif ($Mode -eq "gpu") {
    if ($gpuAvailable) {
        $composeFile = "docker/docker-compose.gpu.yml"
        Write-Host "Mode: GPU (forced)" -ForegroundColor Green
    } else {
        Write-Host "ERROR: GPU mode requested but no GPU found" -ForegroundColor Red
        exit 1
    }
} else {
    $composeFile = "docker/docker-compose.cpu.yml"
    Write-Host "Mode: CPU-only (forced)" -ForegroundColor Green
}

if ($Tier) {
    Write-Host "Tier: $Tier (forced)" -ForegroundColor Green
    $env:SPEECHOS_TIER = $Tier
} else {
    Write-Host "Tier: auto-detect at startup" -ForegroundColor Green
}

Write-Host ""

if ($Dev) {
    Write-Host "Starting in development mode..." -ForegroundColor Blue
    Write-Host ""

    # Create directories
    New-Item -ItemType Directory -Force -Path models, recordings, samples | Out-Null

    # Start API
    Write-Host "Starting API server..." -ForegroundColor Yellow
    $apiJob = Start-Job -ScriptBlock {
        Set-Location $using:PSScriptRoot\api
        uv run python -m src.server
    }

    # Start Web
    Write-Host "Starting web frontend..." -ForegroundColor Yellow
    $webJob = Start-Job -ScriptBlock {
        Set-Location $using:PSScriptRoot\web
        pnpm dev
    }

    Write-Host ""
    Write-Host "Speechos running in dev mode:" -ForegroundColor Green
    Write-Host "  Web:  http://localhost:36301"
    Write-Host "  API:  http://localhost:36300"
    Write-Host "  Docs: http://localhost:36300/docs"
    Write-Host ""
    Write-Host "Press Ctrl+C to stop"

    try {
        while ($true) {
            Start-Sleep -Seconds 1
            # Show any output
            Receive-Job $apiJob -ErrorAction SilentlyContinue
            Receive-Job $webJob -ErrorAction SilentlyContinue
        }
    } finally {
        Stop-Job $apiJob, $webJob -ErrorAction SilentlyContinue
        Remove-Job $apiJob, $webJob -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "Starting Docker containers..." -ForegroundColor Blue
    Write-Host ""

    # Create directories
    New-Item -ItemType Directory -Force -Path models, recordings, samples | Out-Null

    docker compose -f $composeFile up --build -d

    Write-Host ""
    Write-Host "Speechos is running:" -ForegroundColor Green
    Write-Host "  App:  http://localhost (via nginx)"
    Write-Host "  Web:  http://localhost:36301 (direct)"
    Write-Host "  API:  http://localhost:36300 (direct)"
    Write-Host "  Docs: http://localhost:36300/docs (Swagger)"
    Write-Host ""
    Write-Host "Logs: docker compose -f $composeFile logs -f"
    Write-Host "Stop: docker compose -f $composeFile down"
}
