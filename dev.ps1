<#
.SYNOPSIS
  Start Speechos API + Web dev servers with proper process tree cleanup.
.PARAMETER ApiOnly
  Start only the API server.
.PARAMETER WebOnly
  Start only the Web server.
#>
param(
    [switch]$ApiOnly,
    [switch]$WebOnly
)

[System.Collections.ArrayList]$script:childPids = @()

function Clear-Ports {
    foreach ($port in @(36300, 36301)) {
        Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
            ForEach-Object {
                Write-Host "[dev] Killing PID $($_.OwningProcess) on port $port" -ForegroundColor Yellow
                taskkill /PID $_.OwningProcess /T /F 2>$null | Out-Null
            }
    }
}

function Stop-DevProcesses {
    foreach ($cpid in $script:childPids) {
        # /T = kill entire process tree, /F = force
        & taskkill /PID $cpid /T /F 2>$null | Out-Null
    }
    if ($script:childPids.Count -gt 0) {
        Write-Host "`n[dev] Killed process trees: $($script:childPids -join ', ')" -ForegroundColor Yellow
    }
}

try {
    Clear-Ports

    # Load .env file if present
    $envFile = Join-Path $PSScriptRoot ".env"
    $envVars = ""
    if (Test-Path $envFile) {
        Get-Content $envFile | Where-Object { $_ -match '^\s*[A-Z_]+=.+' } | ForEach-Object {
            $envVars += " && set $($_.Trim())"
        }
        Write-Host "[dev] Loaded .env" -ForegroundColor DarkGray
    }

    if (-not $WebOnly) {
        $apiProc = Start-Process cmd.exe -ArgumentList "/c cd /d `"$PSScriptRoot\api`" && set HF_HUB_DISABLE_SYMLINKS_WARNING=1$envVars && uv run python -m src" `
            -NoNewWindow -PassThru
        [void]$script:childPids.Add($apiProc.Id)
        Write-Host "[dev] API starting on http://localhost:36300 (PID $($apiProc.Id))" -ForegroundColor Blue
    }

    if (-not $ApiOnly) {
        $webProc = Start-Process cmd.exe -ArgumentList "/c cd /d `"$PSScriptRoot\web`" && pnpm dev" `
            -NoNewWindow -PassThru
        [void]$script:childPids.Add($webProc.Id)
        Write-Host "[dev] Web starting on http://localhost:36301 (PID $($webProc.Id))" -ForegroundColor Green
    }

    Write-Host "[dev] Press Ctrl+C to stop all.`n" -ForegroundColor Cyan

    while ($true) {
        foreach ($cpid in $script:childPids) {
            if (-not (Get-Process -Id $cpid -ErrorAction SilentlyContinue)) {
                Write-Host "[dev] Process $cpid exited unexpectedly." -ForegroundColor Red
                Stop-DevProcesses
                exit 1
            }
        }
        Start-Sleep -Milliseconds 500
    }
} finally {
    Stop-DevProcesses
}
