# post_dev_cleanup.ps1
# Kill processes on project ports, close dev terminals, clean Python cache
# Run from PowerShell: .\post_dev_cleanup.ps1

# ── 1. Kill processes on project ports ───────────────────────────────────────
$ports = @(8000, 8550, 8551, 8552, 8553, 8554, 8555)

Write-Host "Freeing ports..." -ForegroundColor Yellow

$portPids = $ports | ForEach-Object {
    Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue
} | Select-Object -ExpandProperty OwningProcess -Unique

if ($portPids) {
    foreach ($id in $portPids) {
        $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
        $name = if ($proc) { $proc.Name } else { "unknown" }
        Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
        Write-Host "  Killed PID $id ($name) on project port" -ForegroundColor Yellow
    }
} else {
    Write-Host "  No processes on project ports." -ForegroundColor Gray
}

# ── 2. Kill any remaining python / pythonw processes ────────────────────────
Write-Host "Stopping Python processes..." -ForegroundColor Yellow
Get-Process python  -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process pythonw -ErrorAction SilentlyContinue | Stop-Process -Force

# ── 3. Close dev terminal windows opened by start_all_server_clients_dev.ps1 ─
$titles = @("Server", "Server GUI", "Site 1", "Site 2", "Site 3", "Site 4", "Site 5")

Write-Host "Closing dev terminal windows..." -ForegroundColor Yellow
foreach ($proc in (Get-Process powershell, pwsh -ErrorAction SilentlyContinue)) {
    if ($proc.MainWindowTitle -in $titles) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "  Closed: $($proc.MainWindowTitle)" -ForegroundColor Yellow
    }
}

# ── 4. Clean Python cache / test artifacts ───────────────────────────────────
Write-Host "Cleaning cache files..." -ForegroundColor Yellow
$root = "D:\viral_fl_project"

Get-ChildItem $root -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notlike "*\.venv\*" } |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Get-ChildItem $root -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notlike "*\.venv\*" } |
    Remove-Item -Force -ErrorAction SilentlyContinue

Remove-Item "$root\.coverage"      -Force -ErrorAction SilentlyContinue
Remove-Item "$root\.pytest_cache"  -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$root\htmlcov"        -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "Done. Ports free, terminals closed, cache clean." -ForegroundColor Green
