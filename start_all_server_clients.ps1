# start_all_server_clients.ps1
# Production launcher — no DEV_MODE. Sites read real CSV data from LOCAL_DATA_PATH.
$root = "D:\viral_fl_project"
$venv = "$root\.venv\Scripts\Activate.ps1"

# -- Free project ports before starting -----------------------------------------
$ports = @(8000, 8550, 8551, 8552, 8553, 8554, 8555, 9001, 9002, 9003, 9004, 9005)
Write-Host "Checking project ports..." -ForegroundColor Yellow
$portPids = $ports | ForEach-Object {
    Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue
} | Select-Object -ExpandProperty OwningProcess -Unique

foreach ($id in $portPids) {
    $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
    $name = if ($proc) { $proc.Name } else { "unknown" }
    Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    Write-Host "  Freed port - killed PID $id ($name)" -ForegroundColor Yellow
}
if (-not $portPids) { Write-Host "  All ports free." -ForegroundColor Gray }
Start-Sleep -Seconds 1
# -------------------------------------------------------------------------------

function Start-Pane {
    param(
        [string]$Title,
        [string]$Command,
        [string]$BgColor
    )
    $setup = "`$host.UI.RawUI.WindowTitle = '$Title'; " +
             "`$host.UI.RawUI.BackgroundColor = '$BgColor'; " +
             "Clear-Host; "
    Start-Process powershell -ArgumentList "-NoExit", "-Command", `
        "cd '$root'; & '$venv'; $setup $Command"
}

# -- Server ---------------------------------------------------------------------
Start-Pane -Title "Server"     -Command "python server/main.py"   -BgColor "DarkBlue"
Start-Sleep -Seconds 2

Start-Pane -Title "Server GUI" -Command "python server/ui/app.py" -BgColor "DarkCyan"
Start-Sleep -Seconds 1

# -- Clients (PRODUCTION mode: no DEV_MODE, reads real CSV files) ---------------
# Adjust SITE_ID and LOCAL_DATA_PATH per your deployment environment.
foreach ($i in 1..5) {
    $site = "site_$i"
    Start-Pane -Title "Site $i" `
               -Command "`$env:SITE_ID='$site'; `$env:FLET_CLIENT_PORT='$((8550+$i))'; `$env:CLIENT_STATUS_PORT='$((9000+$i))'; python client/main.py" `
               -BgColor "DarkMagenta"
    Start-Sleep -Milliseconds 500
}

Write-Host "All 7 windows launched (PRODUCTION mode — reads real CSV data)." -ForegroundColor Cyan
