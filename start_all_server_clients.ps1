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

# -- Read per-site secrets from .env (must happen before init_db and client launch) --
$envVars = @{}
Get-Content "$root\.env" | ForEach-Object {
    if ($_ -match '^([^#=\s][^=]*)=(.*)$') {
        $envVars[$matches[1].Trim()] = $matches[2].Trim()
    }
}
$siteSecrets = @{
    "site_1" = $envVars["SITE_1_SECRET"]
    "site_2" = $envVars["SITE_2_SECRET"]
    "site_3" = $envVars["SITE_3_SECRET"]
    "site_4" = $envVars["SITE_4_SECRET"]
    "site_5" = $envVars["SITE_5_SECRET"]
}
# -------------------------------------------------------------------------------

# -- Initialise DB and register sites (idempotent — safe on every start) --------
Write-Host "Initialising database..." -ForegroundColor Yellow
$env:REGISTERED_SITES  = $envVars["REGISTERED_SITES"]
$env:SERVER_DB_URL     = $envVars["SERVER_DB_URL"]
$env:SERVER_SECRET_KEY = $envVars["SERVER_SECRET_KEY"]
& "$root\.venv\Scripts\python.exe" scripts/init_db.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: init_db.py failed. Verify REGISTERED_SITES in .env." -ForegroundColor Red
    exit 1
}
Write-Host "  DB ready." -ForegroundColor Green
# -------------------------------------------------------------------------------

# -- Server ---------------------------------------------------------------------
Start-Pane -Title "Server"     -Command "python server/main.py"   -BgColor "DarkBlue"
Start-Sleep -Seconds 2

Start-Pane -Title "Server GUI" -Command "python server/ui/app.py" -BgColor "DarkCyan"
Start-Sleep -Seconds 1

# -- Clients (PRODUCTION mode: no DEV_MODE, reads real CSV files) ---------------
# Adjust SITE_ID and LOCAL_DATA_PATH per your deployment environment.
foreach ($i in 1..5) {
    $site = "site_$i"
    $secret = $siteSecrets[$site]
    Start-Pane -Title "Site $i" `
               -Command "`$env:SITE_ID='$site'; `$env:SITE_SECRET='$secret'; `$env:FLET_CLIENT_PORT='$((8550+$i))'; `$env:CLIENT_STATUS_PORT='$((9000+$i))'; python client/main.py" `
               -BgColor "DarkMagenta"
    Start-Sleep -Milliseconds 500
}

Write-Host "All 7 windows launched (PRODUCTION mode — reads real CSV data)." -ForegroundColor Cyan
