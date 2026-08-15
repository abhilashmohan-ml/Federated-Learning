$root = "D:\viral_fl_project"
$venv = "$root\.venv\Scripts\Activate.ps1"

# -- Free project ports before starting --------------------------------------
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
# ---------------------------------------------------------------------------


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

# -- Servers -----------------------------------------------------------------
$siteStatusUrls = "site_1:http://localhost:9001,site_2:http://localhost:9002,site_3:http://localhost:9003,site_4:http://localhost:9004,site_5:http://localhost:9005"
Start-Pane -Title "Server"     -Command "`$env:DEV_MODE='true'; `$env:SITE_STATUS_URLS='$siteStatusUrls'; python server/main.py"    -BgColor "DarkBlue"
Start-Sleep -Seconds 2

Start-Pane -Title "Server GUI" -Command "python server/ui/app.py"  -BgColor "DarkCyan"
Start-Sleep -Seconds 1

# -- Read per-site secrets from .env ------------------------------------------------
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
# ---------------------------------------------------------------------------

# -- Per-site physics vars — creates inter-site variance in dev-mode simulation --------
$devPhysics = @{
    "site_1" = "`$env:DEV_J0='150'; `$env:DEV_K1='0.015'; `$env:DEV_K2='0.002'"
    "site_2" = "`$env:DEV_J0='130'; `$env:DEV_K1='0.018'; `$env:DEV_K2='0.003'"
    "site_3" = "`$env:DEV_J0='170'; `$env:DEV_K1='0.012'; `$env:DEV_K2='0.0015'"
    "site_4" = "`$env:DEV_J0='145'; `$env:DEV_K1='0.020'; `$env:DEV_K2='0.0025'"
    "site_5" = "`$env:DEV_J0='160'; `$env:DEV_K1='0.014'; `$env:DEV_K2='0.0018'"
}

# -- Clients -----------------------------------------------------------------
foreach ($i in 1..5) {
    $site = "site_$i"
    $physics = $devPhysics[$site]
    $secret  = $siteSecrets[$site]
    Start-Pane -Title "Site $i" `
               -Command "$physics; `$env:DEV_MODE='true'; `$env:SITE_ID='$site'; `$env:SITE_SECRET='$secret'; `$env:FLET_CLIENT_PORT='$((8550+$i))'; `$env:CLIENT_STATUS_PORT='$((9000+$i))'; python client/main.py" `
               -BgColor "DarkGreen"
    Start-Sleep -Milliseconds 500
}

Write-Host "All 7 windows launched (DEV_MODE - synthetic data, no CSV files needed)." -ForegroundColor Green
