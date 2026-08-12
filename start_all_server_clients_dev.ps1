$root = "D:\viral_fl_project"
$venv = "$root\.venv\Scripts\Activate.ps1"

# -- Free project ports before starting --------------------------------------
$ports = @(8000, 8550, 8551, 8552, 8553, 8554, 8555)
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
Start-Pane -Title "Server"     -Command "python server/main.py"    -BgColor "DarkBlue"
Start-Sleep -Seconds 2

Start-Pane -Title "Server GUI" -Command "python server/ui/app.py"  -BgColor "DarkCyan"
Start-Sleep -Seconds 1

# -- Clients -----------------------------------------------------------------
foreach ($i in 1..5) {
    $site = "site_$i"
    Start-Pane -Title "Site $i" `
               -Command "`$env:SITE_ID='$site'; python client/main.py" `
               -BgColor "DarkGreen"
    Start-Sleep -Milliseconds 500
}

Write-Host "All 7 windows launched." -ForegroundColor Green
