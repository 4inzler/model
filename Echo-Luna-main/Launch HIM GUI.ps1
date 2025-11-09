# H.I.M. Model GUI Launcher for PowerShell
Write-Host "Starting H.I.M. Model GUI..." -ForegroundColor Green
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Launch the GUI
try {
    python him_gui.py
}
catch {
    Write-Host "Error launching GUI: $_" -ForegroundColor Red
    Write-Host "Make sure Python is installed and him_gui.py is in the current directory." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
