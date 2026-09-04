$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*#" -or $_ -match "^\s*$") { return }
        $parts = $_.Split("=", 2)
        if ($parts.Count -eq 2) {
            [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim().Trim('"'))
        }
    }
}
python -m uv run python -m embedbench.benchmark --config configs/models.yaml @args
