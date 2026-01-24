param(
  [int]$MaxSequences = 2,
  [string]$Manifest = "data_sources\manifests\cold_sequences.json"
)

$root = Resolve-Path "."
$backend = Join-Path $root "backend"

Write-Host "Downloading COLD subset using manifest: $Manifest"
Push-Location $backend
python -m app.cli.download_cold_subset --manifest "..\$Manifest" --max_sequences $MaxSequences
Pop-Location
