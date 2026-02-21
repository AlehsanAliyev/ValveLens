param(
  [Parameter(Mandatory=$true)]
  [string]$Url,
  [string]$OutName = "openloris_location.zip"
)

$root = Resolve-Path "."
$downloads = Join-Path $root "data_sources\downloads"
$extracted = Join-Path $root "data_sources\extracted\openloris_location"

New-Item -ItemType Directory -Force -Path $downloads | Out-Null
New-Item -ItemType Directory -Force -Path $extracted | Out-Null

$outPath = Join-Path $downloads $OutName

Write-Host "Downloading OpenLORIS-Location to $outPath"
try {
  Invoke-WebRequest -Uri $Url -OutFile $outPath -UseBasicParsing
} catch {
  Write-Host "Download failed. You can manually place the archive in $downloads"
  exit 1
}

if ((Get-Item $outPath).Length -le 0) {
  Write-Host "Downloaded file is empty."
  exit 1
}

Write-Host "Extracting to $extracted"
if ($outPath.EndsWith(".zip")) {
  Expand-Archive -Path $outPath -DestinationPath $extracted -Force
} elseif ($outPath.EndsWith(".tar.gz") -or $outPath.EndsWith(".tgz")) {
  tar -xzf $outPath -C $extracted
} else {
  Write-Host "Unknown archive type. Extract manually to $extracted"
}

Write-Host "Done."
