param(
  [switch]$DownloadNYCIndoorVPR,
  [switch]$DownloadOpenLORISLocation,
  [switch]$DownloadCOLDSubset,
  [switch]$All,
  [string]$OpenLORISUrl = ""
)

$root = Resolve-Path "."
$downloads = Join-Path $root "data_sources\downloads"
$extracted = Join-Path $root "data_sources\extracted"
$manifests = Join-Path $root "data_sources\manifests"

New-Item -ItemType Directory -Force -Path $downloads | Out-Null
New-Item -ItemType Directory -Force -Path $extracted | Out-Null
New-Item -ItemType Directory -Force -Path $manifests | Out-Null

function Download-File($url, $outPath) {
  try {
    Write-Host "Downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $outPath -UseBasicParsing
    if ((Get-Item $outPath).Length -gt 0) {
      Write-Host "Saved: $outPath"
      return $true
    }
  } catch {
    Write-Host "Download failed: $url"
  }
  return $false
}

if ($All) {
  $DownloadNYCIndoorVPR = $true
  $DownloadOpenLORISLocation = $true
  $DownloadCOLDSubset = $true
}

if ($DownloadNYCIndoorVPR) {
  $nycUrl = "https://huggingface.co/datasets/ai4ce/NYC-Indoor-VPR-Data/resolve/main/indoor_anony1.zip?download=1"
  $nycZip = Join-Path $downloads "nyc_indoor_vpr_indoor_anony1.zip"
  $ok = Download-File $nycUrl $nycZip
  if (-not $ok) {
    Write-Host "Manual download required. Place the zip in $downloads and re-run."
  }
}

if ($DownloadOpenLORISLocation) {
  if ($OpenLORISUrl -ne "") {
    $openLorisZip = Join-Path $downloads "openloris_location.zip"
    $ok = Download-File $OpenLORISUrl $openLorisZip
    if (-not $ok) {
      Write-Host "Manual download required. Place the archive in $downloads and extract to data_sources\extracted\openloris_location"
    }
  } else {
    Write-Host "OpenLORIS-Location requires a direct URL or manual download."
    Write-Host "Pass -OpenLORISUrl <direct_zip_url> or place the archive in $downloads and extract to data_sources\extracted\openloris_location"
  }
}

if ($DownloadCOLDSubset) {
  Write-Host "COLD subset downloads must be configured in data_sources\manifests\cold_sequences.json"
  Write-Host "Paste URLs and run the Python importer to download/extract."
}
