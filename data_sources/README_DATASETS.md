# Dataset-Only Zones (Option B)

This project supports dataset-only zone creation so you do not need home photos. The pipeline downloads public datasets, scans their folder structure, creates zones, ingests keyframes, and rebuilds the FAISS zone index.

## Supported datasets

NYC-Indoor-VPR
- Folder structure: one folder per scene with images inside.
- Zone naming: NYCIndoorVPR_<scene_folder_name>
- Recommended max_per_zone: 300

OpenLORIS-Location
- Folder structure: Scene#/Location#/images
- Zone naming default: OpenLORISLoc_<Scene>_<Location>
- Optional: per-scene zones via --per_scene
- Recommended max_per_zone: 50

COLD (optional, large)
- Use a small subset of sequences.
- Each sequence folder is treated as a zone: COLD_<sequence_name>
- Configure sequence URLs in data_sources/manifests/cold_sequences.json
 - Download with: python -m app.cli.download_cold_subset

## Storage notes

These datasets can be large. Start with a single NYC-Indoor-VPR zip and a small OpenLORIS subset. COLD can be hundreds of MB per sequence, so keep the subset small.

## Troubleshooting

If downloads fail with 401/403, manually place the archive in data_sources/downloads and re-run the importer.

If zones are not found, confirm the extracted folder paths match the expected structure.
