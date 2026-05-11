# ValveLens Data Sources

This folder is for raw dataset archives, extracted source data, and small manifests. The large files are not committed. They are kept locally under `downloads/` and `extracted/`, then converted into the project formats used by the backend and training scripts.

Processed data is stored elsewhere:

- zone embeddings and sqlite records: `backend/data/`
- combined YOLO detector dataset: `data/detection/combined/`
- expanded industrial detector staging: `data/detection/expanded_industrial/`
- focused oil/gas detector staging: `data/detection/oilgas_expanded/`
- detector training outputs: `runs/` and `artifacts/`
- enrolled device references: `backend/data/devices/`

For a compact dataset inventory, see `data_sources/DATASET_REGISTRY.md`.

## Local Datasets Present

The current workspace has these source archives in `data_sources/downloads/`:

- `nyc_indoor_vpr_indoor_anony1.zip`
- `corridor.zip`
- `office.zip`
- `station.zip`
- `Valve Detection.v1i.yolov8.zip`
- `Valve Detection.v6i.yolov8.zip`
- ExDARK files under `exdark/`

The matching extracted folders are:

- `indoor_anony`
- `corridor`
- `office`
- `station`
- `valve_detection_v1i_yolov8`
- `valve_detection_v6i_yolov8`
- `exdark`

## Zone Data

Zone retrieval uses public indoor-place data as a stand-in for industrial areas.

NYC-Indoor-VPR
- Archive: `nyc_indoor_vpr_indoor_anony1.zip`
- Extracted folder: `data_sources/extracted/indoor_anony`
- Import path in code: `scan_nyc_indoor_vpr`
- Zone naming: `NYCIndoorVPR_<scene_folder_name>`
- Usual import limit: `--max_per_zone 300`

OpenLORIS location folders
- Archives: `corridor.zip`, `office.zip`, `station.zip`
- Extracted folders: `data_sources/extracted/corridor`, `office`, `station`
- Project importer: `import_openloris_zones`
- Zone naming examples:
  - `OpenLORIS_corridor_000`
  - `OpenLORIS_office_001`
  - `OpenLORIS_station_008`

Import commands:

```powershell
cd d:\python_works\ValveLens\backend
python -m app.cli.import_zones_from_datasets --dataset nyc_indoor_vpr --root ..\data_sources\extracted\indoor_anony --max_per_zone 300 --rebuild
python -m app.cli.import_openloris_zones --root "D:\python_works\ValveLens\data_sources\extracted" --max_per_zone 300 --rebuild
```

Smoke tests:

```powershell
cd d:\python_works\ValveLens\backend
python -m app.cli.smoke_zones --image "D:\python_works\ValveLens\data_sources\extracted\corridor\000\000.png" --topk 5
python -m app.cli.smoke_zones_aggregate --image "D:\python_works\ValveLens\data_sources\extracted\corridor\000\000.png" --topk 5
```

## Detection Data

The detector training data comes from two YOLO-format valve/gauge archives:

- `Valve Detection.v1i.yolov8.zip`
- `Valve Detection.v6i.yolov8.zip`

The extracted folders are:

- `data_sources/extracted/valve_detection_v1i_yolov8`
- `data_sources/extracted/valve_detection_v6i_yolov8`

Source metadata from the extracted `data.yaml` files:

`Valve Detection.v1i.yolov8.zip`
- Roboflow workspace: `valve-robotics-computer-vision`
- Roboflow project: `valve-detection-xcwdr`
- Version: `1`
- License: `CC BY 4.0`
- Original classes:
  - `Lever Handle Butterfly Valve`
  - `Lever Handle Valve Flanged`
  - `Lever Handle Valve Threaded`
  - `Wheel Handle Butterfly Valve`
  - `Wheel Handle Valve Flanged`
  - `Wheel Handle Valve Threaded`
  - `Wheel Handle Valve Y-Type Flanged`
  - `Wheel Handle Valve Y-Type Threaded`

`Valve Detection.v6i.yolov8.zip`
- Roboflow workspace: `egh455-7trcz`
- Roboflow project: `valve-detection-q2w4n`
- Version: `6`
- License: `CC BY 4.0`
- Original classes:
  - `gauge-pin-centre`
  - `gauge-pin-end`
  - `gauge-pin-end2`
  - `gauge-value-2`
  - `gauge-value-4`
  - `gauge-value-6`
  - `gauge-value-8`
  - `valve-closed`
  - `valve-open`

The preparation script merges both archives into one two-class dataset:

- `0: valve`
- `1: gauge`

Build and inspect the combined dataset:

```powershell
cd d:\python_works\ValveLens
python scripts\prepare_combined_detection_dataset.py
python scripts\inspect_combined_detection_dataset.py
```

Train and verify the detector:

```powershell
python scripts\train_baseline_detector.py --model yolov8n.pt --epochs 30 --imgsz 640 --device 0 --name valvelens_v1_cuda --copy-best
python scripts\evaluate_detector.py --weights models\detector.pt --data artifacts\detection_training\combined_ultralytics.yaml --split test
python scripts\check_backend_detector_integration.py
```

The backend expects the runtime detector here:

```text
models/detector.pt
```

`models/` is ignored by git, so this weight file needs to be restored locally when setting up another machine.

## Device Reference Data

Device references are not a downloaded public dataset. They are local enrollment images used for the v0.3 identity demo.

Recommended first IDs:

- `V-1023`
- `V-2040`
- `PG-45`

Capture 3 to 10 reference images per device. Keep the framing close to the detector crop. For OCR tests, include a clear printed tag in at least some images.

Example local layout:

```text
D:\data\devices\V-1023\
D:\data\devices\V-2040\
D:\data\devices\PG-45\
```

Enrollment commands:

```powershell
cd d:\python_works\ValveLens\backend
python -m app.cli.create_device --device_id "V-1023" --zone_id "<ZONE_ID>" --type valve --desc "Printed test valve"
python -m app.cli.add_device_refs --device_id "V-1023" --folder "D:\data\devices\V-1023"
python -m app.cli.rebuild_device_index
python -m app.cli.smoke_reid --image "D:\data\devices\V-1023\sample.jpg" --topk 5
```

After enrollment, `/debug/status` should show non-zero values for `devices_count`, `device_refs_count`, and `device_faiss_size`.

## ExDARK Low-Light Data

ExDARK is available locally for qualitative low-light preprocessing tests:

- downloaded folder: `data_sources/downloads/exdark`
- extracted folder used by scripts: `data_sources/extracted/exdark`
- image count in the current workspace: `7363`

The current folders are organized by object category, for example `Bicycle`, `Boat`, `Bottle`, `Bus`, `Car`, `Cat`, `Chair`, `Cup`, `Dog`, `Motorbike`, `People`, and `Table`.

Use this dataset for visual low-light enhancement examples and discussion. It is not used for valve/gauge detection metrics unless its annotations are parsed and mapped into the ValveLens label space later.

## Hydraulic Components Detection

Hydraulic Components Detection was staged as a small optional Roboflow dataset for expanded industrial detection experiments. It is not the main graduation-project dataset path now. Do not merge it into the current valve/gauge detector dataset and do not overwrite `models/detector.pt`.

Source:

```text
https://universe.roboflow.com/nattapat-kieuvongngam-rup1a/hydraulic-components-detection
```

Local staging folders:

```text
data_sources/downloads/roboflow/hydraulic_components/
data_sources/extracted/roboflow/hydraulic_components/
data/detection/expanded_industrial/hydraulic_components/
```

Prepare after downloading the YOLOv8 export:

```powershell
cd d:\python_works\ValveLens
python .\scripts\prepare_hydraulic_components_dataset.py --overwrite
```

If Roboflow is configured locally, the script can try automatic staging:

```powershell
python .\scripts\prepare_hydraulic_components_dataset.py --download --overwrite
```

This dataset is only for expanded detector experiments. It is not identity validation data.

## Final Oil/Gas Expanded Dataset Path

The focused graduation-project expansion is now tracked in:

```text
data_sources/DATASET_REGISTRY.md
```

The selected staging targets are:

- Oil Refinery
- Elementos Offshore
- Object_detection_dataset by Anto, for WellHead/gauge/relay/valve
- industrial-multilabel, selected classes only

Local staging folders:

```text
data_sources/downloads/roboflow/oil_refinery/
data_sources/extracted/roboflow/oil_refinery/
data_sources/downloads/roboflow/elementos_offshore/
data_sources/extracted/roboflow/elementos_offshore/
data_sources/downloads/roboflow/wellhead_valve_gauge/
data_sources/extracted/roboflow/wellhead_valve_gauge/
data_sources/downloads/roboflow/industrial_multilabel/
data_sources/extracted/roboflow/industrial_multilabel/
```

Prepared outputs:

```text
data/detection/oilgas_expanded/oil_refinery/
data/detection/oilgas_expanded/elementos_offshore/
data/detection/oilgas_expanded/wellhead_valve_gauge/
data/detection/oilgas_expanded/industrial_multilabel/
```

Inspection commands:

```powershell
cd d:\python_works\ValveLens
python .\scripts\prepare_oilgas_expanded_dataset.py --dataset oil_refinery --overwrite
python .\scripts\prepare_oilgas_expanded_dataset.py --dataset elementos_offshore --overwrite
python .\scripts\prepare_oilgas_expanded_dataset.py --dataset wellhead_valve_gauge --overwrite
python .\scripts\prepare_oilgas_expanded_dataset.py --dataset industrial_multilabel --overwrite --max-images-per-class 200
```

This path is for detection only. It is not identity validation, not OCR validation, and not a reason to retrain before inspection.

## Storage Rules

Do not commit:

- files in `data_sources/downloads/`
- files in `data_sources/extracted/`
- generated FAISS indices
- trained weights
- generated ROI crops
- full training runs

Commit only the scripts, small manifests, documentation, and compact experiment summaries that are needed to reproduce the work.

## Troubleshooting

If a zone import returns no images, check the folder level first. Most import failures happen because the command points one directory too high or too low.

If detector preparation fails, confirm both `Valve Detection...zip` archives are present in `data_sources/downloads/` and extracted under `data_sources/extracted/`.

If ReID returns no device matches, check `/debug/status`. The usual cause is that no device references have been enrolled or the device FAISS index has not been rebuilt.
